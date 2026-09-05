"""
custom-aigw: unit tests for quota_sync (ADR-0004, ai-gateway-ops GH#5).

Payload parsing, sync-loop reconciliation (cooldown TTL / active unblock /
probe-failure changes nothing) and the wall-hit probe hook — all with mocked
httpx; no real keys anywhere.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm.router_utils.cooldown_cache import CooldownCache
from litellm.router_utils.quota_sync import (
    QUOTA_BLIND_COOLDOWN_SECONDS,
    QUOTA_PROBES,
    QUOTA_RESET_BUFFER_SECONDS,
    kimi_probe,
    match_probe,
    parse_kimi_usages,
    parse_reset_epoch,
    run_quota_sync_loop,
    sync_deployment_quota,
)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _payload(weekly: dict | None, five_hour: dict | None) -> dict:
    """Build a Kimi /v1/usages-shaped payload."""
    payload: dict = {"limited": True}
    if weekly is not None:
        payload["usage"] = weekly
    if five_hour is not None:
        payload["limits"] = [{"type": "rolling", "detail": five_hour}]
    return payload


NOW = time.time()
RESET_5H = NOW + 3600  # 5h window resets in 1h
RESET_WEEK = NOW + 7200  # weekly resets later


# ---------------------------------------------------------------------------
# payload parsing
# ---------------------------------------------------------------------------
class TestParseKimiUsages:
    def test_five_hour_exhausted(self):
        dims = parse_kimi_usages(
            _payload(
                weekly={"used": "10", "limit": "100", "remaining": "90", "resetTime": _iso(RESET_WEEK)},
                five_hour={"limit": "200", "used": "200", "remaining": "0", "resetTime": _iso(RESET_5H)},
            )
        )
        by_name = {d["name"]: d for d in dims}
        assert by_name["5h-window"]["exhausted"] is True
        assert by_name["5h-window"]["remaining"] == 0
        assert by_name["weekly"]["exhausted"] is False

    def test_weekly_exhausted_without_remaining_field(self):
        # weekly quota sometimes omits `remaining` -> fall back to limit-used<=0
        dims = parse_kimi_usages(
            _payload(
                weekly={"used": "100", "limit": "100", "resetTime": _iso(RESET_WEEK)},
                five_hour={"limit": "200", "used": "50", "remaining": "150", "resetTime": _iso(RESET_5H)},
            )
        )
        by_name = {d["name"]: d for d in dims}
        assert by_name["weekly"]["exhausted"] is True
        assert by_name["weekly"]["remaining"] == 0
        assert by_name["5h-window"]["exhausted"] is False

    def test_both_exhausted_takes_max_reset(self):
        dims = parse_kimi_usages(
            _payload(
                weekly={"used": "100", "limit": "100", "remaining": "0", "resetTime": _iso(RESET_WEEK)},
                five_hour={"limit": "200", "used": "200", "remaining": "0", "resetTime": _iso(RESET_5H)},
            )
        )
        assert len(dims) == 2
        assert all(d["exhausted"] for d in dims)
        # max-reset selection happens in the cooldown TTL computation
        from litellm.router_utils.quota_sync import _cooldown_ttl_from_dims

        ttl = _cooldown_ttl_from_dims(dims, now=NOW)
        assert ttl == pytest.approx(RESET_WEEK + QUOTA_RESET_BUFFER_SECONDS - NOW, abs=5)

    def test_none_exhausted(self):
        dims = parse_kimi_usages(
            _payload(
                weekly={"used": "1", "limit": "100", "remaining": "99", "resetTime": _iso(RESET_WEEK)},
                five_hour={"limit": "200", "used": "10", "remaining": "190", "resetTime": _iso(RESET_5H)},
            )
        )
        assert len(dims) == 2
        assert all(not d["exhausted"] for d in dims)

    def test_empty_and_malformed(self):
        assert parse_kimi_usages({}) == []
        assert parse_kimi_usages(None) == []
        assert parse_kimi_usages({"usage": {"resetTime": _iso(NOW)}}) == []

    def test_reset_time_parsing(self):
        assert (
            parse_reset_epoch("2026-08-17T12:00:00Z") == datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc).timestamp()
        )
        assert parse_reset_epoch(None) is None
        assert parse_reset_epoch("not-a-date") is None
        assert parse_reset_epoch(123) is None


# ---------------------------------------------------------------------------
# probe registry
# ---------------------------------------------------------------------------
class TestProbeRegistry:
    def test_registry_has_kimi(self):
        assert "api.kimi.com" in QUOTA_PROBES

    def test_match_probe(self):
        assert match_probe("https://api.kimi.com/coding") is kimi_probe
        assert match_probe("https://api.kimi.com:443/coding") is kimi_probe
        assert match_probe("https://dashscope.aliyuncs.com/compatible-mode/v1") is None
        assert match_probe(None) is None
        assert match_probe("") is None

    async def test_kimi_probe_calls_usages_endpoint(self):
        payload = _payload(
            weekly={"used": "0", "limit": "100", "remaining": "100", "resetTime": _iso(RESET_WEEK)},
            five_hour={"limit": "200", "used": "0", "remaining": "200", "resetTime": _iso(RESET_5H)},
        )

        captured = {}

        async def fake_get(self, url, headers=None, **kwargs):
            captured["url"] = url
            captured["headers"] = headers
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            dims = await kimi_probe("https://api.kimi.com/coding", "fake-key")

        assert captured["url"] == "https://api.kimi.com/coding/v1/usages"
        assert captured["headers"]["Authorization"] == "Bearer fake-key"
        assert len(dims) == 2

    async def test_kimi_probe_raises_on_http_error(self):
        async def fake_get(self, url, headers=None, **kwargs):
            return httpx.Response(401, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            with pytest.raises(httpx.HTTPStatusError):
                await kimi_probe("https://api.kimi.com/coding", "fake-key")


# ---------------------------------------------------------------------------
# router fixtures
# ---------------------------------------------------------------------------
def _make_router(num_deployments: int = 1, api_base: str = "https://api.kimi.com/coding"):
    model_list = [
        {
            "model_name": "k3",
            "litellm_params": {
                "model": "anthropic/k3",
                "api_key": f"fake-key-{i}",
                "api_base": api_base,
            },
        }
        for i in range(num_deployments)
    ]
    return litellm.Router(model_list=model_list, cooldown_time=60, num_retries=0)


def _get_cooldown_value(router, model_id):
    """Return the active CooldownCacheValue for model_id, or None."""
    active = router.cooldown_cache.get_active_cooldowns(model_ids=[model_id], parent_otel_span=None)
    return active[0][1] if active else None


# ---------------------------------------------------------------------------
# single-deployment sync (loop unit of work + wall-hit target)
# ---------------------------------------------------------------------------
class TestSyncDeploymentQuota:
    async def test_exhausted_sets_cooldown_to_max_reset_plus_buffer(self):
        router = _make_router()
        model_id = router.get_model_ids()[0]
        dims = parse_kimi_usages(
            _payload(
                weekly={"used": "100", "limit": "100", "remaining": "0", "resetTime": _iso(RESET_WEEK)},
                five_hour={"limit": "200", "used": "200", "remaining": "0", "resetTime": _iso(RESET_5H)},
            )
        )
        probe = AsyncMock(return_value=dims)
        with patch("litellm.router_utils.quota_sync.match_probe", return_value=probe):
            action = await sync_deployment_quota(router, model_id)
        assert action == "cooldown"
        assert probe.await_count == 1
        value = _get_cooldown_value(router, model_id)
        assert value is not None
        # TTL = max(resetTime) + 15min buffer (weekly reset is the later one)
        assert value["cooldown_time"] == pytest.approx(RESET_WEEK + QUOTA_RESET_BUFFER_SECONDS - time.time(), abs=5)

    async def test_recovered_actively_unblocks(self):
        router = _make_router()
        model_id = router.get_model_ids()[0]
        # pre-cool the deployment (as if a wall-hit placed the placeholder)
        router.cooldown_cache.add_deployment_to_cooldown(
            model_id=model_id,
            original_exception=Exception("placeholder"),
            exception_status=429,
            cooldown_time=3600,
        )
        assert _get_cooldown_value(router, model_id) is not None
        dims = parse_kimi_usages(
            _payload(
                weekly={"used": "1", "limit": "100", "remaining": "99", "resetTime": _iso(RESET_WEEK)},
                five_hour={"limit": "200", "used": "5", "remaining": "195", "resetTime": _iso(RESET_5H)},
            )
        )
        with patch("litellm.router_utils.quota_sync.match_probe", return_value=AsyncMock(return_value=dims)):
            action = await sync_deployment_quota(router, model_id)
        assert action == "unblock"
        # cooldown key deleted -> deployment back in the pool immediately
        assert _get_cooldown_value(router, model_id) is None

    async def test_recovered_but_not_cooling_is_ok(self):
        router = _make_router()
        model_id = router.get_model_ids()[0]
        dims = parse_kimi_usages(
            _payload(
                weekly={"used": "1", "limit": "100", "remaining": "99", "resetTime": _iso(RESET_WEEK)},
                five_hour={"limit": "200", "used": "5", "remaining": "195", "resetTime": _iso(RESET_5H)},
            )
        )
        with patch("litellm.router_utils.quota_sync.match_probe", return_value=AsyncMock(return_value=dims)):
            action = await sync_deployment_quota(router, model_id)
        assert action == "ok"
        assert _get_cooldown_value(router, model_id) is None

    async def test_probe_failure_changes_no_state(self):
        router = _make_router()
        model_id = router.get_model_ids()[0]
        router.cooldown_cache.add_deployment_to_cooldown(
            model_id=model_id,
            original_exception=Exception("placeholder"),
            exception_status=429,
            cooldown_time=3600,
        )
        before = _get_cooldown_value(router, model_id)
        with patch(
            "litellm.router_utils.quota_sync.match_probe",
            return_value=AsyncMock(side_effect=httpx.ConnectError("boom")),
        ):
            action = await sync_deployment_quota(router, model_id)
        assert action == "probe-failed"
        # neither unblocked nor re-cooled
        after = _get_cooldown_value(router, model_id)
        assert after is not None
        assert after["cooldown_time"] == before["cooldown_time"]

    async def test_empty_payload_changes_no_state(self):
        router = _make_router()
        model_id = router.get_model_ids()[0]
        with patch("litellm.router_utils.quota_sync.match_probe", return_value=AsyncMock(return_value=[])):
            action = await sync_deployment_quota(router, model_id)
        assert action == "probe-failed"
        assert _get_cooldown_value(router, model_id) is None

    async def test_non_matching_deployment_skipped(self):
        router = _make_router(api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_id = router.get_model_ids()[0]
        action = await sync_deployment_quota(router, model_id)
        assert action == "skipped"
        assert _get_cooldown_value(router, model_id) is None

    async def test_exhausted_without_reset_time_uses_blind_fallback(self):
        router = _make_router()
        model_id = router.get_model_ids()[0]
        dims = parse_kimi_usages(_payload(weekly={"used": "100", "limit": "100", "remaining": "0"}, five_hour=None))
        with patch("litellm.router_utils.quota_sync.match_probe", return_value=AsyncMock(return_value=dims)):
            action = await sync_deployment_quota(router, model_id)
        assert action == "cooldown"
        value = _get_cooldown_value(router, model_id)
        assert value["cooldown_time"] == QUOTA_BLIND_COOLDOWN_SECONDS


# ---------------------------------------------------------------------------
# background loop
# ---------------------------------------------------------------------------
class TestQuotaSyncLoop:
    async def test_loop_exits_immediately_without_matching_deployments(self):
        router = _make_router(api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
        # run_quota_sync_loop must return (not hang) — no probe-registry match
        await asyncio.wait_for(run_quota_sync_loop(router), timeout=5)

    async def test_loop_probes_all_matching_deployments_each_cycle(self):
        router = _make_router(num_deployments=2)
        dims = parse_kimi_usages(
            _payload(
                weekly={"used": "1", "limit": "100", "remaining": "99", "resetTime": _iso(RESET_WEEK)},
                five_hour={"limit": "200", "used": "5", "remaining": "195", "resetTime": _iso(RESET_5H)},
            )
        )
        probe = AsyncMock(return_value=dims)
        with patch("litellm.router_utils.quota_sync.match_probe", return_value=probe):
            task = asyncio.create_task(run_quota_sync_loop(router, sync_interval_seconds=3600))
            await asyncio.wait_for(_wait_for_probe_count(probe, 2), timeout=5)
            task.cancel()
        assert probe.await_count == 2
        for i in range(2):
            assert _get_cooldown_value(router, router.get_model_ids()[i]) is None

    async def test_loop_cooldown_ttl_uses_max_reset_plus_buffer(self):
        router = _make_router()
        model_id = router.get_model_ids()[0]
        dims = parse_kimi_usages(
            _payload(
                weekly={"used": "100", "limit": "100", "remaining": "0", "resetTime": _iso(RESET_WEEK)},
                five_hour={"limit": "200", "used": "200", "remaining": "0", "resetTime": _iso(RESET_5H)},
            )
        )
        probe = AsyncMock(return_value=dims)
        with patch("litellm.router_utils.quota_sync.match_probe", return_value=probe):
            task = asyncio.create_task(run_quota_sync_loop(router, sync_interval_seconds=3600))
            await asyncio.wait_for(_wait_for_probe_count(probe, 1), timeout=5)
            task.cancel()
        value = _get_cooldown_value(router, model_id)
        assert value is not None
        assert value["cooldown_time"] == pytest.approx(RESET_WEEK + QUOTA_RESET_BUFFER_SECONDS - time.time(), abs=5)

    async def test_loop_survives_probe_exception(self):
        router = _make_router(num_deployments=2)

        # first deployment raises, second must still be probed
        async def flaky(api_base, api_key):
            if api_key == "fake-key-0":
                raise RuntimeError("boom")
            return parse_kimi_usages(
                _payload(
                    weekly={"used": "1", "limit": "100", "remaining": "99", "resetTime": _iso(RESET_WEEK)},
                    five_hour={"limit": "200", "used": "5", "remaining": "195", "resetTime": _iso(RESET_5H)},
                )
            )

        with patch("litellm.router_utils.quota_sync.match_probe", return_value=flaky):
            task = asyncio.create_task(run_quota_sync_loop(router, sync_interval_seconds=3600))
            # deployment 1 recovered -> unblocked/ok means no cooldown
            for _ in range(100):
                await asyncio.sleep(0.01)
                if _get_cooldown_value(router, router.get_model_ids()[1]) is None:
                    break
            task.cancel()


async def _wait_for_probe_count(probe: AsyncMock, count: int):
    while probe.await_count < count:
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# cooldown-cache key shape (unblock path must delete the right key)
# ---------------------------------------------------------------------------
def test_unblock_deletes_cooldown_cache_key():
    router = _make_router()
    model_id = router.get_model_ids()[0]
    router.cooldown_cache.add_deployment_to_cooldown(
        model_id=model_id,
        original_exception=Exception("x"),
        exception_status=429,
        cooldown_time=3600,
    )
    key = CooldownCache.get_cooldown_cache_key(model_id)
    assert key == f"deployment:{model_id}:cooldown"
    assert router.cooldown_cache.cache.get_cache(key) is not None
