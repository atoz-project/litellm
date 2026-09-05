"""
custom-aigw: periodic-quota (Kimi Code) cooldown tests.

Production shape (verified via SLS 2026-08-12): the anthropic exception mapper
has no 403 branch, so Kimi's weekly-quota 403 falls through to a generic
APIConnectionError(status 500) with the upstream body embedded — and
APIConnectionError is on the no-cooldown veto list. The fork patch bypasses the
veto for marker-matched errors and cools the deployment immediately with a 1h
blind placeholder TTL (ADR-0004): the "billing cycle" wording is known to
mislead (it is usually just the 5h window), and a fire-and-forget usages probe
overwrites the placeholder with the authoritative resetTime + buffer. Plain
permission 403s and transient connection errors stay uncooled.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.router_utils.cooldown_handlers import (
    _is_cooldown_required,
    _is_periodic_quota_error,
    _set_cooldown_deployments,
)
from litellm.router_utils.quota_sync import QUOTA_BLIND_COOLDOWN_SECONDS

# exact shape seen in production logs (anthropic mapper 403 fallthrough)
_KIMI_QUOTA_WRAPPED = (
    "litellm.APIConnectionError: AnthropicException - "
    '{"error":{"type":"permission_error","message":"You\'ve reached your usage '
    "limit for this billing cycle. Your quota will be refreshed in the next cycle. "
    "To continue now, purchase extra usage or upgrade your plan: "
    'https://www.kimi.com/code/#pricing"},"type":"error"}'
)

# 429 shape for the 5h rolling window (ADR-0004: same treatment as billing-cycle)
_KIMI_QUOTA_429 = (
    "litellm.RateLimitError: AnthropicException - "
    '{"error":{"type":"rate_limit_error","message":"You\'ve reached your usage '
    'limit for this period. Please try again later."},"type":"error"}'
)


def test_periodic_quota_gate():
    # APIConnectionError-wrapped quota error: veto bypassed, cooldown allowed
    assert (
        _is_cooldown_required(
            litellm_router_instance=MagicMock(),
            model_id="dep-1",
            exception_status=500,
            exception_str=_KIMI_QUOTA_WRAPPED,
        )
        is True
    )
    # raw 403 carrying the marker (future mapper-fix shape) also opens the gate
    assert (
        _is_cooldown_required(
            litellm_router_instance=MagicMock(),
            model_id="dep-1",
            exception_status=403,
            exception_str=_KIMI_QUOTA_WRAPPED,
        )
        is True
    )
    # 429 carrying the 5h-window marker opens the gate too
    assert (
        _is_cooldown_required(
            litellm_router_instance=MagicMock(),
            model_id="dep-1",
            exception_status=429,
            exception_str=_KIMI_QUOTA_429,
        )
        is True
    )
    # transient connection error: veto intact, NO cooldown
    assert (
        _is_cooldown_required(
            litellm_router_instance=MagicMock(),
            model_id="dep-1",
            exception_status=500,
            exception_str="litellm.APIConnectionError: AnthropicException - connection reset by peer",
        )
        is False
    )
    # plain permission 403 stays excluded (no false positive)
    assert (
        _is_cooldown_required(
            litellm_router_instance=MagicMock(),
            model_id="dep-1",
            exception_status=403,
            exception_str="403 forbidden: invalid x-api-key",
        )
        is False
    )
    assert _is_periodic_quota_error(None) is False
    assert _is_periodic_quota_error("rate limited, slow down") is False


def _make_k3_router():
    return litellm.Router(
        model_list=[
            {
                "model_name": "round-robin/k3",
                "litellm_params": {
                    "model": "anthropic/k3",
                    "api_key": "fake-key",
                    "api_base": "https://api.kimi.com/coding",
                },
            }
        ],
        cooldown_time=60,
        num_retries=0,
        allowed_fails=1000,  # fail-count policy alone would never cool on 1st failure
    )


@pytest.mark.asyncio
async def test_periodic_quota_cools_immediately_with_placeholder_ttl():
    router = _make_k3_router()
    deployment_id = router.get_model_ids()[0]
    with (
        patch(
            "litellm.router_utils.cooldown_handlers.router_cooldown_event_callback",
            new=AsyncMock(),
        ),
        patch(
            "litellm.router_utils.cooldown_handlers.sync_deployment_quota",
            new=AsyncMock(),
        ) as probe_spy,
        patch.object(
            router.cooldown_cache,
            "add_deployment_to_cooldown",
            wraps=router.cooldown_cache.add_deployment_to_cooldown,
        ) as spy,
    ):
        result = _set_cooldown_deployments(
            litellm_router_instance=router,
            original_exception=Exception(_KIMI_QUOTA_WRAPPED),
            exception_status=500,  # APIConnectionError default, as in production
            deployment=deployment_id,
            time_to_cooldown=60,
        )
        await asyncio.sleep(0)  # let the fire-and-forget probe task run
    assert result is True
    # policy bypassed: cooled on FIRST failure; TTL is the 1h blind placeholder
    # (router 60s overridden; the 6h constant was abolished by ADR-0004)
    assert spy.call_count == 1
    assert spy.call_args.kwargs["cooldown_time"] == QUOTA_BLIND_COOLDOWN_SECONDS
    # wall-hit fires a single-deployment usages sync to overwrite the placeholder
    assert probe_spy.await_count == 1
    assert probe_spy.await_args.args == (router, deployment_id)


@pytest.mark.asyncio
async def test_429_window_marker_gets_same_treatment():
    router = _make_k3_router()
    deployment_id = router.get_model_ids()[0]
    with (
        patch(
            "litellm.router_utils.cooldown_handlers.router_cooldown_event_callback",
            new=AsyncMock(),
        ),
        patch(
            "litellm.router_utils.cooldown_handlers.sync_deployment_quota",
            new=AsyncMock(),
        ) as probe_spy,
        patch.object(
            router.cooldown_cache,
            "add_deployment_to_cooldown",
            wraps=router.cooldown_cache.add_deployment_to_cooldown,
        ) as spy,
    ):
        result = _set_cooldown_deployments(
            litellm_router_instance=router,
            original_exception=Exception(_KIMI_QUOTA_429),
            exception_status=429,
            deployment=deployment_id,
            time_to_cooldown=60,
        )
        await asyncio.sleep(0)
    assert result is True
    assert spy.call_count == 1
    assert spy.call_args.kwargs["cooldown_time"] == QUOTA_BLIND_COOLDOWN_SECONDS
    assert probe_spy.await_count == 1


@pytest.mark.asyncio
async def test_plain_errors_not_cooled():
    router = _make_k3_router()
    deployment_id = router.get_model_ids()[0]
    with (
        patch(
            "litellm.router_utils.cooldown_handlers.router_cooldown_event_callback",
            new=AsyncMock(),
        ),
        patch(
            "litellm.router_utils.cooldown_handlers.sync_deployment_quota",
            new=AsyncMock(),
        ) as probe_spy,
        patch.object(router.cooldown_cache, "add_deployment_to_cooldown") as spy,
    ):
        # plain permission 403
        result_403 = _set_cooldown_deployments(
            litellm_router_instance=router,
            original_exception=Exception("403 forbidden: invalid x-api-key"),
            exception_status=403,
            deployment=deployment_id,
            time_to_cooldown=60,
        )
        # transient connection error
        result_conn = _set_cooldown_deployments(
            litellm_router_instance=router,
            original_exception=Exception("litellm.APIConnectionError: AnthropicException - connection reset by peer"),
            exception_status=500,
            deployment=deployment_id,
            time_to_cooldown=60,
        )
        await asyncio.sleep(0)
    assert result_403 is False
    assert result_conn is False
    assert spy.call_count == 0
    # no quota marker -> no usages probe
    assert probe_spy.await_count == 0
