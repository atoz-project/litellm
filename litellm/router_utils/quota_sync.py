"""
custom-aigw: periodic-quota sync (ADR-0004, ai-gateway-ops issue GH#5).

Kimi Code plan keys carry two periodic quotas: a 5h rolling window and a weekly
quota. When either is exhausted the key rejects traffic until an authoritative
``resetTime``. Rather than guessing a fixed cooldown after hitting the wall,
this module actively syncs quota state:

- A probe registry (``QUOTA_PROBES``) maps api_base host fragments to probe
  functions. Adding a site = one dict entry + one function. Only Kimi is
  implemented this iteration (new-api/Brioi would need account-level
  credentials instead of the deployment's own key — a new attack surface, so
  deliberately deferred).
- A background loop (started from proxy startup, per-replica, uncoordinated)
  probes every matching deployment every 5 minutes using the deployment's own
  api_key:
    - any dimension exhausted -> cooldown with TTL = max(exhausted resetTime)
      + buffer, so the deployment returns to the pool automatically once the
      quota resets;
    - all dimensions recovered while the deployment is cooling down -> delete
      the cooldown key to unblock immediately (covers top-ups / upgrades /
      misattributed cooldowns);
    - probe failure -> change nothing (neither unblock nor re-cool).
- ``sync_deployment_quota`` exposes a single-deployment immediate sync for the
  wall-hit path in ``cooldown_handlers`` (marker-matched errors land a 1h
  placeholder TTL, then a fire-and-forget probe overwrites it with the real
  resetTime + buffer).

The usages endpoint returns 200 even for exhausted keys; payload shape (see
``parse_kimi_usages``): weekly quota in ``usage.{used,limit,remaining,resetTime}``
and the 5h window in ``limits[0].detail.{limit,used,remaining,resetTime}``.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlparse

import httpx
from typing_extensions import TypedDict

from litellm._logging import verbose_router_logger
from litellm.router_utils.cooldown_cache import CooldownCache

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any

# ---------------------------------------------------------------------------
# Constants (ADR-0004)
# ---------------------------------------------------------------------------
QUOTA_SYNC_INTERVAL_SECONDS: Final[float] = 300.0  # probe each key every 5min
QUOTA_RESET_BUFFER_SECONDS: Final[float] = 900.0  # 15min past authoritative resetTime
QUOTA_BLIND_COOLDOWN_SECONDS: Final[float] = (
    3600.0  # placeholder TTL: wall-hit before probe confirms, or exhausted without parseable resetTime
)
QUOTA_PROBE_TIMEOUT_SECONDS: Final[float] = 10.0


class QuotaDimension(TypedDict):
    name: str  # "weekly" | "5h-window"
    exhausted: bool
    remaining: int | None
    limit: int | None
    reset_time: float | None  # epoch seconds (UTC), None when absent/unparseable


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------
def _to_int(value: Any) -> int | None:
    """Kimi returns numbers as strings (e.g. "100"); normalize to int."""
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def parse_reset_epoch(value: Any) -> float | None:
    """Parse an ISO8601 timestamp (trailing Z supported) to UTC epoch seconds."""
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_dimension(name: str, raw: Any) -> QuotaDimension | None:
    """Parse one quota dimension; None when the payload carries no usable numbers.

    Exhaustion rule (ADR-0004): ``remaining == 0``; when ``remaining`` is
    absent (weekly quota sometimes omits it), fall back to ``limit - used <= 0``.
    """
    if not isinstance(raw, dict):
        return None
    limit = _to_int(raw.get("limit"))
    used = _to_int(raw.get("used"))
    remaining = _to_int(raw.get("remaining"))
    if remaining is None and limit is not None and used is not None:
        remaining = limit - used
    if remaining is None and limit is None and used is None:
        return None
    return QuotaDimension(
        name=name,
        exhausted=remaining is not None and remaining <= 0,
        remaining=remaining,
        limit=limit,
        reset_time=parse_reset_epoch(raw.get("resetTime")),
    )


def parse_kimi_usages(payload: Any) -> list[QuotaDimension]:
    """Extract quota dimensions from a Kimi ``/v1/usages`` payload.

    Weekly quota: ``usage.{used,limit,remaining,resetTime}``.
    5h rolling window: ``limits[0].detail.{limit,used,remaining,resetTime}``.
    """
    if not isinstance(payload, dict):
        return []
    dims: list[QuotaDimension] = []
    weekly = _parse_dimension("weekly", payload.get("usage"))
    if weekly is not None:
        dims.append(weekly)
    limits = payload.get("limits")
    if isinstance(limits, list) and limits and isinstance(limits[0], dict):
        five_hour = _parse_dimension("5h-window", limits[0].get("detail"))
        if five_hour is not None:
            dims.append(five_hour)
    return dims


def _format_dims(dims: list[QuotaDimension]) -> str:
    parts = []
    for d in dims:
        remaining = "?" if d["remaining"] is None else str(d["remaining"])
        limit = "?" if d["limit"] is None else str(d["limit"])
        if d["reset_time"] is not None:
            reset = datetime.fromtimestamp(d["reset_time"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            reset = "-"
        parts.append(f"{d['name']}={remaining}/{limit} reset={reset}")
    return " ".join(parts) if parts else "(no dimensions)"


# ---------------------------------------------------------------------------
# Probe registry
# ---------------------------------------------------------------------------
async def kimi_probe(api_base: str, api_key: str) -> list[QuotaDimension]:
    """GET {api_base}/v1/usages (api_base already ends in /coding).

    Raises on any HTTP/parse failure; callers must change no state on failure.
    """
    url = api_base.rstrip("/") + "/v1/usages"
    # trust_env=False: probes run from the SAE pod over NAT egress; host proxy
    # env vars must not divert them.
    async with httpx.AsyncClient(timeout=QUOTA_PROBE_TIMEOUT_SECONDS, trust_env=False) as client:
        response = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        return parse_kimi_usages(response.json())


# Host-fragment -> probe. Adding a site = one entry + one function.
QUOTA_PROBES: Final[dict[str, Callable[[str, str], Awaitable[list[QuotaDimension]]]]] = {
    "api.kimi.com": kimi_probe,
}


def match_probe(api_base: str | None) -> Callable[[str, str], Awaitable[list[QuotaDimension]]] | None:
    """Return the probe whose registry host fragment appears in api_base's host."""
    if not api_base:
        return None
    try:
        host = urlparse(api_base).hostname or ""
    except ValueError:
        return None
    for fragment, probe in QUOTA_PROBES.items():
        if fragment in host:
            return probe
    return None


# ---------------------------------------------------------------------------
# Cooldown reconciliation (shared by the loop and the wall-hit path)
# ---------------------------------------------------------------------------
def _cooldown_ttl_from_dims(dims: list[QuotaDimension], now: float) -> float | None:
    """TTL = max(exhausted resetTime) + buffer - now; blind fallback when no
    exhausted dimension carries a parseable resetTime; None when expired."""
    exhausted = [d for d in dims if d["exhausted"]]
    if not exhausted:
        return None
    resets = [d["reset_time"] for d in exhausted if d["reset_time"] is not None]
    if resets:
        ttl = max(resets) + QUOTA_RESET_BUFFER_SECONDS - now
    else:
        ttl = QUOTA_BLIND_COOLDOWN_SECONDS
    return ttl if ttl > 0 else None


async def _reconcile_deployment_cooldown(
    litellm_router_instance: LitellmRouter,
    model_id: str,
    dims: list[QuotaDimension],
) -> str:
    """Apply a successful probe result to the deployment's cooldown state.

    Returns the action taken, for logging: cooldown | skip-expired | unblock | ok.
    """
    cooldown_cache = litellm_router_instance.cooldown_cache
    now = time.time()
    ttl = _cooldown_ttl_from_dims(dims, now)
    if ttl is not None:
        cooldown_cache.add_deployment_to_cooldown(
            model_id=model_id,
            original_exception=Exception("quota-sync: periodic quota exhausted (usages probe)"),
            exception_status=429,
            cooldown_time=ttl,
        )
        verbose_router_logger.info(
            "quota_sync: model_id=%s %s action=cooldown ttl=%ds",
            model_id,
            _format_dims(dims),
            int(ttl),
        )
        return "cooldown"

    exhausted = [d for d in dims if d["exhausted"]]
    if exhausted:
        # exhausted but resetTime already passed: don't re-cool, let traffic re-probe
        verbose_router_logger.info(
            "quota_sync: model_id=%s %s action=skip-expired",
            model_id,
            _format_dims(dims),
        )
        return "skip-expired"

    # all dimensions recovered: actively unblock if currently cooling down
    active = await cooldown_cache.async_get_active_cooldowns(model_ids=[model_id], parent_otel_span=None)
    if active:
        await cooldown_cache.cache.async_delete_cache(CooldownCache.get_cooldown_cache_key(model_id))
        verbose_router_logger.info(
            "quota_sync: model_id=%s %s action=unblock",
            model_id,
            _format_dims(dims),
        )
        return "unblock"

    verbose_router_logger.info(
        "quota_sync: model_id=%s %s action=ok",
        model_id,
        _format_dims(dims),
    )
    return "ok"


async def sync_deployment_quota(litellm_router_instance: LitellmRouter, model_id: str) -> str:
    """Probe one deployment's quota and reconcile its cooldown state.

    Fire-and-forget target for the wall-hit path and the unit of work for the
    sync loop. Probe failure changes nothing (ADR-0004): no unblock, no re-cool.

    Returns the action for logging: cooldown | skip-expired | unblock | ok |
    probe-failed | skipped.
    """
    try:
        deployment = litellm_router_instance.get_deployment(model_id)
    except Exception:
        deployment = None
    if deployment is None:
        return "skipped"
    api_base = deployment.litellm_params.api_base
    api_key = deployment.litellm_params.api_key
    probe = match_probe(api_base)
    if probe is None or not api_key:
        return "skipped"
    try:
        dims = await probe(api_base, api_key)
    except Exception as e:
        verbose_router_logger.info("quota_sync: model_id=%s probe-failed err=%s action=none", model_id, str(e)[:200])
        return "probe-failed"
    if not dims:
        # empty/malformed payload: treat like a probe failure, change nothing
        verbose_router_logger.info("quota_sync: model_id=%s probe-failed err=empty-payload action=none", model_id)
        return "probe-failed"
    return await _reconcile_deployment_cooldown(litellm_router_instance, model_id, dims)


# ---------------------------------------------------------------------------
# Deployment discovery + background loop
# ---------------------------------------------------------------------------
def get_quota_sync_model_ids(litellm_router_instance: LitellmRouter) -> list[str]:
    """Model ids whose deployment api_base matches the probe registry."""
    matched: list[str] = []
    for model_id in litellm_router_instance.get_model_ids():
        try:
            deployment = litellm_router_instance.get_deployment(model_id)
        except Exception:
            continue
        if deployment is not None and match_probe(deployment.litellm_params.api_base) is not None:
            matched.append(model_id)
    return matched


async def run_quota_sync_loop(
    litellm_router_instance: LitellmRouter,
    sync_interval_seconds: float = QUOTA_SYNC_INTERVAL_SECONDS,
) -> None:
    """Per-replica background loop started at proxy startup (ADR-0004).

    Exits immediately when no deployment matches the probe registry. Each
    deployment is probed with its own api_key; individual failures never kill
    the loop and never change cooldown state.
    """
    if not get_quota_sync_model_ids(litellm_router_instance):
        verbose_router_logger.info("quota_sync: no probe-registry deployments, loop exiting")
        return
    verbose_router_logger.info("quota_sync: loop started interval=%ss", sync_interval_seconds)
    while True:
        for model_id in get_quota_sync_model_ids(litellm_router_instance):
            try:
                await sync_deployment_quota(litellm_router_instance, model_id)
            except Exception as e:
                verbose_router_logger.error("quota_sync: model_id=%s sync error: %s", model_id, e)
        await asyncio.sleep(sync_interval_seconds)
