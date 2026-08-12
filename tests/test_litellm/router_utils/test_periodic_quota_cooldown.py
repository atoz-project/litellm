"""
custom-aigw: periodic-quota (Kimi Code weekly billing-cycle) cooldown tests.

Production shape (verified via SLS 2026-08-12): the anthropic exception mapper
has no 403 branch, so Kimi's weekly-quota 403 falls through to a generic
APIConnectionError(status 500) with the upstream body embedded — and
APIConnectionError is on the no-cooldown veto list. The fork patch bypasses the
veto for marker-matched errors and cools the deployment immediately with a 6h
TTL floor; plain permission 403s and transient connection errors stay uncooled.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.router_utils.cooldown_handlers import (
    PERIODIC_QUOTA_COOLDOWN_SECONDS,
    _is_cooldown_required,
    _is_periodic_quota_error,
    _set_cooldown_deployments,
)

# exact shape seen in production logs (anthropic mapper 403 fallthrough)
_KIMI_QUOTA_WRAPPED = (
    "litellm.APIConnectionError: AnthropicException - "
    '{"error":{"type":"permission_error","message":"You\'ve reached your usage '
    "limit for this billing cycle. Your quota will be refreshed in the next cycle. "
    "To continue now, purchase extra usage or upgrade your plan: "
    'https://www.kimi.com/code/#pricing"},"type":"error"}'
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
async def test_periodic_quota_cools_immediately_with_long_ttl():
    router = _make_k3_router()
    deployment_id = router.get_model_ids()[0]
    with patch(
        "litellm.router_utils.cooldown_handlers.router_cooldown_event_callback",
        new=AsyncMock(),
    ), patch.object(
        router.cooldown_cache,
        "add_deployment_to_cooldown",
        wraps=router.cooldown_cache.add_deployment_to_cooldown,
    ) as spy:
        result = _set_cooldown_deployments(
            litellm_router_instance=router,
            original_exception=Exception(_KIMI_QUOTA_WRAPPED),
            exception_status=500,  # APIConnectionError default, as in production
            deployment=deployment_id,
            time_to_cooldown=60,
        )
    assert result is True
    # policy bypassed: cooled on FIRST failure; TTL floored to 6h (router 60s overridden)
    assert spy.call_count == 1
    assert spy.call_args.kwargs["cooldown_time"] == PERIODIC_QUOTA_COOLDOWN_SECONDS


@pytest.mark.asyncio
async def test_plain_errors_not_cooled():
    router = _make_k3_router()
    deployment_id = router.get_model_ids()[0]
    with patch(
        "litellm.router_utils.cooldown_handlers.router_cooldown_event_callback",
        new=AsyncMock(),
    ), patch.object(
        router.cooldown_cache, "add_deployment_to_cooldown"
    ) as spy:
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
            original_exception=Exception(
                "litellm.APIConnectionError: AnthropicException - connection reset by peer"
            ),
            exception_status=500,
            deployment=deployment_id,
            time_to_cooldown=60,
        )
    assert result_403 is False
    assert result_conn is False
    assert spy.call_count == 0
