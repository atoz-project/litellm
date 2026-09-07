"""Constraint-capability routing filter (issue #11).

Per-deployment capability cards (`litellm_params.capabilities`, CSV) matched against
the request's constraint vector, inside the existing
`Router.async_callback_filter_deployments` chain (after cooldown, before order).

v1 ACTIVE RULE (the only constraint that filters): request carries
**thinking ON × tool_choice FORCED** → keep only deployments whose card declares
`forced_toolchoice_with_thinking`. All other request shapes pass unfiltered.

Semantics (spec: ai-gateway-ops `docs/specs/2026-09-07-constraint-aware-routing.md`):
- No candidate declares any `capabilities` → return the input list object unchanged
  (hot path: no copy, no allocation; un-tagged pools zero impact).
- Untagged deployment inside a pool that HAS cards → dropped by the active rule:
  it carries no proof of satisfying the constraint, and per spec §3.3 a pool whose
  cards don't cover the requested combo is a contract failure, not a fallback.
- Empty survivor set → `BadRequestError` (status_code=400 hard-coded in
  `litellm/exceptions.py`; the callback chain re-raises, so the proxy surfaces 400).
- Kill switch: env `CONSTRAINT_ROUTING_DISABLED` (any non-empty value) → first line
  returns candidates unchanged (preferred fast rollback, spec §4.3 fork 路径).

Runtime key paths (resolved empirically on rebase-2026-09 @ 16da628b78, TBD-①):
- Capability card: `deployment["litellm_params"]["capabilities"]` — the ops TF
  writes it via `additional_litellm_params`, which lands directly in the
  deployment's `litellm_params` at runtime (same channel as `order`/`weight`;
  live `/v1/model/info` shows `litellm_params.order == 0`).
- `/v1/messages` inbound: the anthropic adapter's
  `translate_completion_input_params_with_tool_mapping` runs BEFORE the router, so
  at the filter point `tool_choice` is ALWAYS in OpenAI form
  (`"required"` from `{"type":"any"}`, `{"type":"function","function":{...}}` from
  `{"type":"tool"}`); `thinking` survives as top-level `thinking` only for claude
  targets, otherwise it is translated to top-level `reasoning_effort` at ingress
  (`_target_declares_reasoning_effort` in the same transformation).
- `/v1/chat/completions`: top-level `reasoning_effort`; forced forms as above.
- `/v1/responses`: `reasoning.effort`; forced forms as above.
Anthropic tool_choice forms (`{"type":"tool"|"any"}`) are accepted defensively for
any path that bypasses the adapter translation.
"""

import os
from typing import Any, Final

from litellm._logging import verbose_router_logger
from litellm.exceptions import BadRequestError
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues

FORCED_TOOLCHOICE_WITH_THINKING_CAPABILITY: Final = "forced_toolchoice_with_thinking"
CONSTRAINT_ROUTING_DISABLED_ENV: Final = "CONSTRAINT_ROUTING_DISABLED"
CONSTRAINT_UNSATISFIABLE_MESSAGE: Final = (
    "no deployment in pool satisfies constraints: thinking+forced_toolchoice"
)

_THINKING_ON_TYPES: Final = frozenset({"enabled", "adaptive"})
# Values that explicitly mean "thinking OFF" when they appear in an effort slot.
_EFFORT_OFF_VALUES: Final = frozenset({"none", "disabled", ""})
# Anthropic tool_choice types that mean "must call (a specific) tool" — i.e. forced.
_ANTHROPIC_FORCED_TOOL_CHOICE_TYPES: Final = frozenset({"tool", "any"})


def _is_thinking_on(request_kwargs: dict[str, Any]) -> bool:
    """thinking ON across the three inbound shapes at the filter point (see module docstring)."""
    thinking = request_kwargs.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if isinstance(thinking_type, str) and thinking_type.lower() in _THINKING_ON_TYPES:
            return True
    reasoning_effort = request_kwargs.get("reasoning_effort")
    if (
        isinstance(reasoning_effort, str)
        and reasoning_effort.lower() not in _EFFORT_OFF_VALUES
    ):
        return True
    reasoning = request_kwargs.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort.lower() not in _EFFORT_OFF_VALUES:
            return True
    return False


def _is_forced_tool_choice(request_kwargs: dict[str, Any]) -> bool:
    """tool_choice FORCED across shapes (OpenAI forms primary; Anthropic forms defensive)."""
    tool_choice = request_kwargs.get("tool_choice")
    if isinstance(tool_choice, str):
        return tool_choice.lower() == "required"
    if isinstance(tool_choice, dict):
        tool_choice_type = tool_choice.get("type")
        if isinstance(tool_choice_type, str):
            lowered = tool_choice_type.lower()
            return lowered == "function" or lowered in _ANTHROPIC_FORCED_TOOL_CHOICE_TYPES
    return False


def _declared_capabilities(deployment: dict[str, Any]) -> frozenset[str] | None:
    """per-deployment 能力卡片:`deployment["litellm_params"]["capabilities"]` CSV。

    键缺失/空 = 无声明,返回 None(语义由调用点决定,见模块 docstring)。
    """
    litellm_params = deployment.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return None
    raw = litellm_params.get("capabilities")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


class ConstraintCapabilityCheck(CustomLogger):
    """Filter healthy deployments by per-deployment capability cards (issue #11).

    Stateless; safe to register once process-wide. Registered by
    `Router._ensure_constraint_capability_callback` for every Router instance.
    """

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        # Kill switch (spec §4.3 fork 路径回滚):env 置位即整体失效。
        if os.environ.get(CONSTRAINT_ROUTING_DISABLED_ENV):
            return healthy_deployments

        # 单遍扫描找打标卡。候选池无任何 capabilities 声明 → 机制未启用,
        # 原对象直接返回(零拷贝零分配热路径,未打标池零影响)。
        has_any_card: Final = any(
            _declared_capabilities(deployment) is not None
            for deployment in healthy_deployments
        )
        if not has_any_card:
            return healthy_deployments

        # 有卡才提取请求约束;v1 仅 thinking×forced_toolchoice 触发过滤。
        kwargs: Final = request_kwargs or {}
        if not (_is_thinking_on(kwargs) and _is_forced_tool_choice(kwargs)):
            return healthy_deployments

        survivors: Final = [
            deployment
            for deployment in healthy_deployments
            if (capabilities := _declared_capabilities(deployment)) is not None
            and FORCED_TOOLCHOICE_WITH_THINKING_CAPABILITY in capabilities
        ]
        if not survivors:
            # spec §3.3 决策3:约束本身冲突 = 契约无法满足,fail-closed 400。
            verbose_router_logger.warning(
                "ConstraintCapabilityCheck: thinking+forced_toolchoice request to model=%s "
                "has no capable deployment in the healthy pool (fail-closed 400)",
                model,
            )
            raise BadRequestError(
                message=CONSTRAINT_UNSATISFIABLE_MESSAGE,
                model=model,
                llm_provider="constraint_capability_check",
            )

        verbose_router_logger.debug(
            "ConstraintCapabilityCheck: model=%s thinking+forced_toolchoice → %d/%d deployments survive",
            model,
            len(survivors),
            len(healthy_deployments),
        )
        return survivors
