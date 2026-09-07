"""Unit tests for ConstraintCapabilityCheck (issue #11).

契约来源:ai-gateway-ops docs/specs/2026-09-07-constraint-aware-routing.md。
覆盖:三协议约束提取、幸存过滤、无卡池原样返回(identity)、空幸存 400、
env kill-switch、混合池语义(有卡池内未打标者按 active rule 丢弃)。
"""

import pytest

from litellm.exceptions import BadRequestError
from litellm.router_utils.pre_call_checks.constraint_capability_check import (
    FORCED_TOOLCHOICE_WITH_THINKING_CAPABILITY,
    ConstraintCapabilityCheck,
    _declared_capabilities,
    _is_forced_tool_choice,
    _is_thinking_on,
)


def _deployment(name: str, capabilities: str | None = None) -> dict:
    params = {"model": f"fake/{name}"}
    if capabilities is not None:
        params["capabilities"] = capabilities
    return {"model_info": {"id": f"dep-{name}"}, "litellm_params": params}


# ---------------------------------------------------------------------------
# 约束提取:thinking ON(三协议形态)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # /v1/messages(anthropic 原形态,claude 目标保留 thinking)
        ({"thinking": {"type": "enabled"}}, True),
        ({"thinking": {"type": "adaptive"}}, True),
        ({"thinking": {"type": "disabled"}}, False),
        ({"thinking": {"type": "Enabled"}}, True),  # 大小写不敏感
        ({"thinking": {}}, False),
        # /v1/chat/completions(顶层 reasoning_effort)
        ({"reasoning_effort": "max"}, True),
        ({"reasoning_effort": "low"}, True),
        ({"reasoning_effort": "none"}, False),
        ({"reasoning_effort": ""}, False),
        # /v1/responses(reasoning.effort)
        ({"reasoning": {"effort": "max"}}, True),
        ({"reasoning": {"effort": "xhigh"}}, True),
        ({"reasoning": {"effort": "none"}}, False),
        ({"reasoning": {}}, False),
        # 无约束
        ({}, False),
        ({"messages": [{"role": "user", "content": "hi"}]}, False),
    ],
)
def test_is_thinking_on(kwargs, expected):
    assert _is_thinking_on(kwargs) is expected


# ---------------------------------------------------------------------------
# 约束提取:tool_choice FORCED(三协议形态)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # OpenAI 形态(/v1/messages 经 adapter 翻译后固定为此形态)
        ({"tool_choice": "required"}, True),
        ({"tool_choice": "auto"}, False),
        ({"tool_choice": "none"}, False),
        ({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}, True),
        ({"tool_choice": {"type": "auto"}}, False),
        # Anthropic 原形态(防御:adapter 未翻译的路径)
        ({"tool_choice": {"type": "any"}}, True),
        ({"tool_choice": {"type": "tool", "name": "get_weather"}}, True),
        ({"tool_choice": {"type": "none"}}, False),
        # 无 tool_choice
        ({}, False),
        ({"tool_choice": None}, False),
    ],
)
def test_is_forced_tool_choice(kwargs, expected):
    assert _is_forced_tool_choice(kwargs) is expected


def test_declared_capabilities_parses_csv_and_missing_key():
    assert _declared_capabilities(_deployment("a", "thinking, forced_toolchoice_with_thinking")) == frozenset(
        {"thinking", "forced_toolchoice_with_thinking"}
    )
    assert _declared_capabilities(_deployment("b")) is None
    assert _declared_capabilities(_deployment("c", "")) is None
    assert _declared_capabilities(_deployment("d", "  ")) is None
    assert _declared_capabilities({"model_info": {}}) is None


# ---------------------------------------------------------------------------
# 过滤行为
# ---------------------------------------------------------------------------

MIXED_POOL = [
    _deployment("dashscope", "thinking,toolchoice_auto_none,forced_toolchoice_with_thinking"),
    _deployment("native", "thinking,toolchoice_auto_none"),
    _deployment("untagged"),  # 有卡池内未打标:active rule 下不提供能力证明
]

ALL_UNTAGGED_POOL = [_deployment("a"), _deployment("b")]

THINKING_FORCED_REQUESTS = [
    # /v1/messages(claude 目标形态)
    {"thinking": {"type": "enabled"}, "tool_choice": {"type": "any"}},
    # /v1/messages 经 adapter 翻译后的 openai 形态(过滤点实测形态)
    {"thinking": {"type": "enabled"}, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
    {"reasoning_effort": "max", "tool_choice": "required"},
    # /v1/chat/completions
    {"reasoning_effort": "high", "tool_choice": "required"},
    # /v1/responses
    {"reasoning": {"effort": "max"}, "tool_choice": {"type": "function", "function": {"name": "f"}}},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kwargs", THINKING_FORCED_REQUESTS)
async def test_active_rule_keeps_only_capable_survivors(request_kwargs):
    """三协议 thinking×forced 形态下,幸存者只有声明 forced_toolchoice_with_thinking 的卡。"""
    check = ConstraintCapabilityCheck()
    survivors = await check.async_filter_deployments(
        model="round-robin/k3",
        healthy_deployments=MIXED_POOL,
        messages=None,
        request_kwargs=request_kwargs,
        parent_otel_span=None,
    )
    assert [d["litellm_params"]["model"] for d in survivors] == ["fake/dashscope"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_kwargs",
    [
        # thinking ON 但 tool_choice 非 forced → 不过滤
        {"thinking": {"type": "enabled"}, "tool_choice": {"type": "auto"}},
        {"thinking": {"type": "enabled"}, "tool_choice": {"type": "none"}},
        {"reasoning_effort": "max"},
        {"reasoning": {"effort": "max"}},
        # forced 但 thinking OFF → 不过滤
        {"tool_choice": "required"},
        {"thinking": {"type": "disabled"}, "tool_choice": "required"},
        {"reasoning_effort": "none", "tool_choice": {"type": "any"}},
        # 无约束
        {},
    ],
)
async def test_non_triggering_shapes_pass_through(request_kwargs):
    check = ConstraintCapabilityCheck()
    survivors = await check.async_filter_deployments(
        model="round-robin/k3",
        healthy_deployments=MIXED_POOL,
        messages=None,
        request_kwargs=request_kwargs,
        parent_otel_span=None,
    )
    assert survivors == MIXED_POOL


@pytest.mark.asyncio
async def test_untagged_pool_returns_identical_object():
    """全池无卡:返回输入列表原对象(identity,零拷贝零分配热路径)。"""
    check = ConstraintCapabilityCheck()
    survivors = await check.async_filter_deployments(
        model="round-robin/glm-5.2",
        healthy_deployments=ALL_UNTAGGED_POOL,
        messages=None,
        request_kwargs={"reasoning_effort": "max", "tool_choice": "required"},
        parent_otel_span=None,
    )
    assert survivors is ALL_UNTAGGED_POOL


@pytest.mark.asyncio
async def test_empty_survivors_raise_bad_request_400():
    """池内有卡但无一声明该组合 → fail-closed 400(spec §3.3)。"""
    check = ConstraintCapabilityCheck()
    pool = [
        _deployment("a", "thinking,toolchoice_auto_none"),
        _deployment("b", "thinking"),
    ]
    with pytest.raises(BadRequestError) as exc_info:
        await check.async_filter_deployments(
            model="round-robin/k3",
            healthy_deployments=pool,
            messages=None,
            request_kwargs={"reasoning_effort": "max", "tool_choice": "required"},
            parent_otel_span=None,
        )
    assert exc_info.value.status_code == 400
    assert "no deployment in pool satisfies constraints: thinking+forced_toolchoice" in str(
        exc_info.value.message
    )


@pytest.mark.asyncio
async def test_kill_switch_env_disables_filter(monkeypatch):
    monkeypatch.setenv("CONSTRAINT_ROUTING_DISABLED", "1")
    check = ConstraintCapabilityCheck()
    survivors = await check.async_filter_deployments(
        model="round-robin/k3",
        healthy_deployments=MIXED_POOL,
        messages=None,
        request_kwargs={"reasoning_effort": "max", "tool_choice": "required"},
        parent_otel_span=None,
    )
    assert survivors == MIXED_POOL


@pytest.mark.asyncio
async def test_multiple_survivors_preserve_deployment_dicts():
    """多个幸存卡:原 dict 逐个返回(下游 simple_shuffle 从 dict 重读 weight/rpm/tpm,
    无需归一化,TBD-③)。"""
    check = ConstraintCapabilityCheck()
    pool = [
        _deployment("d1", "thinking,forced_toolchoice_with_thinking"),
        _deployment("d2", "thinking,toolchoice_auto_none"),
        _deployment("d3", "forced_toolchoice_with_thinking"),
    ]
    survivors = await check.async_filter_deployments(
        model="round-robin/k3",
        healthy_deployments=pool,
        messages=None,
        request_kwargs={"reasoning_effort": "max", "tool_choice": "required"},
        parent_otel_span=None,
    )
    assert [d["litellm_params"]["model"] for d in survivors] == ["fake/d1", "fake/d3"]
    for original, survivor in zip((pool[0], pool[2]), survivors):
        assert survivor is original


def test_capability_constant_matches_spec():
    assert FORCED_TOOLCHOICE_WITH_THINKING_CAPABILITY == "forced_toolchoice_with_thinking"
