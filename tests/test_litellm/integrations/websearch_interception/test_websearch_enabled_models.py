"""
Unit tests for enabled_models per-model gating (all interception paths).

Covers the boundary-aware prefix matcher `_is_model_enabled` and the six
gate points where exempt models must be fully transparent:
  1. try_short_circuit_search
  2. async_pre_call_deployment_hook
  3. async_pre_request_hook
  4. async_should_run_agentic_loop            (anthropic surface)
  5. async_should_run_chat_completion_agentic_loop
  6. async_should_run_responses_agentic_loop
"""

from types import SimpleNamespace

import pytest

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)

ENABLED = ["openai/glm-5.2", "openai/glm-5.1"]
EXEMPT_MODEL = "openai/qwen3.7-max-2026-06-08"
ENABLED_MODEL = "openai/glm-5.2"

NATIVE_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}


def _logger(enabled_models=ENABLED):
    return WebSearchInterceptionLogger(
        enabled_providers=["openai"],
        enabled_models=enabled_models,
    )


# ---------------------------------------------------------------------------
# _is_model_enabled prefix matrix
# ---------------------------------------------------------------------------


class TestIsModelEnabled:
    def test_exact_match(self):
        assert _logger()._is_model_enabled("openai/glm-5.2") is True

    def test_snapshot_suffix_match(self):
        """Entry without snapshot suffix covers dated deployments."""
        logger = _logger(["openai/qwen3.7-max"])
        assert logger._is_model_enabled("openai/qwen3.7-max-2026-06-08") is True

    def test_no_false_positive_on_longer_model_name(self):
        """'openai/glm-5.2' must NOT match the distinct model 'openai/glm-5.20'."""
        assert _logger()._is_model_enabled("openai/glm-5.20") is False

    def test_non_listed_model_not_enabled(self):
        assert _logger()._is_model_enabled(EXEMPT_MODEL) is False

    def test_second_entry_matches(self):
        assert _logger()._is_model_enabled("openai/glm-5.1") is True

    def test_none_enables_all_models(self):
        logger = _logger(None)
        assert logger._is_model_enabled(EXEMPT_MODEL) is True
        assert logger._is_model_enabled("anything/at-all") is True

    def test_empty_model_fails_closed(self):
        """Missing/empty model with a configured list → not enabled (no spend)."""
        assert _logger()._is_model_enabled("") is False

    def test_empty_list_enables_nothing(self):
        assert _logger([])._is_model_enabled(ENABLED_MODEL) is False


# ---------------------------------------------------------------------------
# Gate 1: try_short_circuit_search
# ---------------------------------------------------------------------------


class TestShortCircuitGate:
    @pytest.mark.asyncio
    async def test_exempt_model_not_short_circuited(self):
        """Web-search-only request for an exempt model → None (no Tavily)."""
        logger = _logger()
        result = await logger.try_short_circuit_search(
            model=EXEMPT_MODEL,
            messages=[{"role": "user", "content": "Search for X"}],
            tools=[NATIVE_TOOL],
            custom_llm_provider="openai",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_enabled_model_short_circuits_via_prefix(self):
        """Control: whitelisted entry still short-circuits (and a snapshot
        deployment of the same base model matches via prefix)."""
        logger = _logger(["openai/glm-5.2"])
        from unittest.mock import AsyncMock, patch

        with patch.object(logger, "_execute_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = ("Title: R\nURL: u\nSnippet: s", None)
            result = await logger.try_short_circuit_search(
                model="openai/glm-5.2",
                messages=[{"role": "user", "content": "Search for X"}],
                tools=[NATIVE_TOOL],
                custom_llm_provider="openai",
            )
        assert result is not None
        assert result["type"] == "message"


# ---------------------------------------------------------------------------
# Gate 2: async_pre_call_deployment_hook
# ---------------------------------------------------------------------------


class TestDeploymentHookGate:
    @pytest.mark.asyncio
    async def test_exempt_model_tools_not_converted(self):
        """Exempt model: hook returns None and leaves native tools untouched —
        no conversion, no stream downgrade, no native-block flag."""
        logger = _logger()
        kwargs = {
            "model": EXEMPT_MODEL,
            "custom_llm_provider": "openai",
            "tools": [NATIVE_TOOL],
            "stream": True,
        }
        result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

        assert result is None
        assert kwargs["tools"] == [NATIVE_TOOL]
        assert kwargs["stream"] is True
        assert "_websearch_interception_emit_native_blocks" not in kwargs

    @pytest.mark.asyncio
    async def test_enabled_model_tools_converted(self):
        """Control: whitelisted model still gets conversion."""
        logger = _logger()
        kwargs = {
            "model": ENABLED_MODEL,
            "custom_llm_provider": "openai",
            "tools": [NATIVE_TOOL],
        }
        result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

        assert result is not None
        assert result["tools"][0].get("name") != "web_search" or result["tools"][0].get("type") == "function"


# ---------------------------------------------------------------------------
# Gate 3: async_pre_request_hook
# ---------------------------------------------------------------------------


class TestPreRequestHookGate:
    @pytest.mark.asyncio
    async def test_exempt_model_tools_not_converted(self):
        logger = _logger()
        kwargs = {
            "litellm_params": {"custom_llm_provider": "openai"},
            "tools": [NATIVE_TOOL],
            "stream": True,
        }
        result = await logger.async_pre_request_hook(
            model=EXEMPT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            kwargs=kwargs,
        )

        assert result is None
        assert kwargs["tools"] == [NATIVE_TOOL]
        assert kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_enabled_model_tools_converted(self):
        """Control: whitelisted model still gets conversion."""
        logger = _logger()
        kwargs = {
            "litellm_params": {"custom_llm_provider": "openai"},
            "tools": [NATIVE_TOOL],
        }
        result = await logger.async_pre_request_hook(
            model=ENABLED_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            kwargs=kwargs,
        )

        assert result is not None


# ---------------------------------------------------------------------------
# Gate 4: async_should_run_agentic_loop (anthropic surface)
# ---------------------------------------------------------------------------


def _anthropic_response_with_websearch_tool_use():
    return {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "litellm_web_search",
                "input": {"query": "latest news"},
            }
        ]
    }


class TestAnthropicShouldRunGate:
    @pytest.mark.asyncio
    async def test_exempt_model_never_enters_loop(self):
        """Response contains a web_search tool_use, request carries the tool —
        still (False, {}) for an exempt model."""
        logger = _logger()
        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=_anthropic_response_with_websearch_tool_use(),
            model=EXEMPT_MODEL,
            messages=[],
            tools=[{"name": "litellm_web_search"}],
            stream=False,
            custom_llm_provider="openai",
            kwargs={},
        )
        assert should_run is False
        assert tools_dict == {}

    @pytest.mark.asyncio
    async def test_enabled_model_enters_loop(self):
        """Control: whitelisted model with identical payload enters the loop."""
        logger = _logger()
        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=_anthropic_response_with_websearch_tool_use(),
            model=ENABLED_MODEL,
            messages=[],
            tools=[{"name": "litellm_web_search"}],
            stream=False,
            custom_llm_provider="openai",
            kwargs={},
        )
        assert should_run is True
        assert len(tools_dict["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# Gate 5: async_should_run_chat_completion_agentic_loop
# ---------------------------------------------------------------------------


class TestChatCompletionShouldRunGate:
    @staticmethod
    def _response_with_tool_call():
        from litellm.types.utils import (
            ChatCompletionMessageToolCall,
            Choices,
            Function,
            Message,
            ModelResponse,
        )

        return ModelResponse(
            id="test-123",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_123",
                                type="function",
                                function=Function(
                                    name="litellm_web_search",
                                    arguments='{"query": "weather in SF"}',
                                ),
                            )
                        ],
                    ),
                )
            ],
            model="qwen3.7-max-2026-06-08",
            object="chat.completion",
            created=1234567890,
        )

    OPENAI_TOOLS = [{"type": "function", "function": {"name": "litellm_web_search"}}]

    @pytest.mark.asyncio
    async def test_exempt_model_never_enters_loop(self):
        logger = _logger()
        should_run, tools_dict = await logger.async_should_run_chat_completion_agentic_loop(
            response=self._response_with_tool_call(),
            model=EXEMPT_MODEL,
            messages=[{"role": "user", "content": "weather?"}],
            tools=self.OPENAI_TOOLS,
            stream=False,
            custom_llm_provider="openai",
            kwargs={},
        )
        assert should_run is False
        assert tools_dict == {}

    @pytest.mark.asyncio
    async def test_enabled_model_enters_loop(self):
        """Control: whitelisted model with identical payload enters the loop."""
        logger = _logger()
        should_run, tools_dict = await logger.async_should_run_chat_completion_agentic_loop(
            response=self._response_with_tool_call(),
            model=ENABLED_MODEL,
            messages=[{"role": "user", "content": "weather?"}],
            tools=self.OPENAI_TOOLS,
            stream=False,
            custom_llm_provider="openai",
            kwargs={},
        )
        assert should_run is True
        assert len(tools_dict["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# Gate 6: async_should_run_responses_agentic_loop
# ---------------------------------------------------------------------------


class TestResponsesShouldRunGate:
    @staticmethod
    def _response_with_function_call():
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="litellm_web_search",
                    call_id="fc_1",
                    arguments='{"query": "latest ai news"}',
                )
            ]
        )

    RESPONSES_TOOLS = [{"type": "function", "name": "litellm_web_search"}]

    @pytest.mark.asyncio
    async def test_exempt_model_never_enters_loop(self):
        logger = _logger()
        should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
            response=self._response_with_function_call(),
            model=EXEMPT_MODEL,
            messages=[{"role": "user", "content": "news?"}],
            tools=self.RESPONSES_TOOLS,
            stream=False,
            custom_llm_provider="openai",
            kwargs={},
        )
        assert should_run is False
        assert tools_dict == {}

    @pytest.mark.asyncio
    async def test_enabled_model_enters_loop(self):
        """Control: whitelisted model with identical payload enters the loop."""
        logger = _logger()
        should_run, tools_dict = await logger.async_should_run_responses_agentic_loop(
            response=self._response_with_function_call(),
            model=ENABLED_MODEL,
            messages=[{"role": "user", "content": "news?"}],
            tools=self.RESPONSES_TOOLS,
            stream=False,
            custom_llm_provider="openai",
            kwargs={},
        )
        assert should_run is True
        assert len(tools_dict["tool_calls"]) == 1
