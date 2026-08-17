"""
Type definitions for WebSearch Interception integration.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel


class AnthropicSearchQuery(BaseModel):
    """``input`` of an Anthropic ``server_tool_use`` block for a web search."""

    query: str


class AnthropicServerToolUseBlock(BaseModel):
    """
    The ``server_tool_use`` block that must accompany a ``web_search_tool_result``.

    Anthropic requires the pair, with a ``srvtoolu_``-prefixed id shared by both.
    """

    type: Literal["server_tool_use"] = "server_tool_use"
    id: str
    name: Literal["web_search"] = "web_search"
    input: AnthropicSearchQuery


class WebSearchInterceptionConfig(TypedDict, total=False):
    """
    Configuration parameters for WebSearchInterceptionLogger.

    Used in proxy_config.yaml under litellm_settings:
        litellm_settings:
          websearch_interception_params:
            enabled_providers: ["bedrock"]
            search_tool_name: "my-perplexity-search"
    """

    enabled_providers: list[str]
    """List of LLM provider names to enable interception for (e.g., ['bedrock', 'vertex_ai'])"""

    search_tool_name: str | None
    """Name of search tool configured in router's search_tools. If None, uses first available."""

    enabled_models: list[str] | None
    """Optional list of model-name prefixes gating ALL interception paths (tool
    conversion, short-circuit, and every agentic-loop entry). A model is enabled
    when it equals an entry or starts with it followed by '-' (boundary-aware
    prefix); if None, ALL models are eligible (provider filter still applies).
    Non-matching models are fully exempt.
    Match is against the deployment name (router-rewritten, e.g.
    'openai/glm-5.2'), NOT the request model name (e.g. 'round-robin/glm-5.2').
    Prefix matching covers snapshot suffixes: 'openai/qwen3.7-max' also matches
    'openai/qwen3.7-max-2026-06-08' (but 'openai/glm-5.2' does not match the
    distinct model 'openai/glm-5.20')."""
