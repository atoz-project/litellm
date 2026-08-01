"""
Type definitions for WebSearch Interception integration.
"""

from typing import List, Optional, TypedDict


class WebSearchInterceptionConfig(TypedDict, total=False):
    """
    Configuration parameters for WebSearchInterceptionLogger.

    Used in proxy_config.yaml under litellm_settings:
        litellm_settings:
          websearch_interception_params:
            enabled_providers: ["bedrock"]
            search_tool_name: "my-perplexity-search"
    """

    enabled_providers: List[str]
    """List of LLM provider names to enable interception for (e.g., ['bedrock', 'vertex_ai'])"""

    search_tool_name: Optional[str]
    """Name of search tool configured in router's search_tools. If None, uses first available."""

    enabled_models: Optional[List[str]]
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
