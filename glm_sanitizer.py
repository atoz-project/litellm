"""
GLM tool-input sanitizer — CustomLogger for litellm proxy.

Covers the /v1/messages (Anthropic-format) path:
  pre_call_hook is called with call_type='anthropic_messages' from
  litellm/proxy/common_request_processing.py:963.

Logic mirrors the existing Lua plugin-fixed.lua:
  For every message whose content blocks contain a tool_use block where
  input is None / not-a-dict / empty-dict, normalise to {'_': ''}.

Only activates when:
  1. call_type == 'anthropic_messages'   (Anthropic /v1/messages path)
  2. data['model'] has the configured prefix (default 'glm')
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

# litellm imports – available after `pip install litellm`
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth

logger = logging.getLogger(__name__)

GLM_MODEL_PREFIXES = ("glm",)       # matches glm-5.2, glm-4, round-robin/glm-*, …
TARGET_CALL_TYPE = "anthropic_messages"


def _is_empty_input(value: Any) -> bool:
    """Return True when *value* represents an absent / empty tool-use input."""
    if value is None:
        return True
    if not isinstance(value, dict):
        return True
    return len(value) == 0


def _normalize_messages(messages: list) -> bool:
    """
    Walk messages, fix empty tool_use.input in-place.
    Returns True when at least one block was patched.
    """
    changed = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if _is_empty_input(block.get("input")):
                block["input"] = {"_": ""}
                changed = True
    return changed


def _model_is_glm(model: Optional[str]) -> bool:
    if not model:
        return False
    lowered = model.lower()
    return any(lowered.startswith(p) or ("/" + p) in lowered for p in GLM_MODEL_PREFIXES)


class GlmSanitizer(CustomLogger):
    """
    Sanitize empty tool_use.input for GLM models on the /v1/messages path.

    Registration (config.yaml):
        callbacks:
          - python_file: /path/to/glm_sanitizer.py
            callback_name: GlmSanitizer
    Or via litellm.callbacks:
        import litellm
        from glm_sanitizer import GlmSanitizer
        litellm.callbacks = [GlmSanitizer()]
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Union[dict, None]:
        """
        Called by ProxyLogging.pre_call_hook (proxy/utils.py:1493) for every
        registered CustomLogger whose class defines async_pre_call_hook.

        Returns the (possibly mutated) data dict, or None to leave data as-is.
        """
        if call_type != TARGET_CALL_TYPE:
            # Not the Anthropic-format /v1/messages route — skip.
            return None

        model: Optional[str] = data.get("model")
        if not _model_is_glm(model):
            return None

        messages = data.get("messages")
        if not isinstance(messages, list):
            return None

        changed = _normalize_messages(messages)
        if changed:
            logger.debug(
                "GlmSanitizer: patched empty tool_use.input(s) for model=%s", model
            )
        return data
