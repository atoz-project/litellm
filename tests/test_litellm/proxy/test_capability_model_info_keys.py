"""Operator-authoritative reasoning-effort capability keys on the model-management plane.

The capability card (``supports_reasoning`` / ``supports_*_reasoning_effort`` /
``reasoning_effort_levels``) is owned by the operator's model_info declaration:

- update (``update_db_model``): a patch's capability subset REPLACES the stored one.
  Plain merge let boolean-era ``supports_*_reasoning_effort`` flags outlive their
  declarations because they sit outside the SPECIAL_MODEL_INFO_PARAMS null-clear
  whitelist, so a bool->list conversion could never drop them.
- add (``_add_model_to_db``): a create stores exactly the declared model_info — no
  capability keys may appear from the built-in catalog or sibling registrations.
- read (``_overlay_litellm_model_info``, used by /model/info + /v2/model/info):
  catalog values never overlay capability keys onto a DB-declared row, so neither
  the built-in bare-name entry (gpt-6-astra / gpt-5.6-sol carry per-level flags)
  nor a stale in-memory ``litellm.model_cost`` registration (register_model merges
  and never removes keys) can resurrect a dropped flag. config.yaml deployments
  keep the vanilla fill.
"""

import json

import pytest

from litellm.proxy.management_endpoints.model_management_endpoints import (
    _add_model_to_db,
    update_db_model,
)
from litellm.router_utils.reasoning_effort_capability import CAPABILITY_MODEL_INFO_KEYS
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo, updateDeployment

BOOLEAN_ERA_CARD = {
    "supports_reasoning": True,
    "supports_none_reasoning_effort": True,
    "supports_minimal_reasoning_effort": True,
    "supports_xhigh_reasoning_effort": True,
    "supports_max_reasoning_effort": True,
}
LIST_ERA_CARD = {
    "supports_reasoning": True,
    "reasoning_effort_levels": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
}


def _stored_deployment(model_info: dict) -> Deployment:
    return Deployment(
        model_name="round-robin/glm-5.2",
        litellm_params=LiteLLM_Params(model="openai/glm-5.2"),
        model_info=ModelInfo(id="dep-cap", db_model=True, **model_info),
    )


def _stored_capability_keys(result: dict) -> dict:
    stored = json.loads(result["model_info"])
    return {k: v for k, v in stored.items() if k in CAPABILITY_MODEL_INFO_KEYS}


class TestUpdateDbModelCapabilityReplace:
    def test_list_patch_drops_boolean_era_flags(self):
        """The 2026-09-18 bool->list conversion: a patch carrying the list card must
        leave storage holding exactly that card, with every boolean flag gone."""
        result = update_db_model(
            db_model=_stored_deployment(dict(BOOLEAN_ERA_CARD)),
            updated_patch=updateDeployment(model_info=ModelInfo(id="dep-cap", **LIST_ERA_CARD)),
        )
        assert _stored_capability_keys(result) == LIST_ERA_CARD

    def test_patch_without_capability_keys_clears_stored_card(self):
        """A model_info patch that declares no capability keys empties the stored
        subset — replace semantics, not merge."""
        result = update_db_model(
            db_model=_stored_deployment(dict(BOOLEAN_ERA_CARD)),
            updated_patch=updateDeployment(model_info=ModelInfo(id="dep-cap", tier="paid")),
        )
        assert _stored_capability_keys(result) == {}

    def test_patch_without_model_info_keeps_stored_card(self):
        """A patch that carries no model_info at all (e.g. a litellm_params-only
        edit) must not wipe the capability card."""
        result = update_db_model(
            db_model=_stored_deployment(dict(BOOLEAN_ERA_CARD)),
            updated_patch=updateDeployment(model_name="round-robin/glm-5.2-renamed"),
        )
        assert _stored_capability_keys(result) == BOOLEAN_ERA_CARD

    def test_non_capability_model_info_keys_still_merge(self):
        """Replace is scoped to the capability card; every other model_info key keeps
        the existing merge behavior."""
        result = update_db_model(
            db_model=_stored_deployment({**LIST_ERA_CARD, "tier": "paid"}),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-cap", access_groups=["full-access-models"])
            ),
        )
        stored = json.loads(result["model_info"])
        assert stored["access_groups"] == ["full-access-models"]
        # capability keys absent from the patch -> dropped (replace semantics)
        assert _stored_capability_keys(result) == {}


class TestAddModelStoresDeclaredCardOnly:
    @pytest.mark.asyncio
    async def test_create_stores_exactly_the_declared_model_info(self, monkeypatch):
        """A create whose backend model has a built-in catalog entry with per-level
        flags (gpt-6-astra) must not have them appended: storage is the declaration."""
        from litellm.proxy._types import UserAPIKeyAuth

        # encrypt_value_helper needs a salt; same convention as the update_db_model tests.
        monkeypatch.setenv("LITELLM_SALT_KEY", "sk-1234")

        declared = {
            "supports_reasoning": True,
            "reasoning_effort_levels": ["low", "medium", "high", "xhigh", "max"],
        }
        model_params = Deployment(
            model_name="round-robin/gpt-6-astra",
            litellm_params=LiteLLM_Params(model="openai/gpt-6-astra"),
            model_info=ModelInfo(id="dep-astra", db_model=True, **declared),
        )
        row = await _add_model_to_db(
            model_params=model_params,
            user_api_key_dict=UserAPIKeyAuth(user_id="tester"),
            prisma_client=None,  # type: ignore[arg-type]  # unused: should_create_model_in_db=False
            should_create_model_in_db=False,
        )
        stored = json.loads(row.model_info) if isinstance(row.model_info, str) else row.model_info
        assert {k: v for k, v in stored.items() if k in CAPABILITY_MODEL_INFO_KEYS} == declared


class TestReadOverlayCapabilitySkip:
    """The /model/info overlay fills unset keys from the catalog — except capability
    keys on DB-stored rows, where the stored declaration is the whole answer."""

    CATALOG_WITH_FLAGS = {
        "supports_reasoning": True,
        "supports_minimal_reasoning_effort": False,
        "supports_xhigh_reasoning_effort": True,
        "supports_max_reasoning_effort": True,
        "max_tokens": 256000,
    }

    def test_db_row_gets_no_capability_overlay(self):
        from litellm.proxy.proxy_server import _overlay_litellm_model_info

        model_info = {"id": "dep-1", "db_model": True, **LIST_ERA_CARD}
        out = _overlay_litellm_model_info(model_info, dict(self.CATALOG_WITH_FLAGS))
        # capability keys: exactly the declaration (no supports_minimal=False etc.)
        assert {k: v for k, v in out.items() if k in CAPABILITY_MODEL_INFO_KEYS} == LIST_ERA_CARD
        # non-capability catalog keys still fill
        assert out["max_tokens"] == 256000

    def test_db_row_without_declared_card_stays_capability_free(self):
        from litellm.proxy.proxy_server import _overlay_litellm_model_info

        out = _overlay_litellm_model_info(
            {"id": "dep-2", "db_model": True}, dict(self.CATALOG_WITH_FLAGS)
        )
        assert not (CAPABILITY_MODEL_INFO_KEYS & out.keys())
        assert out["max_tokens"] == 256000

    def test_config_yaml_row_keeps_vanilla_overlay(self):
        from litellm.proxy.proxy_server import _overlay_litellm_model_info

        out = _overlay_litellm_model_info({"id": "cfg-1"}, dict(self.CATALOG_WITH_FLAGS))
        assert out["supports_xhigh_reasoning_effort"] is True
        assert out["supports_minimal_reasoning_effort"] is False
        assert out["max_tokens"] == 256000

    def test_declared_key_is_never_overwritten(self):
        from litellm.proxy.proxy_server import _overlay_litellm_model_info

        out = _overlay_litellm_model_info(
            {"max_tokens": 131072}, dict(self.CATALOG_WITH_FLAGS)
        )
        assert out["max_tokens"] == 131072
