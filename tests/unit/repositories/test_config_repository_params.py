"""Pins the CONFIG_PARAMS whitelist consumed by /config/get and /config/param.

Regression (2026-10-13, 1.106 rebase): upstream dropped the class attribute;
both fork endpoints then 500'd with AttributeError in production (terraform
plan on litellm-gateway-tf failed reading router_settings).
"""

from litellm.repositories.config_repository import ConfigRepository


def test_config_params_whitelist_exists_for_config_endpoints():
    assert set(ConfigRepository.CONFIG_PARAMS) == {
        "general_settings",
        "router_settings",
        "litellm_settings",
        "environment_variables",
    }
