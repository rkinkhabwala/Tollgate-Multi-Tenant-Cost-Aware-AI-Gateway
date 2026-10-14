"""Policy schema validation (spec §9) and catalog cost math (spec §11)."""

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from gateway.catalog import ModelCatalog
from gateway.policy.schema import AppPolicy
from gateway.settings import get_settings


def test_default_policy_file_is_valid_and_matches_defaults() -> None:
    raw = json.loads(get_settings().default_policy_json.read_text())
    policy = AppPolicy.model_validate(raw)
    assert policy.limits.rpm == 600
    assert policy.routing.allow_pinning is False
    # Schema defaults mirror the file (except the example custom pattern).
    defaults = AppPolicy().model_dump()
    loaded = policy.model_dump()
    loaded["redaction"]["custom_patterns"] = []
    loaded["routing"]["table_overrides"] = {}
    assert loaded == defaults


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        AppPolicy.model_validate({"limits": {"rpm": 10, "rmp": 5}})
    with pytest.raises(ValidationError):
        AppPolicy.model_validate({"surprise": {}})


def test_tier_order_and_bad_regex_rejected() -> None:
    with pytest.raises(ValidationError):
        AppPolicy.model_validate({"routing": {"min_tier": "large", "max_tier": "small"}})
    with pytest.raises(ValidationError):
        AppPolicy.model_validate({"redaction": {"custom_patterns": [{"name": "X", "regex": "("}]}})


def test_catalog_yaml_loads_and_prices() -> None:
    cat = ModelCatalog.from_yaml(get_settings().models_yaml)
    local = cat.get("ollama/qwen2.5:7b")
    assert local is not None and local.tier == "local"
    assert cat.cost("ollama/qwen2.5:7b", 10_000, 10_000) == Decimal("0.000000")


def test_cost_formula() -> None:
    cat = ModelCatalog.from_yaml(get_settings().models_yaml)
    m = cat.get("anthropic/claude-sonnet-5-5")
    assert m is not None
    expected = (Decimal(1200) * m.input_usd_per_mtok + Decimal(300) * m.output_usd_per_mtok) / Decimal(1_000_000)
    assert cat.cost(m.id, 1200, 300) == expected.quantize(Decimal("0.000001"))
    assert cat.cost("nope/unknown", 1000, 1000) == Decimal(0)
