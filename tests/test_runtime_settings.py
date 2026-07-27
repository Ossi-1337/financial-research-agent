from __future__ import annotations

import json
from pathlib import Path

import pytest

from financial_research_agent.runtime_settings import (
    RuntimeSettingsOverrides,
    RuntimeSettingsStore,
)
from financial_research_agent.settings import Settings


def test_runtime_settings_overrides_are_applied_without_storing_secrets() -> None:
    base = Settings.from_env(
        {
            "FRA_OPENAI_API_KEY": "secret-key",
            "FRA_ANTHROPIC_API_KEY": "anthropic-key",
            "FRA_GEMINI_API_KEY": "gemini-key",
            "FRA_LITELLM_API_KEY": "litellm-key",
            "FRA_ALPHA_VANTAGE_API_KEY": "alpha-key",
        }
    )
    overrides = RuntimeSettingsOverrides(
        llm_provider="local-openai",
        llm_model="test-model",
        llm_base_url="http://127.0.0.1:8080/v1",
        embedding_provider="local-openai",
        retrieval_top_k=9,
        background_max_concurrent_research_runs=2,
    )

    settings = overrides.apply_to(base)
    payload = overrides.to_dict()

    assert settings.provider.llm_provider == "local-openai"
    assert settings.provider.llm_model == "test-model"
    assert settings.provider.llm_base_url == "http://127.0.0.1:8080/v1"
    assert settings.provider.openai_api_key == "secret-key"
    assert settings.provider.anthropic_api_key == "anthropic-key"
    assert settings.provider.gemini_api_key == "gemini-key"
    assert settings.provider.litellm_api_key == "litellm-key"
    assert settings.data_sources.alpha_vantage_api_key == "alpha-key"
    assert settings.retrieval.top_k == 9
    assert settings.background.max_concurrent_research_runs == 2
    assert "openai_api_key" not in payload
    assert "alpha_vantage_api_key" not in payload


def test_runtime_settings_store_persists_versioned_non_secret_overrides(tmp_path: Path) -> None:
    base = Settings.from_env({"FRA_HOME": str(tmp_path), "FRA_OPENAI_API_KEY": "secret-key"})
    store = RuntimeSettingsStore.from_settings(base)

    saved = store.update(
        RuntimeSettingsOverrides(llm_provider="offline-test", llm_model="custom-offline"),
        base_settings=base,
    )
    reloaded = RuntimeSettingsStore.from_settings(base)
    payload = json.loads((tmp_path / "data" / "settings_overrides.json").read_text())

    assert saved.llm_model == "custom-offline"
    assert reloaded.get().llm_model == "custom-offline"
    assert payload["version"] == 1
    assert payload["overrides"]["llm_model"] == "custom-offline"
    assert "secret-key" not in json.dumps(payload)


def test_runtime_settings_reject_unknown_and_secret_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported runtime setting"):
        RuntimeSettingsOverrides.from_mapping({"unknown": "value"})

    with pytest.raises(ValueError, match="Secret settings are environment-only"):
        RuntimeSettingsOverrides.from_mapping({"openai_api_key": "secret"})
    for field in ("anthropic_api_key", "gemini_api_key", "litellm_api_key"):
        with pytest.raises(ValueError, match="Secret settings are environment-only"):
            RuntimeSettingsOverrides.from_mapping({field: "secret"})


def test_runtime_settings_store_clear_removes_overrides(tmp_path: Path) -> None:
    base = Settings.from_env({"FRA_HOME": str(tmp_path)})
    store = RuntimeSettingsStore.from_settings(base)
    store.update(RuntimeSettingsOverrides(llm_model="custom-offline"), base_settings=base)

    cleared = store.clear()

    assert cleared.to_dict() == {}
    assert RuntimeSettingsStore.from_settings(base).get().to_dict() == {}
