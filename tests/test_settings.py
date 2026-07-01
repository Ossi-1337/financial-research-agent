from __future__ import annotations

from pathlib import Path

from financial_research_agent.settings import Settings


def test_settings_defaults_to_local_offline_provider() -> None:
    settings = Settings.from_env({})

    assert settings.environment == "local"
    assert settings.provider.llm_provider == "offline-test"
    assert settings.provider.llm_model == "offline-test"
    assert settings.provider.llm_base_url is None
    assert settings.provider.embedding_provider == "disabled"


def test_settings_reads_environment_overrides(tmp_path: Path) -> None:
    app_home = tmp_path / "fra-home"
    settings = Settings.from_env(
        {
            "FRA_ENV": "test",
            "FRA_HOME": str(app_home),
            "FRA_LLM_PROVIDER": "local-openai",
            "FRA_LLM_MODEL": "qwen3:8b",
            "FRA_LLM_BASE_URL": "http://127.0.0.1:11434/v1",
            "FRA_EMBEDDING_PROVIDER": "local-openai",
            "FRA_EMBEDDING_MODEL": "nomic-embed-text",
        }
    )

    assert settings.environment == "test"
    assert settings.local_paths.app_home == app_home
    assert settings.local_paths.data_dir == app_home / "data"
    assert settings.provider.llm_provider == "local-openai"
    assert settings.provider.llm_model == "qwen3:8b"
    assert settings.provider.llm_base_url == "http://127.0.0.1:11434/v1"
    assert settings.provider.embedding_provider == "local-openai"
    assert settings.provider.embedding_model == "nomic-embed-text"


def test_blank_environment_values_fall_back_to_defaults() -> None:
    settings = Settings.from_env(
        {
            "FRA_ENV": " ",
            "FRA_LLM_PROVIDER": "",
            "FRA_LLM_BASE_URL": " ",
        }
    )

    assert settings.environment == "local"
    assert settings.provider.llm_provider == "offline-test"
    assert settings.provider.llm_base_url is None
