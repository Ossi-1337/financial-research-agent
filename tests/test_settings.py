from __future__ import annotations

from pathlib import Path

from financial_research_agent.settings import ProviderTask, Settings


def test_settings_defaults_to_local_offline_provider() -> None:
    settings = Settings.from_env({})

    assert settings.environment == "local"
    assert settings.provider.llm_provider == "offline-test"
    assert settings.provider.llm_model == "offline-test"
    assert settings.provider.llm_base_url is None
    assert settings.provider.llm_local_runtime == "llama.cpp"
    assert settings.provider.llm_timeout_seconds == 30.0
    assert settings.provider.embedding_provider == "disabled"
    assert settings.provider.selection_for_task(ProviderTask.CHAT).provider == "offline-test"
    assert settings.provider.selection_for_task(ProviderTask.CHAT).model == "offline-test"


def test_settings_reads_environment_overrides(tmp_path: Path) -> None:
    app_home = tmp_path / "fra-home"
    settings = Settings.from_env(
        {
            "FRA_ENV": "test",
            "FRA_HOME": str(app_home),
            "FRA_LLM_PROVIDER": "local-openai",
            "FRA_LLM_MODEL": "qwen3:8b",
            "FRA_LLM_BASE_URL": "http://127.0.0.1:11434/v1",
            "FRA_LLM_LOCAL_RUNTIME": "ollama",
            "FRA_LLM_TIMEOUT_SECONDS": "12.5",
            "FRA_EMBEDDING_PROVIDER": "local-openai",
            "FRA_EMBEDDING_MODEL": "nomic-embed-text",
            "FRA_CHAT_PROVIDER": "online-chat",
            "FRA_CHAT_MODEL": "chat-model",
            "FRA_TOOL_CALLING_PROVIDER": "tool-provider",
            "FRA_TOOL_CALLING_MODEL": "tool-model",
            "FRA_STRUCTURED_OUTPUT_PROVIDER": "json-provider",
            "FRA_STRUCTURED_OUTPUT_MODEL": "json-model",
            "FRA_STREAMING_PROVIDER": "stream-provider",
            "FRA_STREAMING_MODEL": "stream-model",
        }
    )

    assert settings.environment == "test"
    assert settings.local_paths.app_home == app_home
    assert settings.local_paths.data_dir == app_home / "data"
    assert settings.provider.llm_provider == "local-openai"
    assert settings.provider.llm_model == "qwen3:8b"
    assert settings.provider.llm_base_url == "http://127.0.0.1:11434/v1"
    assert settings.provider.llm_local_runtime == "ollama"
    assert settings.provider.llm_timeout_seconds == 12.5
    assert settings.provider.embedding_provider == "local-openai"
    assert settings.provider.embedding_model == "nomic-embed-text"
    assert settings.provider.selection_for_task("chat").provider == "online-chat"
    assert settings.provider.selection_for_task("chat").model == "chat-model"
    assert settings.provider.selection_for_task("tool_calling").provider == "tool-provider"
    assert settings.provider.selection_for_task("tool_calling").model == "tool-model"
    assert settings.provider.selection_for_task("structured_output").provider == "json-provider"
    assert settings.provider.selection_for_task("structured_output").model == "json-model"
    assert settings.provider.selection_for_task("streaming").provider == "stream-provider"
    assert settings.provider.selection_for_task("streaming").model == "stream-model"
    assert settings.provider.selection_for_task("embeddings").provider == "local-openai"
    assert settings.provider.selection_for_task("embeddings").model == "nomic-embed-text"


def test_blank_environment_values_fall_back_to_defaults() -> None:
    settings = Settings.from_env(
        {
            "FRA_ENV": " ",
            "FRA_LLM_PROVIDER": "",
            "FRA_LLM_BASE_URL": " ",
            "FRA_LLM_LOCAL_RUNTIME": " ",
            "FRA_CHAT_PROVIDER": " ",
        }
    )

    assert settings.environment == "local"
    assert settings.provider.llm_provider == "offline-test"
    assert settings.provider.llm_base_url is None
    assert settings.provider.llm_local_runtime == "llama.cpp"
    assert settings.provider.chat_provider is None


def test_invalid_timeout_setting_is_rejected() -> None:
    try:
        Settings.from_env({"FRA_LLM_TIMEOUT_SECONDS": "0"})
    except ValueError as exc:
        assert "FRA_LLM_TIMEOUT_SECONDS must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid timeout setting to be rejected")


def test_task_provider_settings_fall_back_to_global_llm_settings() -> None:
    settings = Settings.from_env(
        {
            "FRA_LLM_PROVIDER": "local-openai",
            "FRA_LLM_MODEL": "qwen3:8b",
            "FRA_LLM_BASE_URL": "http://127.0.0.1:8080/v1",
            "FRA_CHAT_MODEL": "qwen3:14b",
        }
    )

    chat_selection = settings.provider.selection_for_task(ProviderTask.CHAT)
    tool_selection = settings.provider.selection_for_task(ProviderTask.TOOL_CALLING)

    assert chat_selection.provider == "local-openai"
    assert chat_selection.model == "qwen3:14b"
    assert chat_selection.base_url == "http://127.0.0.1:8080/v1"
    assert tool_selection.provider == "local-openai"
    assert tool_selection.model == "qwen3:8b"
