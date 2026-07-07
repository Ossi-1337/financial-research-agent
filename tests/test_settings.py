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
    assert settings.provider.openai_api_key is None
    assert settings.provider.openai_base_url == "https://api.openai.com/v1"
    assert settings.provider.selection_for_task(ProviderTask.CHAT).provider == "offline-test"
    assert settings.provider.selection_for_task(ProviderTask.CHAT).model == "offline-test"
    assert settings.data_sources.financial_statement_provider == "sec-companyfacts"
    assert settings.data_sources.financial_statement_cache_ttl_days == 30
    assert "@" in settings.data_sources.sec_user_agent
    assert "contact-not-configured" not in settings.data_sources.sec_user_agent
    assert settings.data_sources.filing_provider == "sec-edgar"
    assert settings.data_sources.filing_cache_ttl_days == 30
    assert settings.data_sources.filing_max_document_bytes == 8_000_000
    assert settings.storage.provider == "local-json"
    assert settings.retrieval.provider == "local-vector"
    assert settings.retrieval.top_k == 5
    assert settings.retrieval.min_score == 0.0
    assert settings.interoperability.enabled is False
    assert settings.interoperability.local_only is True
    assert settings.interoperability.api_key is None
    assert settings.background.max_concurrent_research_runs == 1


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
            "FRA_OPENAI_API_KEY": "test-key",
            "FRA_OPENAI_BASE_URL": "https://api.openai.test/v1",
            "FRA_OPENAI_ORGANIZATION": "org_123",
            "FRA_OPENAI_PROJECT": "proj_123",
            "FRA_CHAT_PROVIDER": "online-chat",
            "FRA_CHAT_MODEL": "chat-model",
            "FRA_TOOL_CALLING_PROVIDER": "tool-provider",
            "FRA_TOOL_CALLING_MODEL": "tool-model",
            "FRA_STRUCTURED_OUTPUT_PROVIDER": "json-provider",
            "FRA_STRUCTURED_OUTPUT_MODEL": "json-model",
            "FRA_STREAMING_PROVIDER": "stream-provider",
            "FRA_STREAMING_MODEL": "stream-model",
            "FRA_CHAT_HISTORY_RECENT_TURNS": "4",
            "FRA_CHAT_HISTORY_SUMMARY_MAX_CHARS": "500",
            "FRA_COMPANY_LOOKUP_PROVIDER": "sec",
            "FRA_COMPANY_LOOKUP_CACHE_TTL_DAYS": "7",
            "FRA_SEC_USER_AGENT": "financial-research-agent-test/0.1 contact",
            "FRA_MARKET_DATA_PROVIDER": "alpha-vantage",
            "FRA_MARKET_DATA_CACHE_TTL_DAYS": "2",
            "FRA_ALPHA_VANTAGE_API_KEY": "alpha-key",
            "FRA_FILING_PROVIDER": "sec-edgar",
            "FRA_FILING_CACHE_TTL_DAYS": "12",
            "FRA_FILING_MAX_DOCUMENT_BYTES": "5000000",
            "FRA_STORAGE_PROVIDER": "local-json",
            "FRA_RETRIEVAL_PROVIDER": "local-vector",
            "FRA_RETRIEVAL_TOP_K": "7",
            "FRA_RETRIEVAL_MIN_SCORE": "0.25",
            "FRA_INTEROP_ENABLED": "true",
            "FRA_INTEROP_LOCAL_ONLY": "false",
            "FRA_INTEROP_API_KEY": "interop-key",
            "FRA_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS": "2",
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
    assert settings.provider.openai_api_key == "test-key"
    assert settings.provider.openai_base_url == "https://api.openai.test/v1"
    assert settings.provider.openai_organization == "org_123"
    assert settings.provider.openai_project == "proj_123"
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
    assert settings.chat.history_recent_turns == 4
    assert settings.chat.history_summary_max_chars == 500
    assert settings.data_sources.company_lookup_provider == "sec"
    assert settings.data_sources.company_lookup_cache_ttl_days == 7
    assert settings.data_sources.sec_user_agent == "financial-research-agent-test/0.1 contact"
    assert settings.data_sources.market_data_provider == "alpha-vantage"
    assert settings.data_sources.market_data_cache_ttl_days == 2
    assert settings.data_sources.alpha_vantage_api_key == "alpha-key"
    assert settings.data_sources.financial_statement_provider == "sec-companyfacts"
    assert settings.data_sources.financial_statement_cache_ttl_days == 30
    assert settings.data_sources.filing_provider == "sec-edgar"
    assert settings.data_sources.filing_cache_ttl_days == 12
    assert settings.data_sources.filing_max_document_bytes == 5_000_000
    assert settings.storage.provider == "local-json"
    assert settings.retrieval.provider == "local-vector"
    assert settings.retrieval.top_k == 7
    assert settings.retrieval.min_score == 0.25
    assert settings.interoperability.enabled is True
    assert settings.interoperability.local_only is False
    assert settings.interoperability.api_key == "interop-key"
    assert settings.background.max_concurrent_research_runs == 2


def test_blank_environment_values_fall_back_to_defaults() -> None:
    settings = Settings.from_env(
        {
            "FRA_ENV": " ",
            "FRA_LLM_PROVIDER": "",
            "FRA_LLM_BASE_URL": " ",
            "FRA_LLM_LOCAL_RUNTIME": " ",
            "FRA_OPENAI_API_KEY": " ",
            "FRA_CHAT_PROVIDER": " ",
        }
    )

    assert settings.environment == "local"
    assert settings.provider.llm_provider == "offline-test"
    assert settings.provider.llm_base_url is None
    assert settings.provider.llm_local_runtime == "llama.cpp"
    assert settings.provider.openai_api_key is None
    assert settings.provider.chat_provider is None


def test_openai_settings_support_standard_openai_environment_aliases() -> None:
    settings = Settings.from_env(
        {
            "OPENAI_API_KEY": "standard-key",
            "OPENAI_ORG_ID": "org_standard",
            "OPENAI_PROJECT_ID": "proj_standard",
        }
    )

    assert settings.provider.openai_api_key == "standard-key"
    assert settings.provider.openai_organization == "org_standard"
    assert settings.provider.openai_project == "proj_standard"


def test_invalid_timeout_setting_is_rejected() -> None:
    try:
        Settings.from_env({"FRA_LLM_TIMEOUT_SECONDS": "0"})
    except ValueError as exc:
        assert "FRA_LLM_TIMEOUT_SECONDS must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid timeout setting to be rejected")


def test_invalid_chat_history_settings_are_rejected() -> None:
    try:
        Settings.from_env({"FRA_CHAT_HISTORY_RECENT_TURNS": "0"})
    except ValueError as exc:
        assert "FRA_CHAT_HISTORY_RECENT_TURNS must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid recent turns setting to be rejected")

    try:
        Settings.from_env({"FRA_CHAT_HISTORY_SUMMARY_MAX_CHARS": "many"})
    except ValueError as exc:
        assert "FRA_CHAT_HISTORY_SUMMARY_MAX_CHARS must be an integer" in str(exc)
    else:
        raise AssertionError("Expected invalid summary max chars setting to be rejected")


def test_invalid_company_lookup_cache_ttl_is_rejected() -> None:
    try:
        Settings.from_env({"FRA_COMPANY_LOOKUP_CACHE_TTL_DAYS": "0"})
    except ValueError as exc:
        assert "FRA_COMPANY_LOOKUP_CACHE_TTL_DAYS must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid company lookup cache TTL to be rejected")


def test_invalid_market_data_cache_ttl_is_rejected() -> None:
    try:
        Settings.from_env({"FRA_MARKET_DATA_CACHE_TTL_DAYS": "0"})
    except ValueError as exc:
        assert "FRA_MARKET_DATA_CACHE_TTL_DAYS must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid market data cache TTL to be rejected")


def test_financial_statement_settings_are_read_and_validated() -> None:
    settings = Settings.from_env(
        {
            "FRA_FINANCIAL_STATEMENT_PROVIDER": "sec-companyfacts",
            "FRA_FINANCIAL_STATEMENT_CACHE_TTL_DAYS": "14",
        }
    )

    assert settings.data_sources.financial_statement_provider == "sec-companyfacts"
    assert settings.data_sources.financial_statement_cache_ttl_days == 14

    try:
        Settings.from_env({"FRA_FINANCIAL_STATEMENT_CACHE_TTL_DAYS": "0"})
    except ValueError as exc:
        assert "FRA_FINANCIAL_STATEMENT_CACHE_TTL_DAYS must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid financial statement cache TTL to be rejected")


def test_filing_settings_are_read_and_validated() -> None:
    settings = Settings.from_env(
        {
            "FRA_FILING_PROVIDER": "sec-edgar",
            "FRA_FILING_CACHE_TTL_DAYS": "14",
            "FRA_FILING_MAX_DOCUMENT_BYTES": "1000",
        }
    )

    assert settings.data_sources.filing_provider == "sec-edgar"
    assert settings.data_sources.filing_cache_ttl_days == 14
    assert settings.data_sources.filing_max_document_bytes == 1000

    try:
        Settings.from_env({"FRA_FILING_CACHE_TTL_DAYS": "0"})
    except ValueError as exc:
        assert "FRA_FILING_CACHE_TTL_DAYS must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid filing cache TTL to be rejected")

    try:
        Settings.from_env({"FRA_FILING_MAX_DOCUMENT_BYTES": "0"})
    except ValueError as exc:
        assert "FRA_FILING_MAX_DOCUMENT_BYTES must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid filing max bytes to be rejected")


def test_retrieval_settings_are_read_and_validated() -> None:
    settings = Settings.from_env(
        {
            "FRA_RETRIEVAL_PROVIDER": "local-vector",
            "FRA_RETRIEVAL_TOP_K": "3",
            "FRA_RETRIEVAL_MIN_SCORE": "0.1",
        }
    )

    assert settings.retrieval.provider == "local-vector"
    assert settings.retrieval.top_k == 3
    assert settings.retrieval.min_score == 0.1

    try:
        Settings.from_env({"FRA_RETRIEVAL_TOP_K": "0"})
    except ValueError as exc:
        assert "FRA_RETRIEVAL_TOP_K must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid retrieval top_k to be rejected")

    try:
        Settings.from_env({"FRA_RETRIEVAL_MIN_SCORE": "2"})
    except ValueError as exc:
        assert "FRA_RETRIEVAL_MIN_SCORE must be between -1 and 1" in str(exc)
    else:
        raise AssertionError("Expected invalid retrieval min score to be rejected")


def test_alpha_vantage_key_supports_standard_alias() -> None:
    settings = Settings.from_env({"ALPHA_VANTAGE_API_KEY": "standard-alpha-key"})

    assert settings.data_sources.alpha_vantage_api_key == "standard-alpha-key"


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


def test_interoperability_settings_are_read_and_validated() -> None:
    local_settings = Settings.from_env(
        {
            "FRA_INTEROP_ENABLED": "1",
            "FRA_INTEROP_LOCAL_ONLY": "yes",
        }
    )

    assert local_settings.interoperability.enabled is True
    assert local_settings.interoperability.local_only is True
    assert local_settings.interoperability.api_key is None
    assert local_settings.interoperability.to_dict()["api_key_configured"] is False

    remote_settings = Settings.from_env(
        {
            "FRA_INTEROP_ENABLED": "true",
            "FRA_INTEROP_LOCAL_ONLY": "false",
            "FRA_INTEROP_API_KEY": "secret",
        }
    )

    assert remote_settings.interoperability.enabled is True
    assert remote_settings.interoperability.local_only is False
    assert remote_settings.interoperability.api_key == "secret"

    try:
        Settings.from_env({"FRA_INTEROP_ENABLED": "maybe"})
    except ValueError as exc:
        assert "FRA_INTEROP_ENABLED must be a boolean" in str(exc)
    else:
        raise AssertionError("Expected invalid interop boolean to be rejected")

    try:
        Settings.from_env(
            {
                "FRA_INTEROP_ENABLED": "true",
                "FRA_INTEROP_LOCAL_ONLY": "false",
            }
        )
    except ValueError as exc:
        assert "FRA_INTEROP_API_KEY is required" in str(exc)
    else:
        raise AssertionError("Expected remote interop without API key to be rejected")


def test_invalid_background_research_limit_is_rejected() -> None:
    try:
        Settings.from_env({"FRA_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS": "0"})
    except ValueError as exc:
        assert "FRA_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid background research limit to be rejected")
