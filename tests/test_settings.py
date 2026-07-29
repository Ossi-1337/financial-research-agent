from __future__ import annotations

from pathlib import Path

import pytest

from financial_research_agent.settings import ProviderTask, Settings


def test_settings_defaults_to_local_offline_provider() -> None:
    settings = Settings.from_env({})

    assert settings.environment == "local"
    assert settings.provider.llm_provider == "offline-test"
    assert settings.provider.llm_model == "offline-test"
    assert settings.provider.llm_base_url is None
    assert settings.provider.llm_local_runtime == "llama.cpp"
    assert settings.provider.llm_timeout_seconds == 90.0
    assert settings.provider.embedding_provider == "disabled"
    assert settings.provider.openai_api_key is None
    assert settings.provider.openai_base_url == "https://api.openai.com/v1"
    assert settings.provider.anthropic_api_key is None
    assert settings.provider.gemini_api_key is None
    assert settings.provider.litellm_api_key is None
    assert settings.provider.selection_for_task(ProviderTask.CHAT).provider == "offline-test"
    assert settings.provider.selection_for_task(ProviderTask.CHAT).model == "offline-test"
    assert settings.data_sources.financial_statement_provider == "sec-companyfacts"
    assert settings.data_sources.financial_statement_cache_ttl_days == 30
    assert "@" in settings.data_sources.sec_user_agent
    assert "contact-not-configured" not in settings.data_sources.sec_user_agent
    assert settings.data_sources.filing_provider == "sec-edgar"
    assert settings.data_sources.filing_cache_ttl_days == 30
    assert settings.data_sources.filing_max_document_bytes == 8_000_000
    assert settings.storage.provider == "sqlite"
    assert settings.retrieval.provider == "local-vector"
    assert settings.retrieval.top_k == 5
    assert settings.retrieval.min_score == 0.0
    assert settings.a2a.enabled is False
    assert settings.a2a.local_only is True
    assert settings.a2a.max_concurrent_tasks == 1
    assert settings.a2a.max_queued_tasks == 8
    assert settings.a2a.max_input_chars == 4_000
    assert settings.a2a.public_base_url == "http://127.0.0.1:8001"
    assert settings.a2a.financial_report_url == "http://127.0.0.1:8002"
    assert settings.a2a.stock_url == "http://127.0.0.1:8003"
    assert settings.a2a.context_url == "http://127.0.0.1:8004"
    assert settings.a2a.synthesis_url == "http://127.0.0.1:8005"
    assert settings.a2a.delegation_timeout_seconds == 120.0
    assert settings.a2a.delegation_max_attempts == 2
    assert settings.background.max_concurrent_research_runs == 1
    assert settings.performance.prompt_budget_input_tokens == 16_000
    assert settings.performance.prompt_budget_output_tokens == 1_024
    assert settings.performance.agent_max_output_tokens == 2_048
    assert settings.performance.embedding_cache_enabled is True
    assert settings.security.allow_remote_bind is False


def test_remote_a2a_requires_environment_only_bearer_key() -> None:
    with pytest.raises(ValueError, match="FRA_A2A_API_KEY"):
        Settings.from_env(
            {
                "FRA_A2A_ENABLED": "true",
                "FRA_A2A_LOCAL_ONLY": "false",
            }
        )


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
            "FRA_ANTHROPIC_API_KEY": "anthropic-key",
            "FRA_ANTHROPIC_BASE_URL": "https://api.anthropic.test/v1",
            "FRA_ANTHROPIC_API_VERSION": "2023-06-01",
            "FRA_GEMINI_API_KEY": "gemini-key",
            "FRA_GEMINI_BASE_URL": "https://gemini.test/v1beta",
            "FRA_LITELLM_API_KEY": "litellm-key",
            "FRA_LITELLM_BASE_URL": "http://127.0.0.1:4000/v1",
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
            "FRA_A2A_ENABLED": "true",
            "FRA_A2A_LOCAL_ONLY": "false",
            "FRA_A2A_API_KEY": "a2a-key",
            "FRA_A2A_MAX_CONCURRENT_TASKS": "2",
            "FRA_A2A_MAX_QUEUED_TASKS": "5",
            "FRA_A2A_MAX_INPUT_CHARS": "2000",
            "FRA_A2A_PUBLIC_BASE_URL": "https://a2a.example.test",
            "FRA_A2A_FINANCIAL_REPORT_URL": "http://financial.test",
            "FRA_A2A_STOCK_URL": "http://stock.test",
            "FRA_A2A_CONTEXT_URL": "http://context.test",
            "FRA_A2A_SYNTHESIS_URL": "http://synthesis.test",
            "FRA_A2A_DELEGATION_TIMEOUT_SECONDS": "15",
            "FRA_A2A_DELEGATION_MAX_ATTEMPTS": "3",
            "FRA_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS": "2",
            "FRA_PROMPT_BUDGET_INPUT_TOKENS": "12000",
            "FRA_PROMPT_BUDGET_OUTPUT_TOKENS": "800",
            "FRA_AGENT_MAX_OUTPUT_TOKENS": "1400",
            "FRA_EMBEDDING_CACHE_ENABLED": "false",
            "FRA_ALLOW_REMOTE_BIND": "true",
        }
    )

    assert settings.environment == "test"
    assert settings.local_paths.app_home == app_home
    assert settings.local_paths.data_dir == app_home / "data"
    assert settings.provider.llm_provider == "local-openai"
    assert settings.provider.llm_model == "qwen3:8b"
    assert settings.provider.llm_base_url == "http://127.0.0.1:11434/v1"
    assert settings.provider.llm_local_runtime == "ollama"
    assert settings.a2a.enabled is True
    assert settings.a2a.local_only is False
    assert settings.a2a.api_key == "a2a-key"
    assert settings.a2a.max_concurrent_tasks == 2
    assert settings.a2a.max_queued_tasks == 5
    assert settings.a2a.max_input_chars == 2_000
    assert settings.a2a.public_base_url == "https://a2a.example.test"
    assert settings.a2a.financial_report_url == "http://financial.test"
    assert settings.a2a.stock_url == "http://stock.test"
    assert settings.a2a.context_url == "http://context.test"
    assert settings.a2a.synthesis_url == "http://synthesis.test"
    assert settings.a2a.delegation_timeout_seconds == 15.0
    assert settings.a2a.delegation_max_attempts == 3
    assert settings.provider.llm_timeout_seconds == 12.5
    assert settings.provider.embedding_provider == "local-openai"
    assert settings.provider.embedding_model == "nomic-embed-text"
    assert settings.provider.openai_api_key == "test-key"
    assert settings.provider.openai_base_url == "https://api.openai.test/v1"
    assert settings.provider.openai_organization == "org_123"
    assert settings.provider.openai_project == "proj_123"
    assert settings.provider.anthropic_api_key == "anthropic-key"
    assert settings.provider.anthropic_base_url == "https://api.anthropic.test/v1"
    assert settings.provider.gemini_api_key == "gemini-key"
    assert settings.provider.gemini_base_url == "https://gemini.test/v1beta"
    assert settings.provider.litellm_api_key == "litellm-key"
    assert settings.provider.litellm_base_url == "http://127.0.0.1:4000/v1"
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
    assert settings.background.max_concurrent_research_runs == 2
    assert settings.performance.prompt_budget_input_tokens == 12_000
    assert settings.performance.prompt_budget_output_tokens == 800
    assert settings.performance.agent_max_output_tokens == 1_400
    assert settings.performance.embedding_cache_enabled is False
    assert settings.security.allow_remote_bind is True


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


def test_hosted_provider_settings_support_vendor_environment_aliases() -> None:
    settings = Settings.from_env(
        {
            "ANTHROPIC_API_KEY": "anthropic-standard",
            "GEMINI_API_KEY": "gemini-standard",
        }
    )

    assert settings.provider.anthropic_api_key == "anthropic-standard"
    assert settings.provider.gemini_api_key == "gemini-standard"
    payload = settings.provider.to_dict()
    assert payload["anthropic_api_key_configured"] is True
    assert payload["gemini_api_key_configured"] is True
    assert "anthropic-standard" not in str(payload)
    assert "gemini-standard" not in str(payload)


def test_hosted_providers_require_explicit_models() -> None:
    for provider in ("anthropic", "gemini", "litellm", "openai"):
        try:
            Settings.from_env({"FRA_LLM_PROVIDER": provider})
        except ValueError as exc:
            assert "FRA_LLM_MODEL must be explicitly configured" in str(exc)
        else:
            raise AssertionError(f"Expected {provider} without a model to be rejected")

    try:
        Settings.from_env({"FRA_EMBEDDING_PROVIDER": "gemini"})
    except ValueError as exc:
        assert "FRA_EMBEDDING_MODEL must be explicitly configured" in str(exc)
    else:
        raise AssertionError("Expected Gemini embeddings without a model to be rejected")


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


def test_invalid_background_research_limit_is_rejected() -> None:
    try:
        Settings.from_env({"FRA_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS": "0"})
    except ValueError as exc:
        assert "FRA_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid background research limit to be rejected")
