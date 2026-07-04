# Financial Research Agent

Financial Research Agent is a local-first AI agent system for company and stock research.
The project is currently a Python foundation with provider-neutral LLM contracts,
an offline test provider, an OpenAI-compatible local endpoint adapter, and an optional
hosted OpenAI adapter. It also has a deterministic tool registry foundation, but it does
not run multi-agent research yet. It ingests daily market data when a provider key is
configured, SEC financial statements for SEC filers, and SEC HTML/TXT filing documents.
A minimal local chat UI is available for direct LLM chat.

## Status

Implemented:

- Python 3.14 package foundation.
- Environment-driven local settings.
- Health-check CLI.
- Immutable domain models for financial research concepts.
- Async-first LLM provider contracts for chat, tools, structured output, embeddings, streaming, model metadata, token usage, and provider errors.
- Deterministic `offline-test` provider for tests and local development without network access.
- `local-openai` provider adapter for OpenAI-compatible local endpoints.
- `openai` provider adapter for hosted OpenAI Chat Completions and embeddings.
- Local endpoint health checks for reachability, selected model, available models, capabilities, and known limitations.
- Deterministic tool registry and guarded tool-call loop for local function calling.
- Agent prompt contracts, role definitions, and structured JSON output schemas.
- Local FastAPI chat UI with persistent local sessions, bounded context, and direct provider-backed responses.
- SEC company ticker search for reviewable company/security entity candidates with cached source metadata.
- Alpha Vantage daily historical price ingestion with local JSON storage, freshness warnings, and deterministic price metrics.
- SEC companyfacts financial statement ingestion with normalized statements, key ratios, local JSON storage, and source metadata.
- SEC EDGAR filing/document ingestion for primary HTML/TXT filings with local raw document, extracted text, chunk metadata, and source metadata storage.
- Local storage manifest, migration, cache inspection, cache clear, and local data reset commands for `FRA_HOME`.

Not implemented yet:

- Anthropic, Gemini, or gateway provider integrations.
- Agent runtime and orchestration.
- PDF, news, or macro ingestion.
- SQLite/remote database, vector search, or RAG.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run

```powershell
python -m financial_research_agent --pretty
```

The default provider is `offline-test`, so no API keys or local model server are required.

## Run Chat UI

```powershell
python -m financial_research_agent serve --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`. The UI uses the configured chat provider and model.
Sessions are persisted under `FRA_HOME` and can be reopened or cleared locally. The chat
endpoint does not use live financial data tools, RAG, or multi-agent orchestration yet.
Type `@company` in the composer to search SEC company tickers and insert a resolved
company mention chip. Mentions are sent to the LLM as identifier context only; they do not
automatically fetch prices, statements, filings, RAG evidence, or run agents.

## Tools

The tracked `financial_research_agent.tools` package defines deterministic tool contracts,
a guarded registry, basic JSON-schema argument validation, and a provider-neutral tool-call
loop. Current built-in tools are limited to UTC time, simple ratio calculation, a company
lookup tool when a real provider is injected, a company lookup stub when no provider is
injected, and injected in-memory evidence reads. They do not execute shell commands,
browse the web, use a database, or fetch prices/filings/news.

## Company Lookup

The first entity-resolution source is the SEC company ticker list. In the UI, use
`@company` mentions in the chat composer. The API endpoint remains available directly:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/company-search?query=Novo%20Nordisk"
```

Results are candidates for user review, not automatic selection. SEC ticker data includes
CIK, ticker, and company title only, so missing exchange, currency, country, and ISIN fields
are explicit until a later identifier source such as OpenFIGI is added.

## Market Data

The first market data provider is Alpha Vantage daily prices:

```powershell
$env:FRA_ALPHA_VANTAGE_API_KEY = "your-alpha-vantage-key"
Invoke-RestMethod "http://127.0.0.1:8000/api/market-data/history" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"symbol":"NVO","refresh":true}'
```

Historical bars are persisted locally under `FRA_HOME/data/market_data_price_bars.json`.
The API returns source metadata, freshness warnings, and deterministic metrics including
returns, moving averages, volatility, and max drawdown. This is delayed/provider-limited
research data, not a trading feed.

## Financial Statements

The first financial statement provider is the official SEC companyfacts XBRL API for SEC
filers:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/financial-statements" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"cik":"0000320193","fiscal_years":3,"refresh":true}'
```

Statement results are persisted locally under `FRA_HOME/data/financial_statements.json`.
The API returns normalized annual income statement, balance sheet, cash flow, and ratio
rows where supported facts are available, plus source metadata and warnings for missing
periods, restatements, ignored units, and stale cached data.

## Filings

The first filing document provider is SEC EDGAR submissions plus SEC Archives primary
documents:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/filings/ingest" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"cik":"0000320193","forms":["10-K","10-Q"],"limit":1,"refresh":true}'
```

Raw documents, extracted text, chunks, and metadata are stored locally under
`FRA_HOME/data/filings/`. Only SEC primary HTML/TXT documents are extracted at runtime.
PDF extraction, vector search, RAG, document interpretation, and redistribution are
deferred.

## Local Storage And Cache

Local runtime state lives under `FRA_HOME`, which defaults to
`~/.financial-research-agent`. Milestone 16 keeps the existing local JSON/file storage
formats and adds shared inspection and maintenance commands:

```powershell
python -m financial_research_agent storage-status --pretty
python -m financial_research_agent storage-migrate --pretty
python -m financial_research_agent cache-clear --pretty
python -m financial_research_agent data-reset --yes --pretty
```

`cache-clear` removes clearable provider cache entries, such as SEC company ticker cache,
without deleting chat sessions or stored research data. `data-reset --yes` removes local
chat/research data and cache files while leaving logs alone. Secrets are not stored in
ordinary app tables or local JSON stores.

## Agent Prompts

The tracked `financial_research_agent.agents` package defines reusable prompt contracts
for orchestrator, financial report, stock, news/macro, and synthesis roles. These contracts
include allowed tool names and structured JSON output schemas. They do not run agents,
call providers, fetch data, or produce investment recommendations.

## Local Inference

The first documented local runtime path is `llama.cpp` through its OpenAI-compatible
server. Start `llama-server` with your own GGUF model file, then configure:

```powershell
$env:FRA_LLM_PROVIDER = "local-openai"
$env:FRA_LLM_MODEL = "your-local-model-name"
$env:FRA_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
$env:FRA_LLM_LOCAL_RUNTIME = "llama.cpp"
python -m financial_research_agent --pretty
```

The project does not bundle model files, download models automatically, or apply
GPU-specific tuning. Tool calls, structured output, and embeddings depend on the local
server, model, chat template, and runtime flags.

Ollama can also be used through its OpenAI-compatible endpoint by setting
`FRA_LLM_BASE_URL` to `http://127.0.0.1:11434/v1` and
`FRA_LLM_LOCAL_RUNTIME` to `ollama`.

## Hosted OpenAI

The hosted OpenAI adapter is optional and disabled unless selected in the environment.
It uses direct async HTTP through `httpx`, not the OpenAI SDK:

```powershell
$env:FRA_LLM_PROVIDER = "openai"
$env:FRA_LLM_MODEL = "your-openai-model"
$env:FRA_OPENAI_API_KEY = "your-api-key"
python -m financial_research_agent --pretty
```

`OPENAI_API_KEY`, `OPENAI_ORG_ID`, and `OPENAI_PROJECT_ID` are also supported as fallback
aliases. Do not commit real keys. The adapter uses Chat Completions for parity with
local OpenAI-compatible runtimes; newer OpenAI-only workflows can add Responses API
support behind the same provider boundary later.

Runtime references:

- llama.cpp server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Ollama OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility
- OpenAI API overview: https://developers.openai.com/api/reference/overview/
- OpenAI Chat Completions: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/
- OpenAI embeddings: https://developers.openai.com/api/reference/resources/embeddings/methods/create/
- SEC EDGAR APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- SEC EDGAR access policy: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data

## Configuration

Settings are read from real environment variables. `.env.example` documents the supported
`FRA_*` variables. `FRA` means Financial Research Agent. The settings include
task-specific provider/model overrides for chat, tool calling, structured output, and
streaming. Local storage is controlled by `FRA_HOME` and
`FRA_STORAGE_PROVIDER=local-json`. Chat history context is controlled by
`FRA_CHAT_HISTORY_RECENT_TURNS` and
`FRA_CHAT_HISTORY_SUMMARY_MAX_CHARS`. Company lookup is controlled by
`FRA_COMPANY_LOOKUP_PROVIDER`, `FRA_COMPANY_LOOKUP_CACHE_TTL_DAYS`, and
`FRA_SEC_USER_AGENT`. Market data is controlled by `FRA_MARKET_DATA_PROVIDER`,
`FRA_MARKET_DATA_CACHE_TTL_DAYS`, and `FRA_ALPHA_VANTAGE_API_KEY`.
Financial statements are controlled by `FRA_FINANCIAL_STATEMENT_PROVIDER` and
`FRA_FINANCIAL_STATEMENT_CACHE_TTL_DAYS`. Filing ingestion is controlled by
`FRA_FILING_PROVIDER`, `FRA_FILING_CACHE_TTL_DAYS`, and
`FRA_FILING_MAX_DOCUMENT_BYTES`.
`.env.example` is a reference file only; the app does not auto-load `.env` yet.

## Verify

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m compileall -q src tests
git diff --check
```
