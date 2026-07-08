# Financial Research Agent

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker Compose](https://img.shields.io/badge/runtime-Docker%20Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![llama.cpp](https://img.shields.io/badge/local%20LLM-llama.cpp-555555)](https://github.com/ggml-org/llama.cpp)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Financial Research Agent is a local-first research workspace for company and stock analysis.

It combines a lightweight chat UI, provider-neutral LLM adapters, deterministic tools, SEC
filing and statement ingestion, local storage, retrieval, citations, and bounded specialist
workflows. The project is designed so local models can run first, while hosted providers can
be swapped in through the same interfaces when needed.

The goal is not to create an automated trading system. The goal is to make financial
research workflows inspectable, source-aware, and runnable on a developer machine.

## Requirements

- Python 3.14
- Docker Desktop for the recommended local model setup
- Optional NVIDIA GPU support for the default llama.cpp CUDA container
- Optional Alpha Vantage API key for daily market data
- Optional OpenAI API key for hosted LLM usage
- A real SEC User-Agent/contact string for serious SEC EDGAR usage

## Quick Start

Run the full local stack with Docker Compose:

```powershell
docker compose up --build
```

Open the app:

```text
http://127.0.0.1:8000
```

The Compose setup starts:

- `llama-cpp` on `http://127.0.0.1:8080/v1`
- the Financial Research Agent web app on `http://127.0.0.1:8000`
- a persistent Docker volume for local app data

By default, Compose uses:

```text
unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL
```

The first run downloads the llama.cpp image and the selected GGUF model into your Hugging
Face cache. Override the model if needed:

```powershell
$env:FRA_LOCAL_MODEL = "your-huggingface-gguf-repo:quant"
docker compose up --build
```

Stop the stack:

```powershell
docker compose down
```

## Python Development

Install the project locally:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the health check:

```powershell
python -m financial_research_agent --pretty
```

Run the web UI without Docker:

```powershell
python -m financial_research_agent serve --host 127.0.0.1 --port 8000
```

The default provider is `offline-test`, so the Python-only path works without API keys or
a local model server. Configure `local-openai` or `openai` when you want real model calls.

## What Works Today

- Local chat UI with persisted sessions, bounded context, and streamed assistant output.
- Settings panel for local provider/model, endpoint, embeddings, retrieval, cache, and
  background-run controls.
- Inline `@company` mentions backed by SEC company ticker search.
- Provider-neutral LLM contracts for chat, streaming, embeddings, tool calls, structured
  output, model metadata, usage, and provider errors.
- Deterministic `offline-test` provider for tests and local development.
- OpenAI-compatible local provider for llama.cpp, Ollama, and similar local endpoints.
- Optional hosted OpenAI provider.
- Deterministic tool registry with guarded function calling.
- Deny-by-default tool name/permission allowlists and untrusted-document prompt framing.
- SEC company lookup using the official company ticker list.
- Alpha Vantage daily market data ingestion when an API key is configured.
- SEC companyfacts financial statement ingestion for SEC filers.
- SEC EDGAR filing ingestion for primary HTML/TXT documents.
- Local raw document, extracted text, chunk, metadata, and JSON store management.
- Local vector retrieval over stored filing chunks when embeddings are configured.
- Provider-call performance payloads with approximate tokens, latency, and local/offline
  cost estimates.
- Prompt budget defaults, local model hardware profiles, and a hash-only embedding cache.
- Explicit cited-answer workflow over retrieved local evidence.
- Specialist analysis for financial reports, stock price history, and explicit
  news/macro/sector context.
- Bounded orchestrator workflow that coordinates company resolution, data refresh,
  specialist runs, handoffs, and inspectable research run state.
- In-process background research queue for `/research ...` chat commands with status,
  progress, cancellation, and a local concurrency limit.
- Deterministic synthesis reports with current situation, strengths, weaknesses,
  opportunities, risks, upside/downside scenarios, unknowns, confidence, and evidence
  coverage indicators.
- Local observability for orchestrator runs, including redacted trace timelines,
  stored-result replay plans, and exportable debug bundles without hosted telemetry.
- Offline deterministic evaluation harness for fixture-labeled research artifacts,
  citations, source freshness, refusal behavior, guardrails, and traceability.
- Optional interoperability spike with A2A discovery metadata and one local-safe,
  read-only MCP-style status tool. It is disabled by default.

## Not Built Yet

- Anthropic, Gemini, and gateway provider adapters.
- LLM-generated narrative report writing beyond deterministic source-backed synthesis.
- Automatic news or macro ingestion.
- Production A2A agent server or broad MCP tool server.
- PDF extraction.
- SQLite or remote database storage.
- Automatic chat RAG for every message.
- Hosted telemetry, production audit logging, or remote observability exports.
- Paid or hosted LLM-as-judge evaluation as part of the default local test path.
- Public benchmark claims.
- Trading, broker integration, alerts, monitoring, or buy/sell/hold recommendations.

## Architecture

The system is split into replaceable boundaries:

```text
Chat UI / HTTP API
        |
        v
Provider-neutral LLM layer
        |
        +--> offline-test
        +--> local-openai  -> llama.cpp / Ollama / compatible local servers
        +--> openai        -> hosted OpenAI-compatible calls

Research workflow layer
        |
        +--> tools and tool-call runner
        +--> prompt contracts
        +--> retrieval and cited answers
        +--> specialist agents
        +--> orchestrator

Data layer
        |
        +--> SEC company lookup
        +--> Alpha Vantage market data
        +--> SEC companyfacts statements
        +--> SEC EDGAR filings
        +--> local JSON/file stores under FRA_HOME
```

The orchestration policy is intentionally bounded and local-safe. Steps run through declared
providers and stores, persist handoffs separately, and preserve partial results when one
provider or specialist step fails.

## Configuration

Settings are read from environment variables. `.env.example` documents the supported
variables, but the app does not auto-load `.env` yet.

`FRA` means Financial Research Agent.

Common settings:

| Variable | Purpose | Default |
| --- | --- | --- |
| `FRA_HOME` | Local app data directory | `~/.financial-research-agent` |
| `FRA_LLM_PROVIDER` | Main LLM provider | `offline-test` |
| `FRA_LLM_MODEL` | Main LLM model | `offline-test` |
| `FRA_LLM_BASE_URL` | OpenAI-compatible local endpoint | `http://127.0.0.1:8080/v1` |
| `FRA_LLM_LOCAL_RUNTIME` | Local runtime label | `llama.cpp` |
| `FRA_OPENAI_API_KEY` | Hosted OpenAI API key | empty |
| `FRA_SEC_USER_AGENT` | SEC EDGAR User-Agent/contact | project placeholder |
| `FRA_ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | empty |
| `FRA_EMBEDDING_PROVIDER` | Embedding provider for retrieval | `disabled` |
| `FRA_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS` | Local background research concurrency limit | `1` |
| `FRA_PROMPT_BUDGET_INPUT_TOKENS` | Approximate prompt input token budget | `16000` |
| `FRA_PROMPT_BUDGET_OUTPUT_TOKENS` | Default response output budget | `1024` |
| `FRA_EMBEDDING_CACHE_ENABLED` | Cache embeddings by provider/model/text hash | `true` |
| `FRA_ALLOW_REMOTE_BIND` | Permit non-loopback web server binding | `false` |
| `FRA_LOCAL_MODEL` | Docker Compose llama.cpp model | Mistral Small GGUF |

The settings UI can save non-secret runtime overrides under `FRA_HOME`. API keys remain
environment-only and are shown only as configured/not-configured flags.

Example local model configuration outside Docker Compose:

```powershell
$env:FRA_LLM_PROVIDER = "local-openai"
$env:FRA_LLM_MODEL = "your-local-model-name"
$env:FRA_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
$env:FRA_LLM_LOCAL_RUNTIME = "llama.cpp"
python -m financial_research_agent --pretty
```

Example hosted OpenAI configuration:

```powershell
$env:FRA_LLM_PROVIDER = "openai"
$env:FRA_LLM_MODEL = "your-openai-model"
$env:FRA_OPENAI_API_KEY = "your-api-key"
python -m financial_research_agent --pretty
```

Do not commit real API keys.

## Security

- API keys remain environment-only and are never written by the settings UI.
- Retrieved filings and external source text are treated as untrusted data, not instructions.
- Tool calls require both an explicit tool-name allowlist and matching permissions.
- Direct app runs bind to `127.0.0.1` by default. Non-loopback binds require
  `FRA_ALLOW_REMOTE_BIND=true`.
- Docker Compose publishes app and llama.cpp ports on host loopback only.
- No shell, arbitrary code execution, or unrestricted local-file tool exists.

Use remote binding only behind a trusted network boundary with appropriate authentication.

## CLI

```powershell
python -m financial_research_agent --pretty
python -m financial_research_agent serve --host 127.0.0.1 --port 8000
python -m financial_research_agent storage-status --pretty
python -m financial_research_agent storage-migrate --pretty
python -m financial_research_agent cache-clear --pretty
python -m financial_research_agent data-reset --yes --pretty
python -m financial_research_agent retrieval-status --pretty
python -m financial_research_agent retrieval-clear --pretty
python -m financial_research_agent eval --pretty
```

The installed console script is also available after `pip install -e .`:

```powershell
financial-research-agent --pretty
```

## HTTP API

Start the app first:

```powershell
python -m financial_research_agent serve --host 127.0.0.1 --port 8000
```

Core endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | Runtime status without secrets |
| `GET /api/settings` | Inspect active runtime settings without secrets |
| `PUT /api/settings` | Save local non-secret runtime overrides |
| `DELETE /api/settings` | Clear local runtime overrides |
| `GET /api/settings/provider-health` | Check active or selected provider health |
| `POST /api/sessions` | Create a chat session |
| `GET /api/sessions/{session_id}` | Read a chat session |
| `POST /api/sessions/{session_id}/messages/stream` | Stream a chat response |
| `GET /api/company-search?query=...` | Search SEC company ticker candidates |
| `POST /api/market-data/history` | Fetch/store daily market data |
| `POST /api/financial-statements` | Fetch/store SEC companyfacts statements |
| `POST /api/filings/ingest` | Fetch/store SEC filing documents |
| `POST /api/retrieval/index/filings` | Index stored filing chunks |
| `POST /api/retrieval/search` | Search the local vector index |
| `POST /api/sessions/{session_id}/cited-answer` | Ask with retrieved citations |
| `POST /api/financial-report-analysis` | Analyze stored statements and filings |
| `POST /api/stock-price-analysis` | Analyze stored market data |
| `POST /api/context-analysis` | Analyze explicit source-linked context |
| `POST /api/orchestrator/research` | Run the bounded research workflow |
| `POST /api/background/research-runs` | Queue a background research workflow |
| `GET /api/background/research-runs/{job_id}` | Inspect background run status and progress |
| `POST /api/background/research-runs/{job_id}/cancel` | Cancel a queued or running background run |
| `GET /api/orchestrator/runs` | Inspect stored orchestrator runs |
| `GET /api/orchestrator/runs/{run_id}/trace` | Inspect a redacted run timeline |
| `POST /api/orchestrator/runs/{run_id}/replay` | Build a stored-result replay plan |
| `GET /api/orchestrator/runs/{run_id}/debug-bundle` | Export a redacted local debug bundle |
| `POST /api/sessions/{session_id}/synthesis-report` | Append a rendered synthesis report to a chat session |
| `POST /api/interop/mcp` | Optional disabled-by-default read-only MCP-style spike endpoint |

Example orchestrated research request:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/orchestrator/research" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"Apple financial situation","refresh":true}'
```

In the chat UI, an explicit `/research Apple financial situation` message queues the same
bounded workflow in the background, shows progress, and renders the synthesis report when
the run completes. Ordinary chat messages remain direct LLM chat and do not automatically
run research tools.

Synthesis messages include an optional run trace inspector. Trace and debug-bundle
payloads redact configured secrets and sensitive local paths, and replay plans use stored
handoff outputs only; they do not call providers, LLMs, tools, or data sources.

Optional interoperability spike:

```powershell
$env:FRA_INTEROP_ENABLED = "true"
$env:FRA_INTEROP_LOCAL_ONLY = "true"
python -m financial_research_agent serve --host 127.0.0.1 --port 8000
```

When enabled for local use, A2A-style discovery metadata is available at
`/.well-known/agent.json` and `/.well-known/agent-card.json`. The MCP-style endpoint supports
`initialize`, `tools/list`, and `tools/call` for `financial_research_agent.status` only. Remote
interop requires `FRA_INTEROP_API_KEY`; do not expose this endpoint publicly without an
explicit deployment security review.

## Local Data

Local runtime data is stored under `FRA_HOME`, which defaults to:

```text
~/.financial-research-agent
```

The local store includes runtime settings overrides, chat sessions, provider caches, market
data, financial statements, filing documents, extracted text, embedding cache, retrieval indexes,
cited-answer runs, and orchestrator run state. Background job status is in-process; the
linked orchestrator run state is persisted when the workflow writes run snapshots. Secrets
are not written into ordinary JSON data stores.

The embedding cache stores provider/model/text hashes and vectors, not raw prompt or
document text.

Use these commands to inspect or clean local state:

```powershell
python -m financial_research_agent storage-status --pretty
python -m financial_research_agent cache-clear --pretty
python -m financial_research_agent data-reset --yes --pretty
```

`cache-clear` removes clearable provider caches. `data-reset --yes` removes local
chat/research data and caches while leaving logs alone.

## Data Sources

| Area | Primary source | Notes |
| --- | --- | --- |
| Company lookup | SEC company ticker list | SEC filer coverage only |
| Market data | Alpha Vantage daily prices | Requires API key; delayed/provider-limited |
| Financial statements | SEC companyfacts XBRL JSON | SEC filers only |
| Filings | SEC EDGAR submissions and Archives | Primary HTML/TXT extraction only |
| Retrieval | Local vector index | Derived from already stored filing chunks |
| Context | Explicit user/API-provided source items | No automatic news scraping yet |

Every research path is expected to preserve source URLs, retrieval timestamps, provider
labels, and limitations. Fake data is only for tests and clearly labeled fixtures.

## Financial Advice Policy

This project is for research support only. It does not provide personalized financial
advice, trading signals, price targets, or buy/sell/hold recommendations.

Outputs should be treated as source-linked analysis drafts that require human review.
Provider limits, stale data, missing filings, incomplete identifiers, and local model
limitations must be considered before using any result.

## Testing

Run the local verification suite:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m compileall -q src tests
python -m financial_research_agent eval --pretty
git diff --check
```

The default eval command runs offline against fixture-labeled artifacts. It checks schema
paths, citation coverage, source freshness, refusal behavior, hallucination-sensitive
patterns, and trace components. LLM-as-judge evaluation is represented as a separate
skipped check unless a future milestone explicitly configures it.

Validate Docker Compose without starting services:

```powershell
docker compose config
```

## Project Status

Financial Research Agent is in active early development. The current codebase is useful as
a local research foundation and integration testbed, but it is not a production investment
platform.

Near-term direction:

- stronger orchestrated synthesis over stored specialist outputs
- more provider adapters
- richer report generation
- better identifier resolution
- PDF and additional document formats
- stronger persistence options
- more complete retrieval workflows

## References

- [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [OpenAI API reference](https://developers.openai.com/api/reference/overview/)
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [SEC EDGAR access policy](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
- [Alpha Vantage documentation](https://www.alphavantage.co/documentation/)
- [Agent2Agent Protocol](https://github.com/a2aproject/A2A)
- [Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/)

## License

Financial Research Agent is licensed under the [MIT License](LICENSE).
