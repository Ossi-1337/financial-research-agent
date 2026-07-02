# Financial Research Agent

Financial Research Agent is a local-first AI agent system for company and stock research.
The project is currently a Python foundation with provider-neutral LLM contracts,
an offline test provider, an OpenAI-compatible local endpoint adapter, and an optional
hosted OpenAI adapter. It also has a deterministic tool registry foundation, but it does
not ingest financial data yet. A minimal local chat UI is available for direct LLM chat.

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
- Local FastAPI chat UI with in-memory sessions and direct provider-backed responses.

Not implemented yet:

- Anthropic, Gemini, or gateway provider integrations.
- Agent runtime and orchestration.
- Live financial data tools or ingestion.
- Persistent chat history, database, vector search, or RAG.

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
Sessions are in memory only; restarting the server clears them. The chat endpoint does not
use live financial data tools, RAG, or multi-agent orchestration yet.

## Tools

The tracked `financial_research_agent.tools` package defines deterministic tool contracts,
a guarded registry, basic JSON-schema argument validation, and a provider-neutral tool-call
loop. Current built-in tools are limited to UTC time, simple ratio calculation, a company
lookup stub, and injected in-memory evidence reads. They do not fetch live financial data,
execute shell commands, browse the web, or use a database.

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

## Configuration

Settings are read from real environment variables. `.env.example` documents the supported
`FRA_*` variables. `FRA` means Financial Research Agent. The settings include
task-specific provider/model overrides for chat, tool calling, structured output, and
streaming. `.env.example` is a reference file only; the app does not auto-load `.env`
yet.

## Verify

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m compileall -q src tests
git diff --check
```
