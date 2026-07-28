# Financial Research Agent

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker Compose](https://img.shields.io/badge/runtime-Docker%20Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Financial Research Agent is a local-first Python multi-agent system for company and stock
research. One orchestrator interprets every chat message and delegates only the required work over
A2A to internal specialist services for financial reports, stock analysis, context analysis, and
source-backed synthesis.

The system combines real financial sources, local persistence, filing retrieval, evidence-backed
reports, and swappable local or hosted LLM providers. It is research software, not an autonomous
trading system or financial adviser.

## Start

Requirements: Docker Desktop in Linux-container mode. Python 3.14 is required for direct
development and repository checks.

Start the complete local topology without credentials or a model download:

```powershell
docker compose up --build
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The default `offline-test` provider is
deterministic and intended for setup verification. Direct chat works; real research returns
`agent_provider_unavailable` until a local or hosted LLM is selected.

Start with a local llama.cpp model:

```powershell
docker compose --profile cpu up --build -d
docker compose --profile cuda up --build -d
```

Use one model profile at a time. The first run may download the configured GGUF model.

For repository development:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts/dev.py install
python scripts/dev.py check
```

CLI commands load the repository `.env` without overriding existing process environment
variables. Start from `.env.example`; `.env` is ignored by Git.

## Project Status

Implemented:

- Persistent streaming chat with `@company` references and bounded history.
- SEC company lookup, CompanyFacts statements, and EDGAR HTML/TXT filing ingestion.
- Alpha Vantage daily prices and deterministic market calculations.
- SQLite persistence, local filing retrieval, citations, evidence inspection, and report exports.
- Provider adapters for local OpenAI-compatible endpoints, OpenAI, Anthropic, Gemini, and LiteLLM.
- One LLM orchestrator selects from four tool-using A2A specialist services started by default
  through Docker Compose.
- Structured specialist outputs with evidence-ID validation and one bounded schema-repair attempt.
- Offline evaluation, package verification, guarded storage operations, and local CI through Actio.

Current boundaries:

- No trading, broker integration, price targets, alerts, or buy/sell/hold recommendations.
- No automatic news/macro ingestion, PDF extraction, or RAG on every chat message.
- No durable cross-host queue, remote database, public A2A deployment, or hosted telemetry claim.
- Market data may be delayed or provider-limited; SEC coverage is limited to SEC filers.
- Local LLM quality depends on the selected model, runtime, context budget, and hardware.
- Context analysis accepts approved source-linked inputs; automatic news ingestion is not present.

Only the web UI is published to the host. Specialist ports stay inside the Compose network.
Credentials remain environment-only, external documents are treated as untrusted evidence, and
runtime state is stored under `FRA_HOME`.

See [Architecture](docs/architecture.md) for system boundaries, data ownership, persistence, and
the A2A research flow.

Financial Research Agent is licensed under the [MIT License](LICENSE).
