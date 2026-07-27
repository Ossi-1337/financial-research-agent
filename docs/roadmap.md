# Roadmap

This backlog describes possible directions, not delivery dates or commitments. Local-first
operation, source traceability, and deterministic report integrity remain the default design
constraints.

## Provider Coverage

- Keep OpenAI, Anthropic, Gemini, and LiteLLM adapters aligned with provider-neutral conformance
  tests.
- Evaluate automatic routing or fallback only when it can preserve provider capabilities,
  errors, privacy boundaries, and user control.
- Add other hosted providers only when a concrete workflow needs them.

## Agent Interoperability

- Turn the current A2A discovery spike into a production-capable local agent server.
- Define durable task state, authentication, authorization, cancellation, and artifact
  exchange before distributing specialist agents.
- Use MCP for bounded tools and resources where it complements A2A rather than replacing agent
  coordination.
- Keep skills for developer workflows and domain guidance that do not require runtime protocol
  exposure.

## Documents And Retrieval

- Add robust PDF extraction with page-level provenance and layout-aware validation.
- Evaluate SQLite full-text or hybrid retrieval before adopting a separate vector database.
- Add opt-in automatic chat RAG only after relevance, latency, privacy, and citation evaluation.
- Improve cross-document evidence deduplication and contradiction reporting.

## Research Coverage

- Add scheduled, source-licensed news and macro ingestion with freshness and attribution.
- Expand financial statement normalization beyond current SEC/IFRS concepts and currencies.
- Add more reproducible company scenarios without turning scenario IDs into per-company API
  design.
- Add LLM-assisted narrative reports only as a labeled layer over deterministic synthesis.

## Operations

- Add hosted telemetry exporters only with explicit opt-in, redaction, retention, and local
  fallback.
- Evaluate a durable local background queue before remote job infrastructure.
- Add production audit logging only alongside a concrete hosted deployment and threat model.
- Evaluate remote database storage when multi-user operation is a real requirement.

## Evaluation

- Expand real-data regression scenarios and source-shape contract tests.
- Add optional hosted or local LLM-as-judge runs outside the default test path.
- Define reproducible benchmark methodology before publishing any quality, latency, or cost
  claims.

## Explicit Non-Goals

- No automated trading, broker execution, or personalized investment advice.
- No buy/sell/hold recommendations or price targets by default.
- No monitoring or alerting until source licensing, persistence, and user-notification
  semantics are designed.
- No public model or financial-accuracy claims from a single company scenario.
