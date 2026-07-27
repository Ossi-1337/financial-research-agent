# Engineering Notes

This project is a portfolio implementation of a local-first Python AI/backend system. Its main
engineering goal is explicit control over evidence, providers, persistence, and failure modes.

## Provider Abstraction

LLM contracts use frozen dataclasses, enums, protocols, and provider-neutral request/response
objects. `offline-test`, local OpenAI-compatible endpoints, OpenAI, Anthropic, Gemini, and
LiteLLM share the same async chat boundary. Embeddings are registered only for providers that
expose them.

Tradeoff: this requires adapter mapping and capability checks, but avoids leaking SDK-specific
types into orchestration, tools, tests, or web routes.

## Tool Calls

The tool registry validates a bounded JSON Schema subset, permissions, arguments, timeouts, and
result serialization. Unknown or denied calls become structured failures. No shell or arbitrary
filesystem tool exists.

Tradeoff: supported schemas are intentionally smaller than full JSON Schema. This keeps local
execution auditable and is sufficient for current deterministic tools.

## Retrieval And Citations

Retrieval is explicit. Filing chunks are indexed locally; cited answers search stored evidence
and provide source-limited context to the selected model. Citation IDs, snippets, URLs, and
retrieval metadata are persisted.

Automatic RAG on every message is intentionally absent. It would add latency, irrelevant
context, and harder-to-explain data access. Users select cited research when evidence is needed.

## Deterministic Specialists And Synthesis

Financial report, stock, and context specialists operate over stored source data. Synthesis
combines persisted handoffs into fixed report sections, conditional scenarios, confidence,
unknowns, limitations, and evidence coverage.

This is deliberately less fluent than unrestricted narrative generation. The benefit is a
stable, testable report whose facts and completion checks do not depend on model behavior.

## SQLite And Filesystem Split

SQLite owns structured state and relationships. Source documents, extracted text, exports,
vector indexes, caches, backups, and logs remain files.

Tradeoff: the hybrid design requires careful lifecycle operations, but avoids storing large
documents as database blobs and keeps artifacts directly inspectable. Transactions, foreign
keys, WAL, migrations, integrity checks, and maintenance locking protect structured state.

## Evaluation

Default evaluation is offline and deterministic. Fixture-labeled cases test schemas, evidence
coverage, citations, refusal behavior, freshness, guardrails, and traceability. Live acceptance
is separate because providers, quotas, and source data change.

No public accuracy or performance benchmark is claimed. A paid LLM judge would make the default
test path less reproducible and is not currently required.

## Security

- Secrets remain environment-only and are redacted from status, traces, debug bundles, and
  exports.
- External source text is untrusted data, not instructions.
- Tool names and permissions are deny-by-default.
- Generated filenames and IDs prevent request-controlled filesystem paths.
- Web and model ports bind to loopback by default.
- Restore, reset, cleanup, and legacy import require guarded operations.

This is a local development security model, not a multi-user hosted authorization system.

## Local Inference

llama.cpp runs as an external OpenAI-compatible service. Docker profiles provide small CPU and
larger CUDA defaults; the Python package does not bundle models, drivers, or runtime binaries.

Tradeoff: users must manage model suitability and hardware. In return, the app remains
provider-neutral, lightweight, and usable without cloud LLM credentials.

## Real Data And Licensing

The app uses official SEC sources first and a documented market-data provider. Every evidence
path preserves source identity, retrieval time, data-as-of time when available, attribution,
freshness, and limitations.

Cached source documents and market data are local research inputs. Demo assets avoid publishing
provider-derived price values. Users remain responsible for source terms and redistribution
rights.

## Interoperability Direction

The current A2A/MCP surface is a disabled-by-default spike with discovery metadata and one
read-only status tool. It proves boundary placement without claiming production readiness.

A2A is the preferred direction for independently deployable specialist agents. MCP remains
useful for exposing bounded tools or resources to models and external clients. Production work
requires authentication, authorization, durable task state, protocol conformance, and threat
modeling.
