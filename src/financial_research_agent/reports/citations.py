from __future__ import annotations

from financial_research_agent.llm import ChatMessage, MessageRole
from financial_research_agent.reports.contracts import Citation, EvidenceSnippet
from financial_research_agent.retrieval import RetrievalResult
from financial_research_agent.security import (
    UNTRUSTED_CONTENT_INSTRUCTION,
    UntrustedContent,
    build_untrusted_content_payload,
)

MAX_QUOTE_CHARS = 280
MAX_EVIDENCE_SNIPPET_CHARS = 900


def citation_artifacts_from_retrieval(
    result: RetrievalResult,
) -> tuple[tuple[Citation, ...], tuple[EvidenceSnippet, ...]]:
    citations: list[Citation] = []
    evidence: list[EvidenceSnippet] = []
    for index, match in enumerate(result.matches, start=1):
        citation_id = f"C{index}"
        snippet_id = f"evidence:{result.index_id}:{match.chunk.source_id}:{index}"
        quote = _bounded_excerpt(match.chunk.text, MAX_QUOTE_CHARS)
        quote_start = _metadata_int(match.chunk.metadata, "char_start") or 0
        quote_end = quote_start + len(quote)
        citations.append(
            Citation(
                id=citation_id,
                evidence_id=snippet_id,
                source_url=match.chunk.source_url,
                source_id=match.chunk.source_id,
                document_id=match.chunk.document_id,
                chunk_id=match.chunk.id,
                section=match.chunk.section_heading,
                retrieved_at=result.generated_at,
                quote_start=quote_start,
                quote_end=quote_end,
                quote=quote,
                metadata=match.chunk.metadata,
            )
        )
        evidence.append(
            EvidenceSnippet(
                id=snippet_id,
                citation_id=citation_id,
                text=_bounded_excerpt(match.chunk.text, MAX_EVIDENCE_SNIPPET_CHARS),
                source_url=match.chunk.source_url,
                retrieved_at=result.generated_at,
                score=match.score,
                source_id=match.chunk.source_id,
                document_id=match.chunk.document_id,
                chunk_id=match.chunk.id,
                section=match.chunk.section_heading,
                metadata=match.chunk.metadata,
            )
        )
    return tuple(citations), tuple(evidence)


def build_rag_messages(
    query: str,
    evidence: tuple[EvidenceSnippet, ...],
) -> tuple[ChatMessage, ...]:
    if not evidence:
        raise ValueError("evidence is required to build a cited RAG prompt")
    evidence_payload = build_untrusted_content_payload(
        _untrusted_content_from_snippet(snippet) for snippet in evidence
    )
    return (
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are a financial research assistant. Answer using only the evidence "
                "snippets provided in this request. Cite every factual claim with citation "
                "markers like [C1]. If the evidence is insufficient, say what is missing "
                "instead of inventing facts. Do not cite sources that are not provided. "
                "Do not provide buy, sell, or hold recommendations. Provide concise "
                f"reasoning summaries only. {UNTRUSTED_CONTENT_INSTRUCTION}"
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=f"Question:\n{query.strip()}\n\nUntrusted evidence JSON:\n{evidence_payload}",
        ),
    )


def ensure_citation_marker(answer: str, citations: tuple[Citation, ...]) -> str:
    text = answer.strip()
    if not citations:
        return text
    if any(citation.marker in text for citation in citations):
        return text
    markers = " ".join(citation.marker for citation in citations)
    return f"{text}\n\nSources: {markers}"


def missing_evidence_limitation(query: str) -> str:
    return (
        "I could not find stored evidence snippets that support this request, so I cannot "
        f"provide a cited answer for: {query.strip()}"
    )


def _untrusted_content_from_snippet(snippet: EvidenceSnippet) -> UntrustedContent:
    return UntrustedContent(
        source_id=snippet.id,
        source_url=snippet.source_url,
        content=snippet.text,
        metadata={
            "citation_id": snippet.citation_id,
            "citation_marker": f"[{snippet.citation_id}]",
            "retrieved_at": snippet.retrieved_at.isoformat(),
            "section": snippet.section,
        },
    )


def _bounded_excerpt(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "..."


def _metadata_int(metadata, key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None
