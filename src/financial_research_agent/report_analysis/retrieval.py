from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from financial_research_agent.filings import FilingChunk, FilingIngestionResult


class FilingRetrievalMethod(StrEnum):
    LEXICAL = "lexical"
    VECTOR_RERANKED = "vector_reranked"


@dataclass(frozen=True, slots=True)
class FilingChunkMatch:
    chunk: FilingChunk
    score: float
    method: FilingRetrievalMethod


FilingVectorReranker = Callable[[tuple[FilingChunk, ...], str], Mapping[str, float]]


def retrieve_filing_chunks(
    filing_result: FilingIngestionResult | None,
    query_terms: Iterable[str],
    *,
    limit: int,
    vector_reranker: FilingVectorReranker | None = None,
) -> tuple[FilingChunkMatch, ...]:
    if filing_result is None or limit <= 0:
        return ()
    terms = tuple(dict.fromkeys(term.casefold().strip() for term in query_terms if term.strip()))
    lexical: list[tuple[float, FilingChunk]] = []
    for chunk in filing_result.chunks:
        text = chunk.text.casefold()
        count = sum(text.count(term) for term in terms)
        if count:
            lexical.append((count / max(len(text.split()), 1), chunk))
    lexical.sort(key=lambda item: (-item[0], item[1].chunk_index, item[1].id))
    candidates = tuple(chunk for _score, chunk in lexical[: max(limit * 4, limit)])
    if not candidates:
        return ()
    if vector_reranker is None:
        return tuple(
            FilingChunkMatch(
                chunk=chunk,
                score=score,
                method=FilingRetrievalMethod.LEXICAL,
            )
            for score, chunk in lexical[:limit]
        )
    vector_scores = vector_reranker(candidates, " ".join(terms))
    reranked = sorted(
        (
            (float(vector_scores.get(chunk.id, 0.0)), lexical_score, chunk)
            for lexical_score, chunk in lexical
            if chunk in candidates
        ),
        key=lambda item: (-item[0], -item[1], item[2].chunk_index, item[2].id),
    )
    return tuple(
        FilingChunkMatch(
            chunk=chunk,
            score=vector_score,
            method=FilingRetrievalMethod.VECTOR_RERANKED,
        )
        for vector_score, _lexical_score, chunk in reranked[:limit]
    )
