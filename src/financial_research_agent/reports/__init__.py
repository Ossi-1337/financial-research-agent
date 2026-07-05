"""Cited report retrieval contracts, prompt building, and local run storage."""

from financial_research_agent.reports.citations import (
    build_rag_messages,
    citation_artifacts_from_retrieval,
    ensure_citation_marker,
    missing_evidence_limitation,
)
from financial_research_agent.reports.contracts import (
    Citation,
    CitedResearchRun,
    CitedResearchRunStatus,
    EvidenceSnippet,
)
from financial_research_agent.reports.store import CitedResearchRunStore

__all__ = [
    "Citation",
    "CitedResearchRun",
    "CitedResearchRunStatus",
    "CitedResearchRunStore",
    "EvidenceSnippet",
    "build_rag_messages",
    "citation_artifacts_from_retrieval",
    "ensure_citation_marker",
    "missing_evidence_limitation",
]
