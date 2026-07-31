"""Synthesis, risk, and scenario reporting over specialist handoffs."""

from financial_research_agent.synthesis.agent import SynthesisAgent
from financial_research_agent.synthesis.contracts import (
    NO_SYNTHESIS_RECOMMENDATION_NOTICE,
    ConfidenceLabel,
    EvidenceCoverage,
    ScenarioDirection,
    SynthesisPoint,
    SynthesisReport,
    SynthesisReportStatus,
    SynthesisScenario,
    SynthesisSection,
)
from financial_research_agent.synthesis.narrative import (
    MAX_NARRATIVE_PARAGRAPH_CHARS,
    MAX_NARRATIVE_PARAGRAPHS_PER_SECTION,
    NARRATIVE_PROMPT_ID,
    NARRATIVE_PROMPT_VERSION,
    NARRATIVE_SECTION_ORDER,
    NarrativeParagraph,
    NarrativePresentation,
    NarrativePresentationSection,
    NarrativeSection,
    synthesis_sha256,
)
from financial_research_agent.synthesis.store import NarrativePresentationStore

__all__ = [
    "MAX_NARRATIVE_PARAGRAPHS_PER_SECTION",
    "MAX_NARRATIVE_PARAGRAPH_CHARS",
    "NARRATIVE_PROMPT_ID",
    "NARRATIVE_PROMPT_VERSION",
    "NARRATIVE_SECTION_ORDER",
    "NO_SYNTHESIS_RECOMMENDATION_NOTICE",
    "ConfidenceLabel",
    "EvidenceCoverage",
    "NarrativeParagraph",
    "NarrativePresentation",
    "NarrativePresentationSection",
    "NarrativePresentationStore",
    "NarrativeSection",
    "ScenarioDirection",
    "SynthesisAgent",
    "SynthesisPoint",
    "SynthesisReport",
    "SynthesisReportStatus",
    "SynthesisScenario",
    "SynthesisSection",
    "synthesis_sha256",
]
