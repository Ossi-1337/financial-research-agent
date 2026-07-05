"""News, macro, and sector context analysis over explicit source items."""

from financial_research_agent.context_analysis.agent import (
    NewsMacroSectorAgent,
    create_default_context_source_strategy,
)
from financial_research_agent.context_analysis.contracts import (
    ConfidenceLabel,
    ContextAnalysisResult,
    ContextAnalysisStatus,
    ContextFinding,
    ContextRecency,
    ContextScope,
    ContextSourceItem,
    ContextSourceStrategyItem,
    ContextSourceType,
    SourceReliability,
)

__all__ = [
    "ConfidenceLabel",
    "ContextAnalysisResult",
    "ContextAnalysisStatus",
    "ContextFinding",
    "ContextRecency",
    "ContextScope",
    "ContextSourceItem",
    "ContextSourceStrategyItem",
    "ContextSourceType",
    "NewsMacroSectorAgent",
    "SourceReliability",
    "create_default_context_source_strategy",
]
