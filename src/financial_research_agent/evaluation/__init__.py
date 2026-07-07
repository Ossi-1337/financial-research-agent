"""Offline deterministic evaluation harness for research artifacts."""

from financial_research_agent.evaluation.contracts import (
    EVALUATION_SCHEMA_VERSION,
    EvalArtifact,
    EvalArtifactKind,
    EvalCase,
    EvalCaseResult,
    EvalCheckKind,
    EvalCheckResult,
    EvalCheckStatus,
    EvalDatasetLabel,
    EvalSuiteResult,
    EvalSuiteStatus,
)
from financial_research_agent.evaluation.dataset import (
    DEFAULT_EVALUATION_SUITE_ID,
    default_eval_artifacts,
    default_eval_cases,
)
from financial_research_agent.evaluation.runner import (
    evaluate_case,
    run_default_offline_evaluations,
    run_evaluation_suite,
)

__all__ = [
    "DEFAULT_EVALUATION_SUITE_ID",
    "EVALUATION_SCHEMA_VERSION",
    "EvalArtifact",
    "EvalArtifactKind",
    "EvalCase",
    "EvalCaseResult",
    "EvalCheckKind",
    "EvalCheckResult",
    "EvalCheckStatus",
    "EvalDatasetLabel",
    "EvalSuiteResult",
    "EvalSuiteStatus",
    "default_eval_artifacts",
    "default_eval_cases",
    "evaluate_case",
    "run_default_offline_evaluations",
    "run_evaluation_suite",
]
