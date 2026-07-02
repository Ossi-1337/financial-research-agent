from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from financial_research_agent.agents.contracts import (
    AgentOutputSchema,
    AgentRole,
    PromptCatalog,
    PromptContract,
    PromptVersion,
)

PROMPT_VERSION = PromptVersion("1.0.0")

COMMON_SAFETY_RULES = """
Shared rules:
- Separate facts, assumptions, analysis, and opinion in the structured output.
- Every factual claim must cite one or more evidence_ids from supplied evidence or tool outputs.
- If evidence is missing, stale, ambiguous, or incomplete, say so in uncertainty and refusal_notes.
- Refuse unsupported claims instead of inventing identifiers, prices, dates, metrics,
  sources, or citations.
- Do not provide buy, sell, or hold recommendations by default.
- Do not persist or reveal hidden chain-of-thought; provide concise reasoning_summary only.
- This is financial research support, not personalized financial advice.
""".strip()


def create_default_prompt_catalog() -> PromptCatalog:
    return PromptCatalog(
        (
            _contract(
                prompt_id="agent.orchestrator.v1",
                role=AgentRole.ORCHESTRATOR,
                description=(
                    "Coordinates research tasks, tool use, missing evidence, and handoff needs."
                ),
                allowed_tools=(
                    "current_utc_datetime",
                    "resolve_company_stub",
                    "read_local_evidence",
                    "calculate_ratio",
                ),
                role_instructions="""
You are the orchestrator for local-first company research.
Plan which deterministic tools or specialist agents are needed.
Do not perform specialist analysis when evidence should be delegated.
Track partial research, ambiguity, missing tickers, and stale data as explicit uncertainty.
Return only the agreed JSON object.
""".strip(),
            ),
            _contract(
                prompt_id="agent.financial_report_analyst.v1",
                role=AgentRole.FINANCIAL_REPORT_ANALYST,
                description="Analyzes sourced financial report evidence and ratio calculations.",
                allowed_tools=("read_local_evidence", "calculate_ratio", "current_utc_datetime"),
                role_instructions="""
You are the financial report analyst.
Use only supplied filings, statements, evidence, and deterministic calculation results.
Explain financial statement trends, quality, and caveats with evidence_ids.
Never invent line items, reporting periods, currencies, or filing dates.
Return only the agreed JSON object.
""".strip(),
            ),
            _contract(
                prompt_id="agent.stock_analyst.v1",
                role=AgentRole.STOCK_ANALYST,
                description=(
                    "Analyzes sourced price-development evidence without giving recommendations."
                ),
                allowed_tools=("read_local_evidence", "calculate_ratio", "current_utc_datetime"),
                role_instructions="""
You are the stock price development analyst.
Analyze sourced price movement, volatility, valuation context, and uncertainty.
Use evidence_ids for claims about prices, periods, benchmarks, and calculated ratios.
Do not predict a target price or recommend trading action.
Return only the agreed JSON object.
""".strip(),
            ),
            _contract(
                prompt_id="agent.news_macro_analyst.v1",
                role=AgentRole.NEWS_MACRO_ANALYST,
                description="Analyzes sourced news, macro, policy, sector, and event context.",
                allowed_tools=("read_local_evidence", "current_utc_datetime"),
                role_instructions="""
You are the news, macro, and sector context analyst.
Separate confirmed events from assumptions and possible implications.
Use evidence_ids for claims about events, publication dates, macro data, or sector context.
Flag stale, missing, conflicting, or weak evidence clearly.
Return only the agreed JSON object.
""".strip(),
            ),
            _contract(
                prompt_id="agent.synthesis.v1",
                role=AgentRole.SYNTHESIS_AGENT,
                description=(
                    "Synthesizes specialist outputs into evidence-grounded research summaries."
                ),
                allowed_tools=("read_local_evidence", "current_utc_datetime"),
                role_instructions="""
You are the synthesis agent.
Combine specialist outputs into a concise, evidence-grounded research answer.
Distinguish facts, assumptions, analysis, opinion, risks, scenarios, and follow-up needs.
Do not add new facts that are not present in evidence or specialist outputs.
Return only the agreed JSON object.
""".strip(),
            ),
        )
    )


def _contract(
    *,
    prompt_id: str,
    role: AgentRole,
    description: str,
    allowed_tools: tuple[str, ...],
    role_instructions: str,
) -> PromptContract:
    return PromptContract(
        id=prompt_id,
        role=role,
        version=PROMPT_VERSION,
        system_prompt=f"{role_instructions}\n\n{COMMON_SAFETY_RULES}",
        description=description,
        allowed_tools=allowed_tools,
        output_schema=AgentOutputSchema(
            name=f"{role.value}_output",
            schema=_agent_output_schema(role),
        ),
    )


def _agent_output_schema(role: AgentRole) -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "agent_role": {"type": "string", "enum": [role.value]},
            "facts": {"type": "array", "items": _evidence_statement_schema()},
            "assumptions": {"type": "array", "items": _assumption_schema()},
            "analysis": {"type": "array", "items": _evidence_statement_schema()},
            "opinion": {"type": "array", "items": _evidence_statement_schema()},
            "findings": {"type": "array", "items": _finding_schema()},
            "uncertainty": _uncertainty_schema(),
            "risks": {"type": "array", "items": _risk_schema()},
            "scenarios": {"type": "array", "items": _scenario_schema()},
            "follow_up_questions": {"type": "array", "items": {"type": "string"}},
            "refusal_notes": {"type": "array", "items": _refusal_schema()},
            "reasoning_summary": {"type": "string"},
        },
        "required": [
            "agent_role",
            "facts",
            "assumptions",
            "analysis",
            "opinion",
            "findings",
            "uncertainty",
            "risks",
            "scenarios",
            "follow_up_questions",
            "refusal_notes",
            "reasoning_summary",
        ],
        "additionalProperties": False,
    }


def _evidence_statement_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "statement": {"type": "string"},
            "evidence_ids": _evidence_ids_schema(),
            "confidence": _confidence_schema(),
        },
        "required": ["statement", "evidence_ids", "confidence"],
        "additionalProperties": False,
    }


def _finding_schema() -> Mapping[str, Any]:
    schema = dict(_evidence_statement_schema())
    schema["properties"] = {
        **schema["properties"],
        "category": {"type": "string"},
    }
    schema["required"] = ["category", "statement", "evidence_ids", "confidence"]
    return schema


def _assumption_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "statement": {"type": "string"},
            "basis": {"type": "string"},
            "confidence": _confidence_schema(),
        },
        "required": ["statement", "basis", "confidence"],
        "additionalProperties": False,
    }


def _uncertainty_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "missing_evidence": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "confidence": _confidence_schema(),
        },
        "required": ["missing_evidence", "limitations", "confidence"],
        "additionalProperties": False,
    }


def _risk_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
            "evidence_ids": _evidence_ids_schema(),
        },
        "required": ["title", "description", "severity", "evidence_ids"],
        "additionalProperties": False,
    }


def _scenario_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "evidence_ids": _evidence_ids_schema(),
        },
        "required": ["name", "description", "evidence_ids"],
        "additionalProperties": False,
    }


def _refusal_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["claim", "reason"],
        "additionalProperties": False,
    }


def _confidence_schema() -> Mapping[str, Any]:
    return {"type": "string", "enum": ["low", "medium", "high", "unknown"]}


def _evidence_ids_schema() -> Mapping[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "minItems": 1}
