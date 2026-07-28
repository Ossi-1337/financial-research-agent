from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from financial_research_agent.agents import (
    AgentOutputSchema,
    AgentRole,
    PromptCatalog,
    PromptContract,
    PromptVersion,
    create_default_prompt_catalog,
)
from financial_research_agent.llm import ResponseFormatType


def test_prompt_contract_construction_is_immutable_and_response_format_is_json_schema() -> None:
    catalog = create_default_prompt_catalog()
    contract = catalog.by_role(AgentRole.ORCHESTRATOR)

    response_format = contract.response_format()

    assert contract.id == "agent.orchestrator.v1"
    assert contract.version.value == "1.0.0"
    assert contract.skill_ids == ("company-research",)
    assert response_format.format_type == ResponseFormatType.JSON_SCHEMA
    assert response_format.name == "orchestrator_output"
    assert response_format.json_schema == contract.output_schema.schema
    with pytest.raises(FrozenInstanceError):
        contract.description = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        contract.output_schema.schema["type"] = "changed"  # type: ignore[index]


def test_prompt_version_requires_semantic_version() -> None:
    assert PromptVersion("1.2.3").value == "1.2.3"
    with pytest.raises(ValueError, match="semantic version"):
        PromptVersion("v1")


def test_prompt_catalog_rejects_duplicate_prompt_ids_and_roles() -> None:
    catalog = create_default_prompt_catalog()
    contract = catalog.by_role(AgentRole.ORCHESTRATOR)

    with pytest.raises(ValueError, match="already registered"):
        PromptCatalog([contract, contract])

    duplicate_role = PromptContract(
        id="agent.orchestrator.copy",
        role=AgentRole.ORCHESTRATOR,
        version=contract.version,
        system_prompt=contract.system_prompt,
        description=contract.description,
        allowed_tools=contract.allowed_tools,
        output_schema=contract.output_schema,
    )
    with pytest.raises(ValueError, match="role is already registered"):
        PromptCatalog([contract, duplicate_role])


def test_default_catalog_has_one_contract_for_each_required_agent_role() -> None:
    catalog = create_default_prompt_catalog()

    assert {contract.role for contract in catalog.contracts()} == set(AgentRole)
    for role in AgentRole:
        contract = catalog.by_role(role)
        assert catalog.by_id(contract.id) is contract
        assert contract.system_prompt.strip()
        assert contract.description.strip()
        assert contract.allowed_tools


def test_default_prompts_include_required_safety_and_evidence_rules() -> None:
    catalog = create_default_prompt_catalog()

    for contract in catalog.contracts():
        prompt = contract.system_prompt.lower()
        assert "evidence_ids" in prompt
        assert "factual claim" in prompt
        assert "refuse unsupported claims" in prompt
        assert "do not provide buy, sell, or hold recommendations" in prompt
        assert "hidden chain-of-thought" in prompt
        assert "concise reasoning_summary" in prompt
        assert "facts, assumptions, analysis, and opinion" in prompt


def test_default_output_schemas_require_research_sections() -> None:
    expected_required = {
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
    }
    catalog = create_default_prompt_catalog()

    for contract in catalog.contracts():
        schema = contract.output_schema.schema
        assert set(schema["required"]) == expected_required
        assert "evidence_ids" in schema["properties"]["findings"]["items"]["required"]
        assert "missing_evidence" in schema["properties"]["uncertainty"]["required"]
        assert "evidence_ids" in schema["properties"]["risks"]["items"]["required"]
        assert "evidence_ids" in schema["properties"]["scenarios"]["items"]["required"]
        assert "claim" in schema["properties"]["refusal_notes"]["items"]["required"]


def test_agent_output_schema_accepts_labeled_tool_result_fixture_output() -> None:
    catalog = create_default_prompt_catalog()

    for contract in catalog.contracts():
        output = _valid_fixture_output(contract.role)
        assert contract.output_schema.validate_output(output) == ()


def test_agent_output_schema_rejects_finding_without_evidence_ids() -> None:
    contract = create_default_prompt_catalog().by_role(AgentRole.FINANCIAL_REPORT_ANALYST)
    output = _valid_fixture_output(contract.role)
    del output["findings"][0]["evidence_ids"]

    errors = contract.output_schema.validate_output(output)

    assert any("findings[0].evidence_ids is required" in error for error in errors)


def test_agent_output_schema_rejects_empty_evidence_ids() -> None:
    contract = create_default_prompt_catalog().by_role(AgentRole.FINANCIAL_REPORT_ANALYST)
    output = _valid_fixture_output(contract.role)
    output["findings"][0]["evidence_ids"] = []

    errors = contract.output_schema.validate_output(output)

    assert any(
        "findings[0].evidence_ids must contain at least 1 items" in error for error in errors
    )


def test_agent_output_schema_rejects_invalid_agent_role() -> None:
    contract = create_default_prompt_catalog().by_role(AgentRole.SYNTHESIS_AGENT)
    output = _valid_fixture_output(contract.role)
    output["agent_role"] = "orchestrator"

    errors = contract.output_schema.validate_output(output)

    assert any("agent_role" in error for error in errors)


def test_invalid_agent_output_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid agent output schema"):
        AgentOutputSchema(name="bad", schema={"type": "string"})


def _valid_fixture_output(role: AgentRole) -> dict[str, object]:
    evidence_id = "test-tool-result:read_local_evidence:1"
    statement = "Test fixture statement derived from labeled deterministic tool output only."
    return deepcopy(
        {
            "agent_role": role.value,
            "facts": [
                {
                    "statement": statement,
                    "evidence_ids": [evidence_id],
                    "confidence": "high",
                }
            ],
            "assumptions": [
                {
                    "statement": "This is a labeled test fixture, not real market data.",
                    "basis": "Fixture generated from deterministic tool-result-shaped data.",
                    "confidence": "high",
                }
            ],
            "analysis": [
                {
                    "statement": "Analysis is limited to fixture shape validation.",
                    "evidence_ids": [evidence_id],
                    "confidence": "medium",
                }
            ],
            "opinion": [
                {
                    "statement": "No investment opinion is provided in this fixture.",
                    "evidence_ids": [evidence_id],
                    "confidence": "high",
                }
            ],
            "findings": [
                {
                    "category": "test_fixture",
                    "statement": statement,
                    "evidence_ids": [evidence_id],
                    "confidence": "high",
                }
            ],
            "uncertainty": {
                "missing_evidence": ["No live financial data is present in this fixture."],
                "limitations": ["Fixture data validates prompt schema shape only."],
                "confidence": "medium",
            },
            "risks": [
                {
                    "title": "Fixture limitation",
                    "description": "This output must not be treated as real company research.",
                    "severity": "unknown",
                    "evidence_ids": [evidence_id],
                }
            ],
            "scenarios": [
                {
                    "name": "Schema validation only",
                    "description": "A later milestone may provide real sourced evidence.",
                    "evidence_ids": [evidence_id],
                }
            ],
            "follow_up_questions": ["Which real source should be queried in a later milestone?"],
            "refusal_notes": [
                {
                    "claim": "Any real valuation or recommendation",
                    "reason": "The fixture contains no real financial evidence.",
                }
            ],
            "reasoning_summary": "Validated fixture shape without hidden chain-of-thought.",
        }
    )
