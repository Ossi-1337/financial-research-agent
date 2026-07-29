from dataclasses import FrozenInstanceError

import pytest

from financial_research_agent.agents import (
    AgentRole,
    PromptCatalog,
    PromptContract,
    create_default_prompt_catalog,
)
from financial_research_agent.skills import (
    SkillCatalog,
    SkillContract,
    SkillRole,
    SkillVersion,
    create_default_skill_catalog,
)


def test_default_skills_are_versioned_immutable_and_role_bound() -> None:
    catalog = create_default_skill_catalog()

    assert {skill.id for skill in catalog.contracts()} == {
        "company-research",
        "filing-review",
        "source-verification",
    }
    assert catalog.by_id("company-research").version.value == "2.0.0"
    filing = catalog.by_id("filing-review")
    assert filing.version.value == "1.0.0"
    assert filing.role == SkillRole.FINANCIAL_REPORT_ANALYST
    with pytest.raises(FrozenInstanceError):
        filing.description = "changed"  # type: ignore[misc]


def test_skill_catalog_rejects_invalid_versions_duplicates_and_wrong_roles() -> None:
    catalog = create_default_skill_catalog()
    filing = catalog.by_id("filing-review")

    with pytest.raises(ValueError, match="semantic version"):
        SkillVersion("v1")
    with pytest.raises(ValueError, match="already registered"):
        SkillCatalog((filing, filing))
    with pytest.raises(ValueError, match="belongs to"):
        catalog.compose_for_prompt(
            role=SkillRole.SYNTHESIS_AGENT,
            skill_ids=("filing-review",),
            prompt_allowed_tools=filing.allowed_tools,
        )


def test_skill_cannot_expand_prompt_tool_authority() -> None:
    skill = SkillContract(
        id="test-unsafe-skill",
        version=SkillVersion("1.0.0"),
        role=SkillRole.ORCHESTRATOR,
        description="TEST fixture skill.",
        instructions="TEST fixture instructions.",
        required_inputs=("query",),
        allowed_tools=("shell",),
        allowed_resources=(),
        output_contract="test_output",
    )

    with pytest.raises(ValueError, match="exceeds prompt tool authority"):
        SkillCatalog((skill,)).compose_for_prompt(
            role=SkillRole.ORCHESTRATOR,
            skill_ids=(skill.id,),
            prompt_allowed_tools=("resolve_company",),
        )


def test_default_prompt_skills_compose_without_expanding_tools() -> None:
    prompts = create_default_prompt_catalog()
    skills = create_default_skill_catalog()
    expected = {
        AgentRole.ORCHESTRATOR: ("company-research",),
        AgentRole.FINANCIAL_REPORT_ANALYST: ("filing-review",),
        AgentRole.SYNTHESIS_AGENT: ("source-verification",),
    }

    for role, skill_ids in expected.items():
        prompt = prompts.by_role(role)
        instructions, references = skills.compose_for_prompt(
            role=role.value,
            skill_ids=prompt.skill_ids,
            prompt_allowed_tools=prompt.allowed_tools,
        )
        assert prompt.skill_ids == skill_ids
        assert instructions
        assert tuple(reference.id for reference in references) == skill_ids


def test_prompt_contract_without_skills_remains_backward_compatible() -> None:
    original = create_default_prompt_catalog().by_role(AgentRole.STOCK_ANALYST)
    legacy = PromptContract(
        id="test.legacy.prompt",
        role=original.role,
        version=original.version,
        system_prompt=original.system_prompt,
        description=original.description,
        allowed_tools=original.allowed_tools,
        output_schema=original.output_schema,
    )

    catalog = PromptCatalog((legacy,))

    assert catalog.by_id(legacy.id).skill_ids == ()
