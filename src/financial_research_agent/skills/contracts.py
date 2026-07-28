from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

_SEMANTIC_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class SkillRole(StrEnum):
    ORCHESTRATOR = "orchestrator"
    FINANCIAL_REPORT_ANALYST = "financial_report_analyst"
    STOCK_ANALYST = "stock_analyst"
    NEWS_MACRO_ANALYST = "news_macro_analyst"
    SYNTHESIS_AGENT = "synthesis_agent"


@dataclass(frozen=True, slots=True)
class SkillVersion:
    value: str

    def __post_init__(self) -> None:
        value = _require_text("value", self.value)
        if _SEMANTIC_VERSION_PATTERN.fullmatch(value) is None:
            raise ValueError("skill version must use semantic version format x.y.z")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class SkillReference:
    id: str
    version: SkillVersion

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        if not isinstance(self.version, SkillVersion):
            raise ValueError("version must be a SkillVersion")

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "version": self.version.value}


@dataclass(frozen=True, slots=True)
class SkillContract:
    id: str
    version: SkillVersion
    role: SkillRole
    description: str
    instructions: str
    required_inputs: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    allowed_resources: tuple[str, ...]
    output_contract: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        if not isinstance(self.version, SkillVersion):
            raise ValueError("version must be a SkillVersion")
        object.__setattr__(self, "role", SkillRole(self.role))
        object.__setattr__(self, "description", _require_text("description", self.description))
        object.__setattr__(self, "instructions", _require_text("instructions", self.instructions))
        object.__setattr__(
            self,
            "required_inputs",
            _text_tuple("required_inputs", self.required_inputs),
        )
        object.__setattr__(
            self,
            "allowed_tools",
            _text_tuple("allowed_tools", self.allowed_tools),
        )
        object.__setattr__(
            self,
            "allowed_resources",
            _text_tuple("allowed_resources", self.allowed_resources),
        )
        object.__setattr__(
            self,
            "output_contract",
            _require_text("output_contract", self.output_contract),
        )

    def reference(self) -> SkillReference:
        return SkillReference(id=self.id, version=self.version)


class SkillCatalog:
    def __init__(self, contracts: Iterable[SkillContract] = ()) -> None:
        self._by_id: dict[str, SkillContract] = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract: SkillContract) -> SkillCatalog:
        if contract.id in self._by_id:
            raise ValueError(f"Skill contract is already registered: {contract.id}")
        self._by_id[contract.id] = contract
        return self

    def by_id(self, skill_id: str) -> SkillContract:
        return self._by_id[_require_text("skill_id", skill_id)]

    def contracts(self) -> tuple[SkillContract, ...]:
        return tuple(self._by_id.values())

    def compose_for_prompt(
        self,
        *,
        role: SkillRole | str,
        skill_ids: Iterable[str],
        prompt_allowed_tools: Iterable[str],
    ) -> tuple[str, tuple[SkillReference, ...]]:
        selected_role = SkillRole(role)
        allowed_tools = set(prompt_allowed_tools)
        instructions: list[str] = []
        references: list[SkillReference] = []
        for skill_id in skill_ids:
            contract = self.by_id(skill_id)
            if contract.role != selected_role:
                raise ValueError(
                    f"Skill {contract.id} belongs to {contract.role.value}, "
                    f"not {selected_role.value}"
                )
            unauthorized = sorted(set(contract.allowed_tools) - allowed_tools)
            if unauthorized:
                raise ValueError(
                    f"Skill {contract.id} exceeds prompt tool authority: {', '.join(unauthorized)}"
                )
            instructions.append(
                f"Skill {contract.id} v{contract.version.value}:\n{contract.instructions}"
            )
            references.append(contract.reference())
        return "\n\n".join(instructions), tuple(references)


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings")
    return tuple(
        dict.fromkeys(_require_text(f"{name}[{index}]", item) for index, item in enumerate(values))
    )
