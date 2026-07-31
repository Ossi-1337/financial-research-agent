from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

NARRATIVE_PROMPT_ID = "synthesis.narrative-presentation.v1"
NARRATIVE_PROMPT_VERSION = "1.0.1"
NARRATIVE_SCHEMA_VERSION = 1
MAX_NARRATIVE_PARAGRAPHS_PER_SECTION = 2
MAX_NARRATIVE_PARAGRAPH_CHARS = 700


class NarrativeSection(StrEnum):
    CURRENT_SITUATION = "current_situation"
    FINANCIALS = "financials"
    STOCK_BEHAVIOR = "stock_behavior"
    CONTEXT = "context"
    RISKS = "risks"
    SCENARIOS = "scenarios"
    UNKNOWNS = "unknowns"
    LIMITATIONS = "limitations"


NARRATIVE_SECTION_ORDER = tuple(NarrativeSection)


@dataclass(frozen=True, slots=True)
class NarrativeParagraph:
    text: str
    source_point_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()
    source_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        text = _required_text("text", self.text)
        if len(text) > MAX_NARRATIVE_PARAGRAPH_CHARS:
            raise ValueError(f"text must be at most {MAX_NARRATIVE_PARAGRAPH_CHARS} characters")
        object.__setattr__(self, "text", text)
        object.__setattr__(
            self,
            "source_point_ids",
            _text_tuple("source_point_ids", self.source_point_ids),
        )
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(
            self,
            "source_markers",
            _text_tuple("source_markers", self.source_markers),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "source_point_ids": list(self.source_point_ids),
            "evidence_ids": list(self.evidence_ids),
            "source_markers": list(self.source_markers),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> NarrativeParagraph:
        return cls(
            text=str(payload["text"]),
            source_point_ids=_string_items(payload.get("source_point_ids")),
            evidence_ids=_string_items(payload.get("evidence_ids")),
            source_markers=_string_items(payload.get("source_markers")),
        )


@dataclass(frozen=True, slots=True)
class NarrativePresentationSection:
    section: NarrativeSection
    paragraphs: tuple[NarrativeParagraph, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "section", NarrativeSection(self.section))
        paragraphs = tuple(self.paragraphs)
        if len(paragraphs) > MAX_NARRATIVE_PARAGRAPHS_PER_SECTION:
            raise ValueError(
                f"section may contain at most {MAX_NARRATIVE_PARAGRAPHS_PER_SECTION} paragraphs"
            )
        if any(not isinstance(paragraph, NarrativeParagraph) for paragraph in paragraphs):
            raise ValueError("paragraphs must contain NarrativeParagraph values")
        object.__setattr__(self, "paragraphs", paragraphs)

    def to_dict(self) -> dict[str, object]:
        return {
            "section": self.section.value,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
        }


@dataclass(frozen=True, slots=True)
class NarrativePresentation:
    id: str
    run_id: str
    report_id: str
    synthesis_sha256: str
    provider: str
    model: str
    created_at: datetime
    sections: tuple[NarrativePresentationSection, ...]
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    no_recommendation_notice: str = ""
    prompt_id: str = NARRATIVE_PROMPT_ID
    prompt_version: str = NARRATIVE_PROMPT_VERSION
    schema_version: int = NARRATIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("id", "run_id", "report_id", "provider", "model"):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        if not _is_sha256(self.synthesis_sha256):
            raise ValueError("synthesis_sha256 must be a lowercase SHA-256 digest")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        sections = tuple(self.sections)
        if tuple(section.section for section in sections) != NARRATIVE_SECTION_ORDER:
            raise ValueError("narrative sections must use the canonical section order")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        object.__setattr__(
            self,
            "no_recommendation_notice",
            _required_text("no_recommendation_notice", self.no_recommendation_notice),
        )
        if self.prompt_id != NARRATIVE_PROMPT_ID:
            raise ValueError("unsupported narrative prompt id")
        if self.prompt_version != NARRATIVE_PROMPT_VERSION:
            raise ValueError("unsupported narrative prompt version")
        if self.schema_version != NARRATIVE_SCHEMA_VERSION:
            raise ValueError("unsupported narrative schema version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "run_id": self.run_id,
            "report_id": self.report_id,
            "synthesis_sha256": self.synthesis_sha256,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "sections": [section.to_dict() for section in self.sections],
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "no_recommendation_notice": self.no_recommendation_notice,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> NarrativePresentation:
        sections = tuple(
            NarrativePresentationSection(
                section=NarrativeSection(str(value["section"])),
                paragraphs=tuple(
                    NarrativeParagraph.from_dict(item)
                    for item in _mapping_items(value.get("paragraphs"))
                ),
            )
            for value in _mapping_items(payload.get("sections"))
        )
        created_at = datetime.fromisoformat(str(payload["created_at"]))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return cls(
            schema_version=int(payload.get("schema_version", 0)),
            id=str(payload["id"]),
            run_id=str(payload["run_id"]),
            report_id=str(payload["report_id"]),
            synthesis_sha256=str(payload["synthesis_sha256"]),
            prompt_id=str(payload["prompt_id"]),
            prompt_version=str(payload["prompt_version"]),
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            created_at=created_at,
            sections=sections,
            warnings=_string_items(payload.get("warnings")),
            limitations=_string_items(payload.get("limitations")),
            no_recommendation_notice=str(payload["no_recommendation_notice"]),
        )


def synthesis_sha256(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        _json_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings")
    return tuple(_required_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _string_items(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _mapping_items(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value
