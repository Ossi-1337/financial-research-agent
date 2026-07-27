from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.resources import files
from urllib.parse import urlparse

from financial_research_agent.context_analysis import (
    ContextScope,
    ContextSourceItem,
    ContextSourceType,
    SourceReliability,
)

from .contracts import ScenarioError, ScenarioErrorCode


def load_context_snapshot(
    resource_name: str,
    *,
    scenario_id: str,
    now: datetime | None = None,
) -> tuple[ContextSourceItem, ...]:
    current = now or datetime.now(UTC)
    try:
        resource = files("financial_research_agent.scenarios").joinpath("data", resource_name)
        payload = json.loads(resource.read_text(encoding="utf-8"))
        return _parse_snapshot(payload, scenario_id=scenario_id, now=current)
    except ScenarioError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ScenarioError(
            ScenarioErrorCode.INVALID_CONTEXT_SNAPSHOT,
            "Scenario context snapshot could not be loaded safely.",
        ) from exc


def _parse_snapshot(
    payload: object,
    *,
    scenario_id: str,
    now: datetime,
) -> tuple[ContextSourceItem, ...]:
    if not isinstance(payload, Mapping):
        raise _invalid("Context snapshot must be a JSON object.")
    if payload.get("schema_version") != 1 or payload.get("scenario_id") != scenario_id:
        raise _invalid("Context snapshot schema or scenario id is invalid.")
    items = payload.get("source_items")
    if not isinstance(items, list) or not items:
        raise _invalid("Context snapshot must contain source_items.")
    parsed = tuple(_source_item(item, now=now) for item in items)
    scopes = {item.scope for item in parsed}
    if ContextScope.COMPANY not in scopes or not scopes.intersection(
        {ContextScope.MACRO, ContextScope.SECTOR}
    ):
        raise _invalid("Context snapshot requires company and macro or sector sources.")
    return parsed


def _source_item(payload: object, *, now: datetime) -> ContextSourceItem:
    if not isinstance(payload, Mapping):
        raise _invalid("Context source item must be an object.")
    source_url = str(payload.get("source_url", ""))
    parsed_url = urlparse(source_url)
    hostname = (parsed_url.hostname or "").lower()
    if parsed_url.scheme != "https" or not hostname or hostname == "localhost":
        raise _invalid("Context source URLs must use public HTTPS endpoints.")
    if hostname.endswith((".invalid", ".local", ".test")):
        raise _invalid("Fixture and local context source URLs are not allowed.")
    metadata_payload = payload.get("metadata", {})
    if not isinstance(metadata_payload, Mapping):
        raise _invalid("Context source metadata must be an object.")
    metadata = {str(key): str(value) for key, value in metadata_payload.items()}
    marker_fields = (
        payload.get("id"),
        payload.get("title"),
        payload.get("summary"),
        payload.get("source_name"),
        *metadata.values(),
    )
    if any(re.search(r"\bfixture\b", str(value), re.IGNORECASE) for value in marker_fields):
        raise _invalid("Fixture context sources are not allowed.")
    retrieved_at = _aware_datetime(payload.get("retrieved_at"), "retrieved_at")
    published_value = payload.get("published_at")
    if published_value is None:
        raise _invalid("Context sources must include published_at.")
    published_at = _aware_datetime(published_value, "published_at")
    if retrieved_at > now or published_at > now:
        raise _invalid("Future-dated context sources are not allowed.")
    try:
        return ContextSourceItem(
            id=str(payload["id"]),
            title=str(payload["title"]),
            summary=str(payload["summary"]),
            source_url=source_url,
            source_name=str(payload["source_name"]),
            source_type=ContextSourceType(str(payload["source_type"])),
            reliability=SourceReliability(str(payload["reliability"])),
            scope=ContextScope(str(payload["scope"])),
            retrieved_at=retrieved_at,
            published_at=published_at,
            company_symbols=tuple(str(item) for item in payload.get("company_symbols", ())),
            sector=_optional_text(payload.get("sector")),
            region=_optional_text(payload.get("region")),
            topics=tuple(str(item) for item in payload.get("topics", ())),
            metadata=metadata,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _invalid("Context source item is invalid.") from exc


def _aware_datetime(value: object, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise _invalid(f"{name} must be an ISO 8601 datetime.") from exc
    if parsed.tzinfo is None:
        raise _invalid(f"{name} must include a timezone.")
    return parsed


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _invalid(message: str) -> ScenarioError:
    return ScenarioError(ScenarioErrorCode.INVALID_CONTEXT_SNAPSHOT, message)
