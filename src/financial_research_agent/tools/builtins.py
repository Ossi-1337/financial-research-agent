from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from financial_research_agent.entities import CompanySearchError, CompanySearchProvider
from financial_research_agent.tools.contracts import (
    ToolContext,
    ToolErrorCode,
    ToolPermission,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


def create_default_tool_registry(
    company_search_provider: CompanySearchProvider | None = None,
) -> ToolRegistry:
    company_tool = (
        resolve_company_tool(company_search_provider)
        if company_search_provider is not None
        else resolve_company_stub_tool()
    )
    return ToolRegistry(
        (
            current_utc_datetime_tool(),
            calculate_ratio_tool(),
            company_tool,
            read_local_evidence_tool(),
        )
    )


def current_utc_datetime_tool() -> ToolSpec:
    async def handler(_context: ToolContext, _arguments: Mapping[str, Any]) -> ToolResult:
        timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return ToolResult.succeeded(
            tool_call_id="current_utc_datetime",
            tool_name="current_utc_datetime",
            data={"utc_datetime": timestamp, "timezone": "UTC"},
            source="system_clock",
            freshness=timestamp,
        )

    return ToolSpec(
        name="current_utc_datetime",
        description="Return the current UTC date and time from the local system clock.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        permissions=(ToolPermission.CLOCK,),
        timeout_seconds=1.0,
        handler=handler,
    )


def calculate_ratio_tool() -> ToolSpec:
    async def handler(_context: ToolContext, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            numerator = Decimal(str(arguments["numerator"]))
            denominator = Decimal(str(arguments["denominator"]))
            precision = int(arguments.get("precision", 4))
        except (InvalidOperation, ValueError, TypeError) as exc:
            return ToolResult.failed(
                tool_call_id="calculate_ratio",
                tool_name="calculate_ratio",
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
                errors=(f"Ratio arguments are invalid: {exc}",),
            )
        if denominator == 0:
            return ToolResult.failed(
                tool_call_id="calculate_ratio",
                tool_name="calculate_ratio",
                error_code=ToolErrorCode.DIVISION_BY_ZERO,
                errors=("denominator must not be zero",),
            )
        if precision < 0 or precision > 12:
            return ToolResult.failed(
                tool_call_id="calculate_ratio",
                tool_name="calculate_ratio",
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
                errors=("precision must be between 0 and 12",),
            )
        ratio = numerator / denominator
        formatted = f"{ratio:.{precision}f}"
        return ToolResult.succeeded(
            tool_call_id="calculate_ratio",
            tool_name="calculate_ratio",
            data={
                "numerator": str(numerator),
                "denominator": str(denominator),
                "precision": precision,
                "ratio": formatted,
            },
            source="deterministic_calculation",
        )

    return ToolSpec(
        name="calculate_ratio",
        description="Calculate numerator divided by denominator with optional decimal precision.",
        input_schema={
            "type": "object",
            "properties": {
                "numerator": {"type": "number"},
                "denominator": {"type": "number"},
                "precision": {"type": "integer"},
            },
            "required": ["numerator", "denominator"],
            "additionalProperties": False,
        },
        permissions=(ToolPermission.CALCULATION,),
        timeout_seconds=1.0,
        handler=handler,
    )


def resolve_company_stub_tool() -> ToolSpec:
    async def handler(_context: ToolContext, arguments: Mapping[str, Any]) -> ToolResult:
        query = str(arguments["query"]).strip()
        return ToolResult.succeeded(
            tool_call_id="resolve_company_stub",
            tool_name="resolve_company_stub",
            data={
                "query": query,
                "matches": [],
                "real_entity_resolution_available": False,
            },
            source="milestone_08_stub",
            warnings=(
                "Real company entity resolution requires an injected company search provider.",
            ),
        )

    return ToolSpec(
        name="resolve_company_stub",
        description="A structural company lookup stub that returns no real company data.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        permissions=(ToolPermission.ENTITY_LOOKUP,),
        timeout_seconds=1.0,
        handler=handler,
    )


def resolve_company_tool(company_search_provider: CompanySearchProvider) -> ToolSpec:
    async def handler(_context: ToolContext, arguments: Mapping[str, Any]) -> ToolResult:
        query = str(arguments["query"]).strip()
        limit = int(arguments.get("limit", 10))
        try:
            result = await company_search_provider.search(query, limit=limit)
        except (CompanySearchError, ValueError) as exc:
            data: Mapping[str, Any] = {}
            if isinstance(exc, CompanySearchError):
                data = {
                    "code": exc.code.value,
                    "provider": exc.provider,
                    "retryable": exc.retryable,
                }
            return ToolResult.failed(
                tool_call_id="resolve_company",
                tool_name="resolve_company",
                error_code=ToolErrorCode.EXECUTION_FAILED,
                errors=(f"Company search failed: {exc}",),
                data=data,
            )
        source = result.source.provider if result.source is not None else None
        freshness = result.source.retrieved_at.isoformat() if result.source is not None else None
        return ToolResult.succeeded(
            tool_call_id="resolve_company",
            tool_name="resolve_company",
            data=result.to_dict(),
            source=source,
            freshness=freshness,
            warnings=result.warnings,
        )

    return ToolSpec(
        name="resolve_company",
        description=(
            "Resolve a company search query to real candidate companies, securities, tickers, "
            "CIKs, and source metadata. Return candidates for user review instead of guessing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        permissions=(ToolPermission.ENTITY_LOOKUP,),
        timeout_seconds=20.0,
        handler=handler,
    )


def read_local_evidence_tool() -> ToolSpec:
    async def handler(context: ToolContext, arguments: Mapping[str, Any]) -> ToolResult:
        evidence_id = str(arguments["evidence_id"]).strip()
        evidence = context.local_evidence.get(evidence_id)
        if evidence is None:
            return ToolResult.failed(
                tool_call_id="read_local_evidence",
                tool_name="read_local_evidence",
                error_code=ToolErrorCode.NOT_FOUND,
                errors=(f"Local evidence not found: {evidence_id}",),
                data={"evidence_id": evidence_id},
            )
        return ToolResult.succeeded(
            tool_call_id="read_local_evidence",
            tool_name="read_local_evidence",
            data={"evidence_id": evidence_id, "evidence": evidence},
            source="local_evidence",
        )

    return ToolSpec(
        name="read_local_evidence",
        description="Read an evidence item from injected in-memory local evidence.",
        input_schema={
            "type": "object",
            "properties": {"evidence_id": {"type": "string"}},
            "required": ["evidence_id"],
            "additionalProperties": False,
        },
        permissions=(ToolPermission.LOCAL_READ,),
        timeout_seconds=1.0,
        handler=handler,
    )
