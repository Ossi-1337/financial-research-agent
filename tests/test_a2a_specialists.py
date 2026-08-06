from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from financial_research_agent.a2a.specialists import (
    SpecialistExecutionService,
    _stock_evidence_ids,
    _synthesis_handoff_payload,
)
from financial_research_agent.agents import AgentRuntimeResolver
from financial_research_agent.context_analysis import NewsMacroSectorAgent
from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    MessageRole,
    ModelMetadata,
    ProviderCapability,
    ToolCall,
)
from financial_research_agent.llm.registry import ProviderRegistry
from financial_research_agent.market_data import (
    HistoricalPriceBar,
    HistoricalPriceResult,
    MarketDataSource,
    MarketDataStore,
    MarketSecurity,
    calculate_price_metrics,
)
from financial_research_agent.orchestration import (
    AgentHandoff,
    AgentRole,
    DelegationRequest,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorRunStatus,
    OrchestratorStepKind,
    default_orchestrator_plan,
)
from financial_research_agent.report_analysis import FinancialReportAnalysisStatus
from financial_research_agent.settings import Settings
from financial_research_agent.web_research import (
    WebJurisdiction,
    WebResearchResult,
    WebResearchStatus,
    WebSourceEvidence,
    WebSourceReliability,
    WebSourceType,
)

EVIDENCE_ID = "financial:evidence:1"
NOW = datetime(2026, 7, 27, 12, tzinfo=UTC)


def test_a2a_json_integral_numbers_are_accepted_as_positive_integers() -> None:
    from financial_research_agent.a2a.specialists import (
        _positive_int,
        _positive_int_mapping,
    )

    assert _positive_int({"fiscal_years": 3.0}, "fiscal_years") == 3
    assert _positive_int_mapping({"20-F": 1.0, "6-K": 2.0}) == {
        "20-F": 1,
        "6-K": 2,
    }


def test_stock_evidence_allows_missing_benchmark() -> None:
    assert _stock_evidence_ids(
        {
            "security": {"symbol": "NVO"},
            "primary_source": {"provider": "TEST TOOL OUTPUT"},
            "benchmark_security": None,
            "benchmark_source": None,
        }
    ) == ("stock:NVO:primary",)


def test_context_specialist_skips_when_no_approved_sources_exist() -> None:
    settings = Settings.from_env({})
    service = SpecialistExecutionService(
        financial_report_agent=object(),  # type: ignore[arg-type]
        stock_price_agent=object(),  # type: ignore[arg-type]
        context_agent=object(),  # type: ignore[arg-type]
        agent_runtime=AgentRuntimeResolver(settings=lambda: settings),
        now=lambda: NOW,
    )

    handoff = asyncio.run(
        service.execute(
            DelegationRequest(
                role=AgentRole.CONTEXT,
                run_id="run:test",
                step_id="context_analysis",
                correlation_id="run:test",
                expected_kind=OrchestratorStepKind.CONTEXT_ANALYSIS,
                payload={
                    "query": "TEST TOOL OUTPUT context",
                    "company_symbols": ["NVO"],
                    "source_items": [],
                },
            )
        )
    )

    assert handoff.status == OrchestratorHandoffStatus.SKIPPED
    assert handoff.limitations == ("No approved context sources were available.",)


def test_context_specialist_runs_web_research_inside_allowlisted_evidence_tool() -> None:
    provider = ContextSpecialistProvider()
    registry = ProviderRegistry().register_chat_provider("scripted", provider)
    settings = Settings.from_env(
        {"FRA_LLM_PROVIDER": "scripted", "FRA_LLM_MODEL": "scripted-model"}
    )
    web_service = FakeWebResearchService()
    service = SpecialistExecutionService(
        financial_report_agent=object(),  # type: ignore[arg-type]
        stock_price_agent=object(),  # type: ignore[arg-type]
        context_agent=NewsMacroSectorAgent(now=lambda: NOW),
        agent_runtime=AgentRuntimeResolver(
            settings=lambda: settings,
            registry=lambda _current: registry,
        ),
        web_research_service=web_service,  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    handoff = asyncio.run(
        service.execute(
            DelegationRequest(
                role=AgentRole.CONTEXT,
                run_id="run:test",
                step_id="context_analysis",
                correlation_id="run:test",
                expected_kind=OrchestratorStepKind.CONTEXT_ANALYSIS,
                payload={
                    "query": "Current Danish A/S reporting rules",
                    "company_symbols": [],
                    "source_items": [],
                    "web_research": True,
                    "jurisdiction": "DK",
                    "requires_official_source": True,
                    "evidence_required": True,
                },
            )
        )
    )

    request_payload = json.loads(provider.requests[0].messages[-1].content)
    assert web_service.calls == 1
    assert provider.requests[0].tools == ()
    assert request_payload["required_tool_result"]["source"] == "bounded_context_sources"
    assert handoff.evidence_ids == ("web:test",)
    assert "brave" in handoff.output["retrieval_methods"]


def test_market_refresh_reuses_fresh_cache_in_auto_mode() -> None:
    security = MarketSecurity(symbol="NVO", security_id="security:nvo")
    bars = (
        HistoricalPriceBar(
            security=security,
            priced_at=date(2026, 7, 27),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=1000,
        ),
    )
    store = MarketDataStore(stale_after=timedelta(days=1))
    store.save_history(
        HistoricalPriceResult(
            security=security,
            bars=bars,
            source=MarketDataSource(
                provider="alpha-vantage",
                provider_status="TEST TOOL OUTPUT",
                source_url="https://example.invalid/market",
                retrieved_at=NOW,
                attribution="TEST TOOL OUTPUT",
                data_as_of=date(2026, 7, 27),
            ),
            metrics=calculate_price_metrics(bars),
        )
    )
    settings = Settings.from_env({})
    service = SpecialistExecutionService(
        financial_report_agent=object(),  # type: ignore[arg-type]
        stock_price_agent=object(),  # type: ignore[arg-type]
        context_agent=object(),  # type: ignore[arg-type]
        market_data_store=store,
        agent_runtime=AgentRuntimeResolver(settings=lambda: settings),
        now=lambda: NOW,
    )

    handoff = asyncio.run(
        service.execute(
            DelegationRequest(
                role=AgentRole.STOCK,
                run_id="run:test",
                step_id="refresh_market_data",
                correlation_id="run:test",
                expected_kind=OrchestratorStepKind.MARKET_DATA_REFRESH,
                payload={
                    "security_id": security.security_id,
                    "ticker": security.symbol,
                    "outputsize": "compact",
                },
            )
        )
    )

    assert handoff.status == OrchestratorHandoffStatus.SUCCEEDED
    assert "Fresh cached market data reused" in handoff.warnings[-1]


def test_synthesis_handoff_payload_excludes_duplicate_internal_analysis() -> None:
    handoff = AgentHandoff(
        id="handoff:financial",
        step_id="financial_report_analysis",
        kind=OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
        status=OrchestratorHandoffStatus.PARTIAL,
        started_at=NOW,
        completed_at=NOW,
        input_summary={"company_id": "company:test"},
        output={
            "analysis": {"large_internal_payload": "not sent to synthesis LLM"},
            "agent_output": _valid_agent_output(),
        },
        evidence_ids=(EVIDENCE_ID,),
        warnings=("TEST warning.",),
        limitations=("TEST limitation.",),
        confidence=HandoffConfidence.MEDIUM,
    )

    payload = _synthesis_handoff_payload(handoff)

    assert payload["agent_output"] == _valid_agent_output()
    assert payload["evidence_ids"] == [EVIDENCE_ID]
    assert "analysis" not in payload
    assert "input_summary" not in payload
    assert "execution" not in payload
    assert "started_at" not in payload


def test_financial_tool_exposes_analysis_evidence_ids_to_agent_validation() -> None:
    provider = CapturingSpecialistProvider()
    registry = ProviderRegistry().register_chat_provider("scripted", provider)
    settings = Settings.from_env(
        {
            "FRA_LLM_PROVIDER": "scripted",
            "FRA_LLM_MODEL": "scripted-model",
        }
    )
    service = SpecialistExecutionService(
        financial_report_agent=FakeFinancialAgent(),  # type: ignore[arg-type]
        stock_price_agent=object(),  # type: ignore[arg-type]
        context_agent=object(),  # type: ignore[arg-type]
        agent_runtime=AgentRuntimeResolver(
            settings=lambda: settings,
            registry=lambda _current: registry,
        ),
        now=lambda: NOW,
    )

    handoff = asyncio.run(
        service.execute(
            DelegationRequest(
                role=AgentRole.FINANCIAL_REPORT,
                run_id="run:test",
                step_id="financial_report_analysis",
                correlation_id="run:test",
                expected_kind=OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
                payload={
                    "company_id": "company:test",
                    "legal_name": "TEST TOOL OUTPUT COMPANY",
                    "cik": "0000000001",
                },
            )
        )
    )

    agent_payload = json.loads(provider.requests[0].messages[-1].content)
    tool_payload = agent_payload["required_tool_result"]

    assert tool_payload["data"]["evidence_ids"] == [EVIDENCE_ID]
    assert provider.requests[0].tools == ()
    assert provider.requests[0].max_output_tokens == 2048
    assert handoff.status == OrchestratorHandoffStatus.PARTIAL
    assert handoff.evidence_ids == (EVIDENCE_ID,)
    assert handoff.execution is not None
    assert handoff.execution.provider == "scripted"
    assert handoff.execution.model == "scripted-model"
    assert handoff.execution.skill_references == ("filing-review@1.0.0",)


def test_specialist_uses_run_provider_snapshot_after_runtime_setting_changes() -> None:
    initial = CapturingSpecialistProvider(provider="initial", model="initial-model")
    updated = CapturingSpecialistProvider(provider="updated", model="updated-model")
    registry = (
        ProviderRegistry()
        .register_chat_provider("initial", initial)
        .register_chat_provider("updated", updated)
    )
    settings = Settings.from_env(
        {
            "FRA_LLM_PROVIDER": "updated",
            "FRA_LLM_MODEL": "updated-model",
        }
    )
    run = OrchestratedResearchRun(
        id="run:test",
        query="Research TEST TOOL OUTPUT company",
        status=OrchestratorRunStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW,
        execution_policy=OrchestratorExecutionPolicy.DISTRIBUTED_A2A,
        plan=default_orchestrator_plan(("financial-report", "synthesis")),
        specialist_roles=("financial-report", "synthesis"),
        agent_provider="initial",
        agent_model="initial-model",
    )
    service = SpecialistExecutionService(
        financial_report_agent=FakeFinancialAgent(),  # type: ignore[arg-type]
        stock_price_agent=object(),  # type: ignore[arg-type]
        context_agent=object(),  # type: ignore[arg-type]
        run_store=FakeRunStore(run),
        agent_runtime=AgentRuntimeResolver(
            settings=lambda: settings,
            registry=lambda _current: registry,
        ),
        now=lambda: NOW,
    )

    handoff = asyncio.run(service.execute(_financial_request()))

    assert len(initial.requests) == 1
    assert updated.requests == []
    assert handoff.execution is not None
    assert handoff.execution.provider == "initial"
    assert handoff.execution.model == "initial-model"


@dataclass(frozen=True, slots=True)
class FakeEvidence:
    id: str


@dataclass(frozen=True, slots=True)
class FakeFinancialResult:
    status: FinancialReportAnalysisStatus = FinancialReportAnalysisStatus.PARTIAL
    evidence: tuple[FakeEvidence, ...] = (FakeEvidence(EVIDENCE_ID),)
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ("TEST TOOL OUTPUT is intentionally partial.",)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "evidence": [{"id": evidence.id} for evidence in self.evidence],
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


class FakeFinancialAgent:
    def analyze(self, _company) -> FakeFinancialResult:
        return FakeFinancialResult()


class FakeRunStore:
    def __init__(self, run: OrchestratedResearchRun) -> None:
        self.run = run

    def get(self, run_id: str) -> OrchestratedResearchRun | None:
        return self.run if run_id == self.run.id else None


class CapturingSpecialistProvider:
    def __init__(
        self,
        *,
        provider: str = "scripted",
        model: str = "scripted-model",
    ) -> None:
        self.provider = provider
        self.model = model
        self.requests: list[ChatRequest] = []

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            provider=self.provider,
            model=self.model,
            capabilities=(
                ProviderCapability.CHAT,
                ProviderCapability.TOOL_CALLS,
                ProviderCapability.STRUCTURED_OUTPUT,
            ),
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.tools:
            tool_call = ToolCall(
                id="tool-call:financial:1",
                name="load_financial_report_evidence",
                arguments={},
            )
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=(tool_call,),
                ),
                provider=self.provider,
                model=request.model or self.model,
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(tool_call,),
            )
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
            provider=self.provider,
            model=request.model or self.model,
            structured_output=_valid_agent_output(),
        )

    def stream_chat(self, _request: ChatRequest):
        raise NotImplementedError


class ContextSpecialistProvider(CapturingSpecialistProvider):
    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        output = json.loads(json.dumps(_valid_agent_output()).replace(EVIDENCE_ID, "web:test"))
        output["agent_role"] = "news_macro_analyst"
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
            provider=self.provider,
            model=request.model or self.model,
            structured_output=output,
        )


class FakeWebResearchService:
    def __init__(self) -> None:
        self.calls = 0

    async def research(self, _request) -> WebResearchResult:
        self.calls += 1
        return WebResearchResult(
            status=WebResearchStatus.COMPLETE,
            sources=(
                WebSourceEvidence(
                    id="web:test",
                    canonical_url="https://www.retsinformation.dk/example",
                    title="Danish company law",
                    publisher="Retsinformation",
                    quote="TEST TOOL OUTPUT official Danish reporting rules.",
                    source_type=WebSourceType.REGULATORY,
                    reliability=WebSourceReliability.REGULATORY,
                    provider="brave",
                    jurisdiction=WebJurisdiction.DK,
                    retrieved_at=NOW,
                    expires_at=NOW + timedelta(hours=24),
                    content_sha256="a" * 64,
                ),
            ),
        )


def _valid_agent_output() -> dict[str, object]:
    sourced = {
        "statement": "TEST TOOL OUTPUT source-backed finding.",
        "evidence_ids": [EVIDENCE_ID],
        "confidence": "medium",
    }
    return {
        "agent_role": "financial_report_analyst",
        "facts": [dict(sourced)],
        "assumptions": [
            {
                "statement": "Evidence is intentionally partial.",
                "basis": "Tool result limitation.",
                "confidence": "medium",
            }
        ],
        "analysis": [dict(sourced)],
        "opinion": [dict(sourced)],
        "findings": [
            {
                "category": "financial_report",
                **sourced,
            }
        ],
        "uncertainty": {
            "missing_evidence": ["Additional periods are unavailable."],
            "limitations": ["TEST TOOL OUTPUT is intentionally partial."],
            "confidence": "medium",
        },
        "risks": [
            {
                "title": "Evidence limitation",
                "description": "Available evidence is partial.",
                "severity": "unknown",
                "evidence_ids": [EVIDENCE_ID],
            }
        ],
        "scenarios": [
            {
                "name": "Bounded evidence",
                "description": "No projection beyond stored evidence.",
                "evidence_ids": [EVIDENCE_ID],
            }
        ],
        "follow_up_questions": ["Which additional filing should be inspected?"],
        "refusal_notes": [
            {
                "claim": "Investment recommendation",
                "reason": "Research output does not provide recommendations.",
            }
        ],
        "reasoning_summary": "Used one allowlisted evidence tool.",
    }


def _financial_request() -> DelegationRequest:
    return DelegationRequest(
        role=AgentRole.FINANCIAL_REPORT,
        run_id="run:test",
        step_id="financial_report_analysis",
        correlation_id="run:test",
        expected_kind=OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
        payload={
            "company_id": "company:test",
            "legal_name": "TEST TOOL OUTPUT COMPANY",
            "cik": "0000000001",
        },
    )
