from __future__ import annotations

from dataclasses import dataclass, field

from financial_research_agent.a2a.coordination import A2ATaskExecutionCoordinator
from financial_research_agent.a2a.delegations import SQLiteA2ADelegationStore
from financial_research_agent.a2a.dispatcher import A2AResearchStepDispatcher
from financial_research_agent.a2a.specialists import SpecialistExecutionService
from financial_research_agent.a2a.store import SQLiteA2ATaskStore
from financial_research_agent.background import BackgroundResearchRunner
from financial_research_agent.context_analysis import NewsMacroSectorAgent
from financial_research_agent.entities import create_default_company_search_provider
from financial_research_agent.filings import create_default_filing_provider
from financial_research_agent.market_data import create_default_market_data_provider
from financial_research_agent.orchestration import (
    AgentEndpoint,
    AgentRole,
    ResearchOrchestrator,
)
from financial_research_agent.persistence import PersistenceBundle, create_persistence
from financial_research_agent.report_analysis import FinancialReportAnalysisAgent
from financial_research_agent.settings import Settings
from financial_research_agent.statements import create_default_financial_statement_provider
from financial_research_agent.stock_analysis import StockPriceAnalysisAgent


@dataclass(frozen=True, slots=True)
class A2AResearchRuntime:
    orchestrator: ResearchOrchestrator
    background_runner: BackgroundResearchRunner
    task_store: SQLiteA2ATaskStore
    orchestrator_run_store: object
    persistence: PersistenceBundle
    role: AgentRole = AgentRole.COMPANY_RESEARCH
    specialist_service: SpecialistExecutionService | None = None
    delegation_store: SQLiteA2ADelegationStore | None = None
    execution_coordinator: A2ATaskExecutionCoordinator = field(
        default_factory=A2ATaskExecutionCoordinator
    )


def create_default_a2a_runtime(
    settings: Settings,
    *,
    role: AgentRole = AgentRole.COMPANY_RESEARCH,
) -> A2AResearchRuntime:
    persistence = create_persistence(settings)
    if persistence.database is None or persistence.background_jobs is None:
        raise ValueError("A2A task persistence requires FRA_STORAGE_PROVIDER=sqlite")

    company_search = create_default_company_search_provider(settings)
    market_provider = create_default_market_data_provider(settings)
    statement_provider = create_default_financial_statement_provider(settings)
    filing_provider = create_default_filing_provider(settings)
    financial_report_agent = FinancialReportAnalysisAgent(
        statement_store=persistence.financial_statements,
        filing_store=persistence.filings,
        statement_provider=settings.data_sources.financial_statement_provider,
        filing_provider=settings.data_sources.filing_provider,
    )
    stock_price_agent = StockPriceAnalysisAgent(
        market_data_store=persistence.market_data,
        market_data_provider=settings.data_sources.market_data_provider,
    )
    specialist_service = SpecialistExecutionService(
        financial_report_agent=financial_report_agent,
        stock_price_agent=stock_price_agent,
        context_agent=NewsMacroSectorAgent(),
        run_store=persistence.orchestrator_runs,
    )
    delegation_store = SQLiteA2ADelegationStore(persistence.database)
    dispatcher = (
        A2AResearchStepDispatcher(
            endpoints=_agent_endpoints(settings),
            timeout_seconds=settings.a2a.delegation_timeout_seconds,
            max_attempts=settings.a2a.delegation_max_attempts,
            api_key=settings.a2a.api_key,
            delegation_store=delegation_store,
        )
        if role == AgentRole.COMPANY_RESEARCH and settings.a2a.topology == "distributed"
        else None
    )
    orchestrator = ResearchOrchestrator(
        company_search_provider=company_search,
        market_data_provider=market_provider,
        market_data_store=persistence.market_data,
        financial_statement_provider=statement_provider,
        financial_statement_store=persistence.financial_statements,
        filing_provider=filing_provider,
        filing_store=persistence.filings,
        financial_report_agent=financial_report_agent,
        stock_price_agent=stock_price_agent,
        context_agent=NewsMacroSectorAgent(),
        run_store=persistence.orchestrator_runs,
        step_dispatcher=dispatcher,
    )
    background_runner = BackgroundResearchRunner(
        max_concurrent_runs=settings.a2a.max_concurrent_tasks,
        job_store=persistence.background_jobs,
    )
    return A2AResearchRuntime(
        orchestrator=orchestrator,
        background_runner=background_runner,
        task_store=SQLiteA2ATaskStore(persistence.database, owner=role.value),
        orchestrator_run_store=persistence.orchestrator_runs,
        persistence=persistence,
        role=role,
        specialist_service=specialist_service,
        delegation_store=delegation_store,
        execution_coordinator=A2ATaskExecutionCoordinator(),
    )


def _agent_endpoints(settings: Settings) -> dict[AgentRole, AgentEndpoint]:
    return {
        AgentRole.FINANCIAL_REPORT: AgentEndpoint(
            role=AgentRole.FINANCIAL_REPORT,
            service_id="financial-report",
            base_url=settings.a2a.financial_report_url,
            skill_id="financial_report_analysis",
        ),
        AgentRole.STOCK: AgentEndpoint(
            role=AgentRole.STOCK,
            service_id="stock",
            base_url=settings.a2a.stock_url,
            skill_id="stock_price_analysis",
        ),
        AgentRole.CONTEXT: AgentEndpoint(
            role=AgentRole.CONTEXT,
            service_id="context",
            base_url=settings.a2a.context_url,
            skill_id="context_analysis",
        ),
        AgentRole.SYNTHESIS: AgentEndpoint(
            role=AgentRole.SYNTHESIS,
            service_id="synthesis",
            base_url=settings.a2a.synthesis_url,
            skill_id="research_synthesis",
        ),
    }
