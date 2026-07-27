from __future__ import annotations

from dataclasses import dataclass

from financial_research_agent.a2a.store import SQLiteA2ATaskStore
from financial_research_agent.background import BackgroundResearchRunner
from financial_research_agent.context_analysis import NewsMacroSectorAgent
from financial_research_agent.entities import create_default_company_search_provider
from financial_research_agent.filings import create_default_filing_provider
from financial_research_agent.market_data import create_default_market_data_provider
from financial_research_agent.orchestration import ResearchOrchestrator
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


def create_default_a2a_runtime(settings: Settings) -> A2AResearchRuntime:
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
    )
    background_runner = BackgroundResearchRunner(
        max_concurrent_runs=settings.a2a.max_concurrent_tasks,
        job_store=persistence.background_jobs,
    )
    return A2AResearchRuntime(
        orchestrator=orchestrator,
        background_runner=background_runner,
        task_store=SQLiteA2ATaskStore(persistence.database),
        orchestrator_run_store=persistence.orchestrator_runs,
        persistence=persistence,
    )
