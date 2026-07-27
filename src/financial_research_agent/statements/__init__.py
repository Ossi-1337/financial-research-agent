"""Financial statement ingestion contracts, SEC adapter, and local storage."""

from financial_research_agent.settings import Settings
from financial_research_agent.statements.contracts import (
    FinancialStatementCompany,
    FinancialStatementError,
    FinancialStatementErrorCode,
    FinancialStatementPeriod,
    FinancialStatementPeriodType,
    FinancialStatementProvider,
    FinancialStatementProviderName,
    FinancialStatementResult,
    FinancialStatementSource,
    FinancialStatementType,
    NormalizedFinancialStatement,
)
from financial_research_agent.statements.sec_companyfacts import SECCompanyFactsProvider
from financial_research_agent.statements.store import FinancialStatementStore


def create_default_financial_statement_provider(settings: Settings) -> FinancialStatementProvider:
    provider = settings.data_sources.financial_statement_provider
    if provider != FinancialStatementProviderName.SEC_COMPANY_FACTS.value:
        raise ValueError(f"Unsupported financial statement provider: {provider}")
    return SECCompanyFactsProvider(user_agent=settings.data_sources.sec_user_agent)


__all__ = [
    "FinancialStatementCompany",
    "FinancialStatementError",
    "FinancialStatementErrorCode",
    "FinancialStatementPeriod",
    "FinancialStatementPeriodType",
    "FinancialStatementProvider",
    "FinancialStatementProviderName",
    "FinancialStatementResult",
    "FinancialStatementSource",
    "FinancialStatementStore",
    "FinancialStatementType",
    "NormalizedFinancialStatement",
    "SECCompanyFactsProvider",
    "create_default_financial_statement_provider",
]
