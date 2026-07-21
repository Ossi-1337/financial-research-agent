from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from financial_research_agent.entities import (
    CompanySearchError,
    CompanySearchErrorCode,
    CompanySearchStatus,
    EntityIdentifier,
    EntityIdentifierType,
    ResolvedCompany,
    ResolvedSecurity,
    SECCompanyTickerProvider,
    SourceMetadata,
)
from financial_research_agent.entities.contracts import CompanySearchCandidate, CompanySearchResult

SEC_FIXTURE = {
    "0": {"cik_str": 353278, "ticker": "NVO", "title": "NOVO NORDISK A S"},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": 1000045, "ticker": "NVOX", "title": "NOVO INTEGRATED SCIENCES INC"},
}

SEC_EXCHANGE_FIXTURE = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [
        [353278, "NOVO NORDISK A S", "NVO", "NYSE"],
        [353278, "NOVO NORDISK A S", "NONOF", "OTC"],
        [789019, "MICROSOFT CORP", "MSFT", "Nasdaq"],
    ],
}


def test_company_entity_contracts_are_immutable_and_json_ready() -> None:
    retrieved_at = datetime(2026, 7, 4, tzinfo=UTC)
    cik = EntityIdentifier(EntityIdentifierType.CIK, "0000353278", source="sec_company_tickers")
    ticker = EntityIdentifier(EntityIdentifierType.TICKER, "NVO", source="sec_company_tickers")
    company = ResolvedCompany(
        id="sec:cik:0000353278",
        legal_name="NOVO NORDISK A S",
        identifiers=(cik, ticker),
    )
    security = ResolvedSecurity(
        id="sec:ticker:NVO:cik:0000353278",
        company_id=company.id,
        ticker="nvo",
        name="NOVO NORDISK A S",
        identifiers=(ticker, cik),
    )
    source = SourceMetadata(
        provider="sec_company_tickers",
        provider_status="official",
        source_url="https://www.sec.gov/files/company_tickers.json",
        retrieved_at=retrieved_at,
        attribution="SEC fixture",
    )
    candidate = CompanySearchCandidate(
        company=company,
        securities=(security,),
        score=90,
        match_reason="company_name_tokens",
        source=source,
    )

    payload = candidate.to_dict()

    assert payload["company"]["identifiers"][0]["type"] == "cik"
    assert payload["securities"][0]["ticker"] == "NVO"
    assert security.exchange_mic is None
    assert security.currency is None
    with pytest.raises(AttributeError):
        company.legal_name = "changed"  # type: ignore[misc]


def test_sec_company_search_returns_reviewable_novo_candidates(tmp_path) -> None:
    provider = SECCompanyTickerProvider(
        cache_path=tmp_path / "sec_company_tickers.json",
        user_agent="financial-research-agent-test/0.1",
        http_client=_client_with_json(SEC_FIXTURE),
        now=lambda: datetime(2026, 7, 4, tzinfo=UTC),
    )

    result = asyncio.run(provider.search("Novo Nordisk", limit=5))

    assert result.status == CompanySearchStatus.REVIEW_REQUIRED
    assert result.source is not None
    assert result.source.provider == "sec_company_tickers"
    assert result.source.provider_status == "official"
    assert result.source.retrieved_at == datetime(2026, 7, 4, tzinfo=UTC)
    assert "SEC company_tickers_exchange.json covers SEC filer ticker" in result.warnings[0]
    assert result.candidates[0].company.legal_name == "NOVO NORDISK A S"
    assert result.candidates[0].securities[0].ticker == "NVO"
    assert result.candidates[0].securities[0].isin is None


def test_sec_company_search_supports_ticker_match(tmp_path) -> None:
    provider = SECCompanyTickerProvider(
        cache_path=tmp_path / "sec_company_tickers.json",
        user_agent="financial-research-agent-test/0.1",
        http_client=_client_with_json(SEC_FIXTURE),
    )

    result = asyncio.run(provider.search("NVO"))

    assert result.candidates[0].match_reason == "ticker_exact"
    assert result.candidates[0].score == 100.0


def test_sec_company_search_groups_cik_and_prefers_listed_security(tmp_path) -> None:
    provider = SECCompanyTickerProvider(
        cache_path=tmp_path / "sec_company_tickers.json",
        user_agent="financial-research-agent-test/0.1",
        http_client=_client_with_json(SEC_EXCHANGE_FIXTURE),
    )

    result = asyncio.run(provider.search("Novo Nordisk"))
    candidate = result.candidates[0]

    assert [security.ticker for security in candidate.securities] == ["NVO", "NONOF"]
    assert candidate.securities[0].exchange_name == "NYSE"
    assert candidate.securities[0].exchange_mic == "XNYS"
    assert len(result.candidates) == 1


def test_sec_company_search_honors_explicit_otc_ticker_query(tmp_path) -> None:
    provider = SECCompanyTickerProvider(
        cache_path=tmp_path / "sec_company_tickers.json",
        user_agent="financial-research-agent-test/0.1",
        http_client=_client_with_json(SEC_EXCHANGE_FIXTURE),
    )

    result = asyncio.run(provider.search("NONOF"))

    assert result.candidates[0].securities[0].ticker == "NONOF"
    assert result.candidates[0].match_reason == "ticker_exact"


def test_sec_company_search_uses_fresh_cache_without_network(tmp_path) -> None:
    cache_path = tmp_path / "sec_company_tickers.json"
    first_provider = SECCompanyTickerProvider(
        cache_path=cache_path,
        user_agent="financial-research-agent-test/0.1",
        http_client=_client_with_json(SEC_FIXTURE),
        now=lambda: datetime(2026, 7, 4, tzinfo=UTC),
    )
    asyncio.run(first_provider.search("Novo"))

    def fail_if_called(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("fresh cache should avoid network calls")

    second_provider = SECCompanyTickerProvider(
        cache_path=cache_path,
        user_agent="financial-research-agent-test/0.1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail_if_called)),
        now=lambda: datetime(2026, 7, 5, tzinfo=UTC),
    )

    result = asyncio.run(second_provider.search("NVO"))

    assert result.candidates[0].securities[0].ticker == "NVO"
    assert result.source is not None
    assert result.source.retrieved_at == datetime(2026, 7, 4, tzinfo=UTC)


def test_sec_company_search_falls_back_to_stale_cache_on_refresh_failure(tmp_path) -> None:
    cache_path = tmp_path / "sec_company_tickers.json"
    first_provider = SECCompanyTickerProvider(
        cache_path=cache_path,
        cache_ttl=timedelta(days=1),
        user_agent="financial-research-agent-test/0.1",
        http_client=_client_with_json(SEC_FIXTURE),
        now=lambda: datetime(2026, 7, 1, tzinfo=UTC),
    )
    asyncio.run(first_provider.search("Novo"))
    stale_provider = SECCompanyTickerProvider(
        cache_path=cache_path,
        cache_ttl=timedelta(days=1),
        user_agent="financial-research-agent-test/0.1",
        http_client=_client_with_status(503),
        now=lambda: datetime(2026, 7, 4, tzinfo=UTC),
    )

    result = asyncio.run(stale_provider.search("NVO"))

    assert result.candidates[0].securities[0].ticker == "NVO"
    assert result.source is not None
    assert result.source.freshness_warning == "Using stale cached SEC company ticker data."
    assert "stale cached data" in result.warnings[-1]


def test_sec_company_search_maps_malformed_response(tmp_path) -> None:
    provider = SECCompanyTickerProvider(
        cache_path=tmp_path / "sec_company_tickers.json",
        user_agent="financial-research-agent-test/0.1",
        http_client=_client_with_text("not json"),
    )

    with pytest.raises(CompanySearchError) as exc_info:
        asyncio.run(provider.search("Novo"))

    assert exc_info.value.code == CompanySearchErrorCode.MALFORMED_RESPONSE


def test_sec_company_search_returns_no_matches_without_guessing(tmp_path) -> None:
    provider = SECCompanyTickerProvider(
        cache_path=tmp_path / "sec_company_tickers.json",
        user_agent="financial-research-agent-test/0.1",
        http_client=_client_with_json(SEC_FIXTURE),
    )

    result = asyncio.run(provider.search("Definitely Not A Real Fixture Company"))

    assert result.status == CompanySearchStatus.NO_MATCHES
    assert result.candidates == ()
    assert "No SEC company ticker matches found" in result.warnings[-1]


def _client_with_json(payload: object) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"]
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _client_with_text(payload: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, text=payload))
    )


def _client_with_status(status_code: int) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(status_code))
    )


class FakeCompanySearchProvider:
    async def search(self, query: str, *, limit: int = 10) -> CompanySearchResult:
        source = SourceMetadata(
            provider="fake-company-search",
            provider_status="test fixture",
            source_url="https://example.invalid/company-search-fixture",
            retrieved_at=datetime(2026, 7, 4, tzinfo=UTC),
            attribution="test fixture",
        )
        company = ResolvedCompany(
            id="fixture:company:novo",
            legal_name="TEST TOOL OUTPUT NOVO NORDISK",
            identifiers=(EntityIdentifier(EntityIdentifierType.TICKER, "NVO", source="fixture"),),
        )
        security = ResolvedSecurity(
            id="fixture:security:nvo",
            company_id=company.id,
            ticker="NVO",
            name=company.legal_name,
        )
        return CompanySearchResult(
            query=query,
            status=CompanySearchStatus.REVIEW_REQUIRED,
            candidates=(
                CompanySearchCandidate(
                    company=company,
                    securities=(security,),
                    score=90,
                    match_reason="test_fixture",
                    source=source,
                ),
            ),
            source=source,
            warnings=(f"limit={limit}",),
        )
