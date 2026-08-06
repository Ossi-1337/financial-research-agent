from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from financial_research_agent.web_research.contracts import (
    WebJurisdiction,
    WebSearchCandidate,
    WebSourceReliability,
    WebSourceType,
)

_TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_OFFICIAL_DOMAINS = {
    WebJurisdiction.DK: (
        "retsinformation.dk",
        "erhvervsstyrelsen.dk",
        "virk.dk",
        "dst.dk",
        "nationalbanken.dk",
    ),
    WebJurisdiction.EU: ("europa.eu", "eur-lex.europa.eu", "ecb.europa.eu"),
    WebJurisdiction.US: (
        "sec.gov",
        "federalreserve.gov",
        "fred.stlouisfed.org",
        "congress.gov",
    ),
}


@dataclass(frozen=True, slots=True)
class SourceClassification:
    source_type: WebSourceType
    reliability: WebSourceReliability
    jurisdiction: WebJurisdiction | None
    warnings: tuple[str, ...] = ()


class WebSourcePolicy:
    def classify(
        self,
        candidate: WebSearchCandidate,
        *,
        requested_jurisdiction: WebJurisdiction | None,
    ) -> SourceClassification:
        host = _host(candidate.url)
        official_jurisdiction = _official_jurisdiction(host)
        if official_jurisdiction is not None:
            return SourceClassification(
                source_type=WebSourceType.REGULATORY,
                reliability=WebSourceReliability.REGULATORY,
                jurisdiction=official_jurisdiction,
            )
        if host == "finance.yahoo.com" or host.endswith(".finance.yahoo.com"):
            return SourceClassification(
                source_type=WebSourceType.SECONDARY,
                reliability=WebSourceReliability.SECONDARY,
                jurisdiction=requested_jurisdiction,
                warnings=(
                    "Yahoo Finance is secondary context and is not a primary price, filing, "
                    "statement, or regulatory source.",
                ),
            )
        if candidate.provider == "alpha-vantage-news":
            return SourceClassification(
                source_type=WebSourceType.NEWS,
                reliability=WebSourceReliability.DOCUMENTED_API,
                jurisdiction=requested_jurisdiction,
            )
        return SourceClassification(
            source_type=WebSourceType.SECONDARY,
            reliability=WebSourceReliability.SECONDARY,
            jurisdiction=requested_jurisdiction,
            warnings=("Source is secondary and should be corroborated for material claims.",),
        )

    def is_official_for(
        self,
        classification: SourceClassification,
        jurisdiction: WebJurisdiction | None,
    ) -> bool:
        return classification.reliability in {
            WebSourceReliability.OFFICIAL,
            WebSourceReliability.REGULATORY,
        } and (jurisdiction is None or classification.jurisdiction == jurisdiction)

    def official_domains(self, jurisdiction: WebJurisdiction | None) -> tuple[str, ...]:
        if jurisdiction is None:
            return tuple(domain for domains in _OFFICIAL_DOMAINS.values() for domain in domains)
        return _OFFICIAL_DOMAINS[jurisdiction]


def canonicalize_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme.lower() != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("source URL must be an absolute HTTPS URL")
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMETERS
        ],
        doseq=True,
    )
    path = parts.path or "/"
    return urlunsplit(("https", parts.netloc.lower(), path, query, ""))


def _official_jurisdiction(host: str) -> WebJurisdiction | None:
    for jurisdiction, domains in _OFFICIAL_DOMAINS.items():
        if any(host == domain or host.endswith(f".{domain}") for domain in domains):
            return jurisdiction
    return None


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()
