from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from financial_research_agent.documents import DocumentExtractionError, DocumentExtractor
from financial_research_agent.web_research.contracts import (
    WebResearchError,
    WebResearchErrorCode,
    WebSearchCandidate,
)
from financial_research_agent.web_research.policy import SourceClassification, canonicalize_url

HostValidator = Callable[[str], Awaitable[None]]


class BoundedWebSourceFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: float = 15,
        max_bytes: int = 2_000_000,
        client: httpx.AsyncClient | None = None,
        host_validator: HostValidator | None = None,
        pdf_extractor: DocumentExtractor | None = None,
    ) -> None:
        if timeout_seconds <= 0 or max_bytes <= 0:
            raise ValueError("timeout_seconds and max_bytes must be positive")
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self.max_bytes = max_bytes
        self.host_validator = host_validator or ensure_public_https_url
        self._validate_connected_peer = host_validator is None
        self.pdf_extractor = pdf_extractor

    async def fetch(
        self,
        candidate: WebSearchCandidate,
        *,
        classification: SourceClassification,
    ) -> tuple[str, str]:
        url = canonicalize_url(candidate.url)
        for _redirect in range(4):
            await self.host_validator(url)
            try:
                async with self.client.stream(
                    "GET",
                    url,
                    headers={
                        "Accept": "text/html,text/plain,application/pdf",
                        "User-Agent": "financial-research-agent/0.1",
                    },
                    follow_redirects=False,
                ) as response:
                    if self._validate_connected_peer:
                        _ensure_public_peer(response)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise WebResearchError(
                                WebResearchErrorCode.FETCH_FAILED,
                                "Web source redirect was missing a location.",
                            )
                        url = canonicalize_url(urljoin(url, location))
                        continue
                    if response.status_code >= 400:
                        raise WebResearchError(
                            WebResearchErrorCode.FETCH_FAILED,
                            "Web source could not be fetched.",
                        )
                    declared = response.headers.get("content-length")
                    if declared and int(declared) > self.max_bytes:
                        raise WebResearchError(
                            WebResearchErrorCode.CONTENT_TOO_LARGE,
                            "Web source exceeded the configured size limit.",
                        )
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > self.max_bytes:
                            raise WebResearchError(
                                WebResearchErrorCode.CONTENT_TOO_LARGE,
                                "Web source exceeded the configured size limit.",
                            )
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    text = await self._extract(
                        bytes(content),
                        content_type=content_type,
                        classification=classification,
                    )
                    return canonicalize_url(str(response.url)), text
            except httpx.TimeoutException as exc:
                raise WebResearchError(
                    WebResearchErrorCode.TIMEOUT,
                    "Web source fetch timed out.",
                ) from exc
            except httpx.HTTPError as exc:
                raise WebResearchError(
                    WebResearchErrorCode.FETCH_FAILED,
                    "Web source fetch failed.",
                ) from exc
        raise WebResearchError(
            WebResearchErrorCode.FETCH_FAILED,
            "Web source redirected too many times.",
        )

    async def _extract(
        self,
        content: bytes,
        *,
        content_type: str,
        classification: SourceClassification,
    ) -> str:
        if content_type in {"text/html", "application/xhtml+xml"}:
            parser = _VisibleTextParser()
            parser.feed(content.decode("utf-8", errors="replace"))
            return _bounded_text(parser.text)
        if content_type == "text/plain":
            return _bounded_text(content.decode("utf-8", errors="replace"))
        if content_type == "application/pdf" and classification.reliability.value in {
            "official",
            "regulatory",
        }:
            if self.pdf_extractor is None:
                raise WebResearchError(
                    WebResearchErrorCode.UNSUPPORTED_CONTENT,
                    "PDF web sources are unavailable in this runtime.",
                )
            try:
                result = await asyncio.to_thread(
                    self.pdf_extractor.extract,
                    content,
                    content_type=content_type,
                )
            except DocumentExtractionError as exc:
                raise WebResearchError(
                    WebResearchErrorCode.UNSUPPORTED_CONTENT,
                    "Official PDF source could not be extracted safely.",
                ) from exc
            return _bounded_text(result.document.text)
        raise WebResearchError(
            WebResearchErrorCode.UNSUPPORTED_CONTENT,
            "Web source content type is unsupported.",
        )


async def ensure_public_https_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme.lower() != "https" or not parts.hostname or parts.username or parts.password:
        raise WebResearchError(WebResearchErrorCode.UNSAFE_URL, "Web source URL is not allowed.")
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            parts.hostname,
            parts.port or 443,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise WebResearchError(
            WebResearchErrorCode.UNSAFE_URL,
            "Web source host could not be resolved.",
        ) from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise WebResearchError(
                WebResearchErrorCode.UNSAFE_URL,
                "Web source resolved to a private address.",
            )


def _ensure_public_peer(response: httpx.Response) -> None:
    stream = response.extensions.get("network_stream")
    peer = stream.get_extra_info("server_addr") if stream is not None else None
    if not isinstance(peer, tuple) or not peer:
        raise WebResearchError(
            WebResearchErrorCode.UNSAFE_URL,
            "Web source peer address could not be verified.",
        )
    try:
        address = ipaddress.ip_address(str(peer[0]))
    except ValueError as exc:
        raise WebResearchError(
            WebResearchErrorCode.UNSAFE_URL,
            "Web source peer address could not be verified.",
        ) from exc
    if not address.is_global:
        raise WebResearchError(
            WebResearchErrorCode.UNSAFE_URL,
            "Web source connected to a private address.",
        )


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._parts: list[str] = []

    @property
    def text(self) -> str:
        return " ".join(self._parts)

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag.lower() in {"script", "style", "noscript", "template", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "template", "svg"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0 and (text := " ".join(data.split())):
            self._parts.append(text)


def _bounded_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        raise WebResearchError(
            WebResearchErrorCode.UNSUPPORTED_CONTENT,
            "Web source contained no readable text.",
        )
    return text[:20_000]
