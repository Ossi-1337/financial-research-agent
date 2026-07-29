from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum


class ConversationScope(StrEnum):
    FINANCIAL_RESEARCH = "financial_research"
    FINANCIAL_EDUCATION = "financial_education"
    PRODUCT_HELP = "product_help"
    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"


class ConversationPolicyReason(StrEnum):
    ALLOWED = "allowed"
    CODE_GENERATION = "code_generation"
    PROMPT_INJECTION = "prompt_injection"
    SECRET_EXTRACTION = "secret_extraction"
    PERMISSION_ESCALATION = "permission_escalation"
    INVESTMENT_ADVICE = "investment_advice"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class ConversationPolicyDecision:
    scope: ConversationScope
    reason: ConversationPolicyReason
    safe_response: str
    flags: tuple[ConversationPolicyReason, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", ConversationScope(self.scope))
        object.__setattr__(self, "reason", ConversationPolicyReason(self.reason))
        object.__setattr__(
            self,
            "flags",
            tuple(dict.fromkeys(ConversationPolicyReason(flag) for flag in self.flags)),
        )
        if not self.safe_response.strip():
            raise ValueError("safe_response is required")

    @property
    def uses_fixed_response(self) -> bool:
        return (
            self.scope
            in {
                ConversationScope.GREETING,
                ConversationScope.PRODUCT_HELP,
                ConversationScope.OUT_OF_SCOPE,
            }
            or self.reason != ConversationPolicyReason.ALLOWED
        )


class ConversationPolicy:
    MAX_INPUT_CHARS = 4_000
    MAX_MENTIONS = 5

    def normalize_input(self, content: str) -> str:
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        normalized = unicodedata.normalize("NFKC", content)
        normalized = "".join(
            character
            for character in normalized
            if character in "\n\t" or unicodedata.category(character) not in {"Cc", "Cf"}
        )
        lines = (" ".join(line.split()) for line in normalized.splitlines())
        return "\n".join(line for line in lines if line).strip()

    def evaluate_input(
        self,
        content: str,
        *,
        company_references: Iterable[Mapping[str, object]] = (),
    ) -> ConversationPolicyDecision | None:
        normalized = self.normalize_input(content)
        searchable = _searchable(normalized)
        reference_text = _searchable(
            " ".join(
                str(value)
                for reference in company_references
                for value in reference.values()
                if value is not None
            )
        )
        security_text = f"{searchable} {reference_text}".strip()

        checks = (
            (
                ConversationPolicyReason.PROMPT_INJECTION,
                _matches_any(security_text, _PROMPT_INJECTION_PATTERNS),
            ),
            (
                ConversationPolicyReason.SECRET_EXTRACTION,
                _matches_any(security_text, _SECRET_EXTRACTION_PATTERNS),
            ),
            (
                ConversationPolicyReason.PERMISSION_ESCALATION,
                _matches_any(security_text, _PERMISSION_ESCALATION_PATTERNS),
            ),
            (
                ConversationPolicyReason.CODE_GENERATION,
                _matches_any(searchable, _CODE_REQUEST_PATTERNS),
            ),
            (
                ConversationPolicyReason.INVESTMENT_ADVICE,
                _matches_any(searchable, _INVESTMENT_ADVICE_PATTERNS),
            ),
            (
                ConversationPolicyReason.OUT_OF_SCOPE,
                _matches_any(searchable, _OUT_OF_SCOPE_PATTERNS),
            ),
        )
        for reason, matched in checks:
            if matched:
                return self.fixed_decision(
                    scope=ConversationScope.OUT_OF_SCOPE,
                    reason=reason,
                    flags=(reason,),
                )

        if _matches_any(searchable, _GREETING_PATTERNS):
            return self.fixed_decision(scope=ConversationScope.GREETING)
        if _matches_any(searchable, _PRODUCT_HELP_PATTERNS):
            return self.fixed_decision(scope=ConversationScope.PRODUCT_HELP)
        return None

    def validate_agent_decision(
        self,
        *,
        mode: str,
        scope: ConversationScope | str,
        reason: ConversationPolicyReason | str,
        flags: Iterable[ConversationPolicyReason | str] = (),
    ) -> ConversationPolicyDecision:
        resolved_scope = ConversationScope(scope)
        resolved_reason = ConversationPolicyReason(reason)
        resolved_flags = tuple(ConversationPolicyReason(flag) for flag in flags)

        if resolved_reason != ConversationPolicyReason.ALLOWED:
            return self.fixed_decision(
                scope=ConversationScope.OUT_OF_SCOPE,
                reason=resolved_reason,
                flags=resolved_flags or (resolved_reason,),
            )
        if resolved_scope in {ConversationScope.GREETING, ConversationScope.PRODUCT_HELP}:
            return self.fixed_decision(scope=resolved_scope)
        if mode == "research" and resolved_scope == ConversationScope.FINANCIAL_RESEARCH:
            return self.fixed_decision(scope=resolved_scope)
        if mode == "direct_answer" and resolved_scope == ConversationScope.FINANCIAL_EDUCATION:
            return self.fixed_decision(scope=resolved_scope)
        if mode == "clarification" and resolved_scope in {
            ConversationScope.FINANCIAL_RESEARCH,
            ConversationScope.FINANCIAL_EDUCATION,
        }:
            return ConversationPolicyDecision(
                scope=resolved_scope,
                reason=ConversationPolicyReason.ALLOWED,
                safe_response=(
                    "Please clarify the company or financial research question so I can use "
                    "the correct evidence."
                ),
            )
        if mode == "refusal" or resolved_scope == ConversationScope.OUT_OF_SCOPE:
            return self.fixed_decision(
                scope=ConversationScope.OUT_OF_SCOPE,
                reason=ConversationPolicyReason.OUT_OF_SCOPE,
            )
        raise ValueError("agent decision violates conversation scope policy")

    def validate_output(
        self,
        content: str,
        *,
        sensitive_values: Iterable[str] = (),
    ) -> ConversationPolicyDecision | None:
        searchable = _searchable(content)
        if any(value and value in content for value in sensitive_values):
            return self.fixed_decision(
                scope=ConversationScope.OUT_OF_SCOPE,
                reason=ConversationPolicyReason.SECRET_EXTRACTION,
            )
        checks = (
            (ConversationPolicyReason.CODE_GENERATION, _OUTPUT_CODE_PATTERNS),
            (ConversationPolicyReason.PROMPT_INJECTION, _OUTPUT_PROMPT_PATTERNS),
            (ConversationPolicyReason.SECRET_EXTRACTION, _OUTPUT_SECRET_PATTERNS),
            (ConversationPolicyReason.INVESTMENT_ADVICE, _OUTPUT_ADVICE_PATTERNS),
            (ConversationPolicyReason.OUT_OF_SCOPE, _OUTPUT_OFF_TOPIC_PATTERNS),
        )
        for reason, patterns in checks:
            if _matches_any(searchable, patterns):
                return self.fixed_decision(
                    scope=ConversationScope.OUT_OF_SCOPE,
                    reason=reason,
                    flags=(reason,),
                )
        return None

    def fixed_decision(
        self,
        *,
        scope: ConversationScope,
        reason: ConversationPolicyReason = ConversationPolicyReason.ALLOWED,
        flags: tuple[ConversationPolicyReason, ...] = (),
    ) -> ConversationPolicyDecision:
        return ConversationPolicyDecision(
            scope=scope,
            reason=reason,
            safe_response=_safe_response(scope, reason),
            flags=flags,
        )


def build_untrusted_user_payload(
    *,
    content: str,
    company_references: Iterable[Mapping[str, object]],
) -> str:
    return json.dumps(
        {
            "trust_boundary": "untrusted_user_input",
            "instruction_authority": "none",
            "request": content,
            "resolved_company_references": [dict(reference) for reference in company_references],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_response(
    scope: ConversationScope,
    reason: ConversationPolicyReason,
) -> str:
    if scope == ConversationScope.GREETING:
        return (
            "Hello. Ask about a company, stock, filing, financial statement, or financial concept."
        )
    if scope == ConversationScope.PRODUCT_HELP:
        return (
            "Use @company to select a company, then ask about its financial statements, "
            "filings, stock performance, risks, or evidence."
        )
    if reason == ConversationPolicyReason.INVESTMENT_ADVICE:
        return (
            "I can provide source-backed financial research and scenarios, but not personalized "
            "buy, sell, or hold recommendations."
        )
    if reason in {
        ConversationPolicyReason.PROMPT_INJECTION,
        ConversationPolicyReason.SECRET_EXTRACTION,
        ConversationPolicyReason.PERMISSION_ESCALATION,
    }:
        return (
            "I cannot change system instructions, expand permissions, or reveal protected "
            "configuration. I can help with financial research."
        )
    return (
        "I can only help with financial research, company analysis, markets, filings, "
        "accounting, and application guidance."
    )


def _searchable(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        character if unicodedata.category(character) not in {"Cc", "Cf"} else " "
        for character in normalized
    )
    return " ".join(normalized.split())


def _matches_any(value: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(value) is not None for pattern in patterns)


def _patterns(*values: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(value, re.IGNORECASE) for value in values)


_PROMPT_INJECTION_PATTERNS = _patterns(
    r"\bignore\s*(all\s+)?(?:(previous|prior)(\s+(system|developer))?|system|developer)"
    r"\s+instructions?\b",
    r"\b(disregard|override|bypass)\s+(the\s+)?(system|developer|safety|policy)\b",
    r"\b(ignorer|tilsidesaet|tilsidesæt)\s+.*\b(instruktioner|systemprompt)\b",
    r"\b(reveal|show|print|repeat|leak)\s+.*\b(system|developer)\s+prompt\b",
    r"\b(vis|udskriv|gentag|afslor|afslør)\s+.*\b(systemprompt|instruktioner)\b",
    r"\byou\s+are\s+now\s+(in\s+)?(developer|admin|unrestricted)\s+mode\b",
)

_SECRET_EXTRACTION_PATTERNS = _patterns(
    r"\b(reveal|show|print|give|list|extract)\s+.*"
    r"\b(api[ _-]?key|password|secret|credential|\.env)\b",
    r"\b(vis|udskriv|giv|list|udtraek|udtræk)\s+.*\b(api[ _-]?key|kodeord|hemmelighed|\.env)\b",
    r"\bwhat\s+is\s+your\s+(api[ _-]?key|password|secret|credential)\b",
)

_PERMISSION_ESCALATION_PATTERNS = _patterns(
    r"\b(call|enable|invoke|select|switch)\s+.*\b(tool|provider|agent|shell|browser)\b",
    r"\b(use|read|open)\s+.*\b(arbitrary|local|system)\s+(path|file|directory)\b",
    r"\b(kald|aktiver|vaelg|vælg|skift)\s+.*\b(tool|provider|agent|shell|browser)\b",
)

_CODE_REQUEST_PATTERNS = _patterns(
    r"\b(make|write|create|generate|build|produce|code)\s+(me\s+)?(a\s+)?"
    r"(python|javascript|typescript|java|c#|c\+\+|sql|powershell|bash|shell|code|script|program)\b",
    r"\b(lav|skriv|generer|byg|kod)\s+(mig\s+)?(et\s+|en\s+)?"
    r"(python|javascript|typescript|java|c#|c\+\+|sql|powershell|bash|shell|kode|script|program)\b",
    r"\bimplement\s+.*\b(function|class|script|program)\b",
)

_INVESTMENT_ADVICE_PATTERNS = _patterns(
    r"\bshould\s+i\s+(buy|sell|hold)\b",
    r"\b(is|would)\s+.*\b(good|bad)\s+time\s+to\s+(buy|sell)\b",
    r"\b(recommend|advise)\s+.*\b(buying|selling|holding)\b",
    r"\bskal\s+jeg\s+(kobe|købe|saelge|sælge|holde)\b",
    r"\ber\s+det\s+.*\b(godt|darligt|dårligt)\s+tidspunkt\s+at\s+(kobe|købe|saelge|sælge)\b",
)

_OUT_OF_SCOPE_PATTERNS = _patterns(
    r"\b(tell|write|make|create)\s+.*\b(joke|poem|story|song)\b",
    r"\b(fortael|fortæl|skriv|lav)\s+.*\b(joke|vittighed|digt|historie|sang)\b",
)

_GREETING_PATTERNS = _patterns(
    r"^(hi|hello|hey|hej|hejsa|godmorgen|goddag|godaften)"
    r"(\s+from\s+[\w -]{1,40})?[!,. ]*$",
)

_PRODUCT_HELP_PATTERNS = _patterns(
    r"^(help|hjaelp|hjælp)[!?. ]*$",
    r"\b(what\s+can\s+you\s+do|how\s+do\s+i\s+use\s+(this|the)\s+(app|application))\b",
    r"\b(hvad\s+kan\s+du|hvordan\s+bruger\s+jeg\s+(appen|programmet))\b",
)

_OUTPUT_CODE_PATTERNS = _patterns(
    r"```|~~~",
    r"(?m)^\s*(def|class|import|from\s+\w+\s+import|function)\s+",
    r"\b(console\.log|print|subprocess\.(run|popen)|os\.system)\s*\(",
    r"<script\b|javascript:",
)

_OUTPUT_PROMPT_PATTERNS = _patterns(
    r"\b(system|developer)\s+prompt\b",
    r"\b(my|the)\s+(hidden|internal)\s+instructions?\b",
)

_OUTPUT_SECRET_PATTERNS = _patterns(
    r"\b(api[ _-]?key|token|secret|password)\s*[:=]\s*\S+",
    r"\bbearer\s+[a-z0-9._~-]{12,}\b",
    r"\bsk-[a-z0-9_-]{12,}\b",
)

_OUTPUT_ADVICE_PATTERNS = _patterns(
    r"\byou\s+should\s+(buy|sell|hold)\b",
    r"\bi\s+(recommend|advise)\s+(buying|selling|holding)\b",
    r"\bthis\s+is\s+a\s+(buy|sell|hold)\b",
)

_OUTPUT_OFF_TOPIC_PATTERNS = _patterns(
    r"\bhere(?:'s| is)\s+(a\s+)?(joke|poem|story|song)\b",
)
