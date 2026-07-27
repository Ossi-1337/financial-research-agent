from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

LINK_PATTERN = re.compile(r"!?\[[^\]]*]\(([^)\n]+)\)")
WINDOWS_PATH_PATTERN = re.compile(r"(?i)(?:^|[\s(`])(?:[a-z]:\\|\\\\)")
PLACEHOLDER_PATTERN = re.compile(r"(?i)\b(?:TODO|TBD|FIXME|CHANGEME|REPLACE[_ -]?ME)\b|<[^>\n]+>")
SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
SENSITIVE_ENV_NAMES = (
    "FRA_ALPHA_VANTAGE_API_KEY",
    "FRA_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "FRA_INTEROP_API_KEY",
    "FRA_SEC_USER_AGENT",
)


def public_markdown_files(root: Path) -> tuple[Path, ...]:
    return (
        root / "README.md",
        *sorted((root / "docs").rglob("*.md")),
    )


def validate_repository(root: Path) -> tuple[str, ...]:
    root = root.resolve()
    errors: list[str] = []
    secret_values = _configured_secret_values(root)
    for document in public_markdown_files(root):
        if not document.is_file():
            errors.append(f"{document.relative_to(root)}: missing public document")
            continue
        text = document.read_text(encoding="utf-8")
        relative = document.relative_to(root)
        lowered = text.casefold()
        if ".codex/" in lowered or ".codex\\" in lowered:
            errors.append(f"{relative}: private .codex path is not allowed")
        if "file://" in lowered:
            errors.append(f"{relative}: file:// link is not allowed")
        if WINDOWS_PATH_PATTERN.search(text):
            errors.append(f"{relative}: absolute Windows path is not allowed")
        if PLACEHOLDER_PATTERN.search(text):
            errors.append(f"{relative}: unresolved placeholder is not allowed")
        if SECRET_PATTERN.search(text):
            errors.append(f"{relative}: possible API secret is not allowed")
        for value in secret_values:
            if value in text:
                errors.append(f"{relative}: configured secret value is not allowed")
        errors.extend(_link_errors(root, document, text))
    return tuple(dict.fromkeys(errors))


def _link_errors(root: Path, document: Path, text: str) -> tuple[str, ...]:
    errors: list[str] = []
    for match in LINK_PATTERN.finditer(text):
        raw_target = match.group(1).strip()
        target = _target_without_title(raw_target)
        lowered = target.casefold()
        if lowered.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if lowered.startswith("file://") or ".codex/" in lowered or ".codex\\" in lowered:
            continue
        path_text = unquote(urlsplit(target).path)
        if path_text == "":
            continue
        candidate = (document.parent / path_text).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(f"{document.relative_to(root)}: link escapes repository: {target}")
            continue
        if not candidate.exists():
            errors.append(f"{document.relative_to(root)}: broken link: {target}")
    return tuple(errors)


def _target_without_title(raw_target: str) -> str:
    if raw_target.startswith("<") and ">" in raw_target:
        return raw_target[1 : raw_target.index(">")]
    return raw_target.split(maxsplit=1)[0]


def _configured_secret_values(root: Path) -> tuple[str, ...]:
    values = {value for name in SENSITIVE_ENV_NAMES if (value := os.environ.get(name, "").strip())}
    env_path = root / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", maxsplit=1)
            if name.strip() in SENSITIVE_ENV_NAMES and (cleaned := value.strip().strip("'\"")):
                values.add(cleaned)
    return tuple(value for value in values if len(value) >= 8)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = validate_repository(root)
    if errors:
        for error in errors:
            print(f"docs error: {error}", file=sys.stderr)
        return 1
    print(f"Documentation check passed ({len(public_markdown_files(root))} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
