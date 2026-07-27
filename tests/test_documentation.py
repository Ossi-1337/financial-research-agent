from __future__ import annotations

from pathlib import Path

from scripts.check_docs import validate_repository

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_documentation_has_valid_links_and_no_private_or_secret_content() -> None:
    assert validate_repository(PROJECT_ROOT) == ()


def test_documentation_checker_rejects_private_paths_placeholders_and_broken_links(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "README.md").write_text(
        "See [.codex](.codex/private.md), [missing](docs/missing.md), and TODO.\n",
        encoding="utf-8",
    )

    errors = validate_repository(tmp_path)

    assert any("private .codex path" in error for error in errors)
    assert any("broken link" in error for error in errors)
    assert any("unresolved placeholder" in error for error in errors)


def test_readme_documents_runnable_entrypoints() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "docker compose up --build" in readme
    assert "python scripts/dev.py install" in readme
    assert "python scripts/dev.py run" in readme
    assert "scenario-run novo-nordisk --pretty" in readme
    assert "/scenario novo-nordisk --with-local-qa" in readme
    assert "python scripts/dev.py check" in readme
    assert "docker compose up --build" in readme
    assert "four internal A2A specialist services" in readme
    assert "[Architecture](docs/architecture.md)" in readme
