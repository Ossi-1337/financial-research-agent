from __future__ import annotations

import struct
from pathlib import Path

from scripts.check_docs import validate_repository

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ASSETS = {
    "novo-report-desktop.jpg": (1440, 900),
    "novo-evidence-desktop.jpg": (1440, 900),
    "novo-local-qa-mobile.jpg": (390, 844),
}


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
    assert "financial_research_agent a2a-serve" in readme
    assert "docker compose --profile a2a up --build" in readme
    assert "[LLM providers](docs/providers.md)" in readme


def test_demo_assets_are_real_browser_images_at_documented_viewports() -> None:
    asset_root = PROJECT_ROOT / "docs" / "assets" / "demo"
    for filename, expected_size in DEMO_ASSETS.items():
        path = asset_root / filename
        assert path.stat().st_size > 10_000
        assert _jpeg_dimensions(path) == expected_size


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    with path.open("rb") as stream:
        assert stream.read(2) == b"\xff\xd8"
        while marker_prefix := stream.read(1):
            if marker_prefix != b"\xff":
                continue
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            marker_value = marker[0]
            if marker_value in start_of_frame_markers:
                stream.read(3)
                height, width = struct.unpack(">HH", stream.read(4))
                return width, height
            if marker_value in {0xD8, 0xD9}:
                continue
            segment_length = struct.unpack(">H", stream.read(2))[0]
            stream.seek(segment_length - 2, 1)
    raise AssertionError(f"JPEG dimensions not found in {path.name}")
