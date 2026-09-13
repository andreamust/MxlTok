"""Shared fixtures: the MusicXML corpus under ``tests/fixtures/mxl``."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "mxl"
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
FIXTURES = sorted(
    p for p in FIXTURES_DIR.iterdir() if p.suffix in {".mxl", ".musicxml", ".xml"}
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite golden token files instead of comparing against them.",
    )


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-golden"))


@pytest.fixture(params=FIXTURES, ids=lambda p: p.stem[:40])
def fixture_path(request: pytest.FixtureRequest) -> Path:
    return request.param
