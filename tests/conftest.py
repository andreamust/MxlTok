"""Shared fixtures.

The reference files under ``fixtures/golden`` come from mxlGPT's tokenizer
(see ``scripts/mxlgpt_parity.py``): MxlTok must reproduce them exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SCORES = sorted((FIXTURES / "mxl").iterdir())
GOLDEN = FIXTURES / "golden"
# Scores that mxlGPT encodes (the others have content outside the vocabulary).
ENCODABLE = [s for s in SCORES if (GOLDEN / f"{s.stem}.part_first.lmx").exists()]


def golden_tokens(name: str) -> list[str]:
    return (GOLDEN / name).read_text(encoding="utf-8").split()


def score_id(path: Path) -> str:
    return path.stem[:30]


@pytest.fixture(params=ENCODABLE, ids=score_id)
def score_path(request: pytest.FixtureRequest) -> Path:
    return request.param  # type: ignore[no-any-return]
