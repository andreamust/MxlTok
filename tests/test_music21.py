"""Semantic round trip checked with music21 (skipped when not installed)."""

from collections import Counter
from pathlib import Path

import pytest

music21 = pytest.importorskip("music21")

from mxltok import MXL, TokenizerConfig  # noqa: E402
from mxltok.musicxml import dump, load  # noqa: E402
from tests.conftest import FIXTURES  # noqa: E402

SMALL = [p for p in FIXTURES if p.stat().st_size < 60_000]


def _content(path: Path) -> list[Counter]:
    """Per part: multiset of (pitches or rest, quarter length) of every event."""
    score = music21.converter.parse(str(path))
    parts = []
    for part in score.parts:
        events: Counter = Counter()
        for element in part.recurse().notesAndRests:
            if element.isChord:
                key = tuple(sorted(p.midi for p in element.pitches))
            elif element.isNote:
                key = (element.pitch.midi,)
            else:
                key = "rest"
            events[(key, element.quarterLength)] += 1
        parts.append(events)
    return parts


@pytest.mark.parametrize("arrangement", ["part_first", "measure_first"])
def test_music21_semantic_roundtrip(
    tmp_path: Path, arrangement: str, fixture_path: Path
):
    if fixture_path not in SMALL:
        pytest.skip("slow fixture")
    tokenizer = MXL(TokenizerConfig(arrangement=arrangement))
    score = load(fixture_path)
    original = tmp_path / "original.musicxml"
    dump(score, original)
    decoded = tmp_path / "decoded.musicxml"
    tokenizer.decode(
        tokenizer.encode(score, strict=True), output_path=decoded, strict=True
    )
    original_parts, decoded_parts = _content(original), _content(decoded)
    assert len(decoded_parts) == len(original_parts)
    for before, after in zip(original_parts, decoded_parts, strict=True):
        assert after == before
