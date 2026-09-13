"""Splitting scores into bar-aligned training chunks."""

from pathlib import Path

import pytest

from mxltok import MXL, split_files_for_training
from mxltok.musicxml import load
from mxltok.score import Attributes, Note
from mxltok.utils import bar_boundaries, get_files_paths, split_score_per_bars
from tests.conftest import FIXTURES

SMALL = [p for p in FIXTURES if p.stat().st_size < 60_000]


def _notes(score) -> int:
    return sum(
        isinstance(e, Note) for p in score.parts for m in p.measures for e in m.elements
    )


def test_bar_boundaries():
    assert bar_boundaries([10] * 10, 35, num_overlap_bars=0) == [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 10),
    ]
    assert bar_boundaries([10] * 10, 35, num_overlap_bars=1) == [
        (0, 3),
        (2, 5),
        (4, 7),
        (6, 9),
        (8, 10),
    ]
    assert bar_boundaries([10] * 10, 35, num_overlap_bars=1, min_seq_len=30) == [
        (0, 3),
        (2, 5),
        (4, 7),
        (6, 9),
    ]
    assert bar_boundaries([50, 5, 5], 20) == [(0, 1), (1, 3)]  # oversized bar cut alone
    assert bar_boundaries([5, 5], 100) == [(0, 2)]
    assert bar_boundaries([10] * 4, 25, overhead=5) == [(0, 2), (1, 3), (2, 4)]
    with pytest.raises(ValueError):
        bar_boundaries([1], 10, num_overlap_bars=-1)


def test_split_score_per_bars_carries_attributes():
    score = load(SMALL[0])
    chunk = split_score_per_bars(score, 3, 5)
    assert [len(p.measures) for p in chunk.parts] == [2] * len(score.parts)
    for part in chunk.parts:
        head = part.measures[0].elements[0]
        assert isinstance(head, Attributes)
        assert head.key is not None and head.time is not None and head.clefs
    assert (
        split_score_per_bars(score, 0, 2).parts[0].measures
        == score.parts[0].measures[:2]
    )


def test_split_files_for_training(tmp_path: Path):
    tok = MXL()
    max_seq_len = 256
    chunks = split_files_for_training(SMALL, tok, tmp_path / "chunks", max_seq_len)
    assert chunks and all(c.suffix == ".musicxml" for c in chunks)
    assert sorted(chunks) == get_files_paths(tmp_path / "chunks")
    for chunk in chunks:
        seq = tok.encode(chunk, strict=True)
        assert len(seq) <= max_seq_len
    total_notes = sum(_notes(load(p)) for p in SMALL)
    # overlapping bars duplicate notes, so chunks carry at least the original notes
    assert sum(_notes(load(c)) for c in chunks) >= total_notes
    no_overlap = split_files_for_training(
        SMALL,
        tok,
        tmp_path / "no_overlap",
        max_seq_len,
        num_overlap_bars=0,
        min_seq_len=0,
    )
    assert sum(_notes(load(c)) for c in no_overlap) == total_notes
    # second call is a no-op returning the existing chunks
    assert sorted(
        split_files_for_training(SMALL, tok, tmp_path / "chunks", max_seq_len)
    ) == sorted(chunks)
    with pytest.raises(ValueError):
        split_files_for_training([], tok, tmp_path, max_seq_len)
