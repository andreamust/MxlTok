"""Helpers: file discovery and the training iterator."""

from __future__ import annotations

import hashlib
import os
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from mxltok.constants import ALTER_TO_ACCIDENTAL, STEP_SEMITONES
from mxltok.musicxml import dump, load
from mxltok.score import Attributes, Harmony, Note, Part, Score, assign_accidentals

if TYPE_CHECKING:
    from mxltok.tokenizer import MusicTokenizer

MUSICXML_EXTENSIONS = frozenset({".mxl", ".musicxml", ".xml"})


def get_files_paths(directory: str | Path) -> list[Path]:
    """All MusicXML files under ``directory`` (recursive), sorted."""
    return sorted(
        p for p in Path(directory).rglob("*") if p.suffix.lower() in MUSICXML_EXTENSIONS
    )


def deepest_common_dir(paths: Sequence[str | Path]) -> Path:
    """Deepest directory containing all ``paths``."""
    return Path(os.path.commonpath([Path(p).resolve().parent for p in paths]))


class TokTrainingIterator:
    """Iterable of byte strings (one per file / bar chunk) to train a model on.

    Files failing to load or encode are skipped.
    """

    def __init__(
        self, tokenizer: MusicTokenizer, files_paths: Sequence[str | Path]
    ) -> None:
        self.tokenizer = tokenizer
        self.files_paths = [Path(p) for p in files_paths]
        self._index = 0

    def load_file(self, path: Path) -> list[str]:
        try:
            seq = self.tokenizer.encode(path, encode_ids=False)
        except Exception:
            return []
        if self.tokenizer.config.encode_ids_split == "bar":
            chunks = [chunk for chunk in seq.split_per_bars() if len(chunk) > 0]
        else:
            chunks = [seq]
        return [self.tokenizer._ids_to_bytes(chunk.ids) for chunk in chunks]

    def __len__(self) -> int:
        return len(self.files_paths)

    def __getitem__(self, index: int) -> list[str]:
        return self.load_file(self.files_paths[index])

    def __iter__(self) -> TokTrainingIterator:
        self._index = 0
        return self

    def __next__(self) -> list[str]:
        if self._index >= len(self):
            raise StopIteration
        self._index += 1
        return self[self._index - 1]


# --------------------------------------------------------------------------- #
# Splitting scores into training chunks
# --------------------------------------------------------------------------- #


def bar_boundaries(
    tokens_per_bar: Sequence[int],
    max_seq_len: int,
    num_overlap_bars: int = 1,
    min_seq_len: int | None = None,
    overhead: int = 0,
) -> list[tuple[int, int]]:
    """Bar ranges ``(start, end)`` whose token counts fit in ``max_seq_len``.

    ``overhead`` counts the tokens every chunk carries regardless of its bars
    (e.g. the ``Part`` tokens). Consecutive chunks share ``num_overlap_bars``
    bars; a final chunk shorter than ``min_seq_len`` (default
    ``max_seq_len // 4``) is dropped.
    """
    if num_overlap_bars < 0:
        raise ValueError("num_overlap_bars must be >= 0")
    if min_seq_len is None:
        min_seq_len = max_seq_len // 4
    budget = max_seq_len - overhead
    ranges: list[tuple[int, int]] = []
    start = 0
    current = 0
    for index, count in enumerate(tokens_per_bar):
        if current + count <= budget:
            current += count
            continue
        if index == start:  # a single bar exceeds the budget: keep it alone
            end, next_start = index + 1, index + 1
        else:  # cut before this bar, overlapping as much as fits
            end, next_start = index, max(start + 1, index - num_overlap_bars)
            while (
                next_start < index
                and sum(tokens_per_bar[next_start : index + 1]) > budget
            ):
                next_start += 1
        ranges.append((start, end))
        start = next_start
        current = sum(tokens_per_bar[start : index + 1])
    if current + overhead >= min_seq_len or not ranges:
        ranges.append((start, len(tokens_per_bar)))
    return ranges


def split_score_per_bars(score: Score, start: int, end: int) -> Score:
    """Measures ``start:end`` of each part, plus the attributes in force at start."""
    parts: list[Part] = []
    for part in score.parts:
        state = Attributes()
        for measure in part.measures[:start]:
            for element in measure.elements:
                if isinstance(element, Attributes):
                    _merge_attributes(state, element)
        measures = deepcopy(part.measures[start:end])
        if measures and start > 0 and state != Attributes():
            first = measures[0].elements
            head = next((e for e in first if isinstance(e, Attributes)), None)
            if head is None:
                head = Attributes()
                first.insert(0, head)
            for element in first:
                if isinstance(element, Attributes):
                    _merge_attributes(state, element)  # explicit ones win
            head.key, head.time, head.staves = state.key, state.time, state.staves
            head.clefs = [
                c
                for c in state.clefs
                if not any(h.number == c.number for h in head.clefs)
            ] + head.clefs
        parts.append(replace(part, measures=measures))
    return Score(parts)


def _max_injected_attribute_tokens(score: Score) -> int:
    """Upper bound of the Key/TimeSig/Clef/Staff tokens a chunk start may add."""
    total = 0
    for part in score.parts:
        clefs = max(
            (
                len(e.clefs)
                for m in part.measures
                for e in m.elements
                if isinstance(e, Attributes)
            ),
            default=1,
        )
        total += 2 + 2 * clefs
    return total


def _merge_attributes(state: Attributes, element: Attributes) -> None:
    if element.key is not None:
        state.key = element.key
    if element.time is not None:
        state.time = element.time
    if element.staves is not None:
        state.staves = element.staves
    for clef in element.clefs:
        state.clefs = [c for c in state.clefs if c.number != clef.number] + [clef]
    state.clefs.sort(key=lambda c: c.number)


def split_files_for_training(
    files_paths: Sequence[str | Path],
    tokenizer: MusicTokenizer,
    save_dir: str | Path,
    max_seq_len: int,
    num_overlap_bars: int = 1,
    min_seq_len: int | None = None,
) -> list[Path]:
    """Split MusicXML files at bars into chunks of at most ``max_seq_len`` tokens.

    Chunks are saved as ``<stem>_<i>.musicxml`` under ``save_dir`` (mirroring
    the source tree). A hidden marker file records a finished split so that
    calling again with the same files returns the existing chunks.
    """
    paths = [Path(p) for p in files_paths]
    if not paths:
        raise ValueError("No music file provided to split for training.")
    save_dir = Path(save_dir)
    digest = hashlib.sha1("\n".join(sorted(map(str, paths))).encode()).hexdigest()[:16]
    marker = save_dir / f".split_{digest}"
    if marker.is_file():
        return get_files_paths(save_dir)
    root_dir = deepest_common_dir(paths)
    new_paths: list[Path] = []
    for path in paths:
        try:
            score = load(path)
            seq = tokenizer.encode(score, encode_ids=False)
        except Exception as exc:
            warnings.warn(f"Skipping {path}: {exc}", stacklevel=2)
            continue
        num_bars = max((len(p.measures) for p in score.parts), default=0)
        tokens_per_bar = [0] * num_bars
        overhead = sum(1 for event in seq.events if event.time < 0)
        for event in seq.events:
            if event.time >= 0:
                tokens_per_bar[event.time] += 1
        overhead += _max_injected_attribute_tokens(score)
        ranges = bar_boundaries(
            tokens_per_bar, max_seq_len, num_overlap_bars, min_seq_len, overhead
        )
        chunks = (
            [score]
            if ranges == [(0, num_bars)]
            else [split_score_per_bars(score, a, b) for a, b in ranges]
        )
        for index, chunk in enumerate(chunks):
            if not any(
                isinstance(e, Note)
                for p in chunk.parts
                for m in p.measures
                for e in m.elements
            ):
                continue
            out_path = save_dir / path.resolve().relative_to(root_dir).with_name(
                f"{path.stem}_{index}.musicxml"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            dump(chunk, out_path)
            new_paths.append(out_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{len(paths)} files split\n")
    return new_paths


# --------------------------------------------------------------------------- #
# Transposition
# --------------------------------------------------------------------------- #

# Semitone shift -> shift on the circle of fifths (mod 12)
_SEMITONES_TO_FIFTHS = {
    0: 0,
    1: -5,
    2: 2,
    3: -3,
    4: 4,
    5: -1,
    6: 6,
    7: 1,
    8: -4,
    9: 3,
    10: -2,
    11: 5,
}
_SHARP_SPELLING = {
    0: ("C", 0),
    1: ("C", 1),
    2: ("D", 0),
    3: ("D", 1),
    4: ("E", 0),
    5: ("F", 0),
    6: ("F", 1),
    7: ("G", 0),
    8: ("G", 1),
    9: ("A", 0),
    10: ("A", 1),
    11: ("B", 0),
}
_FLAT_SPELLING = {
    0: ("C", 0),
    1: ("D", -1),
    2: ("D", 0),
    3: ("E", -1),
    4: ("E", 0),
    5: ("F", 0),
    6: ("G", -1),
    7: ("G", 0),
    8: ("A", -1),
    9: ("A", 0),
    10: ("B", -1),
    11: ("B", 0),
}


def transpose_key(fifths: int, semitones: int) -> int:
    """Key signature (in fifths, kept within -7..7) after a transposition."""
    new_fifths = fifths + _SEMITONES_TO_FIFTHS[semitones % 12]
    while new_fifths > 7:
        new_fifths -= 12
    while new_fifths < -7:
        new_fifths += 12
    return new_fifths


def transpose(score: Score, semitones: int) -> Score:
    """Copy of ``score`` transposed by ``semitones``, respelled after the new keys.

    Pitches, key signatures and chord symbols are shifted; accidentals are
    recomputed so that the notation stays consistent with the new key.
    """
    score = deepcopy(score)
    if semitones == 0:
        return score
    for part in score.parts:
        use_flats = False
        for measure in part.measures:
            for element in measure.elements:
                if isinstance(element, Attributes) and element.key is not None:
                    element.key = transpose_key(element.key, semitones)
                    use_flats = element.key < 0
                elif (
                    isinstance(element, Note)
                    and not element.rest
                    and not element.unpitched
                ):
                    if element.step is None or element.octave is None:
                        continue
                    midi = (
                        STEP_SEMITONES[element.step]
                        + int(element.alter)
                        + 12 * element.octave
                    )
                    target = midi + semitones
                    element.step, element.alter = _spell(target % 12, use_flats)
                    element.octave = max(0, min(9, target // 12))
                elif isinstance(element, Harmony):
                    element.root, element.root_alter = _transpose_step(
                        element.root, element.root_alter, semitones, use_flats
                    )
                    if element.bass is not None:
                        element.bass, element.bass_alter = _transpose_step(
                            element.bass, element.bass_alter, semitones, use_flats
                        )
        assign_accidentals(part, ALTER_TO_ACCIDENTAL)
    return score


def _spell(pitch_class: int, use_flats: bool) -> tuple[str, int]:
    return (_FLAT_SPELLING if use_flats else _SHARP_SPELLING)[pitch_class]


def _transpose_step(
    step: str, alter: int, semitones: int, use_flats: bool
) -> tuple[str, int]:
    return _spell((STEP_SEMITONES.get(step, 0) + alter + semitones) % 12, use_flats)
