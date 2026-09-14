"""Transposition of scores, for pitch-shift data augmentation."""

from __future__ import annotations

from copy import deepcopy

from mxltok.score import Attributes, Harmony, Note, Score, assign_accidentals

_STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_SPELLINGS = (
    ("C", 0),
    ("C", 1),
    ("D", 0),
    ("D", 1),
    ("E", 0),
    ("F", 0),
    ("F", 1),
    ("G", 0),
    ("G", 1),
    ("A", 0),
    ("A", 1),
    ("B", 0),
)
_FLAT_SPELLINGS = (
    ("C", 0),
    ("D", -1),
    ("D", 0),
    ("E", -1),
    ("E", 0),
    ("F", 0),
    ("G", -1),
    ("G", 0),
    ("A", -1),
    ("A", 0),
    ("B", -1),
    ("B", 0),
)
# Transposing by n semitones moves a key by this many fifths.
_SEMITONES_TO_FIFTHS = (0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5)


def transpose_key(fifths: int, semitones: int) -> int:
    """Return a key signature (in fifths, within -7..7) after a transposition."""
    fifths += _SEMITONES_TO_FIFTHS[semitones % 12]
    while fifths > 7:
        fifths -= 12
    while fifths < -7:
        fifths += 12
    return fifths


def transpose(score: Score, semitones: int) -> Score:
    """Return a copy of a score transposed by ``semitones``.

    Pitches, key signatures and chord symbols are moved. Notes are respelled
    with sharps or flats after the key they are in, octaves are clamped to
    0-9, and printed accidentals are recomputed to match the new keys.
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
                elif isinstance(element, Note) and element.is_pitched:
                    assert element.step is not None and element.octave is not None
                    semitone = (
                        12 * element.octave
                        + _STEP_SEMITONES[element.step]
                        + round(element.alter)
                        + semitones
                    )
                    element.step, element.alter = _spell(semitone, use_flats)
                    element.octave = max(0, min(9, semitone // 12))
                elif isinstance(element, Harmony):
                    element.root, element.root_alter = _move(
                        element.root, element.root_alter, semitones, use_flats
                    )
                    if element.bass is not None:
                        element.bass, element.bass_alter = _move(
                            element.bass, element.bass_alter, semitones, use_flats
                        )
        assign_accidentals(part)
    return score


def _spell(semitone: int, use_flats: bool) -> tuple[str, int]:
    return (_FLAT_SPELLINGS if use_flats else _SHARP_SPELLINGS)[semitone % 12]


def _move(step: str, alter: int, semitones: int, use_flats: bool) -> tuple[str, int]:
    return _spell(_STEP_SEMITONES.get(step, 0) + alter + semitones, use_flats)
