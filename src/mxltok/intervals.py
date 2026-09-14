"""Conversions between pitch tokens and interval tokens.

In ``pitch_mode="interval"`` every pitch but the first of a measure is written
as ``interval:<semitones>`` from the previous pitch (``interval:-3``,
``interval:0``, ``interval:7``).
"""

from __future__ import annotations

from mxltok.vocabulary import INTERVAL_PREFIX

_STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SUFFIX_SEMITONES = {"##": 2, "bb": -2, "#": 1, "b": -1, "n": 0}  # longest first
_SEMITONE_SPELLINGS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def split_pitch_token(token: str) -> tuple[str, int, str]:
    """Split a pitch token: ``"D5bb"`` -> ``("D", 5, "bb")``."""
    for suffix in _SUFFIX_SEMITONES:
        if token.endswith(suffix):
            return token[0], int(token[1 : -len(suffix)]), suffix
    return token[0], int(token[1:]), ""


def pitch_to_semitone(token: str) -> int:
    """Return the semitone number of a pitch token (``C0`` is 0, ``C0#`` is 1)."""
    step, octave, suffix = split_pitch_token(token)
    return 12 * octave + _STEP_SEMITONES[step] + _SUFFIX_SEMITONES.get(suffix, 0)


def semitone_to_pitch(semitone: int) -> str:
    """Return the pitch token of a semitone number, spelled with sharps.

    The result is clamped to the range C0-B9.
    """
    octave, pitch_class = divmod(max(0, min(119, semitone)), 12)
    spelling = _SEMITONE_SPELLINGS[pitch_class]
    return f"{spelling[0]}{octave}{spelling[1:]}"


def absolute_to_interval(pitch: str, previous_pitch: str | None) -> str:
    """Return the interval token from ``previous_pitch`` to ``pitch``.

    Without a previous pitch, the pitch token itself is returned.
    """
    if previous_pitch is None:
        return pitch
    semitones = pitch_to_semitone(pitch) - pitch_to_semitone(previous_pitch)
    return f"{INTERVAL_PREFIX}{semitones}"


def interval_to_absolute(interval: str, previous_pitch: str) -> str:
    """Return the pitch token an interval token leads to from ``previous_pitch``."""
    semitones = int(interval.removeprefix(INTERVAL_PREFIX))
    return semitone_to_pitch(pitch_to_semitone(previous_pitch) + semitones)
