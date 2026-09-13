"""Typed in-memory representation of a MusicXML score.

Durations are :class:`fractions.Fraction` values in quarter notes; the MusicXML
``<divisions>`` unit is resolved on load and recomputed on write. The model is a
faithful *subset* of MusicXML: layout, lyrics, text and playback-only data are
dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

DirectionKind = Literal["dynamics", "wedge", "pedal", "tempo"]


@dataclass(slots=True)
class Clef:
    sign: str
    line: int | None = None
    number: int = 1


@dataclass(slots=True)
class Attributes:
    key: int | None = None
    time: tuple[int, int] | None = None
    staves: int | None = None
    clefs: list[Clef] = field(default_factory=list)


@dataclass(slots=True)
class Note:
    """A note, rest or unpitched note. Grace notes have a zero duration."""

    duration: Fraction = Fraction(0)
    type_: str | None = None
    step: str | None = None
    octave: int | None = None
    alter: float = 0.0
    accidental: str | None = None
    rest: bool = False
    rest_measure: bool = False
    chord: bool = False
    unpitched: bool = False
    grace: bool = False
    grace_slash: bool = False
    print_object: bool = True
    voice: int | None = None
    staff: int | None = None
    dots: int = 0
    stem: str | None = None
    time_mod: tuple[int, int] | None = None
    ties: tuple[str, ...] = ()
    slurs: tuple[str, ...] = ()
    tuplets: tuple[str, ...] = ()
    beams: tuple[str, ...] = ()
    articulations: tuple[str, ...] = ()
    ornaments: tuple[str, ...] = ()
    notations: tuple[str, ...] = ()


@dataclass(slots=True)
class Backup:
    duration: Fraction


@dataclass(slots=True)
class Forward:
    duration: Fraction


@dataclass(slots=True)
class Direction:
    kind: DirectionKind
    value: str


@dataclass(slots=True)
class Harmony:
    root: str
    root_alter: int = 0
    kind: str = "major"
    bass: str | None = None
    bass_alter: int = 0


@dataclass(slots=True)
class Barline:
    location: str = "right"
    repeat: str | None = None
    style: str | None = None


Element = Note | Backup | Forward | Attributes | Direction | Harmony | Barline


@dataclass(slots=True)
class Measure:
    elements: list[Element] = field(default_factory=list)


@dataclass(slots=True)
class Part:
    id: str
    name: str = ""
    midi_program: int | None = None
    midi_channel: int | None = None
    measures: list[Measure] = field(default_factory=list)


@dataclass(slots=True)
class Score:
    parts: list[Part] = field(default_factory=list)


PitchKey = tuple[int, str, int]  # (staff, step, octave)
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"


def key_alter(fifths: int, step: str) -> int:
    """Alteration implied by a key signature for a step (``-1``, ``0`` or ``1``)."""
    if fifths > 0 and step in _SHARP_ORDER[:fifths]:
        return 1
    if fifths < 0 and step in _FLAT_ORDER[:-fifths]:
        return -1
    return 0


def pitch_key(note: Note) -> PitchKey | None:
    """Identity of a pitched note's staff line, or ``None`` for rests/unpitched."""
    if note.rest or note.unpitched or note.step is None or note.octave is None:
        return None
    return (note.staff or 1, note.step, note.octave)


def measure_onsets(measure: Measure) -> list[Fraction]:
    """Onset (quarters from the measure start) of every element of ``measure``."""
    onsets: list[Fraction] = []
    cursor = Fraction(0)
    last_note_onset = Fraction(0)
    for element in measure.elements:
        onset = cursor
        match element:
            case Note(chord=True):
                onset = last_note_onset
            case Note():
                last_note_onset = cursor
                cursor += element.duration
            case Backup():
                cursor -= element.duration
            case Forward():
                cursor += element.duration
        onsets.append(onset)
    return onsets


class AlterTracker:
    """Resolve the alteration of notes from key, in-measure accidentals and ties.

    Printed accidentals are :meth:`record`-ed per measure; :meth:`expected` then
    gives the alteration a note carries when it prints no accidental itself.
    """

    def __init__(self) -> None:
        self.key = 0
        self._carried: dict[PitchKey, list[tuple[Fraction, float]]] = {}
        self._ties: dict[PitchKey, float] = {}
        self._previous_ties: dict[PitchKey, float] = {}

    def start_measure(self) -> None:
        self._carried = {}
        self._previous_ties, self._ties = self._ties, {}

    def record(self, key: PitchKey, onset: Fraction, alter: float) -> None:
        self._carried.setdefault(key, []).append((onset, alter))

    def tie_start(self, key: PitchKey, alter: float) -> None:
        self._ties[key] = alter

    def expected(self, key: PitchKey, onset: Fraction, tie_stop: bool) -> float:
        if tie_stop:
            for ties in (self._ties, self._previous_ties):
                if key in ties:
                    return ties.pop(key)
        best: tuple[Fraction, float] | None = None
        for carried in self._carried.get(key, ()):
            if carried[0] <= onset and (best is None or carried[0] >= best[0]):
                best = carried
        if best is not None:
            return best[1]
        return key_alter(self.key, key[1])


def resolve_alters(part: Part, accidental_alters: Mapping[str, int]) -> None:
    """Set ``Note.alter`` of every note from key signatures, accidentals and ties."""
    tracker = AlterTracker()
    for measure in part.measures:
        tracker.start_measure()
        onsets = measure_onsets(measure)
        for element, onset in zip(measure.elements, onsets, strict=True):
            if isinstance(element, Attributes) and element.key is not None:
                tracker.key = element.key
            elif isinstance(element, Note) and (key := pitch_key(element)):
                alter = accidental_alters.get(element.accidental or "")
                if alter is not None:
                    tracker.record(key, onset, alter)
        for element, onset in zip(measure.elements, onsets, strict=True):
            if isinstance(element, Note) and (key := pitch_key(element)):
                printed = accidental_alters.get(element.accidental or "")
                if printed is not None:  # a printed accidental always wins
                    element.alter = printed
                else:
                    element.alter = tracker.expected(key, onset, "stop" in element.ties)
                if "start" in element.ties:
                    tracker.tie_start(key, element.alter)


def assign_accidentals(part: Part, alter_accidentals: Mapping[int, str]) -> None:
    """Set ``Note.accidental`` so that ``resolve_alters`` recovers ``Note.alter``.

    Accidentals are printed only where the alteration is not implied by the
    key signature, an earlier accidental of the measure or a tie.
    """
    tracker = AlterTracker()
    for measure in part.measures:
        tracker.start_measure()
        onsets = measure_onsets(measure)
        for element in measure.elements:
            if isinstance(element, Attributes) and element.key is not None:
                tracker.key = element.key
        # Accidentals carry by onset, not document order (voices are sequential)
        by_onset = sorted(
            zip(measure.elements, onsets, strict=True), key=lambda x: x[1]
        )
        for element, onset in by_onset:
            if isinstance(element, Note) and (key := pitch_key(element)):
                expected = tracker.expected(key, onset, "stop" in element.ties)
                if element.alter == expected:
                    element.accidental = None
                else:
                    element.accidental = alter_accidentals[int(element.alter)]
                    tracker.record(key, onset, element.alter)
                if "start" in element.ties:
                    tracker.tie_start(key, element.alter)
