"""Typed in-memory model of the MusicXML subset that LMX encodes.

A :class:`Score` holds parts, a part holds measures, and a measure holds its
elements in document order. Durations are :class:`~fractions.Fraction` values
in quarter notes: the MusicXML ``<divisions>`` unit is resolved when reading
and recomputed when writing. Layout, lyrics, text and playback-only data are
not modelled.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction


@dataclass(slots=True)
class Clef:
    """A clef; ``sign`` is ``None`` when the MusicXML ``<sign>`` is missing."""

    sign: str | None
    line: int | None = None
    number: int = 1


@dataclass(slots=True)
class Attributes:
    """Key signature (in fifths), time signature, staff count and clefs."""

    key: int | None = None
    time: tuple[int, int] | None = None
    staves: int | None = None
    clefs: list[Clef] = field(default_factory=list)


@dataclass(slots=True)
class Note:
    """A note, rest or unpitched note.

    ``accidental`` is the *printed* accidental while ``alter`` is the sounding
    alteration in semitones. Grace notes have a zero duration. ``ornaments``
    lists the ``<ornaments>`` children in order, where ``"tremolo"`` stands
    for the ``(type, marks)`` pair stored in ``tremolo``.
    """

    duration: Fraction = Fraction(0)
    type_: str | None = None
    dots: int = 0
    step: str | None = None
    octave: int | None = None
    alter: float = 0
    accidental: str | None = None
    rest: bool = False
    measure_rest: bool = False
    unpitched: bool = False
    chord: bool = False
    grace: bool = False
    grace_slash: bool = False
    print_object: bool = True
    voice: int | None = None
    staff: int | None = None
    stem: str | None = None
    time_modification: tuple[int, int] | None = None
    ties: tuple[str, ...] = ()
    beams: tuple[str, ...] = ()
    slurs: tuple[str, ...] = ()
    tuplets: tuple[str, ...] = ()
    articulations: tuple[str, ...] = ()
    ornaments: tuple[str, ...] = ()
    tremolo: tuple[str, str] | None = None
    fermata: bool = False
    arpeggiate: bool = False
    glissandos: tuple[str, ...] = ()

    @property
    def is_pitched(self) -> bool:
        """Whether the note has a pitch (it is neither a rest nor unpitched)."""
        return not (self.rest or self.unpitched or self.step is None)


@dataclass(slots=True)
class Backup:
    """Move the measure cursor back in time (to write another voice)."""

    duration: Fraction


@dataclass(slots=True)
class Forward:
    """Move the measure cursor forward in time (an invisible rest)."""

    duration: Fraction


@dataclass(slots=True)
class Direction:
    """A dynamic, wedge, pedal or tempo marking.

    ``value`` is the dynamic name (``"mf"``), the wedge or pedal type
    (``"crescendo"``, ``"start"``) or the tempo in BPM (``"120"``).
    """

    kind: str
    value: str


@dataclass(slots=True)
class Harmony:
    """A chord symbol: root, MusicXML ``<kind>`` and optional bass."""

    root: str
    root_alter: int = 0
    kind: str = "major"
    bass: str | None = None
    bass_alter: int = 0


@dataclass(slots=True)
class Barline:
    """A barline with an optional repeat direction and bar style."""

    location: str = "right"
    repeat: str | None = None
    style: str | None = None


Element = Attributes | Note | Backup | Forward | Direction | Harmony | Barline


@dataclass(slots=True)
class Measure:
    """The elements of one measure, in document order."""

    elements: list[Element] = field(default_factory=list)


@dataclass(slots=True)
class Part:
    """A part: its name, MIDI instrument and measures."""

    id: str
    name: str = ""
    midi_program: int | None = None
    midi_channel: int | None = None
    measures: list[Measure] = field(default_factory=list)


@dataclass(slots=True)
class Score:
    """A score: a list of parts."""

    parts: list[Part] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Alterations
# --------------------------------------------------------------------------- #

# Semitones of the accidentals the alteration logic understands.
ACCIDENTAL_ALTERS: dict[str, int] = {
    "natural": 0,
    "sharp": 1,
    "flat": -1,
    "double-sharp": 2,
    "flat-flat": -2,
    "natural-sharp": 1,
    "natural-flat": -1,
}
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"

# A pitched note's place on the staff: (staff, step, octave).
PitchKey = tuple[int, str, int | None]


def key_alter(fifths: int, step: str) -> int:
    """Return the alteration (-1, 0 or 1) a key signature gives to a step."""
    if fifths > 0 and step in _SHARP_ORDER[:fifths]:
        return 1
    if fifths < 0 and step in _FLAT_ORDER[:-fifths]:
        return -1
    return 0


def pitch_key(note: Note) -> PitchKey | None:
    """Return the staff position of a pitched note, ``None`` for other notes."""
    if not note.is_pitched:
        return None
    assert note.step is not None
    return (note.staff or 1, note.step, note.octave)


def measure_onsets(measure: Measure) -> list[Fraction]:
    """Return the onset (in quarters from the measure start) of each element."""
    onsets: list[Fraction] = []
    cursor = last_onset = Fraction(0)
    for element in measure.elements:
        onset = cursor
        if isinstance(element, Note):
            if element.chord:
                onset = last_onset
            else:
                last_onset = cursor
                cursor += element.duration
        elif isinstance(element, Backup):
            cursor -= element.duration
        elif isinstance(element, Forward):
            cursor += element.duration
        onsets.append(onset)
    return onsets


def _pitched_notes(measure: Measure) -> Iterator[tuple[Note, PitchKey, Fraction]]:
    for element, onset in zip(measure.elements, measure_onsets(measure), strict=True):
        if isinstance(element, Note) and (key := pitch_key(element)) is not None:
            yield element, key, onset


def resolve_alters(part: Part) -> None:
    """Set the sounding ``alter`` of every note from its notated context.

    A note takes the alteration of the latest accidental printed on its staff
    position earlier in the measure, else the one of the key signature. A note
    ending a tie takes the alteration of the note starting it.
    """
    fifths = 0
    tie_starts: dict[PitchKey, Note] = {}
    for measure in part.measures:
        printed: dict[PitchKey, list[tuple[Fraction, int]]] = {}
        for note, key, onset in _pitched_notes(measure):
            if note.accidental in ACCIDENTAL_ALTERS:
                alter = ACCIDENTAL_ALTERS[note.accidental]
                printed.setdefault(key, []).append((onset, alter))
        for accidentals in printed.values():
            accidentals.sort(key=lambda accidental: accidental[0])
        previous_tie_starts, tie_starts = tie_starts, {}
        onsets = measure_onsets(measure)
        for element, onset in zip(measure.elements, onsets, strict=True):
            if isinstance(element, Attributes) and element.key is not None:
                fifths = element.key
            if not isinstance(element, Note) or (place := pitch_key(element)) is None:
                continue
            carried = [alter for at, alter in printed.get(place, []) if at <= onset]
            element.alter = carried[-1] if carried else key_alter(fifths, place[1])
            if "stop" in element.ties:
                origin = tie_starts if place in tie_starts else previous_tie_starts
                tied_from = origin.pop(place, None)
                if tied_from is not None:
                    element.alter = tied_from.alter
            if "start" in element.ties:
                tie_starts[place] = element


def assign_accidentals(part: Part) -> None:
    """Set the printed ``accidental`` of every note from its ``alter``.

    The inverse of :func:`resolve_alters`: an accidental is printed only where
    the alteration is not implied by the key signature, an earlier accidental
    of the measure or a tie.
    """
    alter_to_accidental = {
        alter: name for name, alter in reversed(ACCIDENTAL_ALTERS.items())
    }
    fifths = 0
    tie_alters: dict[PitchKey, float] = {}
    for measure in part.measures:
        for element in measure.elements:
            if isinstance(element, Attributes) and element.key is not None:
                fifths = element.key
        carried: dict[PitchKey, float] = {}
        previous_tie_alters, tie_alters = tie_alters, {}
        for note, key, _ in sorted(_pitched_notes(measure), key=lambda x: x[2]):
            expected: float = carried.get(key, key_alter(fifths, key[1]))
            if "stop" in note.ties:
                expected = tie_alters.pop(key, previous_tie_alters.pop(key, expected))
            if note.alter == expected:
                note.accidental = None
            else:
                note.accidental = alter_to_accidental.get(round(note.alter))
                carried[key] = note.alter
            if "start" in note.ties:
                tie_alters[key] = note.alter
