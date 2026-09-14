"""Encode one part of a :class:`~mxltok.score.Score` into LMX tokens.

A part becomes a flat token list: each measure starts with ``measure`` and
lists its elements in document order. A note is written as prefix tokens, one
root token (its type) and suffix tokens::

    [print-object:no] [grace [grace:slash]] [chord] <pitch | rest> [voice:N]
    <type | rest:measure>
    [<ratio>] [dot]* [stem:S] [staff:N] [beam:B]* [tied:T]* [tuplet:T]*
    [<marking>]* [slur:T]*

``voice``, ``stem`` and ``staff`` are delta-encoded: they are only written
when they change within the measure. See ``docs/format.md`` for the whole
specification.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fractions import Fraction

from mxltok.config import TokenizerConfig
from mxltok.errors import MxlTokError
from mxltok.intervals import absolute_to_interval
from mxltok.score import (
    Attributes,
    Backup,
    Barline,
    Direction,
    Forward,
    Harmony,
    Measure,
    Note,
    Part,
)
from mxltok.vocabulary import (
    ACCIDENTAL_TO_SUFFIX,
    ALTER_TO_ACCIDENTAL,
    BACKUP,
    DYNAMICS,
    FORWARD,
    HARMONY_ALTERS,
    HARMONY_KINDS,
    MEASURE,
    NOTE_MARKINGS,
    NOTE_TYPE_QUARTERS,
    NOTE_TYPES,
    PEDALS,
    PERCUSSION_CLEF,
    PITCHES,
    REPEAT_BACKWARD,
    REPEAT_FORWARD,
    STEMS,
    TEMPO_GRID,
    WEDGES,
)

_PITCHES = frozenset(PITCHES)
_BEAM_TOKENS = {
    "forward hook": "beam:forward-hook",
    "backward hook": "beam:backward-hook",
}


class _Unencodable(Exception):
    """An element that cannot be encoded: it is dropped with a warning."""


@dataclass
class _State:
    """What the encoder remembers while walking a part."""

    # Part scope
    clefs: dict[int, str] = field(default_factory=dict)  # clef token per staff
    staves: int | None = None  # highest staff number announced by a clef

    # Measure scope (delta encoding)
    voice: int | None = None
    stem: str | None = None
    staff: int | None = None
    previous_pitch: str | None = None  # for pitch_mode="interval"

    def start_measure(self) -> None:
        self.voice = self.stem = self.staff = self.previous_pitch = None


class PartEncoder:
    """Turns the measures of a :class:`~mxltok.score.Part` into LMX tokens."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config

    def encode(self, part: Part) -> list[str]:
        """Return the LMX tokens of a part (without the ``part:*`` tokens).

        Elements that cannot be encoded (an unknown accidental or stem, a
        cursor move that is not a sum of note types) are skipped with a
        warning. Content that makes the whole part unencodable (a pitch
        outside A0-B9, an unknown note type) raises :class:`MxlTokError`.
        """
        state = _State()
        tokens: list[str] = []
        for measure in part.measures:
            state.start_measure()
            repeats = [
                element.repeat
                for element in measure.elements
                if isinstance(element, Barline)
            ]
            # Repeats precede the measure token of their measure: backward ones
            # always, forward ones only when the measure has content.
            tokens += [REPEAT_BACKWARD] * repeats.count("backward")
            events = self._measure_events(measure, state)
            if events:
                tokens += [REPEAT_FORWARD] * repeats.count("forward")
                tokens.append(MEASURE)
                tokens += [token for event in events for token in event]
        return tokens

    def _measure_events(self, measure: Measure, state: _State) -> list[list[str]]:
        """Return the tokens of each encodable element of a measure.

        An element that fails to encode still counts as an event (with no
        tokens), so that its measure keeps its ``measure`` token.
        """
        events: list[list[str] | None] = []
        for element in measure.elements:
            if isinstance(element, Attributes):
                events += self._attributes(element, state)
                continue
            try:
                if isinstance(element, Note):
                    events.append(self._note(element, state))
                elif isinstance(element, Backup | Forward):
                    events.append(self._cursor_move(element, state))
                elif isinstance(element, Direction):
                    events.append(self._direction(element))
                elif isinstance(element, Harmony):
                    events.append(self._harmony(element))
            except _Unencodable as exc:
                warnings.warn(str(exc), stacklevel=2)
                events.append([])
        return [event for event in events if event is not None]

    # ------------------------------------------------------------- notes

    def _note(self, note: Note, state: _State) -> list[str] | None:
        """Return the tokens of a note, or ``None`` if it is not encoded."""
        if (note.grace and not self.config.include_grace_notes) or (
            note.rest and not self.config.include_rests
        ):
            return None
        if not note.rest and not note.is_pitched:
            return None  # unpitched (percussion) notes are not encoded
        note_type = note.type_
        if note_type is None:
            if note.measure_rest:
                note_type = "whole"
            elif note.grace:
                note_type = "eighth"
            else:
                return None

        tokens: list[str] = []
        if not note.print_object:
            tokens.append("print-object:no")
        if note.grace:
            tokens.append("grace")
            if note.grace_slash:
                tokens.append("grace:slash")
        if note.chord:
            tokens.append("chord")
        if note.rest:
            tokens.append("rest")
        else:
            tokens.append(self._pitch(note, state))

        if note.voice is not None:
            if note.voice != state.voice:
                tokens.append(f"voice:{note.voice}")
            state.voice = note.voice

        if note.measure_rest:
            tokens.append("rest:measure")
        elif note_type in NOTE_TYPE_QUARTERS:
            tokens.append(note_type)
        else:
            raise MxlTokError(f"Unknown note type: {note_type!r}")

        if note.time_modification is not None:
            actual, normal = note.time_modification
            tokens.append(f"{actual}in{normal}")
        tokens += ["dot"] * note.dots
        if note.stem is not None:
            if note.stem not in STEMS:
                raise _Unencodable(f"Unknown stem type '{note.stem}'.")
            if note.stem != state.stem:
                tokens.append(f"stem:{note.stem}")
                state.stem = note.stem
        if state.staves is not None and state.staves >= 2:
            staff = 1 if note.staff is None else note.staff
            if staff != state.staff:
                tokens.append(f"staff:{staff}")
                state.staff = staff
        for beam in note.beams:
            if beam != "continue":
                tokens.append(_BEAM_TOKENS.get(beam, f"beam:{beam}"))
        tokens += [f"tied:{tie}" for tie in note.ties if tie in ("start", "stop")]
        tokens += [f"tuplet:{tuplet}" for tuplet in note.tuplets]
        tokens += [mark for mark in _markings(note) if _is_note_marking(mark)]
        tokens += [f"slur:{slur}" for slur in note.slurs]
        return tokens

    def _pitch(self, note: Note, state: _State) -> str:
        """Return the pitch token of a note: pitch plus accidental suffix.

        The suffix is the printed accidental, else the one implied by the
        sounding alteration. In interval mode, every pitch but the first of
        the measure is written as an interval from the previous one.
        """
        pitch = f"{note.step}{note.octave}"
        if pitch not in _PITCHES:
            raise MxlTokError(f"Pitch out of range: {pitch!r}")
        accidental = note.accidental or ALTER_TO_ACCIDENTAL.get(note.alter)
        if accidental is not None:
            if accidental not in ACCIDENTAL_TO_SUFFIX:
                raise _Unencodable(f"Unsupported accidental type: {accidental}")
            pitch += ACCIDENTAL_TO_SUFFIX[accidental]
        if self.config.pitch_mode == "interval":
            token = absolute_to_interval(pitch, state.previous_pitch)
            state.previous_pitch = pitch
            return token
        return pitch

    # -------------------------------------------------- other elements

    def _attributes(self, attributes: Attributes, state: _State) -> list[list[str]]:
        """Return one event per key, time signature and clef.

        A clef without ``<sign>`` repeats the last clef of its staff. A clef is
        followed by its staff number when the attributes declare ``<staves>``.
        """
        events: list[list[str]] = []
        if attributes.key is not None:
            events.append([f"key:fifths:{attributes.key}"])
        if attributes.time is not None:
            beats, beat_type = attributes.time
            events.append([f"time:num:{beats}", f"time:den:{beat_type}"])
        for clef in attributes.clefs:
            if clef.sign is not None:
                state.clefs[clef.number] = (
                    PERCUSSION_CLEF
                    if clef.sign.lower() == "percussion"
                    else f"clef:{clef.sign}{'' if clef.line is None else clef.line}"
                )
        for clef in attributes.clefs:
            if clef.number not in state.clefs:
                continue
            event = [state.clefs[clef.number]]
            if attributes.staves is not None:
                event.append(f"staff:{clef.number}")
                state.staves = max(state.staves or 0, clef.number)
            events.append(event)
        return events

    @staticmethod
    def _cursor_move(move: Backup | Forward, state: _State) -> list[str]:
        """Return a backup/forward as ``backup <type>`` pairs summing up to it.

        A backup resets the delta encoding: the next voice states its voice,
        stem and staff again.
        """
        tag = BACKUP if isinstance(move, Backup) else FORWARD
        note_types, exact = split_duration(move.duration)
        if isinstance(move, Backup):
            state.voice = state.staff = state.stem = None
        tokens = [token for note_type in note_types for token in (tag, note_type)]
        if not exact:
            warnings.warn(
                f"The {tag} duration {move.duration} is not a sum of note types "
                "(most likely a tuplet cursor move); it is only partially encoded.",
                stacklevel=2,
            )
        return tokens

    @staticmethod
    def _direction(direction: Direction) -> list[str] | None:
        """Return the token of a direction, or ``None`` if it is not encoded."""
        known = {"dynamics": DYNAMICS, "wedge": WEDGES, "pedal": PEDALS}
        if direction.kind == "tempo":
            return [f"tempo:{quantize_tempo(int(direction.value))}"]
        if direction.value in known.get(direction.kind, ()):
            return [f"{direction.kind}:{direction.value}"]
        return None

    @staticmethod
    def _harmony(harmony: Harmony) -> list[str]:
        quality, extension = HARMONY_KINDS.get(harmony.kind, (harmony.kind, None))
        tokens = [
            f"harmony:root:{harmony.root}{HARMONY_ALTERS.get(harmony.root_alter, '')}",
            f"harmony:quality:{quality}",
        ]
        if extension is not None:
            tokens.append(f"harmony:ext:{extension}")
        if harmony.bass is not None:
            bass_alter = HARMONY_ALTERS.get(harmony.bass_alter, "")
            tokens.append(f"harmony:bass:{harmony.bass}{bass_alter}")
        return tokens


def _markings(note: Note) -> list[str]:
    """Return the candidate marking tokens of a note, in encoding order."""
    markings = list(note.articulations)
    if note.fermata:
        markings.append("fermata")
    if note.arpeggiate:
        markings.append("arpeggiate")
    for ornament in note.ornaments:
        if ornament == "tremolo" and note.tremolo is not None:
            tremolo_type, marks = note.tremolo
            markings += [f"tremolo:{tremolo_type}", f"tremolo:{marks}"]
        else:
            markings.append(ornament)
    markings += [f"glissando:{glissando}" for glissando in note.glissandos]
    return markings


def _is_note_marking(token: str) -> bool:
    return token in NOTE_MARKINGS or token.startswith(("tremolo:", "glissando:"))


def split_duration(duration: Fraction) -> tuple[list[str], bool]:
    """Split a duration into distinct note types, longest first.

    Returns:
        The note types and whether they sum up to ``duration`` exactly.
    """
    note_types = []
    remainder = duration
    for note_type in reversed(NOTE_TYPES):
        if remainder <= 0:
            break
        if NOTE_TYPE_QUARTERS[note_type] <= remainder:
            note_types.append(note_type)
            remainder -= NOTE_TYPE_QUARTERS[note_type]
    return note_types, remainder == 0


def quantize_tempo(bpm: int) -> int:
    """Return the tempo of :data:`~mxltok.vocabulary.TEMPO_GRID` nearest to ``bpm``."""
    return min(TEMPO_GRID, key=lambda grid_bpm: abs(grid_bpm - bpm))
