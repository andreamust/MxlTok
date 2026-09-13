"""Score IR -> token events, one part at a time."""

from __future__ import annotations

import warnings
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction

from mxltok.classes import Event, MxlTokError, TokenizerConfig
from mxltok.constants import (
    ACCIDENTAL_ALTER,
    ACCIDENTAL_TO_SUFFIX,
    ALTER_TO_ACCIDENTAL,
    ARTICULATIONS,
    BEAMS,
    CLEFS,
    DYNAMICS,
    HARMONY_ALTERS,
    HARMONY_KINDS,
    NOTATIONS,
    NOTE_TYPE_QUARTERS,
    NOTE_TYPES,
    ORNAMENTS,
    PEDALS,
    PITCH_STEPS,
    STEMS,
    TREMOLO_MARKS,
    TREMOLO_TYPES,
    WEDGES,
)
from mxltok.score import (
    AlterTracker,
    Attributes,
    Backup,
    Barline,
    Direction,
    Forward,
    Harmony,
    Measure,
    Note,
    Part,
    measure_onsets,
    pitch_key,
)

_DEFAULT_CLEF_LINES = {"G": 2, "F": 4, "C": 3}
_MAX_TYPE = NOTE_TYPE_QUARTERS[NOTE_TYPES[-1]]


def pitch_token(step: str, octave: int, accidental: str | None) -> str:
    """``Pitch`` token value, e.g. ``C4``, ``F4#``, ``B3bb``."""
    suffix = ACCIDENTAL_TO_SUFFIX[accidental] if accidental else ""
    return f"{step}{octave}{suffix}"


def note_type_of(duration: Fraction) -> tuple[str, int] | None:
    """Infer ``(type, dots)`` from an undotted/dotted plain duration, if any."""
    for dots in range(4):
        factor = 2 - Fraction(1, 2**dots)
        base = duration / factor
        for name, quarters in NOTE_TYPE_QUARTERS.items():
            if quarters == base:
                return name, dots
    return None


def diatonic_index(step: str, octave: int) -> int:
    """Position of a note on the diatonic scale (C0 = 0, one per step)."""
    return octave * 7 + PITCH_STEPS.index(step)


@dataclass(slots=True)
class _State:
    staves: int = 1
    voice: int | None = None
    staff: int | None = None
    stem: str | None = None
    prev_pitch: int | None = None  # diatonic index of the previous pitched note

    def reset(self) -> None:
        self.voice = self.staff = self.stem = None

    def new_measure(self) -> None:
        self.reset()
        self.prev_pitch = None


class PartEncoder:
    """Turn a :class:`Part` into per-measure lists of :class:`Event`."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self._tempos = config.tempos
        self._time_signatures = set(config.time_signatures)
        self._note_types = set(config.note_types)
        self._tuplet_ratios = set(config.tuplet_ratios)

    @staticmethod
    def _problem(message: str, strict: bool) -> None:
        if strict:
            raise MxlTokError(message)
        warnings.warn(message, stacklevel=3)

    def _clamp(self, name: str, value: int, maximum: int, strict: bool) -> int:
        if value <= maximum:
            return max(value, 1)
        self._problem(
            f"{name} {value} exceeds the configured maximum {maximum}", strict
        )
        return maximum

    def encode(self, part: Part, strict: bool = False) -> list[list[Event]]:
        state = _State()
        tracker = AlterTracker()
        measures: list[list[Event]] = []
        for index, measure in enumerate(part.measures):
            state.new_measure()
            tracker.start_measure()
            events: list[Event] = []
            onsets = measure_onsets(measure)
            self._record_accidentals(measure, onsets, tracker)
            for element, onset in zip(measure.elements, onsets, strict=True):
                match element:
                    case Note():
                        events += self._note(element, state, tracker, onset, strict)
                    case Backup():
                        events += self._chunks("Backup", element.duration, strict)
                        state.reset()
                    case Forward():
                        events += self._chunks("Forward", element.duration, strict)
                    case Attributes():
                        events += self._attributes(element, state, tracker, strict)
                    case Direction():
                        events += self._direction(element, strict)
                    case Harmony():
                        events += self._harmony(element, strict)
                    case Barline():
                        events += self._barline(element)
            for event in events:
                event.time = index
            measures.append(events)
        return measures

    @staticmethod
    def _record_accidentals(
        measure: Measure, onsets: list[Fraction], tracker: AlterTracker
    ) -> None:
        for element, onset in zip(measure.elements, onsets, strict=True):
            if isinstance(element, Attributes) and element.key is not None:
                tracker.key = element.key
            elif isinstance(element, Note) and (key := pitch_key(element)):
                alter = ACCIDENTAL_ALTER.get(element.accidental or "")
                if alter is not None:
                    tracker.record(key, onset, alter)

    def _accidental(
        self, note: Note, tracker: AlterTracker, onset: Fraction, strict: bool
    ) -> str | None:
        """Printed accidental, or the one needed when ``alter`` is not implied."""
        if note.accidental is not None:
            if note.accidental in ACCIDENTAL_TO_SUFFIX:
                accidental = note.accidental
            else:
                self._problem(f"Unsupported accidental {note.accidental!r}", strict)
                accidental = None
        else:
            accidental = None
            key = pitch_key(note)
            if key is not None:
                expected = tracker.expected(key, onset, "stop" in note.ties)
                if note.alter != expected:
                    accidental = ALTER_TO_ACCIDENTAL.get(int(note.alter))
                    if accidental is None:
                        self._problem(f"Unsupported alteration {note.alter}", strict)
                    else:
                        tracker.record(key, onset, note.alter)
        key = pitch_key(note)
        if key is not None and "start" in note.ties:
            tracker.tie_start(key, note.alter)
        return accidental

    def _note(
        self,
        note: Note,
        state: _State,
        tracker: AlterTracker,
        onset: Fraction,
        strict: bool,
    ) -> list[Event]:
        cfg = self.config
        if (note.grace and not cfg.use_grace_notes) or (
            note.rest and not cfg.use_rests
        ):
            return []
        if not note.rest and (note.step is None or note.octave is None):
            self._problem("Note without pitch skipped", strict)
            return []
        root = self._root(note, strict)
        if root is None:
            return []
        events: list[Event] = []
        if not note.print_object and cfg.use_print_object:
            events.append(Event("Hidden", "None"))
        if note.grace:
            events.append(Event("Grace", "slash" if note.grace_slash else "None"))
        if note.chord:
            events.append(Event("Chord", "None"))
        if note.rest:
            if not note.rest_measure:
                events.append(Event("Rest", "None"))
        else:
            assert note.step is not None and note.octave is not None
            lo, hi = cfg.octave_range
            if not lo <= note.octave <= hi:
                self._problem(
                    f"Octave {note.octave} outside {cfg.octave_range}", strict
                )
                return []
            accidental = self._accidental(note, tracker, onset, strict)
            events.append(self._pitch_event(note.step, note.octave, accidental, state))
        if cfg.use_voices and note.voice is not None:
            voice = self._clamp("Voice", note.voice, cfg.max_voices, strict)
            if voice != state.voice:
                events.append(Event("Voice", voice))
                state.voice = voice
        events.extend(root)
        events.extend(self._note_suffixes(note, state, strict))
        return events

    def _pitch_event(
        self, step: str, octave: int, accidental: str | None, state: _State
    ) -> Event:
        index = diatonic_index(step, octave)
        previous, state.prev_pitch = state.prev_pitch, index
        if self.config.pitch_mode == "interval" and previous is not None:
            steps = index - previous
            if abs(steps) <= self.config.max_pitch_interval:
                suffix = ACCIDENTAL_TO_SUFFIX[accidental] if accidental else ""
                return Event("Interval", f"{steps:+d}{suffix}")
        return Event("Pitch", pitch_token(step, octave, accidental))

    def _root(self, note: Note, strict: bool) -> list[Event] | None:
        """``Duration``/``TimeMod``/``Dot`` events, or a ``RestMeasure`` root."""
        if note.rest_measure:
            return [Event("RestMeasure", "None")]
        type_, dots = note.type_, note.dots
        if type_ is None:
            inferred = note_type_of(note.duration) if not note.grace else ("eighth", 0)
            if inferred is None:
                self._problem(
                    f"Cannot infer a note type for duration {note.duration}", strict
                )
                return None
            type_, dots = inferred
        if type_ not in self._note_types:
            self._problem(f"Unsupported note type {type_!r}", strict)
            return None
        events = [Event("Duration", type_)]
        if note.time_mod is not None:
            ratio = f"{note.time_mod[0]}in{note.time_mod[1]}"
            if ratio not in self._tuplet_ratios:
                self._problem(f"Unsupported tuplet ratio {ratio}", strict)
                return None
            events.append(Event("TimeMod", ratio))
        if dots:
            events.append(
                Event("Dot", self._clamp("Dot", dots, self.config.max_dots, strict))
            )
        return events

    def _note_suffixes(self, note: Note, state: _State, strict: bool) -> list[Event]:
        cfg = self.config
        events: list[Event] = []
        if cfg.use_stems and note.stem is not None:
            if note.stem not in STEMS:
                self._problem(f"Unsupported stem {note.stem!r}", strict)
            elif note.stem != state.stem:
                events.append(Event("Stem", note.stem))
                state.stem = note.stem
        if cfg.use_staves and note.staff is not None and state.staves > 1:
            staff = self._clamp("Staff", note.staff, cfg.max_staves, strict)
            if staff != state.staff:
                events.append(Event("Staff", staff))
                state.staff = staff
        if cfg.use_beams:
            for beam in note.beams:
                value = beam.replace(" ", "-")
                if value == "continue":
                    continue
                if value not in BEAMS:
                    self._problem(f"Unsupported beam {beam!r}", strict)
                    continue
                events.append(Event("Beam", value))
        if cfg.use_ties:
            events += [Event("Tie", t) for t in note.ties if t in ("start", "stop")]
        if cfg.use_tuplets:
            events += [
                Event("Tuplet", t) for t in note.tuplets if t in ("start", "stop")
            ]
        if cfg.use_slurs:
            events += [Event("Slur", t) for t in note.slurs if t in ("start", "stop")]
        if cfg.use_articulations:
            events += self._known(
                "Articulation", note.articulations, ARTICULATIONS, strict
            )
        if cfg.use_ornaments:
            for ornament in note.ornaments:
                if ornament.startswith("tremolo:"):
                    events += self._tremolo(ornament, strict)
                else:
                    events += self._known("Ornament", (ornament,), ORNAMENTS, strict)
        if cfg.use_notations:
            events += self._known("Notation", note.notations, NOTATIONS, strict)
        return events

    def _known(
        self,
        type_: str,
        values: tuple[str, ...],
        allowed: tuple[str, ...],
        strict: bool,
    ) -> list[Event]:
        events: list[Event] = []
        for value in values:
            if value in allowed:
                events.append(Event(type_, value))
            else:
                self._problem(f"Unsupported {type_.lower()} {value!r}", strict)
        return events

    def _tremolo(self, ornament: str, strict: bool) -> list[Event]:
        _, kind, marks = ornament.split(":")
        if (
            kind not in TREMOLO_TYPES
            or not marks.isdigit()
            or int(marks) not in TREMOLO_MARKS
        ):
            self._problem(f"Unsupported tremolo {ornament!r}", strict)
            return []
        return [Event("Tremolo", f"{kind}-{int(marks)}")]

    def _chunks(self, type_: str, duration: Fraction, strict: bool) -> list[Event]:
        """Decompose a backup/forward duration into note-type (+ tuplet) chunks."""
        parts = self._decompose(duration)
        if parts is None:
            self._problem(
                f"{type_} duration {duration} cannot be written with note types", strict
            )
            rounded = round(duration * 256) / 256
            parts = self._decompose(Fraction(rounded)) or []
        events: list[Event] = []
        for note_type, ratio in parts:
            events += [Event(type_, "None"), Event("Duration", note_type)]
            if ratio is not None:
                events.append(Event("TimeMod", ratio))
        return events

    def _decompose(self, duration: Fraction) -> list[tuple[str, str | None]] | None:
        if duration <= 0:
            return [] if duration == 0 else None
        if _is_dyadic(duration):
            return [(t, None) for t in self._binary(duration)]
        for ratio in self.config.tuplet_ratios:
            actual, normal = map(int, ratio.split("in"))
            scaled = duration * actual / normal
            if _is_dyadic(scaled):
                return [(t, ratio) for t in self._binary(scaled)]
        largest = next(
            (
                t
                for t in reversed(self.config.note_types)
                if NOTE_TYPE_QUARTERS[t] <= duration
            ),
            None,
        )
        if largest is None:
            return None
        rest = self._decompose(duration - NOTE_TYPE_QUARTERS[largest])
        return None if rest is None else [(largest, None), *rest]

    def _binary(self, duration: Fraction) -> list[str]:
        """Greedy decomposition of a dyadic duration into allowed note types."""
        chunks: list[str] = []
        for note_type in reversed(self.config.note_types):
            quarters = NOTE_TYPE_QUARTERS[note_type]
            while duration >= quarters:
                chunks.append(note_type)
                duration -= quarters
        return chunks

    def _attributes(
        self, attributes: Attributes, state: _State, tracker: AlterTracker, strict: bool
    ) -> list[Event]:
        cfg = self.config
        events: list[Event] = []
        if attributes.key is not None:
            tracker.key = attributes.key
            if cfg.use_key_signatures:
                lo, hi = cfg.key_range
                if not lo <= attributes.key <= hi:
                    self._problem(
                        f"Key {attributes.key} outside {cfg.key_range}", strict
                    )
                events.append(Event("Key", min(max(attributes.key, lo), hi)))
        if attributes.time is not None and cfg.use_time_signatures:
            value = f"{attributes.time[0]}/{attributes.time[1]}"
            if value in self._time_signatures:
                events.append(Event("TimeSig", value))
            else:
                self._problem(f"Time signature {value} not in the vocabulary", strict)
        state.staves = max(
            state.staves,
            attributes.staves or 0,
            *(clef.number for clef in attributes.clefs),
        )
        if cfg.use_clefs:
            for clef in attributes.clefs:
                if clef.sign.lower() == "percussion":
                    value = "percussion"
                else:
                    line = clef.line or _DEFAULT_CLEF_LINES.get(clef.sign, 2)
                    value = f"{clef.sign}{line}"
                if value not in CLEFS:
                    self._problem(f"Unsupported clef {value!r}", strict)
                    continue
                events.append(Event("Clef", value))
                if cfg.use_staves and state.staves > 1:
                    events.append(
                        Event(
                            "Staff",
                            self._clamp("Staff", clef.number, cfg.max_staves, strict),
                        )
                    )
        return events

    def _direction(self, direction: Direction, strict: bool) -> list[Event]:
        cfg = self.config
        match direction.kind:
            case "dynamics" if cfg.use_dynamics:
                return self._known("Dynamics", (direction.value,), DYNAMICS, strict)
            case "wedge" if cfg.use_wedges:
                return self._known("Wedge", (direction.value,), WEDGES, strict)
            case "pedal" if cfg.use_pedals:
                return self._known("Pedal", (direction.value,), PEDALS, strict)
            case "tempo" if cfg.use_tempos:
                return [Event("Tempo", self._nearest_tempo(int(direction.value)))]
        return []

    def _nearest_tempo(self, bpm: int) -> int:
        tempos = self._tempos
        index = bisect_left(tempos, bpm)
        if index == 0:
            return tempos[0]
        if index == len(tempos):
            return tempos[-1]
        before, after = tempos[index - 1], tempos[index]
        return before if bpm - before <= after - bpm else after

    def _harmony(self, harmony: Harmony, strict: bool) -> list[Event]:
        if not self.config.use_harmonies:
            return []
        if (
            harmony.root_alter not in HARMONY_ALTERS
            or harmony.bass_alter not in HARMONY_ALTERS
        ):
            self._problem("Harmony root/bass alterations must be within -1..1", strict)
            return []
        kind = harmony.kind
        if kind not in HARMONY_KINDS:
            self._problem(f"Unsupported harmony kind {kind!r}", strict)
            kind = "other"
        events = [
            Event("HarmonyRoot", f"{harmony.root}{HARMONY_ALTERS[harmony.root_alter]}"),
            Event("HarmonyKind", kind),
        ]
        if harmony.bass is not None:
            events.append(
                Event(
                    "HarmonyBass", f"{harmony.bass}{HARMONY_ALTERS[harmony.bass_alter]}"
                )
            )
        return events

    def _barline(self, barline: Barline) -> list[Event]:
        if self.config.use_repeats and barline.repeat in ("forward", "backward"):
            return [Event("Repeat", barline.repeat)]
        return []


def _is_dyadic(duration: Fraction) -> bool:
    denominator = duration.denominator
    return denominator <= 256 and denominator & (denominator - 1) == 0
