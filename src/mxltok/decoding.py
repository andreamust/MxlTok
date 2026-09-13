"""Tokens -> score IR, one part at a time."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import cast

from mxltok.classes import MxlTokError, TokenizerConfig
from mxltok.constants import (
    ACCIDENTAL_ALTER,
    NOTE_TYPE_QUARTERS,
    NOTE_TYPES,
    PITCH_STEPS,
    SUFFIX_TO_ACCIDENTAL,
)
from mxltok.score import (
    Attributes,
    Backup,
    Barline,
    Clef,
    Direction,
    DirectionKind,
    Element,
    Forward,
    Harmony,
    Measure,
    Note,
    Part,
    resolve_alters,
)

PREFIX_TYPES = frozenset(
    {
        "Hidden",
        "Grace",
        "Chord",
        "Rest",
        "Pitch",
        "Interval",
        "Voice",
        "Backup",
        "Forward",
    }
)
ROOT_TYPES = frozenset(
    {
        "Duration",
        "RestMeasure",
        "Key",
        "TimeSig",
        "Clef",
        "Tempo",
        "Dynamics",
        "Wedge",
        "Pedal",
        "HarmonyRoot",
        "Repeat",
    }
)
SUFFIX_TYPES = frozenset(
    {
        "TimeMod",
        "Dot",
        "Stem",
        "Staff",
        "Beam",
        "Tie",
        "Slur",
        "Tuplet",
        "Articulation",
        "Ornament",
        "Tremolo",
        "Notation",
        "HarmonyKind",
        "HarmonyBass",
    }
)
_NOTE_ROOTS = frozenset({"Duration", "RestMeasure"})
_STEMMED_TYPES = frozenset(NOTE_TYPES[: NOTE_TYPES.index("half") + 1])

Token = tuple[str, str]


def parse_pitch_token(value: str) -> tuple[str, int, str | None]:
    """``"F4#"`` -> ``("F", 4, "sharp")``."""
    step, rest = value[0], value[1:]
    digits = ""
    while rest and rest[0].isdigit():
        digits, rest = digits + rest[0], rest[1:]
    if (
        not digits
        or step not in "CDEFGAB"
        or (rest and rest not in SUFFIX_TO_ACCIDENTAL)
    ):
        raise MxlTokError(f"Invalid pitch token value {value!r}")
    return step, int(digits), SUFFIX_TO_ACCIDENTAL.get(rest)


def parse_interval_token(value: str) -> tuple[int, str | None]:
    """``"+3#"`` -> ``(3, "sharp")``."""
    digits = value.rstrip("#bn")
    suffix = value[len(digits) :]
    if not digits or digits[0] not in "+-" or not digits[1:].isdigit():
        raise MxlTokError(f"Invalid interval token value {value!r}")
    if suffix and suffix not in SUFFIX_TO_ACCIDENTAL:
        raise MxlTokError(f"Invalid interval token value {value!r}")
    return int(digits), SUFFIX_TO_ACCIDENTAL.get(suffix)


def duration_of(type_: str, dots: int, time_mod: tuple[int, int] | None) -> Fraction:
    """Duration in quarters of a note type with dots and tuplet ratio."""
    duration = NOTE_TYPE_QUARTERS[type_] * (2 - Fraction(1, 2**dots))
    if time_mod is not None:
        duration = duration * time_mod[1] / time_mod[0]
    return duration


@dataclass(slots=True)
class Group:
    """A root token with the prefix and suffix tokens attached to it."""

    root: Token
    prefixes: list[Token] = field(default_factory=list)
    suffixes: list[Token] = field(default_factory=list)

    def values(self, type_: str, source: list[Token]) -> list[str]:
        return [value for t, value in source if t == type_]

    def prefix(self, type_: str) -> str | None:
        return next((v for t, v in self.prefixes if t == type_), None)

    def suffix(self, type_: str) -> str | None:
        return next((v for t, v in self.suffixes if t == type_), None)


@dataclass(slots=True)
class _State:
    time: tuple[int, int] = (4, 4)
    voice: int | None = None
    staff: int | None = None
    stem: str | None = None
    open_beams: int = 0
    open_beams_grace: int = 0
    prev_pitch: int | None = None  # diatonic index of the previous pitched note
    clefs: dict[int, str] = field(default_factory=dict)

    def reset(self) -> None:
        self.voice = self.staff = self.stem = None
        self.open_beams = self.open_beams_grace = 0

    def new_measure(self) -> None:
        self.reset()
        self.prev_pitch = None


class PartDecoder:
    """Rebuild the measures of a :class:`Part` from their tokens."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config

    @staticmethod
    def _problem(message: str, strict: bool) -> None:
        if strict:
            raise MxlTokError(message)
        warnings.warn(message, stacklevel=3)

    def decode(self, measures: list[list[str]], strict: bool = False) -> list[Measure]:
        state = _State()
        part = Part(id="")
        for tokens in measures:
            state.new_measure()
            part.measures.append(self._measure(tokens, state, strict))
        resolve_alters(part, ACCIDENTAL_ALTER)
        self._finalize(part, state)
        return part.measures

    @staticmethod
    def _finalize(part: Part, state: _State) -> None:
        if not part.measures:
            return
        staves = max(state.clefs, default=1)
        if staves > 1:
            first = part.measures[0].elements
            attributes = next((e for e in first if isinstance(e, Attributes)), None)
            if attributes is None:
                attributes = Attributes()
                first.insert(0, attributes)
            attributes.staves = max(attributes.staves or 0, staves)
        last = part.measures[-1].elements
        if not any(isinstance(e, Barline) and e.location == "right" for e in last):
            last.append(Barline("right", None, "light-heavy"))

    def _group(self, tokens: list[str], strict: bool) -> list[Group]:
        groups: list[Group] = []
        pending: list[Token] = []
        current: Group | None = None
        for token in tokens:
            type_, _, value = token.partition("_")
            if type_ in ROOT_TYPES:
                if current is not None:
                    groups.append(current)
                current = Group((type_, value), pending)
                pending = []
            elif type_ in PREFIX_TYPES:
                if current is not None:  # a prefix always starts a new group
                    groups.append(current)
                    current = None
                pending.append((type_, value))
            elif type_ in SUFFIX_TYPES:
                if current is None:
                    self._problem(f"Suffix token {token} without a root token", strict)
                    continue
                current.suffixes.append((type_, value))
            else:
                self._problem(f"Unknown token {token}", strict)
        if current is not None:
            groups.append(current)
        if pending:
            self._problem(f"Dangling prefix tokens {pending}", strict)
        return groups

    def _measure(self, tokens: list[str], state: _State, strict: bool) -> Measure:
        elements: list[Element] = []
        for group in self._group(tokens, strict):
            root_type, value = group.root
            element: Element | None
            if root_type in _NOTE_ROOTS:
                if (
                    group.prefix("Backup") is not None
                    or group.prefix("Forward") is not None
                ):
                    element = self._backup_forward(group, state, strict)
                    if isinstance(element, Backup):
                        state.reset()
                else:
                    element = self._note(group, state, strict)
            elif root_type in ("Key", "TimeSig", "Clef"):
                previous = elements[-1] if elements else None
                if not isinstance(previous, Attributes):
                    previous = Attributes()
                    elements.append(previous)
                self._attributes(group, previous, state, strict)
                continue
            elif root_type == "HarmonyRoot":
                element = self._harmony(group, strict)
            elif root_type == "Repeat":
                if value == "forward":
                    elements.insert(0, Barline("left", "forward", "heavy-light"))
                else:
                    elements.append(Barline("right", "backward", "light-heavy"))
                continue
            else:
                self._check_bare(group, strict)
                element = Direction(cast(DirectionKind, root_type.lower()), value)
            if element is not None:
                elements.append(element)
        return Measure(elements)

    def _check_bare(self, group: Group, strict: bool) -> None:
        if group.prefixes or group.suffixes:
            self._problem(f"Unexpected tokens attached to {group.root}", strict)

    def _attributes(
        self, group: Group, attributes: Attributes, state: _State, strict: bool
    ) -> None:
        root_type, value = group.root
        if root_type == "Key":
            attributes.key = int(value)
        elif root_type == "TimeSig":
            beats, beat_type = value.split("/")
            attributes.time = state.time = (int(beats), int(beat_type))
        else:
            number = int(group.suffix("Staff") or 1)
            if value == "percussion":
                clef = Clef("percussion", None, number)
            else:
                clef = Clef(value[0], int(value[1:]), number)
            attributes.clefs.append(clef)
            state.clefs[number] = value
        if group.prefixes or [t for t, _ in group.suffixes if t != "Staff"]:
            self._problem(f"Unexpected tokens attached to {group.root}", strict)

    def _duration(
        self, group: Group, strict: bool
    ) -> tuple[str, int, tuple[int, int] | None] | None:
        type_ = group.root[1]
        dots = int(group.suffix("Dot") or 0)
        time_mod: tuple[int, int] | None = None
        if (ratio := group.suffix("TimeMod")) is not None:
            actual, normal = ratio.split("in")
            time_mod = (int(actual), int(normal))
        return type_, dots, time_mod

    def _backup_forward(
        self, group: Group, state: _State, strict: bool
    ) -> Backup | Forward | None:
        if group.root[0] == "RestMeasure":
            self._problem("RestMeasure cannot be a backup/forward root", strict)
            return None
        type_, dots, time_mod = self._duration(group, strict) or ("quarter", 0, None)
        duration = duration_of(type_, dots, time_mod)
        return (
            Backup(duration)
            if group.prefix("Backup") is not None
            else Forward(duration)
        )

    def _note(self, group: Group, state: _State, strict: bool) -> Note | None:
        note = Note()
        note.print_object = group.prefix("Hidden") is None
        if (grace := group.prefix("Grace")) is not None:
            note.grace = True
            note.grace_slash = grace == "slash"
        note.chord = group.prefix("Chord") is not None
        pitch = group.prefix("Pitch")
        interval = group.prefix("Interval")
        if pitch is not None:
            note.step, note.octave, note.accidental = parse_pitch_token(pitch)
        elif interval is not None and state.prev_pitch is not None:
            steps, note.accidental = parse_interval_token(interval)
            index = max(0, min(state.prev_pitch + steps, 7 * 10 - 1))
            note.step, note.octave = PITCH_STEPS[index % 7], index // 7
        elif group.prefix("Rest") is not None or group.root[0] == "RestMeasure":
            note.rest = True
        else:
            what = "an interval without a previous pitch" if interval else "no pitch"
            self._problem(f"Note group {group.root} has {what}; using a rest", strict)
            note.rest = True
        if note.step is not None and note.octave is not None:
            state.prev_pitch = note.octave * 7 + PITCH_STEPS.index(note.step)
        if (voice := group.prefix("Voice")) is not None:
            state.voice = int(voice)
        note.voice = state.voice
        if group.root[0] == "RestMeasure":
            note.rest_measure = True
            note.duration = Fraction(state.time[0] * 4, state.time[1])
        else:
            type_, dots, time_mod = self._duration(group, strict) or (
                "quarter",
                0,
                None,
            )
            note.type_, note.dots, note.time_mod = type_, dots, time_mod
            note.duration = (
                Fraction(0) if note.grace else duration_of(type_, dots, time_mod)
            )
        if (stem := group.suffix("Stem")) is not None:
            state.stem = stem
        if not note.rest and note.type_ in _STEMMED_TYPES:
            note.stem = state.stem
        if (staff := group.suffix("Staff")) is not None:
            state.staff = int(staff)
        note.staff = state.staff
        if not note.rest and state.clefs.get(note.staff or 1) == "percussion":
            note.unpitched = True
        if not note.chord:
            beams = group.values("Beam", group.suffixes)
            if note.grace:
                note.beams, state.open_beams_grace = _number_beams(
                    beams, state.open_beams_grace
                )
            else:
                note.beams, state.open_beams = _number_beams(beams, state.open_beams)
        note.ties = tuple(group.values("Tie", group.suffixes))
        note.slurs = tuple(group.values("Slur", group.suffixes))
        note.tuplets = tuple(group.values("Tuplet", group.suffixes))
        note.articulations = tuple(group.values("Articulation", group.suffixes))
        ornaments: list[str] = []
        for type_, value in group.suffixes:
            if type_ == "Ornament":
                ornaments.append(value)
            elif type_ == "Tremolo":
                kind, marks = value.rsplit("-", 1)
                ornaments.append(f"tremolo:{kind}:{marks}")
        note.ornaments = tuple(ornaments)
        note.notations = tuple(group.values("Notation", group.suffixes))
        return note

    def _harmony(self, group: Group, strict: bool) -> Harmony:
        root = group.root[1]
        bass = group.suffix("HarmonyBass")
        harmony = Harmony(
            root=root[0],
            root_alter=_harmony_alter(root[1:]),
            kind=group.suffix("HarmonyKind") or "major",
            bass=bass[0] if bass else None,
            bass_alter=_harmony_alter(bass[1:]) if bass else 0,
        )
        if group.prefixes or any(
            t not in ("HarmonyKind", "HarmonyBass") for t, _ in group.suffixes
        ):
            self._problem(f"Unexpected tokens attached to {group.root}", strict)
        return harmony


def _harmony_alter(suffix: str) -> int:
    return {"": 0, "#": 1, "b": -1}[suffix]


def _number_beams(values: list[str], open_count: int) -> tuple[tuple[str, ...], int]:
    """Rebuild numbered beams from ``begin``/``end``/hook tokens and the open count."""
    delta = values.count("begin") - values.count("end")
    if delta < -open_count:
        open_count = -delta
    beams: list[str] = []
    if delta >= 0:
        beams += ["continue"] * open_count + ["begin"] * delta
        open_count += delta
    else:
        beams += ["continue"] * (open_count + delta) + ["end"] * (-delta)
        open_count += delta
    beams += [v.replace("-", " ") for v in values if "hook" in v]
    return tuple(beams), max(open_count, 0)
