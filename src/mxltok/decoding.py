"""Decode the LMX tokens of one part back into :class:`~mxltok.score.Measure` s.

Decoding mirrors :mod:`mxltok.encoding`. The tokens of a measure are grouped
around their *root* token (a note type, a signature, a direction or a chord
symbol), with the prefix tokens before it and the suffix tokens after it.
Each group then becomes a score element. Model-generated sequences need not
be well formed: malformed groups are skipped with a warning, or raise
:class:`~mxltok.errors.MxlTokError` in strict mode.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from mxltok.errors import MxlTokError
from mxltok.intervals import interval_to_absolute, split_pitch_token
from mxltok.score import (
    Attributes,
    Backup,
    Barline,
    Clef,
    Direction,
    Element,
    Forward,
    Harmony,
    Measure,
    Note,
    Part,
    resolve_alters,
)
from mxltok.vocabulary import (
    BACKUP,
    BEAM_TOKENS,
    CLEF_TOKENS,
    DYNAMICS_TOKENS,
    FORWARD,
    GLISSANDO_TOKENS,
    HARMONY_BASS_TOKENS,
    HARMONY_EXT_TOKENS,
    HARMONY_KINDS,
    HARMONY_QUALITY_TOKENS,
    HARMONY_ROOT_TOKENS,
    INTERVAL_TOKENS,
    KEY_TOKENS,
    MEASURE,
    NOTE_TYPE_QUARTERS,
    ORNAMENTS,
    PEDAL_TOKENS,
    PERCUSSION_CLEF,
    PITCH_TOKENS,
    REPEAT_BACKWARD,
    REPEAT_FORWARD,
    SPICCATO_TOKENS,
    STAFF_TOKENS,
    STEM_TOKENS,
    STRESS_TOKENS,
    SUFFIX_TO_ACCIDENTAL,
    TIME_DEN_TOKENS,
    TIME_MODIFICATIONS,
    TIME_NUM_TOKENS,
    TREMOLO_MARK_TOKENS,
    TREMOLO_TYPE_TOKENS,
    VOICE_TOKENS,
    WEDGE_TOKENS,
)

# --------------------------------------------------------------------------- #
# Token roles
# --------------------------------------------------------------------------- #

_NOTE_ROOTS = frozenset({*NOTE_TYPE_QUARTERS, "rest:measure"})
_SIGNATURE_ROOTS = frozenset({*TIME_NUM_TOKENS, *KEY_TOKENS, *CLEF_TOKENS})
_DIRECTION_ROOTS = frozenset({*DYNAMICS_TOKENS, *WEDGE_TOKENS, *PEDAL_TOKENS})
_HARMONY_ROOTS = frozenset(HARMONY_ROOT_TOKENS)
_TEMPO_RE = re.compile(r"^tempo:\d+$")

_PITCHES = frozenset(PITCH_TOKENS)
_INTERVALS = frozenset(INTERVAL_TOKENS)
_VOICES = frozenset(VOICE_TOKENS)
_PREFIXES = frozenset(
    {
        "print-object:no",
        "grace",
        "grace:slash",
        "chord",
        "rest",
        BACKUP,
        FORWARD,
        *_PITCHES,
        *_INTERVALS,
        *_VOICES,
    }
)

_RATIOS = frozenset(TIME_MODIFICATIONS)
_STEMS = frozenset(STEM_TOKENS)
_STAVES = frozenset(STAFF_TOKENS)
_BEAMS = frozenset(BEAM_TOKENS)
_TREMOLO_TYPES = frozenset(TREMOLO_TYPE_TOKENS)
_TREMOLO_MARKS = frozenset(TREMOLO_MARK_TOKENS)
_ORNAMENTS = frozenset(ORNAMENTS)
_GLISSANDOS = frozenset(GLISSANDO_TOKENS)
_TIME_DENOMINATORS = frozenset(TIME_DEN_TOKENS)
_SLURS = frozenset({"slur:start", "slur:stop"})
_SUFFIXES = frozenset(
    {
        *_RATIOS,
        "dot",
        *_STEMS,
        *_STAVES,
        *_BEAMS,
        "tied:start",
        "tied:stop",
        "tuplet:start",
        "tuplet:stop",
        *_SLURS,
        "fermata",
        "arpeggiate",
        "staccato",
        "accent",
        "strong-accent",
        "tenuto",
        "trill-mark",
        *_TREMOLO_TYPES,
        *_TREMOLO_MARKS,
        *STRESS_TOKENS,
        *SPICCATO_TOKENS,
        *_ORNAMENTS,
        "breath-mark",
        "caesura",
        *_GLISSANDOS,
        *HARMONY_QUALITY_TOKENS,
        *HARMONY_EXT_TOKENS,
        *HARMONY_BASS_TOKENS,
        *_TIME_DENOMINATORS,
    }
)

# Articulations in the order MuseScore writes them.
_ARTICULATION_ORDER = ("accent", "strong-accent", "tenuto", "staccato")
_KIND_OF_SPLIT = {split: kind for kind, split in HARMONY_KINDS.items()}
_HARMONY_SUFFIX_ALTERS = {"bb": -2, "b": -1, "#": 1, "##": 2}
_DEFAULT_MEASURE_DURATION = Fraction(4)  # when no time signature is known


def is_root(token: str) -> bool:
    """Whether a token is the root of a group (a note type, a signature...)."""
    return (
        token in _NOTE_ROOTS
        or token in _SIGNATURE_ROOTS
        or token in _DIRECTION_ROOTS
        or token in _HARMONY_ROOTS
        or _TEMPO_RE.match(token) is not None
    )


def is_prefix(token: str) -> bool:
    """Whether a token comes before the root of its group."""
    return token in _PREFIXES


def is_suffix(token: str) -> bool:
    """Whether a token comes after the root of its group."""
    return token in _SUFFIXES


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #


class _Malformed(Exception):
    """A group of tokens that does not form a valid element."""


@dataclass
class _Group:
    """A root token with the prefix and suffix tokens attached to it."""

    root: str
    prefixes: list[str] = field(default_factory=list)
    suffixes: list[str] = field(default_factory=list)

    def prefix(self, accepted: Collection[str]) -> str | None:
        """Remove and return the prefix of ``accepted``, if any."""
        return _take_one(self.prefixes, accepted, self.root)

    def suffix(self, accepted: Collection[str]) -> str | None:
        """Remove and return the suffix of ``accepted``, if any."""
        return _take_one(self.suffixes, accepted, self.root)

    def all_suffixes(self, accepted: Collection[str]) -> list[str]:
        """Remove and return every suffix of ``accepted``, in order."""
        taken = [token for token in self.suffixes if token in accepted]
        self.suffixes = [token for token in self.suffixes if token not in accepted]
        return taken

    def has_suffix(self, token: str) -> bool:
        """Remove ``token`` from the suffixes and tell whether it was there."""
        return self.suffix({token}) is not None

    def check_consumed(self) -> None:
        """Raise if a prefix or suffix was not used."""
        if self.prefixes or self.suffixes:
            leftovers = " ".join([*self.prefixes, *self.suffixes])
            raise _Malformed(f"Unexpected tokens for '{self.root}': {leftovers}")


def _take_one(tokens: list[str], accepted: Collection[str], root: str) -> str | None:
    found = [token for token in tokens if token in accepted]
    if len(found) > 1:
        raise _Malformed(f"Conflicting tokens {found} for '{root}'")
    if found:
        tokens.remove(found[0])
    return found[0] if found else None


def _value(token: str | None, transform: Callable[[str], int] = int) -> int | None:
    """Return the value after the last colon of a token, if there is a token."""
    return None if token is None else transform(token.rsplit(":", 1)[1])


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #


@dataclass
class _State:
    """What the decoder remembers while reading a part."""

    # Part scope
    measure_duration: Fraction | None = None  # from the last time signature
    previous_pitch: str | None = None  # resolves interval tokens

    # Measure scope
    voice: int | None = None
    stem: str | None = None
    staff: int | None = None
    open_beams: int = 0
    open_grace_beams: int = 0
    attributes: Attributes | None = None  # extended by consecutive signatures
    last_element: Element | None = None  # merges consecutive cursor moves

    def start_measure(self) -> None:
        self.voice = self.stem = self.staff = None
        self.open_beams = self.open_grace_beams = 0
        self.attributes = self.last_element = None


class PartDecoder:
    """Turns the LMX tokens of a part back into measures.

    Args:
        strict: Raise :class:`MxlTokError` on malformed tokens instead of
            skipping them with a warning.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def decode(self, tokens: Sequence[str]) -> list[Measure]:
        """Return the measures of a part, given its tokens (without ``part:*``)."""
        state = _State()
        part = Part(id="")
        for measure_tokens in self._split_measures(tokens):
            part.measures.append(self._measure(measure_tokens, state))
        _declare_staves(part)
        resolve_alters(part)
        _end_with_final_barline(part)
        return part.measures

    def _problem(self, message: str) -> None:
        if self.strict:
            raise MxlTokError(message)
        warnings.warn(message, stacklevel=2)

    @staticmethod
    def _split_measures(tokens: Sequence[str]) -> list[list[str]]:
        """Split the tokens at each ``measure`` token.

        Repeat tokens come right before the ``measure`` token of the measure
        they belong to, as the encoder writes them; trailing ones belong to
        the last measure. (mxlGPT's decoder attached backward repeats to the
        previous measure, moving them one measure early.) Tokens before the
        first ``measure`` form a measure of their own; without any
        ``measure`` token there is no measure.
        """
        measures: list[list[str]] = []
        leading: list[str] = []
        current: list[str] = []
        repeats: list[str] = []
        started = False
        for token in tokens:
            if token == MEASURE:
                if started:
                    measures.append(current)
                else:
                    leading = current
                current, repeats = repeats, []
                started = True
            elif token in (REPEAT_FORWARD, REPEAT_BACKWARD):
                repeats.append(token)
            else:
                current.append(token)
        if not started:
            return []
        measures.append(current + repeats)
        return [leading, *measures] if leading else measures

    def _measure(self, tokens: list[str], state: _State) -> Measure:
        state.start_measure()
        measure = Measure()
        body = [t for t in tokens if t not in (REPEAT_FORWARD, REPEAT_BACKWARD)]
        for group in self._groups(body):
            if group.root not in _SIGNATURE_ROOTS:
                state.attributes = None
            if group.root not in _NOTE_ROOTS:
                state.last_element = None
            try:
                self._decode_group(group, measure, state)
            except _Malformed as exc:
                self._problem(str(exc))
        forward = Barline("left", "forward", "heavy-light")
        backward = Barline("right", "backward", "light-heavy")
        measure.elements = [
            *[forward] * tokens.count(REPEAT_FORWARD),
            *measure.elements,
            *[backward] * tokens.count(REPEAT_BACKWARD),
        ]
        return measure

    def _groups(self, tokens: Sequence[str]) -> list[_Group]:
        """Attach every prefix and suffix token to its root.

        Prefixes attach to the next root, and a prefix after a note root
        closes that note. Suffixes attach to the current root, or to the next
        one when the current group is closed.
        """
        groups: list[_Group] = []
        current: _Group | None = None
        prefixes: list[str] = []
        suffixes: list[str] = []
        for token in tokens:
            if is_root(token):
                current = _Group(token, prefixes, suffixes)
                groups.append(current)
                prefixes, suffixes = [], []
            elif is_prefix(token):
                if current is not None and current.root in _NOTE_ROOTS:
                    current = None
                prefixes.append(token)
            elif is_suffix(token):
                (current.suffixes if current is not None else suffixes).append(token)
            else:
                self._problem(f"Unexpected token in a measure: {token}")
        return groups

    def _decode_group(self, group: _Group, measure: Measure, state: _State) -> None:
        """Decode a group into an element of ``measure``."""
        if group.root in _SIGNATURE_ROOTS:
            if state.attributes is None:
                state.attributes = Attributes()
                measure.elements.append(state.attributes)
            self._signature(group, state.attributes, state)
        elif group.root in _NOTE_ROOTS:
            element = self._note_or_cursor_move(group, state)
            if not _merge_cursor_moves(state.last_element, element):
                measure.elements.append(element)
                state.last_element = element
        elif group.root in _HARMONY_ROOTS:
            measure.elements.append(self._harmony(group))
        else:  # dynamics, wedge, pedal or tempo
            kind, value = group.root.split(":", 1)
            measure.elements.append(Direction(kind, value))

    # ------------------------------------------------------------- notes

    def _note_or_cursor_move(
        self, group: _Group, state: _State
    ) -> Note | Backup | Forward:
        for tag, move in ((BACKUP, Backup), (FORWARD, Forward)):
            if group.prefix({tag}) is not None:
                # A cursor move breaks the beams: the next voice starts afresh.
                state.open_beams = state.open_grace_beams = 0
                if group.root == "rest:measure":
                    raise _Malformed(f"A measure rest cannot be a {tag}.")
                ratio = group.suffix(_RATIOS)
                group.check_consumed()
                return move(note_duration(group.root, 0, ratio))
        return self._note(group, state)

    def _note(self, group: _Group, state: _State) -> Note:
        """Decode a note group; prefixes and suffixes are read in a fixed order."""
        note = Note(
            print_object=group.prefix({"print-object:no"}) is None,
            grace=group.prefix({"grace"}) is not None,
            grace_slash=group.prefix({"grace:slash"}) is not None,
            chord=group.prefix({"chord"}) is not None,
            rest=group.prefix({"rest"}) is not None,
        )
        pitch = group.prefix(_PITCHES)
        interval = group.prefix(_INTERVALS) if pitch is None else None
        voice = _value(group.prefix(_VOICES))
        ratio = group.suffix(_RATIOS)
        dots = len(group.all_suffixes({"dot"}))
        stem = group.suffix(_STEMS)
        staff = _value(group.suffix(_STAVES))
        beams = group.all_suffixes(_BEAMS)
        note.ties = tuple(t for t in ("stop", "start") if group.has_suffix(f"tied:{t}"))
        note.tuplets = tuple(
            t for t in ("stop", "start") if group.has_suffix(f"tuplet:{t}")
        )
        slurs = [s.removeprefix("slur:") for s in group.all_suffixes(_SLURS)]
        note.slurs = tuple(sorted(slurs, key=lambda slur: slur != "stop"))
        note.fermata = group.has_suffix("fermata")
        note.arpeggiate = group.has_suffix("arpeggiate")
        articulations = [a for a in _ARTICULATION_ORDER if group.has_suffix(a)]
        tremolo_type = group.suffix(_TREMOLO_TYPES)
        tremolo_marks = group.suffix(_TREMOLO_MARKS)
        ornaments = ["trill-mark"] if group.has_suffix("trill-mark") else []
        if group.suffix(STRESS_TOKENS) is not None:
            articulations.append("stress")
        if group.suffix(SPICCATO_TOKENS) is not None:
            articulations.append("spiccato")
        ornaments += group.all_suffixes(_ORNAMENTS)
        articulations += [a for a in ("breath-mark", "caesura") if group.has_suffix(a)]
        glissandos = group.all_suffixes(_GLISSANDOS)
        group.check_consumed()

        if interval is not None:
            if state.previous_pitch is None:
                self._problem(f"No pitch before '{interval}': decoded as a rest.")
            else:
                pitch = interval_to_absolute(interval, state.previous_pitch)
        if pitch is not None:
            state.previous_pitch = pitch
        if pitch is None or note.rest:
            note.rest = True
        else:
            step, octave, suffix = split_pitch_token(pitch)
            note.step, note.octave = step, octave
            note.accidental = SUFFIX_TO_ACCIDENTAL.get(suffix)

        if group.root == "rest:measure":
            note.rest = note.measure_rest = True
            measure_duration = state.measure_duration or _DEFAULT_MEASURE_DURATION
            note.duration = Fraction(0) if note.grace else measure_duration
        else:
            note.type_, note.dots = group.root, dots
            if not note.grace:
                note.duration = note_duration(group.root, dots, ratio)
        if ratio is not None:
            actual, normal = ratio.split("in")
            note.time_modification = (int(actual), int(normal))

        # Voice, stem and staff carry over from the previous note of the measure.
        state.voice = voice if voice is not None else state.voice
        note.voice = state.voice
        if not note.rest and note.type_ != "whole":
            state.stem = stem.removeprefix("stem:") if stem is not None else state.stem
            note.stem = state.stem
        state.staff = staff if staff is not None else state.staff
        note.staff = state.staff
        if not note.chord and not note.rest:
            if note.grace:
                note.beams, state.open_grace_beams = number_beams(
                    beams, state.open_grace_beams
                )
            else:
                note.beams, state.open_beams = number_beams(beams, state.open_beams)

        note.articulations = tuple(articulations)
        if tremolo_type is not None:
            marks = tremolo_marks.removeprefix("tremolo:") if tremolo_marks else "3"
            note.tremolo = (tremolo_type.removeprefix("tremolo:"), marks)
            ornaments.append("tremolo")
        note.ornaments = tuple(ornaments)
        note.glissandos = tuple(g.removeprefix("glissando:") for g in glissandos)
        return note

    # -------------------------------------------------- other elements

    @staticmethod
    def _signature(group: _Group, attributes: Attributes, state: _State) -> None:
        """Decode a key, time signature or clef into ``attributes``."""
        root = group.root
        if root in TIME_NUM_TOKENS:
            beats = _value(root)
            beat_type = _value(group.suffix(_TIME_DENOMINATORS)) or 4
            assert beats is not None
            attributes.time = (beats, beat_type)
            state.measure_duration = Fraction(4 * beats, beat_type)
        elif root in KEY_TOKENS:
            attributes.key = _value(root)
        else:
            number = _value(group.suffix(_STAVES)) or 1
            if root == PERCUSSION_CLEF:
                attributes.clefs.append(Clef("percussion", number=number))
            else:
                sign, line = (
                    root.removeprefix("clef:")[0],
                    root.removeprefix("clef:")[1:],
                )
                attributes.clefs.append(Clef(sign, int(line), number))
        group.check_consumed()

    @staticmethod
    def _harmony(group: _Group) -> Harmony:
        """Decode a chord symbol; unrelated suffixes are ignored."""

        def suffix_value(prefix: str) -> str | None:
            token = next((t for t in group.suffixes if t.startswith(prefix)), None)
            return None if token is None else token.removeprefix(prefix)

        quality = suffix_value("harmony:quality:")
        extension = suffix_value("harmony:ext:")
        kind = _KIND_OF_SPLIT.get((quality, extension)) if quality else None
        if kind is None:
            kind = f"{quality}-{extension}" if extension else quality or "major"
        root, root_alter = _harmony_pitch(group.root.removeprefix("harmony:root:"))
        harmony = Harmony(root, root_alter, kind)
        bass = suffix_value("harmony:bass:")
        if bass is not None:
            harmony.bass, harmony.bass_alter = _harmony_pitch(bass)
        return harmony


def note_duration(note_type: str, dots: int, ratio: str | None) -> Fraction:
    """Return the duration in quarters of a dotted, possibly tuplet, note type."""
    duration = NOTE_TYPE_QUARTERS[note_type] * (2 - Fraction(1, 2**dots))
    if ratio is not None:
        actual, normal = ratio.split("in")
        duration *= Fraction(int(normal), int(actual))
    return duration


def number_beams(tokens: Sequence[str], open_beams: int) -> tuple[tuple[str, ...], int]:
    """Return the MusicXML ``<beam>`` values of a note and the new open beam count.

    ``beam:begin`` and ``beam:end`` open and close beams, the beams left open
    are continued, and hooks come last.
    """
    delta = tokens.count("beam:begin") - tokens.count("beam:end")
    open_beams = max(open_beams, -delta)  # cannot close more beams than are open
    if delta >= 0:
        beams = ["continue"] * open_beams + ["begin"] * delta
    else:
        beams = ["continue"] * (open_beams + delta) + ["end"] * -delta
    beams += [
        token.removeprefix("beam:").replace("-", " ")
        for token in tokens
        if token.endswith("-hook")
    ]
    return tuple(beams), open_beams + delta


def _merge_cursor_moves(last: Element | None, element: Element) -> bool:
    """Extend ``last`` by ``element`` if both are backups or both forwards."""
    for move in (Backup, Forward):
        if isinstance(last, move) and isinstance(element, move):
            last.duration += element.duration
            return True
    return False


def _harmony_pitch(name: str) -> tuple[str, int]:
    """Split a chord symbol pitch: ``"Bb"`` -> ``("B", -1)``."""
    return name[0], _HARMONY_SUFFIX_ALTERS.get(name[1:], 0)


def _declare_staves(part: Part) -> None:
    """Declare the staff count of a multi-staff part in its first measure."""
    staves = max(
        (
            clef.number
            for measure in part.measures
            for element in measure.elements
            if isinstance(element, Attributes)
            for clef in element.clefs
        ),
        default=1,
    )
    if staves < 2:
        return
    elements = part.measures[0].elements
    position = next(
        (i for i, e in enumerate(elements) if not isinstance(e, Barline)), len(elements)
    )
    head = elements[position] if position < len(elements) else None
    if not isinstance(head, Attributes):
        head = Attributes()
        elements.insert(position, head)
    head.staves = staves


def _end_with_final_barline(part: Part) -> None:
    """End the last measure with a final barline, unless it has a right one."""
    if not part.measures:
        return
    elements = part.measures[-1].elements
    if not any(isinstance(e, Barline) and e.location == "right" for e in elements):
        elements.append(Barline("right", None, "light-heavy"))
