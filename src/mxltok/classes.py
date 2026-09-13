"""Core data classes: errors, events, token sequences and the tokenizer config."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from mxltok.constants import (
    ARRANGEMENT,
    ENCODE_IDS_SPLIT,
    GM_CATEGORIES,
    KEY_RANGE,
    MANDATORY_SPECIAL_TOKENS,
    MAX_DOTS,
    MAX_PITCH_INTERVAL,
    MAX_STAVES,
    MAX_VOICES,
    NOTE_TYPES,
    NUM_TEMPOS,
    OCTAVE_RANGE,
    PITCH_MODE,
    SPECIAL_TOKENS,
    TEMPO_RANGE,
    TIME_SIGNATURE_RANGE,
    TUPLET_RATIOS,
)

Arrangement = Literal["part_first", "measure_first"]
PitchMode = Literal["absolute", "interval"]
IdsSplit = Literal["bar", "no"]

# Keys written by ``MusicTokenizer.to_dict`` that are not config parameters.
IGNORED_CONFIG_KEYS = ("mxltok_version", "hf_tokenizers_version")
_TUPLET_RE = re.compile(r"^\d+in\d+$")


class MxlTokError(ValueError):
    """Raised for malformed MusicXML input or invalid token sequences."""


@dataclass
class Event:
    """A token as its type and value; ``str(event)`` gives ``"Type_Value"``."""

    type_: str
    value: str | int
    time: int = -1
    program: int = 0
    desc: str | int = 0

    def __str__(self) -> str:
        return f"{self.type_}_{self.value}"

    def __repr__(self) -> str:
        return f"Event(type={self.type_}, value={self.value}, time={self.time})"


@dataclass
class TokSequence:
    """A token sequence in its several forms: tokens, ids, events, bytes.

    Empty attributes mean "not filled"; ``MusicTokenizer.complete_sequence``
    fills the missing ones. ``are_ids_encoded`` tells whether ``ids`` went
    through a trained BPE/Unigram/WordPiece model.
    """

    tokens: list[str] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)
    bytes: str = ""
    events: list[Event] = field(default_factory=list)
    are_ids_encoded: bool = False
    _ids_decoded: list[int] = field(default_factory=list)

    _ATTRS = ("tokens", "ids", "bytes", "events", "_ids_decoded")

    def split_per_bars(self) -> list[TokSequence]:
        """Split at every ``Bar`` token; the first chunk keeps any leading tokens."""
        tokens = self.tokens or [str(e) for e in self.events]
        if not tokens:
            msg = "split_per_bars needs the tokens or events of the sequence."
            raise ValueError(msg)
        starts = [i for i, tok in enumerate(tokens) if tok.startswith("Bar_")]
        if not starts or starts[0] != 0:
            starts.insert(0, 0)
        return [self._slice(slice(a, b)) for a, b in pairwise([*starts, len(tokens)])]

    def __len__(self) -> int:
        """Length of the first filled form, in order: ids, tokens, events, bytes."""
        for attr in ("ids", "tokens", "events", "bytes"):
            if (length := len(getattr(self, attr))) > 0:
                return length
        return 0

    def __getitem__(self, val: int | slice) -> int | str | Event | TokSequence:
        if isinstance(val, slice):
            return self._slice(val)
        for attr in ("ids", "tokens", "events", "bytes"):
            if len(getattr(self, attr)) > 0:
                return getattr(self, attr)[val]
        msg = "This TokSequence is not initialized: all its attributes are empty."
        raise ValueError(msg)

    def _slice(self, val: slice) -> TokSequence:
        seq = replace(self)
        for attr in self._ATTRS:
            if len(getattr(self, attr)) > 0:
                setattr(seq, attr, getattr(self, attr)[val])
        return seq

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokSequence):
            return False
        common = [
            attr
            for attr in ("tokens", "ids", "bytes", "events")
            if len(getattr(self, attr)) > 0 and len(getattr(other, attr)) > 0
        ]
        return bool(common) and all(
            getattr(self, attr) == getattr(other, attr) for attr in common
        )

    def __add__(self, other: TokSequence) -> TokSequence:
        seq = replace(self)
        seq += other
        return seq

    def __iadd__(self, other: TokSequence) -> TokSequence:
        if not isinstance(other, TokSequence):
            msg = f"Cannot add {other.__class__.__name__} to a TokSequence."
            raise TypeError(msg)
        for attr in self._ATTRS:
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))
        return self

    def __radd__(self, other: TokSequence | int) -> TokSequence:
        if other == 0:  # allows sum()
            return self
        if not isinstance(other, TokSequence):
            return NotImplemented
        return other.__add__(self)


def _format_special_token(token: str) -> str:
    parts = token.split("_")
    if len(parts) == 1:
        parts.append("None")
    elif len(parts) > 2:
        parts = ["-".join(parts[:-1]), parts[-1]]
        warnings.warn(
            f"Special token {token} must contain one underscore; "
            f"it will be saved as {'_'.join(parts)}.",
            stacklevel=3,
        )
    return "_".join(parts)


class TokenizerConfig:
    """Tokenizer configuration: which notation families to encode and their ranges.

    Every ``use_*`` flag gates a token family in the vocabulary, the encoder and
    the decoder. Unknown keyword arguments are kept in ``additional_params`` for
    tokenizer-specific options.

    :param arrangement: ``"part_first"`` serializes all measures of each part
        in turn (``<score-partwise>``); ``"measure_first"`` serializes all parts
        of each measure in turn (``<score-timewise>``).
    :param pitch_mode: ``"absolute"`` (``Pitch_C4#``) or ``"interval"``:
        ``Interval_+3#`` is three diatonic steps above the previous note of
        the measure, with the printed accidental; the first note of a measure
        is absolute.
    :param max_pitch_interval: intervals (in diatonic steps) beyond this are
        written as absolute ``Pitch`` tokens.
    :param octave_range: (lowest, highest) octave, inclusive.
    :param special_tokens: special tokens; ``PAD`` is always included.
    :param encode_ids_split: split token ids at bars before a trained model.
    :param max_voices: highest ``Voice`` value.
    :param max_staves: highest ``Staff`` value.
    :param max_parts: keep only the first parts of a score (``None`` = all).
    :param max_dots: highest ``Dot`` value.
    :param tempo_range: (min, max) BPM of the ``num_tempos`` linear tempo grid.
    :param time_signature_range: ``{denominator: (min, max) | [numerators]}``.
    :param key_range: (min, max) key signature in fifths, inclusive.
    :param note_types: allowed ``Duration`` values, subset of the MusicXML types.
    :param tuplet_ratios: allowed ``TimeMod`` values as ``"actualinnormal"``.
    :param instrument_categories: allowed ``Part`` values.
    """

    def __init__(
        self,
        arrangement: Arrangement = ARRANGEMENT,
        pitch_mode: PitchMode = PITCH_MODE,
        max_pitch_interval: int = MAX_PITCH_INTERVAL,
        octave_range: tuple[int, int] = OCTAVE_RANGE,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        encode_ids_split: IdsSplit = ENCODE_IDS_SPLIT,
        use_rests: bool = True,
        use_grace_notes: bool = True,
        use_voices: bool = True,
        use_staves: bool = True,
        use_stems: bool = True,
        use_beams: bool = True,
        use_ties: bool = True,
        use_slurs: bool = True,
        use_tuplets: bool = True,
        use_articulations: bool = True,
        use_ornaments: bool = True,
        use_notations: bool = True,
        use_dynamics: bool = True,
        use_wedges: bool = True,
        use_pedals: bool = True,
        use_tempos: bool = True,
        use_time_signatures: bool = True,
        use_key_signatures: bool = True,
        use_clefs: bool = True,
        use_harmonies: bool = True,
        use_repeats: bool = True,
        use_programs: bool = True,
        use_print_object: bool = True,
        max_voices: int = MAX_VOICES,
        max_staves: int = MAX_STAVES,
        max_parts: int | None = None,
        max_dots: int = MAX_DOTS,
        tempo_range: tuple[int, int] = TEMPO_RANGE,
        num_tempos: int = NUM_TEMPOS,
        time_signature_range: Mapping[int, Sequence[int]] = TIME_SIGNATURE_RANGE,
        key_range: tuple[int, int] = KEY_RANGE,
        note_types: Sequence[str] = NOTE_TYPES,
        tuplet_ratios: Sequence[str] = TUPLET_RATIOS,
        instrument_categories: Sequence[str] = GM_CATEGORIES,
        **kwargs: Any,
    ) -> None:
        if arrangement not in ("part_first", "measure_first"):
            raise ValueError(f"Unknown arrangement {arrangement!r}")
        if pitch_mode not in ("absolute", "interval"):
            raise ValueError(f"Unknown pitch_mode {pitch_mode!r}")
        if encode_ids_split not in ("bar", "no"):
            raise ValueError(f"Unknown encode_ids_split {encode_ids_split!r}")
        if not 0 <= octave_range[0] < octave_range[1] <= 9:
            raise ValueError(
                f"octave_range must satisfy 0 <= lo < hi <= 9: {octave_range}"
            )
        for name, value in (
            ("max_pitch_interval", max_pitch_interval),
            ("max_voices", max_voices),
            ("max_staves", max_staves),
            ("max_dots", max_dots),
            ("num_tempos", num_tempos),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if max_parts is not None and max_parts < 1:
            raise ValueError(f"max_parts must be >= 1 or None, got {max_parts}")
        if not tempo_range[0] < tempo_range[1]:
            raise ValueError(f"tempo_range must be increasing: {tempo_range}")
        if not -7 <= key_range[0] <= key_range[1] <= 7:
            raise ValueError(f"key_range must be within (-7, 7): {key_range}")
        unknown_types = set(note_types) - set(NOTE_TYPES)
        if unknown_types:
            raise ValueError(f"Unknown note types: {sorted(unknown_types)}")
        bad_ratios = [r for r in tuplet_ratios if not _TUPLET_RE.match(r)]
        if bad_ratios:
            raise ValueError(f"tuplet_ratios must look like '3in2': {bad_ratios}")
        if not use_staves:
            warnings.warn(
                "use_staves=False: multi-staff parts will collapse to one staff "
                "when decoding.",
                stacklevel=2,
            )

        self.arrangement: Arrangement = arrangement
        self.pitch_mode: PitchMode = pitch_mode
        self.max_pitch_interval = max_pitch_interval
        self.octave_range = tuple(octave_range)
        self.special_tokens = self._format_special_tokens(special_tokens)
        self.encode_ids_split: IdsSplit = encode_ids_split
        self.use_rests = use_rests
        self.use_grace_notes = use_grace_notes
        self.use_voices = use_voices
        self.use_staves = use_staves
        self.use_stems = use_stems
        self.use_beams = use_beams
        self.use_ties = use_ties
        self.use_slurs = use_slurs
        self.use_tuplets = use_tuplets
        self.use_articulations = use_articulations
        self.use_ornaments = use_ornaments
        self.use_notations = use_notations
        self.use_dynamics = use_dynamics
        self.use_wedges = use_wedges
        self.use_pedals = use_pedals
        self.use_tempos = use_tempos
        self.use_time_signatures = use_time_signatures
        self.use_key_signatures = use_key_signatures
        self.use_clefs = use_clefs
        self.use_harmonies = use_harmonies
        self.use_repeats = use_repeats
        self.use_programs = use_programs
        self.use_print_object = use_print_object
        self.max_voices = max_voices
        self.max_staves = max_staves
        self.max_parts = max_parts
        self.max_dots = max_dots
        self.tempo_range = tuple(tempo_range)
        self.num_tempos = num_tempos
        self.time_signature_range = self._normalize_time_signatures(
            time_signature_range
        )
        self.key_range = tuple(key_range)
        self.note_types = tuple(note_types)
        self.tuplet_ratios = tuple(tuplet_ratios)
        self.instrument_categories = tuple(instrument_categories)
        self.additional_params: dict[str, Any] = kwargs

    @staticmethod
    def _format_special_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
        formatted: list[str] = []
        for token in tokens:
            token = _format_special_token(token)
            if token in formatted:
                warnings.warn(f"Duplicate special token {token} ignored.", stacklevel=3)
                continue
            formatted.append(token)
        for token in MANDATORY_SPECIAL_TOKENS:
            token = _format_special_token(token)
            if token not in formatted:
                formatted.append(token)
        return tuple(formatted)

    @staticmethod
    def _normalize_time_signatures(
        time_signature_range: Mapping[Any, Sequence[int]],
    ) -> dict[int, tuple[int, ...]]:
        normalized: dict[int, tuple[int, ...]] = {}
        for denominator, numerators in time_signature_range.items():
            denominator = int(denominator)
            if denominator < 1 or denominator & (denominator - 1):
                raise ValueError(
                    f"Time signature denominators must be powers of two: {denominator}"
                )
            values = tuple(numerators)
            if (
                isinstance(numerators, tuple)
                and len(values) == 2
                and values[0] <= values[1]
            ):
                values = tuple(range(values[0], values[1] + 1))
            if any(n < 1 for n in values):
                raise ValueError(f"Time signature numerators must be >= 1: {values}")
            normalized[denominator] = tuple(sorted(set(values)))
        return normalized

    @property
    def tempos(self) -> tuple[int, ...]:
        """The tempo grid (BPM) covered by the ``Tempo`` tokens."""
        lo, hi = self.tempo_range
        if self.num_tempos == 1:
            return (lo,)
        step = (hi - lo) / (self.num_tempos - 1)
        return tuple(sorted({round(lo + i * step) for i in range(self.num_tempos)}))

    @property
    def time_signatures(self) -> tuple[str, ...]:
        """All ``TimeSig`` values as ``"numerator/denominator"``."""
        return tuple(
            f"{n}/{d}"
            for d, numerators in self.time_signature_range.items()
            for n in numerators
        )

    def to_dict(self, serialize: bool = False) -> dict[str, Any]:
        """Dump the config; ``serialize`` makes the result JSON-friendly."""
        dict_config = deepcopy(self.__dict__)
        if serialize:
            dict_config["time_signature_range"] = {
                str(d): list(n) for d, n in self.time_signature_range.items()
            }
        return dict_config

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any], **kwargs: Any) -> TokenizerConfig:
        """Instantiate a config from the output of :meth:`to_dict`."""
        input_dict = dict(input_dict)
        input_dict.update(input_dict.pop("additional_params", {}))
        for key in IGNORED_CONFIG_KEYS:
            input_dict.pop(key, None)
        return cls(**input_dict, **kwargs)

    def save_to_json(self, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as outfile:
            json.dump(self.to_dict(serialize=True), outfile, indent=4)

    @classmethod
    def load_from_json(cls, config_file_path: str | Path) -> TokenizerConfig:
        with Path(config_file_path).open() as param_file:
            return cls.from_dict(json.load(param_file))

    def copy(self) -> TokenizerConfig:
        return deepcopy(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenizerConfig):
            return False
        return self.to_dict(serialize=True) == other.to_dict(serialize=True)

    def __repr__(self) -> str:
        return f"TokenizerConfig({self.to_dict()})"
