"""Tokenizer configuration."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

from mxltok.categories import TokenCategory

Arrangement = Literal["part_first", "measure_first"]
PitchMode = Literal["absolute", "interval"]

# Keys of configurations saved by mxlGPT that never affected LMX tokens.
_LEGACY_KEYS = frozenset(
    {
        "duration_mode",
        "include_chords",
        "ticks_per_quarter",
        "max_bar_embedding",
        "max_bars",
        "features",
        "pitch_shift_range",
        "octuple_active_extended_types",
    }
)


@dataclass
class SpecialTokens:
    """The special tokens, placed first in the vocabulary."""

    pad: str = "<PAD>"
    bos: str = "<BOS>"
    eos: str = "<EOS>"
    unk: str = "<UNK>"
    mask: str = "<MASK>"

    def as_list(self) -> list[str]:
        """Return the special tokens in vocabulary order."""
        return [self.pad, self.bos, self.eos, self.unk, self.mask]


@dataclass
class TokenizerConfig:
    """How scores are turned into tokens.

    Attributes:
        arrangement: ``"part_first"`` writes every measure of a part before the
            next part; ``"measure_first"`` writes every part of a measure
            before the next measure.
        pitch_mode: ``"absolute"`` writes pitches as ``C4#``; ``"interval"``
            writes every pitch but the first of a measure as the semitones
            from the previous one (``interval:-3``).
        include_rests: Encode rests.
        include_grace_notes: Encode grace notes.
        prepend_instruments: Start the sequence with ``num_parts:N`` and the
            instrument of every part.
        add_bos_eos: Wrap the sequence in the BOS and EOS tokens.
        max_parts: Encode only the first parts of a score (0 means all).
        excluded_categories: Token categories (:class:`TokenCategory` values,
            such as ``"stem"``) removed from the encoded sequence.
        token_order: Reorder the tokens of each note by category priority
            (``{"duration": 0, "pitch": 1, ...}``); an empty mapping uses
            :data:`~mxltok.categories.DEFAULT_TOKEN_ORDER`.
        special_tokens: The special tokens.
    """

    arrangement: Arrangement = "part_first"
    pitch_mode: PitchMode = "absolute"
    include_rests: bool = True
    include_grace_notes: bool = True
    prepend_instruments: bool = True
    add_bos_eos: bool = True
    max_parts: int = 0
    excluded_categories: set[str] = field(default_factory=set)
    token_order: dict[str, int] | None = None
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)

    def __post_init__(self) -> None:
        if self.arrangement not in ("part_first", "measure_first"):
            raise ValueError(f"Unknown arrangement: {self.arrangement!r}")
        if self.pitch_mode not in ("absolute", "interval"):
            raise ValueError(f"Unknown pitch_mode: {self.pitch_mode!r}")
        if self.max_parts < 0:
            raise ValueError(f"max_parts must be >= 0, got {self.max_parts}")
        categories = {category.value for category in TokenCategory}
        self.excluded_categories = {str(c) for c in self.excluded_categories}
        unknown = self.excluded_categories - categories
        unknown |= set(self.token_order or {}) - categories
        if unknown:
            raise ValueError(f"Unknown token categories: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serializable dictionary."""
        data = asdict(self)
        data["excluded_categories"] = sorted(self.excluded_categories)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenizerConfig:
        """Build a configuration from :meth:`to_dict` output.

        Configurations saved by mxlGPT load too; their options that never
        affected LMX tokens are ignored.
        """
        data = dict(data)
        if tuple(data.get("pitch_shift_range", (0, 0))) != (0, 0):
            warnings.warn(
                "pitch_shift_range is ignored: transpose scores with "
                "mxltok.transpose() instead.",
                stacklevel=2,
            )
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _LEGACY_KEYS
        if unknown:
            warnings.warn(
                f"Ignoring unknown config keys: {sorted(unknown)}", stacklevel=2
            )
        data = {key: value for key, value in data.items() if key in known}
        if isinstance(data.get("special_tokens"), dict):
            data["special_tokens"] = SpecialTokens(**data["special_tokens"])
        if "excluded_categories" in data:
            data["excluded_categories"] = set(data["excluded_categories"])
        return cls(**data)
