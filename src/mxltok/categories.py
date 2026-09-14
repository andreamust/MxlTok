"""Token categories, and filtering or reordering of tokens by category."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from enum import Enum


class TokenCategory(str, Enum):
    """The semantic category of an LMX token."""

    # Structure
    PART = "part"
    MEASURE = "measure"
    STAFF = "staff"
    VOICE = "voice"
    BACKUP = "backup"
    FORWARD = "forward"
    # Attributes
    CLEF = "clef"
    KEY = "key"
    TIME = "time"
    # Notes and rests
    PITCH = "pitch"
    ACCIDENTAL = "accidental"
    REST = "rest"
    CHORD = "chord"
    GRACE = "grace"
    DURATION = "duration"
    DOT = "dot"
    # Notations
    STEM = "stem"
    BEAM = "beam"
    TIED = "tied"
    SLUR = "slur"
    TUPLET = "tuplet"
    TREMOLO = "tremolo"
    ARTICULATION = "articulation"
    ARPEGGIATE = "arpeggiate"
    ORNAMENT = "ornament"
    GLISSANDO = "glissando"
    # Directions
    DYNAMICS = "dynamics"
    WEDGE = "wedge"
    TEMPO = "tempo"
    PEDAL = "pedal"
    HARMONY = "harmony"
    # Barlines and display
    REPEAT = "repeat"
    PRINT_OBJECT = "print_object"
    # Special tokens (and anything unrecognized)
    SPECIAL = "special"


_PITCH_RE = re.compile(r"^[A-G]\d(?:[#b]|##|bb|n)?$")
_TUPLET_RATIO_RE = re.compile(r"^\d+in\d+$")
_DURATIONS = frozenset(
    {
        "whole",
        "half",
        "quarter",
        "eighth",
        "16th",
        "32nd",
        "64th",
        "128th",
        "256th",
        "512th",
        "1024th",
        "breve",
        "long",
        "maxima",
    }
)
_ARTICULATIONS = frozenset(
    {
        "staccato",
        "accent",
        "tenuto",
        "trill-mark",
        "fermata",
        "strong-accent",
        "breath-mark",
        "caesura",
    }
)
_ORNAMENTS = frozenset({"turn", "inverted-turn", "mordent", "inverted-mordent"})

# Categories identified by a token prefix, checked in order after the exact
# matches of ``classify_token``.
_PREFIXES: tuple[tuple[str, TokenCategory], ...] = (
    ("stem:", TokenCategory.STEM),
    ("beam:", TokenCategory.BEAM),
    ("tied:", TokenCategory.TIED),
    ("slur:", TokenCategory.SLUR),
    ("tuplet:", TokenCategory.TUPLET),
    ("tremolo:", TokenCategory.TREMOLO),
)
_LATE_PREFIXES: tuple[tuple[str, TokenCategory], ...] = (
    ("glissando:", TokenCategory.GLISSANDO),
    ("dynamics:", TokenCategory.DYNAMICS),
    ("wedge:", TokenCategory.WEDGE),
    ("tempo:", TokenCategory.TEMPO),
    ("pedal:", TokenCategory.PEDAL),
    ("harmony:", TokenCategory.HARMONY),
    ("repeat:", TokenCategory.REPEAT),
    ("print-object:", TokenCategory.PRINT_OBJECT),
)


def classify_token(token: str) -> TokenCategory:
    """Return the category of an LMX token (``SPECIAL`` if unrecognized)."""
    if token.startswith("part:"):
        return TokenCategory.PART
    if token == "measure" or token.startswith("bar:"):
        return TokenCategory.MEASURE
    if token.startswith("position:"):
        return TokenCategory.FORWARD
    if token.startswith("staff:"):
        return TokenCategory.STAFF
    if token.startswith("voice:"):
        return TokenCategory.VOICE
    if token == "backup":
        return TokenCategory.BACKUP
    if token == "forward":
        return TokenCategory.FORWARD
    if token.startswith("clef:"):
        return TokenCategory.CLEF
    if token.startswith("key:"):
        return TokenCategory.KEY
    if token == "time" or token.startswith(("time:", "beats:", "beat-type:")):
        return TokenCategory.TIME
    if _PITCH_RE.match(token) or token.startswith("interval:"):
        return TokenCategory.PITCH
    if token in ("rest", "rest:measure"):
        return TokenCategory.REST
    if token == "chord":
        return TokenCategory.CHORD
    if token in ("grace", "grace:slash"):
        return TokenCategory.GRACE
    if token.rstrip(".") in _DURATIONS or _TUPLET_RATIO_RE.match(token):
        return TokenCategory.DURATION
    if token == "dot":
        return TokenCategory.DOT
    for prefix, category in _PREFIXES:
        if token.startswith(prefix):
            return category
    if token in _ARTICULATIONS:
        return TokenCategory.ARTICULATION
    if token in _ORNAMENTS:
        return TokenCategory.ORNAMENT
    if token == "arpeggiate":
        return TokenCategory.ARPEGGIATE
    for prefix, category in _LATE_PREFIXES:
        if token.startswith(prefix):
            return category
    return TokenCategory.SPECIAL


DEFAULT_TOKEN_ORDER: dict[TokenCategory, int] = {
    category: priority
    for priority, category in enumerate(
        (
            TokenCategory.PART,
            TokenCategory.MEASURE,
            TokenCategory.REPEAT,
            TokenCategory.CLEF,
            TokenCategory.KEY,
            TokenCategory.TIME,
            TokenCategory.TEMPO,
            TokenCategory.DYNAMICS,
            TokenCategory.WEDGE,
            TokenCategory.PEDAL,
            TokenCategory.HARMONY,
            TokenCategory.STAFF,
            TokenCategory.VOICE,
            TokenCategory.BACKUP,
            TokenCategory.FORWARD,
            TokenCategory.GRACE,
            TokenCategory.CHORD,
            TokenCategory.REST,
            TokenCategory.PITCH,
            TokenCategory.DURATION,
            TokenCategory.DOT,
            TokenCategory.STEM,
            TokenCategory.BEAM,
            TokenCategory.TIED,
            TokenCategory.SLUR,
            TokenCategory.TUPLET,
            TokenCategory.TREMOLO,
            TokenCategory.ARTICULATION,
            TokenCategory.ORNAMENT,
            TokenCategory.GLISSANDO,
            TokenCategory.ARPEGGIATE,
            TokenCategory.PRINT_OBJECT,
            TokenCategory.SPECIAL,
        )
    )
}

# Categories that start a new group of tokens when reordering.
_GROUP_STARTS = frozenset(
    {
        TokenCategory.MEASURE,
        TokenCategory.BACKUP,
        TokenCategory.FORWARD,
        TokenCategory.PART,
        TokenCategory.VOICE,
    }
)
_UNRANKED = 999


def filter_tokens(tokens: Sequence[str], excluded: Collection[str]) -> list[str]:
    """Return the tokens whose category is not in ``excluded``."""
    return [token for token in tokens if classify_token(token).value not in excluded]


def reorder_tokens(tokens: Sequence[str], order: Mapping[str, int]) -> list[str]:
    """Stably sort the tokens of each group by category priority.

    A group starts at every measure, backup, forward, part and voice token,
    which stays first in its group. Categories missing from ``order`` go
    last; an empty ``order`` means :data:`DEFAULT_TOKEN_ORDER`.
    """
    priorities: Mapping[TokenCategory, int] = {
        TokenCategory(name): priority for name, priority in order.items()
    } or DEFAULT_TOKEN_ORDER

    def priority(token: str) -> int:
        return priorities.get(classify_token(token), _UNRANKED)

    groups: list[list[str]] = []
    for token in tokens:
        if not groups or classify_token(token) in _GROUP_STARTS:
            groups.append([])
        groups[-1].append(token)
    reordered: list[str] = []
    for group in groups:
        if classify_token(group[0]) in _GROUP_STARTS:
            reordered += [group[0], *sorted(group[1:], key=priority)]
        else:
            reordered += sorted(group, key=priority)
    return reordered
