"""LMX token constants and the deterministic vocabulary.

The vocabulary order defines the token ids, so :func:`build_vocabulary` must
stay stable: trained models depend on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from mxltok.instruments import GM_CATEGORIES

# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

PART_START = "part:start"
PART_END = "part:end"
INSTRUMENT_PREFIX = "part:instrument:"
NUM_PARTS_PREFIX = "num_parts:"
MEASURE = "measure"
BACKUP = "backup"
FORWARD = "forward"
REPEAT_FORWARD = "repeat:forward"
REPEAT_BACKWARD = "repeat:backward"
MAX_PARTS_TOKEN = 16

# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #

STEPS = "CDEFGAB"

# Note types from the shortest to the longest, each twice the previous one.
NOTE_TYPES: tuple[str, ...] = (
    "1024th",
    "512th",
    "256th",
    "128th",
    "64th",
    "32nd",
    "16th",
    "eighth",
    "quarter",
    "half",
    "whole",
    "breve",
    "long",
    "maxima",
)
NOTE_TYPE_QUARTERS: dict[str, Fraction] = {
    note_type: Fraction(2**index, 256) for index, note_type in enumerate(NOTE_TYPES)
}

# Pitches from A0 to B9, in ascending order.
PITCHES: tuple[str, ...] = tuple(
    f"{step}{octave}"
    for octave in range(10)
    for step in STEPS
    if octave > 0 or step in "AB"
)

# A pitch token is a pitch plus an optional accidental suffix: ``C4#``, ``B3bb``.
ACCIDENTAL_SUFFIXES: tuple[str, ...] = ("#", "b", "##", "bb", "n")
ACCIDENTAL_TO_SUFFIX: dict[str, str] = {
    "sharp": "#",
    "flat": "b",
    "double-sharp": "##",
    "flat-flat": "bb",
    "natural": "n",
    "natural-sharp": "#",
    "natural-flat": "b",
}
SUFFIX_TO_ACCIDENTAL: dict[str, str] = {
    "#": "sharp",
    "b": "flat",
    "##": "double-sharp",
    "bb": "flat-flat",
    "n": "natural",
}
# The accidental implied by a ``<pitch><alter>`` when none is printed.
ALTER_TO_ACCIDENTAL: dict[float, str] = {
    -2: "flat-flat",
    -1: "flat",
    1: "sharp",
    2: "double-sharp",
}
PITCH_TOKENS: tuple[str, ...] = (
    *PITCHES,
    *(f"{pitch}{suffix}" for pitch in PITCHES for suffix in ACCIDENTAL_SUFFIXES),
)

# Interval tokens (``pitch_mode="interval"``): semitones from the previous note.
MAX_INTERVAL = 127
INTERVAL_PREFIX = "interval:"
INTERVAL_TOKENS: tuple[str, ...] = tuple(
    f"{INTERVAL_PREFIX}{i}" for i in range(-MAX_INTERVAL, MAX_INTERVAL + 1)
)

# Tuplet ratios as ``<actual-notes>in<normal-notes>``.
TIME_MODIFICATIONS: tuple[str, ...] = (
    "3in2",
    "6in4",
    "2in1",
    "2in3",
    "5in4",
    "7in8",
    "7in6",
    "9in8",
    "4in3",
    "7in4",
    "4in6",
    "13in8",
    "22in16",
    "10in4",
    "12in8",
    "9in4",
    "10in8",
    "18in4",
    "16in8",
    "15in8",
    "5in3",
    "11in8",
    "11in12",
    "5in2",
    "8in2",
    "4in2",
    "7in1",
    "35in16",
    "9in2",
)

STEMS: tuple[str, ...] = ("up", "down", "none")
BEAMS: tuple[str, ...] = ("begin", "end", "forward-hook", "backward-hook")
TREMOLO_TYPES: tuple[str, ...] = ("single", "start", "stop", "unmeasured")
TREMOLO_MARKS: tuple[str, ...] = ("1", "2", "3", "4")
ARTICULATIONS: tuple[str, ...] = (
    "staccato",
    "accent",
    "tenuto",
    "trill-mark",
    "fermata",
    "strong-accent",
)
STRESS_TOKENS: tuple[str, ...] = ("stress:above", "stress:below", "stress")
SPICCATO_TOKENS: tuple[str, ...] = ("spiccato:above", "spiccato:below", "spiccato")
ORNAMENTS: tuple[str, ...] = ("turn", "inverted-turn", "mordent", "inverted-mordent")

# Note markings encoded from <articulations>, <ornaments> and <notations>.
NOTE_MARKINGS: frozenset[str] = frozenset(
    {
        "fermata",
        "arpeggiate",
        "staccato",
        "accent",
        "strong-accent",
        "tenuto",
        "trill-mark",
        "spiccato",
        "stress",
        *ORNAMENTS,
        "breath-mark",
        "caesura",
    }
)

MAX_STAFF = 4
MAX_VOICE = 16

# --------------------------------------------------------------------------- #
# Attributes, directions and harmony
# --------------------------------------------------------------------------- #

CLEF_SIGNS = "GFC"
PERCUSSION_CLEF = "clef:PERCUSSION"
KEY_RANGE = range(-7, 8)
TIME_NUMERATORS = range(1, 17)
TIME_DENOMINATORS: tuple[int, ...] = (2, 4, 8, 16)

DYNAMICS: tuple[str, ...] = (
    "ppp",
    "pp",
    "p",
    "mp",
    "mf",
    "f",
    "ff",
    "fff",
    "sfz",
    "fp",
    "sf",
    "rfz",
)
WEDGES: tuple[str, ...] = ("crescendo", "diminuendo", "stop")
PEDALS: tuple[str, ...] = ("start", "stop", "change")

# Tempo tokens of the vocabulary; some are never emitted by the encoder.
VOCABULARY_TEMPOS: tuple[int, ...] = (
    *range(40, 60, 2),
    60,
    63,
    66,
    69,
    72,
    76,
    80,
    84,
    88,
    92,
    96,
    100,
    104,
    108,
    112,
    116,
    120,
    126,
    132,
    138,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
)
# The grid tempo markings are quantized to (ties go to the slower tempo).
TEMPO_GRID: tuple[int, ...] = (
    40,
    50,
    60,
    66,
    72,
    76,
    80,
    84,
    88,
    92,
    96,
    100,
    104,
    108,
    112,
    116,
    120,
    126,
    132,
    138,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
)

HARMONY_ALTERS: dict[int, str] = {0: "", -1: "b", 1: "#", -2: "bb", 2: "##"}
HARMONY_PITCH_NAMES: tuple[str, ...] = tuple(
    f"{step}{alter}" for step in STEPS for alter in ("", "b", "#")
)
HARMONY_QUALITIES: tuple[str, ...] = (
    "major",
    "minor",
    "augmented",
    "diminished",
    "dominant",
    "half",
    "suspended",
    "power",
    "pedal",
    "none",
    "other",
    "Tristan",
)
HARMONY_EXTENSIONS: tuple[str, ...] = (
    "seventh",
    "sixth",
    "ninth",
    "11th",
    "13th",
    "second",
    "fourth",
    "diminished",
    "minor",
)
# MusicXML <kind> values and their (quality, extension) split.
HARMONY_KINDS: dict[str, tuple[str, str | None]] = {
    "major": ("major", None),
    "minor": ("minor", None),
    "augmented": ("augmented", None),
    "diminished": ("diminished", None),
    "dominant": ("dominant", None),
    "half-diminished": ("half", "diminished"),
    "major-seventh": ("major", "seventh"),
    "minor-seventh": ("minor", "seventh"),
    "diminished-seventh": ("diminished", "seventh"),
    "augmented-seventh": ("augmented", "seventh"),
    "augmented-ninth": ("augmented", "ninth"),
    "major-sixth": ("major", "sixth"),
    "minor-sixth": ("minor", "sixth"),
    "dominant-ninth": ("dominant", "ninth"),
    "dominant-11th": ("dominant", "11th"),
    "dominant-13th": ("dominant", "13th"),
    "major-ninth": ("major", "ninth"),
    "minor-ninth": ("minor", "ninth"),
    "major-11th": ("major", "11th"),
    "minor-11th": ("minor", "11th"),
    "major-13th": ("major", "13th"),
    "minor-13th": ("minor", "13th"),
    "major-minor": ("major", "minor"),
    "suspended-second": ("suspended", "second"),
    "suspended-fourth": ("suspended", "fourth"),
    "power": ("power", None),
    "pedal": ("pedal", None),
    "none": ("none", None),
    "other": ("other", None),
    "Tristan": ("Tristan", None),
}


# --------------------------------------------------------------------------- #
# Token families
# --------------------------------------------------------------------------- #

NUM_PARTS_TOKENS: tuple[str, ...] = tuple(
    f"{NUM_PARTS_PREFIX}{n}" for n in range(1, MAX_PARTS_TOKEN + 1)
)
INSTRUMENT_TOKENS: tuple[str, ...] = tuple(
    f"{INSTRUMENT_PREFIX}{category}" for category in GM_CATEGORIES
)
CLEF_TOKENS: tuple[str, ...] = (
    *(f"clef:{sign}{line}" for sign in CLEF_SIGNS for line in range(1, 6)),
    PERCUSSION_CLEF,
)
KEY_TOKENS: tuple[str, ...] = tuple(f"key:fifths:{fifths}" for fifths in KEY_RANGE)
TIME_NUM_TOKENS: tuple[str, ...] = tuple(f"time:num:{n}" for n in TIME_NUMERATORS)
TIME_DEN_TOKENS: tuple[str, ...] = tuple(f"time:den:{d}" for d in TIME_DENOMINATORS)
STEM_TOKENS: tuple[str, ...] = tuple(f"stem:{stem}" for stem in STEMS)
BEAM_TOKENS: tuple[str, ...] = tuple(f"beam:{beam}" for beam in BEAMS)
TREMOLO_TYPE_TOKENS: tuple[str, ...] = tuple(f"tremolo:{t}" for t in TREMOLO_TYPES)
TREMOLO_MARK_TOKENS: tuple[str, ...] = tuple(f"tremolo:{m}" for m in TREMOLO_MARKS)
GLISSANDO_TOKENS: tuple[str, ...] = ("glissando:start", "glissando:stop")
STAFF_TOKENS: tuple[str, ...] = tuple(f"staff:{n}" for n in range(1, MAX_STAFF + 1))
VOICE_TOKENS: tuple[str, ...] = tuple(f"voice:{n}" for n in range(1, MAX_VOICE + 1))
DYNAMICS_TOKENS: tuple[str, ...] = tuple(f"dynamics:{d}" for d in DYNAMICS)
WEDGE_TOKENS: tuple[str, ...] = tuple(f"wedge:{wedge}" for wedge in WEDGES)
PEDAL_TOKENS: tuple[str, ...] = tuple(f"pedal:{pedal}" for pedal in PEDALS)
TEMPO_TOKENS: tuple[str, ...] = tuple(f"tempo:{bpm}" for bpm in VOCABULARY_TEMPOS)
HARMONY_ROOT_TOKENS: tuple[str, ...] = tuple(
    f"harmony:root:{name}" for name in HARMONY_PITCH_NAMES
)
HARMONY_QUALITY_TOKENS: tuple[str, ...] = tuple(
    f"harmony:quality:{quality}" for quality in HARMONY_QUALITIES
)
HARMONY_EXT_TOKENS: tuple[str, ...] = tuple(
    f"harmony:ext:{extension}" for extension in HARMONY_EXTENSIONS
)
HARMONY_BASS_TOKENS: tuple[str, ...] = tuple(
    f"harmony:bass:{name}" for name in HARMONY_PITCH_NAMES
)


def build_vocabulary(
    special_tokens: Sequence[str], pitch_mode: str = "absolute"
) -> list[str]:
    """Return every LMX token, special tokens first, in id order.

    Args:
        special_tokens: The special tokens (``<PAD>``, ``<BOS>``, ...).
        pitch_mode: ``"interval"`` adds the interval tokens.
    """
    vocab = [
        *special_tokens,
        PART_START,
        PART_END,
        MEASURE,
        BACKUP,
        FORWARD,
        *NUM_PARTS_TOKENS,
        *INSTRUMENT_TOKENS,
        *CLEF_TOKENS,
        *KEY_TOKENS,
        *TIME_NUM_TOKENS,
        *TIME_DEN_TOKENS,
        *PITCH_TOKENS,
        *(INTERVAL_TOKENS if pitch_mode == "interval" else ()),
        *NOTE_TYPES,
        "dot",
        *TIME_MODIFICATIONS,
        "rest",
        "rest:measure",
        "chord",
        "grace",
        "grace:slash",
        *STEM_TOKENS,
        *BEAM_TOKENS,
        "tied:start",
        "tied:stop",
        "slur:start",
        "slur:stop",
        "tuplet:start",
        "tuplet:stop",
        *TREMOLO_TYPE_TOKENS,
        *TREMOLO_MARK_TOKENS,
        *ARTICULATIONS,
        "arpeggiate",
        *STRESS_TOKENS,
        *SPICCATO_TOKENS,
        *ORNAMENTS,
        "breath-mark",
        "caesura",
        *GLISSANDO_TOKENS,
        *STAFF_TOKENS,
        *VOICE_TOKENS,
        REPEAT_FORWARD,
        REPEAT_BACKWARD,
        "print-object:no",
        *DYNAMICS_TOKENS,
        *WEDGE_TOKENS,
        *PEDAL_TOKENS,
        *TEMPO_TOKENS,
        *HARMONY_ROOT_TOKENS,
        *HARMONY_QUALITY_TOKENS,
        *HARMONY_EXT_TOKENS,
        *HARMONY_BASS_TOKENS,
    ]
    return list(dict.fromkeys(vocab))  # drop duplicates, keeping the first one
