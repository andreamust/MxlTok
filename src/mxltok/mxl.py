"""The ``MXL`` tokenizer: prefix/root/suffix note groups, part- or measure-first."""

from __future__ import annotations

import re
from itertools import zip_longest
from pathlib import Path

from mxltok.classes import Event, TokenizerConfig, TokSequence
from mxltok.constants import (
    ARTICULATIONS,
    BEAMS,
    CATEGORY_TO_MIDI_PROGRAM,
    CLEFS,
    DRUMS_MIDI_CHANNEL,
    DYNAMICS,
    GM_CATEGORIES,
    HARMONY_ALTERS,
    HARMONY_KINDS,
    INSTRUMENT_NAME_TO_CATEGORY,
    NOTATIONS,
    ORNAMENTS,
    PEDALS,
    PITCH_STEPS,
    STEMS,
    SUFFIX_TO_ACCIDENTAL,
    TREMOLO_MARKS,
    TREMOLO_TYPES,
    WEDGES,
)
from mxltok.decoding import PartDecoder
from mxltok.encoding import PartEncoder
from mxltok.score import Part, Score
from mxltok.tokenizer import MusicTokenizer

NOTE_PREFIX_TYPES = frozenset(
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
NOTE_ROOT_TYPES = frozenset({"Duration", "RestMeasure"})
NOTE_SUFFIX_TYPES = frozenset(
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
    }
)
ATTRIBUTE_ROOT_TYPES = frozenset({"Key", "TimeSig", "Clef"})
DIRECTION_ROOT_TYPES = frozenset({"Tempo", "Dynamics", "Wedge", "Pedal"})
_NAME_WORDS = re.compile(r"[a-z][a-z'\-]*")


def instrument_category(part: Part, categories: tuple[str, ...]) -> str:
    """General MIDI family of a part: MIDI channel/program first, then its name."""
    category = "Other"
    if part.midi_channel == DRUMS_MIDI_CHANNEL:
        category = "Drums"
    elif part.midi_program is not None and 1 <= part.midi_program <= 128:
        category = GM_CATEGORIES[(part.midi_program - 1) // 8]
    else:
        name = part.name.lower().strip()
        if name in INSTRUMENT_NAME_TO_CATEGORY:
            category = INSTRUMENT_NAME_TO_CATEGORY[name]
        else:
            for word in _NAME_WORDS.findall(name):
                if word in INSTRUMENT_NAME_TO_CATEGORY:
                    category = INSTRUMENT_NAME_TO_CATEGORY[word]
                    break
    if category not in categories:
        category = "Other" if "Other" in categories else categories[-1]
    return category


class MXL(MusicTokenizer):
    """MusicXML tokenizer of the MxlTok paper.

    Each note is a group of prefix tokens (``Hidden``, ``Grace``, ``Chord``,
    ``Rest``/``Pitch``, ``Voice``), one root (``Duration`` or ``RestMeasure``)
    and suffix tokens (``TimeMod``, ``Dot``, ``Stem``, ``Staff``, ``Beam``,
    ``Tie``, ``Slur``, ``Tuplet``, ``Articulation``, ``Ornament``, ``Tremolo``,
    ``Notation``). Attributes, directions, harmonies and repeats are emitted in
    document order; ``Voice``/``Staff``/``Stem`` are delta-encoded per measure.
    ``config.arrangement`` selects the part-first or measure-first order.
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig | None = None,
        params: str | Path | None = None,
    ) -> None:
        super().__init__(tokenizer_config, params)
        self._encoder = PartEncoder(self.config)
        self._decoder = PartDecoder(self.config)

    # -------------------------------------------------------- vocabulary

    def _create_base_vocabulary(self) -> list[str]:
        cfg = self.config
        vocab: list[str] = []
        if cfg.use_programs:
            vocab += [f"Part_{category}" for category in cfg.instrument_categories]
        else:
            vocab.append("Part_None")
        vocab.append("Bar_None")
        if cfg.use_key_signatures:
            vocab += [f"Key_{k}" for k in range(cfg.key_range[0], cfg.key_range[1] + 1)]
        if cfg.use_time_signatures:
            vocab += [f"TimeSig_{ts}" for ts in cfg.time_signatures]
        if cfg.use_clefs:
            vocab += [f"Clef_{clef}" for clef in CLEFS]
        if cfg.use_staves:
            vocab += [f"Staff_{i}" for i in range(1, cfg.max_staves + 1)]
        if cfg.use_tempos:
            vocab += [f"Tempo_{tempo}" for tempo in cfg.tempos]
        for octave in range(cfg.octave_range[0], cfg.octave_range[1] + 1):
            for step in PITCH_STEPS:
                vocab += [
                    f"Pitch_{step}{octave}{suffix}"
                    for suffix in ("", *SUFFIX_TO_ACCIDENTAL)
                ]
        if cfg.pitch_mode == "interval":
            vocab += [
                f"Interval_{i:+d}{suffix}"
                for i in range(-cfg.max_pitch_interval, cfg.max_pitch_interval + 1)
                for suffix in ("", *SUFFIX_TO_ACCIDENTAL)
            ]
        if cfg.use_rests:
            vocab += ["Rest_None", "RestMeasure_None"]
        vocab.append("Chord_None")
        if cfg.use_grace_notes:
            vocab += ["Grace_None", "Grace_slash"]
        if cfg.use_print_object:
            vocab.append("Hidden_None")
        vocab += ["Backup_None", "Forward_None"]
        if cfg.use_voices:
            vocab += [f"Voice_{i}" for i in range(1, cfg.max_voices + 1)]
        vocab += [f"Duration_{t}" for t in cfg.note_types]
        vocab += [f"TimeMod_{r}" for r in cfg.tuplet_ratios]
        vocab += [f"Dot_{i}" for i in range(1, cfg.max_dots + 1)]
        if cfg.use_stems:
            vocab += [f"Stem_{s}" for s in STEMS]
        if cfg.use_beams:
            vocab += [f"Beam_{b}" for b in BEAMS]
        if cfg.use_ties:
            vocab += ["Tie_start", "Tie_stop"]
        if cfg.use_slurs:
            vocab += ["Slur_start", "Slur_stop"]
        if cfg.use_tuplets:
            vocab += ["Tuplet_start", "Tuplet_stop"]
        if cfg.use_articulations:
            vocab += [f"Articulation_{a}" for a in ARTICULATIONS]
        if cfg.use_ornaments:
            vocab += [f"Ornament_{o}" for o in ORNAMENTS]
            vocab += [f"Tremolo_{t}-{m}" for t in TREMOLO_TYPES for m in TREMOLO_MARKS]
        if cfg.use_notations:
            vocab += [f"Notation_{n}" for n in NOTATIONS]
        if cfg.use_dynamics:
            vocab += [f"Dynamics_{d}" for d in DYNAMICS]
        if cfg.use_wedges:
            vocab += [f"Wedge_{w}" for w in WEDGES]
        if cfg.use_pedals:
            vocab += [f"Pedal_{p}" for p in PEDALS]
        if cfg.use_harmonies:
            roots = [
                f"{step}{alter}"
                for step in PITCH_STEPS
                for alter in HARMONY_ALTERS.values()
            ]
            vocab += [f"HarmonyRoot_{r}" for r in roots]
            vocab += [f"HarmonyKind_{k}" for k in HARMONY_KINDS]
            vocab += [f"HarmonyBass_{r}" for r in roots]
        if cfg.use_repeats:
            vocab += ["Repeat_forward", "Repeat_backward"]
        return vocab

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        types = set(self._token_types_indexes)
        prefixes = NOTE_PREFIX_TYPES & types
        roots = NOTE_ROOT_TYPES & types
        suffixes = NOTE_SUFFIX_TYPES & types
        item_starts = (
            prefixes
            | roots
            | ATTRIBUTE_ROOT_TYPES
            | DIRECTION_ROOT_TYPES
            | {"HarmonyRoot", "Repeat"}
        ) & types
        next_item = item_starts | {"Bar", "Part"}
        graph: dict[str, set[str]] = {
            "Part": set(next_item),
            "Bar": set(next_item),
            "HarmonyRoot": {"HarmonyKind"},
            "HarmonyKind": {"HarmonyBass"} | next_item,
            "HarmonyBass": set(next_item),
        }
        for type_ in prefixes:
            graph[type_] = set(prefixes | roots)
        graph["Backup"] = graph["Forward"] = {"Duration"}
        for type_ in roots | suffixes:
            graph[type_] = set(suffixes | next_item)
        for type_ in (ATTRIBUTE_ROOT_TYPES | DIRECTION_ROOT_TYPES | {"Repeat"}) & types:
            graph[type_] = set(next_item)
        if "Clef" in types and "Staff" in types:
            graph["Clef"].add("Staff")
        return {
            type_: set(successors & (types | {"Bar", "Part"}))
            for type_, successors in graph.items()
            if type_ in types
        }

    # ---------------------------------------------------------- encoding

    def _score_to_tokens(self, score: Score, strict: bool) -> TokSequence:
        cfg = self.config
        parts = score.parts[: cfg.max_parts] if cfg.max_parts else score.parts
        encoded = [
            (self._part_value(part), self._encoder.encode(part, strict))
            for part in parts
        ]
        if cfg.arrangement == "measure_first":
            events = self._arrange_measure_first(encoded)
        else:
            events = self._arrange_part_first(encoded)
        return TokSequence(events=events)

    def _part_value(self, part: Part) -> str:
        if not self.config.use_programs:
            return "None"
        return instrument_category(part, self.config.instrument_categories)

    @staticmethod
    def _arrange_part_first(parts: list[tuple[str, list[list[Event]]]]) -> list[Event]:
        events: list[Event] = []
        for category, measures in parts:
            events.append(Event("Part", category))
            for index, measure in enumerate(measures):
                events.append(Event("Bar", "None", time=index))
                events += measure
        return events

    @staticmethod
    def _arrange_measure_first(
        parts: list[tuple[str, list[list[Event]]]],
    ) -> list[Event]:
        events: list[Event] = []
        columns = zip_longest(*(measures for _, measures in parts), fillvalue=[])
        for index, column in enumerate(columns):
            events.append(Event("Bar", "None", time=index))
            for (category, _), measure in zip(parts, column, strict=True):
                events.append(Event("Part", category, time=index))
                events += measure
        return events

    # ---------------------------------------------------------- decoding

    def _tokens_to_score(self, tokens: list[str], strict: bool) -> Score:
        special = set(self.config.special_tokens)
        tokens = [token for token in tokens if token not in special]
        if self.config.arrangement == "measure_first":
            blocks = self._split_parts_measure_first(tokens)
        else:
            blocks = self._split_parts_part_first(tokens)
        parts: list[Part] = []
        for index, (category, measures) in enumerate(blocks, start=1):
            part = Part(id=f"P{index}", name=category or f"Part {index}")
            if category is not None and category != "None":
                part.midi_program = CATEGORY_TO_MIDI_PROGRAM.get(category)
                part.midi_channel = DRUMS_MIDI_CHANNEL if category == "Drums" else 1
            part.measures = self._decoder.decode(measures, strict)
            parts.append(part)
        return Score(parts=parts)

    @staticmethod
    def _split_parts_part_first(
        tokens: list[str],
    ) -> list[tuple[str | None, list[list[str]]]]:
        parts: list[tuple[str | None, list[list[str]]]] = []
        for token in tokens:
            type_, value = token.split("_", 1)
            if type_ == "Part":
                parts.append((value, []))
                continue
            if not parts:
                parts.append((None, []))
            measures = parts[-1][1]
            if type_ == "Bar" or not measures:
                measures.append([])
            if type_ != "Bar":
                measures[-1].append(token)
        return parts

    @staticmethod
    def _split_parts_measure_first(
        tokens: list[str],
    ) -> list[tuple[str | None, list[list[str]]]]:
        groups: list[list[tuple[str | None, list[str]]]] = []
        for token in tokens:
            type_, value = token.split("_", 1)
            if type_ == "Bar" or not groups:
                groups.append([])
            if type_ == "Bar":
                continue
            blocks = groups[-1]
            if type_ == "Part":
                blocks.append((value, []))
                continue
            if not blocks:
                blocks.append((None, []))
            blocks[-1][1].append(token)
        num_parts = max((len(blocks) for blocks in groups), default=0)
        parts: list[tuple[str | None, list[list[str]]]] = []
        for index in range(num_parts):
            category = next(
                (
                    blocks[index][0]
                    for blocks in groups
                    if index < len(blocks) and blocks[index][0]
                ),
                None,
            )
            measures = [
                blocks[index][1] if index < len(blocks) else [] for blocks in groups
            ]
            while measures and not measures[-1]:
                measures.pop()
            parts.append((category, measures))
        return parts
