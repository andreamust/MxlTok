"""Vocabulary, part-first encoding and decoding round trips."""

from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from mxltok import MXL, MxlTokError, TokenizerConfig, TokSequence
from mxltok.musicxml import load
from mxltok.score import (
    Attributes,
    Backup,
    Barline,
    Direction,
    Forward,
    Harmony,
    Note,
    Score,
)
from tests.conftest import GOLDEN_DIR
from tests.test_musicxml import FULL_XML

GOLDEN_STEMS = {
    "860",
    "spiccato",
    "stress",
    "2800_single_voice_text_and_chords_and_repeats",
}


@pytest.fixture(scope="module")
def tokenizer() -> MXL:
    return MXL()


def _notes(score: Score) -> list[Note]:
    return [
        e
        for p in score.parts
        for m in p.measures
        for e in m.elements
        if isinstance(e, Note)
    ]


def _structure(score: Score) -> list:
    """Per-measure content that must survive an encode/decode round trip."""
    out = []
    for part in score.parts:
        for measure in part.measures:
            items = []
            for e in measure.elements:
                match e:
                    case Note():
                        items.append(
                            (
                                "N",
                                e.step,
                                e.octave,
                                e.alter,
                                e.duration,
                                e.rest,
                                e.chord,
                                e.grace,
                                e.voice,
                                e.staff,
                                e.ties,
                                e.slurs,
                                e.tuplets,
                                e.beams,
                                e.articulations,
                                e.ornaments,
                                e.notations,
                                e.print_object,
                            )
                        )
                    case Backup() | Forward():
                        kind = type(e).__name__
                        if items and items[-1][0] == kind:  # consecutive = one
                            items[-1] = (kind, items[-1][1] + e.duration)
                        else:
                            items.append((kind, e.duration))
                    case Attributes():
                        items.append(
                            (
                                "A",
                                e.key,
                                e.time,
                                [(c.sign, c.line, c.number) for c in e.clefs],
                            )
                        )
                    case Direction():
                        items.append(
                            ("D", e.kind, e.value if e.kind != "tempo" else None)
                        )
                    case Harmony():
                        items.append(
                            ("H", e.root, e.root_alter, e.kind, e.bass, e.bass_alter)
                        )
                    case Barline():
                        if e.repeat:
                            items.append(("R", e.repeat))
            out.append(items)
    return out


# ------------------------------------------------------------------ vocab


def test_vocab_layout(tokenizer: MXL):
    assert tokenizer["PAD_None"] == 0 and tokenizer[0] == "PAD_None"
    assert tokenizer.pad_token_id == 0
    assert tokenizer.special_tokens_ids == [0, 1, 2, 3]
    assert sorted(tokenizer.vocab.values()) == list(range(len(tokenizer)))
    assert len(tokenizer.token_ids_of_type("Pitch")) == 10 * 7 * 6
    assert tokenizer.token_id_type(tokenizer["Duration_quarter"]) == "Duration"
    assert not tokenizer.is_trained
    assert "Tokenizer" not in repr(tokenizer) or "MXL" in repr(tokenizer)


@pytest.mark.parametrize(
    ("flag", "types"),
    [
        ("use_rests", {"Rest", "RestMeasure"}),
        ("use_grace_notes", {"Grace"}),
        ("use_voices", {"Voice"}),
        ("use_stems", {"Stem"}),
        ("use_beams", {"Beam"}),
        ("use_ties", {"Tie"}),
        ("use_slurs", {"Slur"}),
        ("use_tuplets", {"Tuplet"}),
        ("use_articulations", {"Articulation"}),
        ("use_ornaments", {"Ornament", "Tremolo"}),
        ("use_notations", {"Notation"}),
        ("use_dynamics", {"Dynamics"}),
        ("use_wedges", {"Wedge"}),
        ("use_pedals", {"Pedal"}),
        ("use_tempos", {"Tempo"}),
        ("use_time_signatures", {"TimeSig"}),
        ("use_key_signatures", {"Key"}),
        ("use_clefs", {"Clef"}),
        ("use_harmonies", {"HarmonyRoot", "HarmonyKind", "HarmonyBass"}),
        ("use_repeats", {"Repeat"}),
        ("use_print_object", {"Hidden"}),
    ],
)
def test_flags_remove_exactly_their_types(tokenizer: MXL, flag: str, types: set[str]):
    reduced = MXL(TokenizerConfig(**{flag: False}))
    full_types = set(tokenizer._token_types_indexes)
    assert full_types - set(reduced._token_types_indexes) == types
    for type_ in types:
        assert reduced.token_ids_of_type(type_) == []


def test_use_programs_false_gives_single_part_token():
    tok = MXL(TokenizerConfig(use_programs=False))
    assert tok.token_ids_of_type("Part") == [tok["Part_None"]]
    seq = tok.encode(FULL_XML.encode() and load(FULL_XML.encode()))
    assert seq.tokens[0] == "Part_None"


# --------------------------------------------------------------- encoding


def test_fixtures_encode_strictly_within_vocab(tokenizer: MXL, fixture_path: Path):
    seq = tokenizer.encode(fixture_path, strict=True)
    assert seq.tokens and seq.ids and seq.events
    assert all(token in tokenizer.vocab for token in seq.tokens)
    assert tokenizer._ids_to_tokens(seq.ids) == seq.tokens
    assert tokenizer.tokens_errors(seq) == 0.0
    assert tokenizer.tokens_errors(seq.ids) == 0.0
    assert seq.tokens[0].startswith("Part_")
    assert seq.tokens.count("Bar_None") == sum(
        len(p.measures) for p in load(fixture_path).parts
    )


def test_golden_tokens(tokenizer: MXL, fixture_path: Path, update_golden: bool):
    if fixture_path.stem not in GOLDEN_STEMS:
        pytest.skip("no golden file for this fixture")
    golden = GOLDEN_DIR / f"{fixture_path.stem}.txt"
    tokens = tokenizer.encode(fixture_path, strict=True).tokens
    if update_golden:
        golden.write_text("\n".join(tokens) + "\n")
    assert golden.exists(), "run pytest --update-golden to create the golden files"
    assert tokens == golden.read_text().split()


def test_full_document_tokens(tokenizer: MXL):
    tokens = tokenizer.encode(load(FULL_XML.encode()), strict=True).tokens
    assert tokens[:12] == [
        "Part_Piano",
        "Bar_None",
        "Repeat_forward",
        "Key_-1",
        "TimeSig_3/4",
        "Clef_G2",
        "Staff_1",
        "Clef_F4",
        "Staff_2",
        "Tempo_120",
        "Dynamics_mp",
        "HarmonyRoot_C",
    ]
    assert tokens[12:14] == ["HarmonyKind_major-seventh", "HarmonyBass_Eb"]
    note = tokens[14:]
    assert note[:6] == [
        "Pitch_C4n",
        "Voice_1",
        "Duration_quarter",
        "Stem_up",
        "Staff_1",
        "Beam_begin",
    ]
    assert "Tremolo_single-3" in tokens and "Ornament_trill-mark" in tokens
    assert tokens.count("Chord_None") == 1 and "Grace_slash" in tokens
    assert "TimeMod_3in2" in tokens and "Dot_1" in tokens and "Hidden_None" in tokens
    assert tokens.count("Backup_None") == 1
    assert "RestMeasure_None" in tokens and "Notation_glissando-start" in tokens
    assert tokens[-1] == "Repeat_backward"


def test_articulation_fixtures(tokenizer: MXL):
    for name in ("spiccato", "stress"):
        tokens = tokenizer.encode(
            Path("tests/fixtures/mxl") / f"{name}.musicxml", strict=True
        ).tokens
        assert tokens.count(f"Articulation_{name}") == 4


def test_tuplet_backup_chunks(tokenizer: MXL):
    events = tokenizer._encoder._chunks("Backup", Fraction(5, 3), True)
    tokens = [str(e) for e in events]
    assert tokens == [
        "Backup_None",
        "Duration_half",
        "TimeMod_3in2",
        "Backup_None",
        "Duration_eighth",
        "TimeMod_3in2",
    ]
    assert [
        str(e) for e in tokenizer._encoder._chunks("Forward", Fraction(3, 8), True)
    ] == ["Forward_None", "Duration_16th", "Forward_None", "Duration_32nd"]
    with pytest.raises(MxlTokError):
        tokenizer._encoder._chunks("Backup", Fraction(1, 17), True)
    with pytest.warns(UserWarning):
        assert tokenizer._encoder._chunks("Backup", Fraction(1, 17), False)


def test_strict_raises_and_lenient_clamps():
    xml = FULL_XML.replace("<voice>5</voice>", "<voice>9</voice>")
    score = load(xml.encode())
    with pytest.raises(MxlTokError, match="Voice 9"):
        MXL().encode(score, strict=True)
    with pytest.warns(UserWarning, match="Voice 9"):
        tokens = MXL().encode(score).tokens
    assert "Voice_8" in tokens


def test_alter_without_accidental_gets_one(tokenizer: MXL):
    xml = FULL_XML.replace("<accidental>sharp</accidental>", "")
    tokens = tokenizer.encode(load(xml.encode()), strict=True).tokens
    assert "Pitch_F4#" in tokens
    xml = FULL_XML.replace(
        "<accidental>flat</accidental>", ""
    )  # Eb: not in F major, needs a flat
    assert "Pitch_E4b" in tokenizer.encode(load(xml.encode()), strict=True).tokens


def test_call_dispatch(tokenizer: MXL, tmp_path: Path):
    path = Path("tests/fixtures/mxl/860.mxl")
    seq = tokenizer(path)
    assert isinstance(seq, TokSequence)
    assert tokenizer(load(path)) == seq
    score = tokenizer(seq.ids)
    assert isinstance(score, Score)
    assert len(_notes(score)) == len(_notes(load(path)))
    out = tmp_path / "out.mxl"
    tokenizer.decode(seq.tokens, output_path=out)
    assert load(out) == score


# --------------------------------------------------------------- decoding


def test_token_fixpoint(tokenizer: MXL, fixture_path: Path):
    seq = tokenizer.encode(fixture_path, strict=True)
    score = tokenizer.decode(seq, strict=True)
    assert tokenizer.encode(score, strict=True).tokens == seq.tokens


def test_structure_preserved(tokenizer: MXL, fixture_path: Path):
    original = load(fixture_path)
    decoded = tokenizer.decode(tokenizer.encode(original, strict=True), strict=True)
    assert [len(p.measures) for p in decoded.parts] == [
        len(p.measures) for p in original.parts
    ]
    assert _structure(decoded) == _structure(original)


def test_full_document_structure(tokenizer: MXL):
    original = load(FULL_XML.encode())
    decoded = tokenizer.decode(tokenizer.encode(original, strict=True), strict=True)
    assert _structure(decoded) == _structure(original)
    (part,) = decoded.parts
    assert (part.name, part.midi_program, part.midi_channel) == ("Piano", 1, 1)
    first = part.measures[0].elements[1]
    assert isinstance(first, Attributes) and first.staves == 2


def test_beam_and_slur_reconstruction(tokenizer: MXL):
    from mxltok.decoding import _number_beams

    assert _number_beams(["begin"], 0) == (("begin",), 1)
    assert _number_beams([], 1) == (("continue",), 1)
    assert _number_beams(["begin"], 1) == (("continue", "begin"), 2)
    assert _number_beams(["end"], 2) == (("continue", "end"), 1)
    assert _number_beams(["end", "end"], 2) == (("end", "end"), 0)
    assert _number_beams(["end"], 0) == (("end",), 0)
    assert _number_beams(["backward-hook"], 1) == (("continue", "backward hook"), 1)


def test_alter_resolution(tokenizer: MXL):
    tokens = [
        "Part_Piano",
        "Bar_None",
        "Key_2",
        "TimeSig_4/4",
        "Pitch_F4",
        "Duration_quarter",  # F# from key
        "Pitch_F4n",
        "Duration_quarter",  # natural cancels
        "Pitch_F4",
        "Duration_quarter",  # carried natural
        "Pitch_C5#",
        "Duration_quarter",
        "Tie_start",
        "Bar_None",
        "Pitch_C5",
        "Duration_quarter",
        "Tie_stop",  # tied: still sharp (key too)
        "Pitch_B4b",
        "Duration_quarter",
        "Pitch_B4",
        "Duration_half",  # carried flat
    ]
    notes = _notes(tokenizer.decode(tokens, strict=True))
    assert [n.alter for n in notes] == [1, 0, 0, 1, 1, -1, -1]
    assert [n.accidental for n in notes] == [
        None,
        "natural",
        None,
        "sharp",
        None,
        "flat",
        None,
    ]


def test_decode_strict_negatives(tokenizer: MXL):
    for tokens in (
        ["Part_Piano", "Bar_None", "Dot_1"],
        ["Part_Piano", "Bar_None", "Pitch_C4"],
        ["Part_Piano", "Bar_None", "Duration_quarter"],
        ["Part_Piano", "Bar_None", "Key_0", "Pitch_C4"],
        ["Part_Piano", "Bar_None", "Tempo_120", "Dot_1"],
        ["Bar_None", "Backup_None", "RestMeasure_None"],
    ):
        with pytest.raises(MxlTokError):
            tokenizer.decode(tokens, strict=True)


def test_decode_lenient_never_raises(tokenizer: MXL):
    import random

    rng = random.Random(0)
    vocab = list(tokenizer.vocab)
    for _ in range(200):
        tokens = [rng.choice(vocab) for _ in range(rng.randint(0, 40))]
        with warnings_ignored():
            score = tokenizer.decode(tokens)
        assert isinstance(score, Score)


class warnings_ignored:
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)


def test_decoded_parts_metadata(tokenizer: MXL):
    score = tokenizer.decode(
        [
            "Part_Drums",
            "Bar_None",
            "Pitch_C4",
            "Duration_quarter",
            "Part_Voice",
            "Bar_None",
        ]
    )
    assert [(p.id, p.name, p.midi_channel) for p in score.parts] == [
        ("P1", "Drums", 10),
        ("P2", "Voice", 1),
    ]
    assert Counter(type(e).__name__ for e in score.parts[0].measures[0].elements) == {
        "Note": 1,
        "Barline": 1,
    }
