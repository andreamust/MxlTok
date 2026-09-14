from fractions import Fraction

import pytest
from conftest import FIXTURES, golden_tokens

from mxltok import MXLTokenizer, MxlTokError, TokenizerConfig, read_musicxml
from mxltok.encoding import quantize_tempo, split_duration

RAW = TokenizerConfig(add_bos_eos=False)


def encode(path, **options):
    config = TokenizerConfig(add_bos_eos=False, **options)
    return MXLTokenizer(config).encode(path).tokens


def encode_xml(measures: str, prepend_instruments: bool = False) -> list[str]:
    """Encode a one-part score whose measures are given as MusicXML."""
    xml = f"""<score-partwise><part-list><score-part id="P1">
        <part-name>Piano</part-name></score-part></part-list>
        <part id="P1">{measures}</part></score-partwise>"""
    config = TokenizerConfig(add_bos_eos=False, prepend_instruments=prepend_instruments)
    return MXLTokenizer(config).encode(read_musicxml(xml.encode())).tokens


NOTE = "<note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>"
HEAD = "<attributes><divisions>1</divisions></attributes>"


# --------------------------------------------------------------------------- #
# Parity with mxlGPT
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arrangement", ["part_first", "measure_first"])
def test_tokens_match_mxlgpt(score_path, arrangement):
    expected = golden_tokens(f"{score_path.stem}.{arrangement}.lmx")
    assert encode(score_path, arrangement=arrangement) == expected


def test_interval_tokens_match_mxlgpt(score_path):
    expected = golden_tokens(f"{score_path.stem}.interval.lmx")
    assert encode(score_path, pitch_mode="interval") == expected


def test_filtered_and_reordered_tokens_match_mxlgpt(score_path):
    config = TokenizerConfig(excluded_categories={"stem", "beam"}, token_order={})
    expected = golden_tokens(f"{score_path.stem}.filtered.lmx")
    assert MXLTokenizer(config).encode(score_path).tokens == expected


def test_scores_beyond_the_vocabulary_are_rejected():
    # 18 parts, while the vocabulary stops at num_parts:16, as in mxlGPT.
    path = FIXTURES / "mxl" / "Stabat_Mater_Rossini_1_-_Introduzione.mxl"
    with pytest.raises(MxlTokError, match="num_parts:18"):
        MXLTokenizer().encode(path)


def test_encoding_a_path_or_a_score_is_the_same(score_path):
    tokenizer = MXLTokenizer()
    assert tokenizer.encode(score_path) == tokenizer.encode(read_musicxml(score_path))


# --------------------------------------------------------------------------- #
# Sequence layout
# --------------------------------------------------------------------------- #


def test_full_sequence_layout():
    tokens = encode_xml(f"<measure>{HEAD}{NOTE}<type>quarter</type></note></measure>")
    assert tokens == [
        "part:start",
        "part:instrument:Piano",
        "measure",
        "C4",
        "quarter",
        "part:end",
    ]


def test_bos_eos_and_instrument_header():
    tokenizer = MXLTokenizer()
    sequence = tokenizer.encode(FIXTURES / "mxl" / "860.mxl")
    instruments = ["part:instrument:Ensemble"] * 4
    assert sequence.tokens[:6] == ["<BOS>", "num_parts:4", *instruments]
    assert sequence.tokens[-1] == "<EOS>"
    assert sequence.ids[0] == tokenizer.bos_id and sequence.ids[-1] == tokenizer.eos_id


def test_max_parts_keeps_the_first_parts():
    path = FIXTURES / "mxl" / "4480_string_quintet.mxl"
    tokens = encode(path, max_parts=2)
    assert tokens[0] == "num_parts:2"
    assert tokens.count("part:start") == 2


# --------------------------------------------------------------------------- #
# Encoding rules
# --------------------------------------------------------------------------- #


def test_voice_and_stem_are_delta_encoded():
    note = f"{NOTE}<voice>1</voice><type>quarter</type><stem>up</stem></note>"
    tokens = encode_xml(
        f"<measure>{HEAD}{note}{note}</measure><measure>{note}</measure>"
    )
    assert tokens == [
        "part:start", "part:instrument:Piano",
        "measure", "C4", "voice:1", "quarter", "stem:up", "C4", "quarter",
        "measure", "C4", "voice:1", "quarter", "stem:up",
        "part:end",
    ]  # fmt: skip


def test_accidental_falls_back_to_the_alteration():
    flat = "<note><pitch><step>E</step><alter>-1</alter><octave>5</octave></pitch>"
    printed = f"{flat}<duration>1</duration><type>quarter</type><accidental>natural</accidental></note>"
    implied = f"{flat}<duration>1</duration><type>quarter</type></note>"
    tokens = encode_xml(f"<measure>{HEAD}{printed}{implied}</measure>")
    assert tokens[3:7] == ["E5n", "quarter", "E5b", "quarter"]


def test_repeats_surround_the_measure_token():
    forward = '<barline location="left"><repeat direction="forward"/></barline>'
    backward = '<barline location="right"><repeat direction="backward"/></barline>'
    note = f"{NOTE}<type>whole</type></note>"
    tokens = encode_xml(
        f"<measure>{forward}{HEAD}{note}</measure><measure>{note}{backward}</measure>"
    )
    # A backward repeat is written before the measure token of its measure.
    assert tokens[2:] == [
        "repeat:forward", "measure", "C4", "whole",
        "repeat:backward", "measure", "C4", "whole",
        "part:end",
    ]  # fmt: skip


def test_staff_tokens_follow_a_multi_staff_declaration():
    attributes = """<attributes><divisions>1</divisions><staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef></attributes>"""
    lower = f"{NOTE}<type>whole</type><staff>2</staff></note>"
    tokens = encode_xml(f"<measure>{attributes}{lower}</measure>")
    assert tokens[3:] == [
        "clef:G2",
        "staff:1",
        "clef:F4",
        "staff:2",
        "C4",
        "whole",
        "staff:2",
        "part:end",
    ]


def test_backup_is_split_into_note_types():
    backup = "<backup><duration>3</duration></backup>"
    tokens = encode_xml(
        f"<measure>{HEAD}{NOTE}<type>quarter</type></note>{backup}</measure>"
    )
    assert tokens[5:7] == ["backup", "half"] and tokens[7:9] == ["backup", "quarter"]


def test_unencodable_elements_are_skipped_with_a_warning():
    note = f"{NOTE}<type>quarter</type><stem>double</stem></note>"
    with pytest.warns(UserWarning, match="stem"):
        tokens = encode_xml(f"<measure>{HEAD}{note}</measure>")
    assert tokens == ["part:start", "part:instrument:Piano", "measure", "part:end"]


def test_unpitched_notes_are_not_encoded():
    drum = "<note><unpitched><display-step>C</display-step><display-octave>5</display-octave></unpitched><duration>1</duration><type>quarter</type></note>"
    tokens = encode_xml(f"<measure>{HEAD}{drum}</measure>")
    assert tokens == ["part:start", "part:instrument:Piano", "part:end"]


def test_grace_notes_can_be_excluded():
    grace = "<note><grace slash='yes'/><pitch><step>D</step><octave>4</octave></pitch><type>eighth</type></note>"
    measure = f"<measure>{HEAD}{grace}{NOTE}<type>quarter</type></note></measure>"
    assert "grace:slash" in encode_xml(measure)
    xml = f"""<score-partwise><part-list><score-part id="P1"/></part-list>
        <part id="P1">{measure}</part></score-partwise>"""
    config = TokenizerConfig(add_bos_eos=False, include_grace_notes=False)
    assert (
        "grace" not in MXLTokenizer(config).encode(read_musicxml(xml.encode())).tokens
    )


@pytest.mark.parametrize(
    ("duration", "types", "exact"),
    [
        (Fraction(3), ["half", "quarter"], True),
        (Fraction(7, 2), ["half", "quarter", "eighth"], True),
        (Fraction(1, 3), ["16th", "64th", "256th", "1024th"], False),
        (Fraction(0), [], True),
    ],
)
def test_split_duration(duration, types, exact):
    assert split_duration(duration) == (types, exact)


@pytest.mark.parametrize(
    ("bpm", "quantized"), [(119, 120), (123, 120), (45, 40), (300, 208)]
)
def test_quantize_tempo(bpm, quantized):
    assert quantize_tempo(bpm) == quantized
