"""Part-first vs measure-first serialization."""

from pathlib import Path

import pytest

from mxltok import MXL, TokenizerConfig
from mxltok.musicxml import load
from tests.test_encode import _structure

TWO_PART_XML = """<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Violin</part-name></score-part>
    <score-part id="P2"><part-name>Cello</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note>
    </measure>
    <measure number="3">
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>F</sign><line>4</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>3</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note>
    </measure>
  </part>
</score-partwise>
"""


@pytest.fixture(scope="module")
def part_first() -> MXL:
    return MXL()


@pytest.fixture(scope="module")
def measure_first() -> MXL:
    return MXL(TokenizerConfig(arrangement="measure_first"))


def test_part_first_layout(part_first: MXL):
    tokens = part_first.encode(load(TWO_PART_XML.encode()), strict=True).tokens
    assert tokens.count("Part_Strings") == 2
    parts = [i for i, t in enumerate(tokens) if t == "Part_Strings"]
    bars = [i for i, t in enumerate(tokens) if t == "Bar_None"]
    assert len(bars) == 5
    assert bars[:3] < [parts[1]] * 3 < bars[3:]  # 3 bars of part 1 before part 2


def test_measure_first_layout(measure_first: MXL):
    tokens = measure_first.encode(load(TWO_PART_XML.encode()), strict=True).tokens
    assert tokens[0] == "Bar_None"
    bars = [i for i, t in enumerate(tokens) if t == "Bar_None"]
    assert len(bars) == 3
    for start, end in zip(bars, [*bars[1:], len(tokens)], strict=True):
        assert tokens[start:end].count("Part_Strings") == 2
    assert (
        tokens[-2:] == ["Part_Strings", "Part_Strings"] or tokens[-1] == "Part_Strings"
    )
    assert tokens[bars[0] + 1 : bars[0] + 6] == [
        "Part_Strings",
        "Key_0",
        "TimeSig_4/4",
        "Clef_G2",
        "Pitch_C5",
    ]


def test_measure_first_roundtrip_matches_part_first(
    part_first: MXL, measure_first: MXL
):
    score = load(TWO_PART_XML.encode())
    via_mf = measure_first.decode(measure_first.encode(score, strict=True), strict=True)
    via_pf = part_first.decode(part_first.encode(score, strict=True), strict=True)
    assert via_mf == via_pf
    assert [len(p.measures) for p in via_mf.parts] == [3, 2]
    assert _structure(via_mf) == _structure(score)


def test_measure_first_fixtures(
    part_first: MXL, measure_first: MXL, fixture_path: Path
):
    score = load(fixture_path)
    seq = measure_first.encode(score, strict=True)
    assert measure_first.tokens_errors(seq) == 0.0
    decoded = measure_first.decode(seq, strict=True)
    assert measure_first.encode(decoded, strict=True).tokens == seq.tokens
    assert decoded == part_first.decode(
        part_first.encode(score, strict=True), strict=True
    )


def test_measure_first_split_edge_cases(measure_first: MXL):
    split = measure_first._split_parts_measure_first
    # no leading Bar, missing Part token, parts of different lengths
    tokens = [
        "Pitch_C4",
        "Duration_quarter",
        "Bar_None",
        "Part_Piano",
        "Rest_None",
        "Duration_half",
        "Part_Voice",
        "Pitch_D4",
        "Duration_quarter",
        "Bar_None",
        "Part_Piano",
        "Bar_None",
        "Part_Piano",
        "Part_Voice",
    ]
    assert split(tokens) == [
        ("Piano", [["Pitch_C4", "Duration_quarter"], ["Rest_None", "Duration_half"]]),
        ("Voice", [[], ["Pitch_D4", "Duration_quarter"]]),
    ]
    assert split([]) == []
    assert split(["Bar_None"]) == []
    assert split(["Part_Piano", "Pitch_C4", "Duration_quarter"]) == [
        ("Piano", [["Pitch_C4", "Duration_quarter"]])
    ]


def test_part_first_split_edge_cases(part_first: MXL):
    split = part_first._split_parts_part_first
    assert split(
        ["Pitch_C4", "Duration_quarter", "Bar_None", "Rest_None", "Duration_whole"]
    ) == [(None, [["Pitch_C4", "Duration_quarter"], ["Rest_None", "Duration_whole"]])]
    assert split(["Part_Piano", "Bar_None", "Bar_None", "Part_Voice"]) == [
        ("Piano", [[], []]),
        ("Voice", []),
    ]


def test_max_parts(measure_first: MXL):
    tok = MXL(TokenizerConfig(max_parts=1))
    tokens = tok.encode(load(TWO_PART_XML.encode()), strict=True).tokens
    assert tokens.count("Part_Strings") == 1 and tokens.count("Bar_None") == 3
