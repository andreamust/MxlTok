"""MusicXML reader/writer: fixtures round-trip through the IR without loss."""

import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction
from pathlib import Path

import pytest

from mxltok.classes import MxlTokError
from mxltok.musicxml import dump, load
from mxltok.score import Attributes, Backup, Barline, Direction, Harmony, Note

FULL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
      <score-instrument id="P1-I1">
        <instrument-name>Piano</instrument-name>
      </score-instrument>
      <midi-instrument id="P1-I1">
        <midi-channel>1</midi-channel>
        <midi-program>1</midi-program>
      </midi-instrument>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <barline location="left">
        <bar-style>heavy-light</bar-style>
        <repeat direction="forward" />
      </barline>
      <attributes>
        <divisions>12</divisions>
        <key>
          <fifths>-1</fifths>
        </key>
        <time>
          <beats>3</beats>
          <beat-type>4</beat-type>
        </time>
        <staves>2</staves>
        <clef number="1">
          <sign>G</sign>
          <line>2</line>
        </clef>
        <clef number="2">
          <sign>F</sign>
          <line>4</line>
        </clef>
      </attributes>
      <direction>
        <direction-type>
          <metronome>
            <beat-unit>quarter</beat-unit>
            <per-minute>120</per-minute>
          </metronome>
        </direction-type>
        <sound tempo="120" />
      </direction>
      <direction>
        <direction-type>
          <dynamics>
            <mp />
          </dynamics>
        </direction-type>
      </direction>
      <harmony>
        <root>
          <root-step>C</root-step>
        </root>
        <kind>major-seventh</kind>
        <bass>
          <bass-step>E</bass-step>
          <bass-alter>-1</bass-alter>
        </bass>
      </harmony>
      <note>
        <pitch>
          <step>C</step>
          <octave>4</octave>
        </pitch>
        <duration>12</duration>
        <tie type="start" />
        <voice>1</voice>
        <type>quarter</type>
        <accidental>natural</accidental>
        <stem>up</stem>
        <staff>1</staff>
        <beam number="1">begin</beam>
        <notations>
          <tied type="start" />
          <slur type="start" number="1" />
          <articulations>
            <staccato />
          </articulations>
          <ornaments>
            <trill-mark />
            <tremolo type="single">3</tremolo>
          </ornaments>
        </notations>
      </note>
      <note>
        <chord />
        <pitch>
          <step>E</step>
          <alter>-1</alter>
          <octave>4</octave>
        </pitch>
        <duration>12</duration>
        <voice>1</voice>
        <type>quarter</type>
        <accidental>flat</accidental>
        <stem>up</stem>
        <staff>1</staff>
      </note>
      <note>
        <grace slash="yes" />
        <pitch>
          <step>D</step>
          <octave>4</octave>
        </pitch>
        <voice>1</voice>
        <type>16th</type>
        <stem>up</stem>
        <staff>1</staff>
      </note>
      <note>
        <pitch>
          <step>F</step>
          <alter>1</alter>
          <octave>4</octave>
        </pitch>
        <duration>4</duration>
        <tie type="stop" />
        <voice>1</voice>
        <type>eighth</type>
        <accidental>sharp</accidental>
        <time-modification>
          <actual-notes>3</actual-notes>
          <normal-notes>2</normal-notes>
        </time-modification>
        <stem>up</stem>
        <staff>1</staff>
        <beam number="1">end</beam>
        <notations>
          <tied type="stop" />
          <slur type="stop" number="1" />
          <tuplet type="start" />
          <fermata />
          <arpeggiate />
        </notations>
      </note>
      <note print-object="no">
        <rest />
        <duration>9</duration>
        <voice>1</voice>
        <type>eighth</type>
        <dot />
        <staff>1</staff>
      </note>
      <direction>
        <direction-type>
          <wedge type="crescendo" />
        </direction-type>
      </direction>
      <direction>
        <direction-type>
          <pedal type="start" />
        </direction-type>
      </direction>
      <backup>
        <duration>24</duration>
      </backup>
      <note>
        <rest measure="yes" />
        <duration>36</duration>
        <voice>5</voice>
        <staff>2</staff>
      </note>
      <forward>
        <duration>3</duration>
      </forward>
      <note>
        <unpitched>
          <display-step>E</display-step>
          <display-octave>4</display-octave>
        </unpitched>
        <duration>3</duration>
        <voice>5</voice>
        <type>16th</type>
        <staff>2</staff>
        <notations>
          <glissando type="start" />
        </notations>
      </note>
      <barline location="right">
        <bar-style>light-heavy</bar-style>
        <repeat direction="backward" />
      </barline>
    </measure>
  </part>
</score-partwise>
"""

TIMEWISE_XML = """<score-timewise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>A</part-name></score-part>
    <score-part id="P2"><part-name>B</part-name></score-part>
  </part-list>
  <measure number="1">
    <part id="P1">
      <attributes><divisions>1</divisions></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>whole</type></note>
    </part>
    <part id="P2">
      <attributes><divisions>2</divisions></attributes>
      <note><rest/><duration>8</duration><type>whole</type></note>
    </part>
  </measure>
  <measure number="2">
    <part id="P1">
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><type>whole</type></note>
    </part>
    <part id="P2">
      <note><rest/><duration>8</duration><type>whole</type></note>
    </part>
  </measure>
</score-timewise>
"""


def _canonical(xml: str) -> str:
    return ET.canonicalize(xml, strip_text=True)


def test_load_all_fixtures_have_parts_and_notes(fixture_path: Path):
    score = load(fixture_path)
    assert score.parts
    notes = [
        e
        for part in score.parts
        for m in part.measures
        for e in m.elements
        if isinstance(e, Note)
    ]
    assert notes
    assert all(part.measures for part in score.parts)


def test_ir_fixpoint(fixture_path: Path):
    score = load(fixture_path)
    assert load(dump(score).encode()) == score


def test_full_document_canonical_roundtrip():
    score = load(FULL_XML.encode())
    assert _canonical(dump(score)) == _canonical(FULL_XML)


def test_full_document_ir_content():
    (part,) = load(FULL_XML.encode()).parts
    assert (part.name, part.midi_program, part.midi_channel) == ("Piano", 1, 1)
    elements = part.measures[0].elements
    assert elements[0] == Barline("left", "forward", "heavy-light")
    attrs = elements[1]
    assert isinstance(attrs, Attributes)
    assert (attrs.key, attrs.time, attrs.staves) == (-1, (3, 4), 2)
    assert [(c.sign, c.line, c.number) for c in attrs.clefs] == [
        ("G", 2, 1),
        ("F", 4, 2),
    ]
    assert elements[2:4] == [Direction("tempo", "120"), Direction("dynamics", "mp")]
    assert elements[4] == Harmony("C", 0, "major-seventh", "E", -1)
    first = elements[5]
    assert isinstance(first, Note)
    assert (first.step, first.octave, first.duration) == ("C", 4, Fraction(1))
    assert first.ties == ("start",) and first.slurs == ("start",)
    assert first.beams == ("begin",) and first.articulations == ("staccato",)
    assert first.ornaments == ("trill-mark", "tremolo:single:3")
    chord = elements[6]
    assert isinstance(chord, Note) and chord.chord and chord.alter == -1
    grace = elements[7]
    assert isinstance(grace, Note) and grace.grace and grace.grace_slash
    assert grace.duration == 0
    tuplet = elements[8]
    assert isinstance(tuplet, Note)
    assert tuplet.time_mod == (3, 2) and tuplet.duration == Fraction(1, 3)
    assert tuplet.notations == ("fermata", "arpeggiate")
    hidden = elements[9]
    assert isinstance(hidden, Note) and hidden.rest and not hidden.print_object
    assert hidden.dots == 1 and hidden.duration == Fraction(3, 4)
    assert elements[12] == Backup(Fraction(2))
    measure_rest = elements[13]
    assert isinstance(measure_rest, Note) and measure_rest.rest_measure
    unpitched = elements[15]
    assert isinstance(unpitched, Note) and unpitched.unpitched
    assert unpitched.notations == ("glissando-start",)
    assert elements[16] == Barline("right", "backward", "light-heavy")


def test_timewise_is_converted_to_partwise():
    score = load(TIMEWISE_XML.encode())
    assert [p.id for p in score.parts] == ["P1", "P2"]
    assert [len(p.measures) for p in score.parts] == [2, 2]
    second = score.parts[1].measures[1].elements[0]
    assert isinstance(second, Note) and second.rest and second.duration == 4


def test_mxl_write_and_read(tmp_path: Path):
    score = load(FULL_XML.encode())
    out = tmp_path / "out.mxl"
    dump(score, out)
    with zipfile.ZipFile(out) as archive:
        assert "META-INF/container.xml" in archive.namelist()
    assert load(out) == score


def test_musicxml_write_to_path(tmp_path: Path):
    score = load(FULL_XML.encode())
    out = tmp_path / "out.musicxml"
    text = dump(score, out)
    assert out.read_text() == text
    assert load(out) == score


def test_divisions_inserted_when_first_measure_has_no_attributes():
    xml = (
        '<score-partwise><part-list><score-part id="P1"><part-name/></score-part>'
        '</part-list><part id="P1"><measure number="1"><note><pitch><step>C</step>'
        "<octave>4</octave></pitch><duration>1</duration></note></measure></part>"
        "</score-partwise>"
    )
    score = load(xml.encode())
    assert load(dump(score).encode()) == score
    assert "<divisions>1</divisions>" in dump(score)


def test_bad_root_raises():
    with pytest.raises(MxlTokError):
        load(b"<opus/>")


def test_bad_integer_raises():
    with pytest.raises(MxlTokError):
        load(
            b'<score-partwise><part id="P1"><measure><note><duration>x</duration>'
            b"</note></measure></part></score-partwise>"
        )
