import zipfile

import pytest
from conftest import FIXTURES

from mxltok import MXLTokenizer, MxlTokError, read_musicxml, write_musicxml
from mxltok.score import Backup, Note


def test_writing_preserves_the_tokens(score_path, tmp_path):
    """Writing then reading a score loses nothing that LMX encodes."""
    tokenizer = MXLTokenizer()
    score = read_musicxml(score_path)
    path = tmp_path / "score.musicxml"
    write_musicxml(score, path)
    assert tokenizer.encode(path).tokens == tokenizer.encode(score).tokens


def test_compressed_files_round_trip(tmp_path):
    score = read_musicxml(FIXTURES / "mxl" / "860.mxl")
    path = tmp_path / "score.mxl"
    write_musicxml(score, path)
    assert zipfile.is_zipfile(path)
    assert read_musicxml(path) == score


def test_durations_are_in_quarters():
    xml = b"""<score-partwise><part-list/><part id="P1"><measure>
        <attributes><divisions>2</divisions></attributes>
        <note><pitch><step>C</step><octave>4</octave></pitch><duration>3</duration></note>
        <backup><duration>1</duration></backup>
    </measure></part></score-partwise>"""
    # Attributes with nothing but <divisions> are not kept.
    note, backup = read_musicxml(xml).parts[0].measures[0].elements
    assert isinstance(note, Note) and note.duration == 1.5
    assert backup == Backup(0.5)


def test_cursor_moves_without_divisions_are_dropped():
    xml = b"""<score-partwise><part id="P1"><measure>
        <backup><duration>1</duration></backup></measure></part></score-partwise>"""
    assert read_musicxml(xml).parts[0].measures[0].elements == []


def test_timewise_scores_are_converted():
    xml = b"""<score-timewise><part-list><score-part id="P1"><part-name>Flute</part-name>
        </score-part></part-list><measure number="1"><part id="P1">
        <note><rest/><duration>4</duration></note></part></measure></score-timewise>"""
    (part,) = read_musicxml(xml).parts
    assert part.name == "Flute" and len(part.measures) == 1


@pytest.mark.parametrize(
    "content",
    [
        b"<score-partwise><part",
        b"<opus/>",
        b"<score-partwise><part id='P1'><measure>"
        b"<note><pitch><step>C</step><octave>4</octave></pitch><voice>x</voice></note>"
        b"</measure></part></score-partwise>",
    ],
)
def test_invalid_files_raise(content):
    with pytest.raises(MxlTokError):
        read_musicxml(content)
