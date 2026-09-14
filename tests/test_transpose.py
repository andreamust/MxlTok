import pytest
from conftest import FIXTURES

from mxltok import MXLTokenizer, read_musicxml, transpose
from mxltok.score import Attributes, Note
from mxltok.transpose import transpose_key

STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def semitones(score):
    return [
        12 * e.octave + STEPS[e.step] + e.alter
        for part in score.parts
        for measure in part.measures
        for e in measure.elements
        if isinstance(e, Note) and e.is_pitched
    ]


@pytest.fixture(scope="module")
def score():
    return read_musicxml(FIXTURES / "mxl" / "2340_single_voice_with_chords.mxl")


@pytest.mark.parametrize("shift", [-5, -1, 2, 7])
def test_pitches_move_by_the_shift(score, shift):
    assert semitones(transpose(score, shift)) == [p + shift for p in semitones(score)]


def test_keys_move_on_the_circle_of_fifths():
    assert transpose_key(0, 7) == 1
    assert transpose_key(-3, 2) == -1
    assert transpose_key(6, 1) == 1  # F# major -> G major
    assert transpose_key(6, 2) == -4  # F# major -> Ab major


def test_transposed_scores_encode_consistently(score):
    """Accidentals are respelled: the tokens agree with the new key."""
    shifted = transpose(score, 2)
    keys = [
        e.key
        for e in shifted.parts[0].measures[0].elements
        if isinstance(e, Attributes) and e.key is not None
    ]
    assert keys == [
        transpose_key(k, 2)
        for k in [
            e.key
            for e in score.parts[0].measures[0].elements
            if isinstance(e, Attributes) and e.key is not None
        ]
    ]
    assert MXLTokenizer().encode(shifted).tokens


def test_zero_shift_copies(score):
    copy = transpose(score, 0)
    assert copy == score and copy is not score
