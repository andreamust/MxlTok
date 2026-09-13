"""Transposition on the IR and interval pitch mode."""

from pathlib import Path

import pytest

from mxltok import MXL, TokenizerConfig, transpose
from mxltok.constants import STEP_SEMITONES
from mxltok.musicxml import load
from mxltok.score import Attributes, Harmony, Note
from mxltok.utils import transpose_key
from tests.conftest import FIXTURES
from tests.test_encode import _structure
from tests.test_musicxml import FULL_XML

SMALL = [p for p in FIXTURES if p.stat().st_size < 60_000]


def _midi_numbers(score) -> list[int]:
    return [
        STEP_SEMITONES[e.step] + int(e.alter) + 12 * e.octave
        for p in score.parts
        for m in p.measures
        for e in m.elements
        if isinstance(e, Note) and not e.rest and not e.unpitched
    ]


def _keys(score) -> list[int]:
    return [
        e.key
        for p in score.parts
        for m in p.measures
        for e in m.elements
        if isinstance(e, Attributes) and e.key is not None
    ]


def test_transpose_key():
    assert transpose_key(0, 7) == 1 and transpose_key(0, 5) == -1
    assert (
        transpose_key(-1, 1) == -6
        and transpose_key(5, 2) == 7
        and transpose_key(6, 1) == 1
    )
    assert transpose_key(3, 12) == 3 and transpose_key(3, -12) == 3


@pytest.mark.parametrize("semitones", [-6, -1, 1, 3, 6])
def test_transpose_shifts_pitches_and_roundtrips(semitones: int, fixture_path: Path):
    if fixture_path not in SMALL:
        pytest.skip("slow fixture")
    original = load(fixture_path)
    up = transpose(original, semitones)
    assert _midi_numbers(up) == [n + semitones for n in _midi_numbers(original)]
    assert [transpose_key(k, semitones) for k in _keys(original)] == _keys(up)
    back = transpose(up, -semitones)
    assert _midi_numbers(back) == _midi_numbers(original)
    assert all(
        (a - b) % 12 == 0 for a, b in zip(_keys(back), _keys(original), strict=True)
    )
    # the transposed notation is consistent: tokens decode to the same alters
    tok = MXL()
    decoded = tok.decode(tok.encode(up, strict=True), strict=True)
    assert _midi_numbers(decoded) == _midi_numbers(up)
    assert transpose(original, 0) == original


def test_transpose_harmony_and_spelling():
    score = transpose(load(FULL_XML.encode()), 1)  # F major -> Gb major (flats)
    harmony = next(
        e for e in score.parts[0].measures[0].elements if isinstance(e, Harmony)
    )
    assert (harmony.root, harmony.root_alter, harmony.bass, harmony.bass_alter) == (
        "D",
        -1,
        "E",
        0,
    )
    assert _keys(score) == [-6]
    notes = [
        e
        for e in score.parts[0].measures[0].elements
        if isinstance(e, Note) and not e.rest
    ]
    assert (notes[0].step, notes[0].alter, notes[0].accidental) == (
        "D",
        -1,
        None,
    )  # Db in Gb major
    sharp_side = transpose(load(FULL_XML.encode()), 2)  # G major: sharps
    notes = [
        e
        for e in sharp_side.parts[0].measures[0].elements
        if isinstance(e, Note) and not e.rest
    ]
    assert (notes[0].step, notes[0].alter) == ("D", 0)


@pytest.fixture(scope="module")
def interval_tokenizer() -> MXL:
    return MXL(TokenizerConfig(pitch_mode="interval", max_pitch_interval=12))


def test_interval_vocab_and_tokens(interval_tokenizer: MXL):
    tok = interval_tokenizer
    assert len(tok.token_ids_of_type("Interval")) == 25 * 6
    assert "Interval_+0" in tok.vocab and "Interval_-12bb" in tok.vocab
    tokens = tok.encode(load(FULL_XML.encode()), strict=True).tokens
    assert (
        "Pitch_C4n" in tokens and "Interval_+2b" in tokens and "Interval_-1" in tokens
    )
    assert "Interval_+2#" in tokens
    assert tok.tokens_errors(tokens) == 0.0


def test_interval_fixpoint(interval_tokenizer: MXL, fixture_path: Path):
    if fixture_path not in SMALL:
        pytest.skip("slow fixture")
    original = load(fixture_path)
    seq = interval_tokenizer.encode(original, strict=True)
    decoded = interval_tokenizer.decode(seq, strict=True)
    assert interval_tokenizer.encode(decoded, strict=True).tokens == seq.tokens
    assert _structure(decoded) == _structure(original)


def test_interval_beyond_max_is_absolute():
    tok = MXL(TokenizerConfig(pitch_mode="interval", max_pitch_interval=1))
    tokens = tok.encode(load(FULL_XML.encode()), strict=True).tokens
    # C4 -> Eb4 (+2) and D4 -> F#4 (+2) exceed the limit: absolute; the -1 steps stay
    assert "Pitch_E4b" in tokens and "Pitch_F4#" in tokens
    assert "Interval_+2b" not in tokens and tokens.count("Interval_-1") == 2
    (part,) = tok.decode(tokens, strict=True).parts
    notes = [e for e in part.measures[0].elements if isinstance(e, Note) and not e.rest]
    assert [(n.step, n.octave) for n in notes[:4]] == [
        ("C", 4),
        ("E", 4),
        ("D", 4),
        ("F", 4),
    ]


def test_interval_without_previous_pitch(interval_tokenizer: MXL):
    from mxltok import MxlTokError

    tokens = ["Part_Piano", "Bar_None", "Interval_+2", "Duration_quarter"]
    with pytest.raises(MxlTokError):
        interval_tokenizer.decode(tokens, strict=True)
    with pytest.warns(UserWarning):
        score = interval_tokenizer.decode(tokens)
    note = score.parts[0].measures[0].elements[0]
    assert isinstance(note, Note) and note.rest
