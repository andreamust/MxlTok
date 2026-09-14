import pytest
from conftest import FIXTURES, golden_tokens

from mxltok import (
    MXLTokenizer,
    MxlTokError,
    TokenizerConfig,
    read_musicxml,
    write_musicxml,
)
from mxltok.decoding import note_duration, number_beams
from mxltok.score import Attributes, Barline, Note


def round_trip(tokenizer, tokens):
    """Decode tokens, write MusicXML, read it back and encode it again."""
    xml = write_musicxml(tokenizer.decode(tokens))
    return tokenizer.encode(read_musicxml(xml.encode())).tokens


def without_backward_repeats(tokens):
    return [token for token in tokens if token != "repeat:backward"]


def backward_repeat_measures(tokens):
    """Return the index of the measure token after each backward repeat."""
    measures = [tokens[: i + 1].count("measure") for i, t in enumerate(tokens)]
    return [measures[i] for i, token in enumerate(tokens) if token == "repeat:backward"]


@pytest.mark.parametrize("arrangement", ["part_first", "measure_first"])
def test_round_trip_matches_mxlgpt(score_path, arrangement):
    """Decoding agrees with mxlGPT, except for one bug fix.

    mxlGPT decoded a backward repeat into the measure before its own one:
    MxlTok keeps it where the encoder put it.
    """
    tokenizer = MXLTokenizer(
        TokenizerConfig(arrangement=arrangement, add_bos_eos=False)
    )
    tokens = golden_tokens(f"{score_path.stem}.{arrangement}.lmx")
    expected = golden_tokens(f"{score_path.stem}.{arrangement}.roundtrip.lmx")
    actual = round_trip(tokenizer, tokens)
    assert without_backward_repeats(actual) == without_backward_repeats(expected)
    assert backward_repeat_measures(actual) == backward_repeat_measures(tokens)


def test_round_trip_is_stable(score_path):
    """Decoding is lossless on what it decodes: a second round trip is a no-op."""
    tokenizer = MXLTokenizer(TokenizerConfig(add_bos_eos=False))
    once = round_trip(tokenizer, tokenizer.encode(score_path).tokens)
    assert round_trip(tokenizer, once) == once


def test_ids_and_tokens_decode_alike():
    tokenizer = MXLTokenizer()
    sequence = tokenizer.encode(FIXTURES / "mxl" / "860.mxl")
    from_ids = write_musicxml(tokenizer.decode(sequence.ids))
    assert from_ids == write_musicxml(tokenizer.decode(sequence.tokens))
    assert from_ids == write_musicxml(tokenizer.decode(sequence))


def test_decoded_parts_carry_their_instrument():
    tokenizer = MXLTokenizer()
    score = tokenizer.decode(
        [
            "part:start",
            "part:instrument:Drums",
            "measure",
            "rest",
            "rest:measure",
            "part:end",
        ]
    )
    (part,) = score.parts
    assert (part.name, part.midi_channel) == ("Drums", 10)
    (rest, barline) = part.measures[0].elements
    assert isinstance(rest, Note) and rest.measure_rest
    assert isinstance(barline, Barline) and barline.style == "light-heavy"


def test_malformed_groups_warn_or_raise():
    tokenizer = MXLTokenizer()
    two_voices = ["measure", "C4", "voice:1", "voice:2", "quarter"]
    with pytest.warns(UserWarning, match="Conflicting"):
        score = tokenizer.decode(two_voices)
    assert score.parts[0].measures[0].elements == [
        Barline("right", None, "light-heavy")
    ]
    with pytest.raises(MxlTokError, match="Conflicting"):
        tokenizer.decode(two_voices, strict=True)


def test_tokens_before_the_first_measure_form_a_measure():
    tokenizer = MXLTokenizer()
    score = tokenizer.decode(["C4", "quarter", "measure", "D4", "quarter"])
    assert len(score.parts[0].measures) == 2


def test_measure_first_tolerates_missing_part_markers():
    tokenizer = MXLTokenizer(TokenizerConfig(arrangement="measure_first"))
    tokens = [
        "part:start", "part:instrument:Piano", "key:fifths:2", "C4", "whole",
        "measure", "part:start", "D4", "whole", "part:end",
        "part:start", "part:instrument:Strings", "E4", "whole",
    ]  # fmt: skip
    score = tokenizer.decode(tokens)
    assert [part.name for part in score.parts] == ["Piano", "Strings"]
    # The first measure of the second part inherits the key of the first part.
    first_measure = score.parts[1].measures[0].elements
    assert any(isinstance(e, Attributes) and e.key == 2 for e in first_measure)


def test_note_duration():
    assert note_duration("quarter", 1, None) == 1.5
    assert note_duration("eighth", 0, "3in2") * 3 == 1


def test_number_beams():
    assert number_beams(["beam:begin"], 0) == (("begin",), 1)
    assert number_beams([], 1) == (("continue",), 1)
    assert number_beams(["beam:end", "beam:backward-hook"], 1) == (
        ("end", "backward hook"),
        0,
    )
