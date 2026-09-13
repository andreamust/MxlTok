"""TokenizerConfig validation/serialization and TokSequence behaviour."""

import warnings
from pathlib import Path

import pytest

from mxltok.classes import Event, TokenizerConfig, TokSequence


def test_defaults_and_special_token_formatting():
    cfg = TokenizerConfig()
    assert cfg.arrangement == "part_first"
    assert cfg.special_tokens == ("PAD_None", "BOS_None", "EOS_None", "MASK_None")
    assert cfg.additional_params == {}
    assert cfg.time_signature_range[4] == tuple(range(1, 17))
    assert len(cfg.tempos) == cfg.num_tempos
    assert cfg.tempos[0] == 40 and cfg.tempos[-1] == 250
    assert "3/4" in cfg.time_signatures


def test_pad_is_mandatory_and_duplicates_warn():
    cfg = TokenizerConfig(special_tokens=["BOS", "EOS"])
    assert cfg.special_tokens == ("BOS_None", "EOS_None", "PAD_None")
    with pytest.warns(UserWarning, match="Duplicate"):
        cfg = TokenizerConfig(special_tokens=["PAD", "PAD_None", "A_B_C"])
    assert cfg.special_tokens == ("PAD_None", "A-B_C")


def test_explicit_numerators_and_ranges():
    cfg = TokenizerConfig(time_signature_range={8: [6, 3, 12], "4": (2, 4)})
    assert cfg.time_signature_range == {8: (3, 6, 12), 4: (2, 3, 4)}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arrangement": "voice_first"},
        {"pitch_mode": "relative"},
        {"encode_ids_split": "beat"},
        {"octave_range": (3, 2)},
        {"octave_range": (0, 10)},
        {"max_voices": 0},
        {"max_parts": 0},
        {"tempo_range": (100, 50)},
        {"key_range": (-8, 7)},
        {"note_types": ["quarter", "crotchet"]},
        {"tuplet_ratios": ["3:2"]},
        {"time_signature_range": {3: (1, 4)}},
        {"time_signature_range": {4: [0]}},
    ],
)
def test_validation_errors(kwargs):
    with pytest.raises(ValueError):
        TokenizerConfig(**kwargs)


def test_use_staves_false_warns():
    with pytest.warns(UserWarning, match="use_staves"):
        TokenizerConfig(use_staves=False)


def test_json_roundtrip(tmp_path: Path):
    cfg = TokenizerConfig(
        arrangement="measure_first",
        use_harmonies=False,
        time_signature_range={4: [3, 4]},
        my_option=3,
    )
    assert cfg.additional_params == {"my_option": 3}
    path = tmp_path / "config.json"
    cfg.save_to_json(path)
    loaded = TokenizerConfig.load_from_json(path)
    assert loaded == cfg
    assert loaded.additional_params == {"my_option": 3}
    assert loaded.time_signature_range == {4: (3, 4)}
    assert TokenizerConfig.from_dict(cfg.to_dict()) == cfg
    assert TokenizerConfig.from_dict({**cfg.to_dict(), "mxltok_version": "0"}) == cfg


def test_copy_is_independent():
    cfg = TokenizerConfig()
    other = cfg.copy()
    other.additional_params["x"] = 1
    assert cfg == TokenizerConfig()
    assert cfg != other


def test_event_str():
    assert str(Event("Pitch", "C4#", time=2)) == "Pitch_C4#"


def _seq() -> TokSequence:
    tokens = [
        "Part_Piano",
        "Bar_None",
        "Pitch_C4",
        "Duration_quarter",
        "Bar_None",
        "Rest_None",
    ]
    return TokSequence(tokens=tokens, ids=list(range(len(tokens))))


def test_toksequence_len_getitem_slice():
    seq = _seq()
    assert len(seq) == 6
    assert seq[2] == 2  # ids take precedence
    assert TokSequence(tokens=["a"])[0] == "a"
    sub = seq[1:3]
    assert isinstance(sub, TokSequence)
    assert sub.tokens == ["Bar_None", "Pitch_C4"] and sub.ids == [1, 2]
    with pytest.raises(ValueError):
        TokSequence()[0]


def test_toksequence_eq_and_add():
    a, b = _seq(), _seq()
    assert a == b
    assert a != TokSequence(tokens=["x"])
    assert TokSequence(ids=[1]) != TokSequence(tokens=["x"])  # nothing in common
    total = a + b
    assert len(total) == 12 and total.tokens[6] == "Part_Piano"
    assert sum([a, b]) == total
    with pytest.raises(TypeError):
        a += [1]


def test_split_per_bars():
    chunks = _seq().split_per_bars()
    assert [c.tokens for c in chunks] == [
        ["Part_Piano"],
        ["Bar_None", "Pitch_C4", "Duration_quarter"],
        ["Bar_None", "Rest_None"],
    ]
    assert [c.ids for c in chunks] == [[0], [1, 2, 3], [4, 5]]
    with pytest.raises(ValueError):
        TokSequence(ids=[1]).split_per_bars()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TokSequence(
            events=[Event("Bar", "None"), Event("Rest", "None")]
        ).split_per_bars()
