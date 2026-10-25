import pytest

from mxltok import SpecialTokens, TokenizerConfig


def test_round_trip_through_a_dict():
    config = TokenizerConfig(
        arrangement="measure_first",
        excluded_categories={"stem"},
        token_order={"pitch": 0},
        special_tokens=SpecialTokens(pad="[PAD]"),
    )
    assert TokenizerConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "options",
    [
        {"arrangement": "column_first"},
        {"pitch_mode": "relative"},
        {"max_parts": -1},
        {"excluded_categories": {"colour"}},
        {"token_order": {"colour": 1}},
    ],
)
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        TokenizerConfig(**options)


def test_unknown_keys_warn():
    with pytest.warns(UserWarning, match="colour"):
        TokenizerConfig.from_dict({"colour": "blue"})


def test_pitch_shift_range_warns():
    with pytest.warns(UserWarning, match="transpose"):
        TokenizerConfig.from_dict({"pitch_shift_range": [-2, 2]})
