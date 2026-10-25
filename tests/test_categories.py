import pytest

from mxltok import TokenCategory, classify_token
from mxltok.categories import filter_tokens, reorder_tokens


@pytest.mark.parametrize(
    ("token", "category"),
    [
        ("part:start", TokenCategory.PART),
        ("measure", TokenCategory.MEASURE),
        ("C4#", TokenCategory.PITCH),
        ("interval:-3", TokenCategory.PITCH),
        ("quarter", TokenCategory.DURATION),
        ("3in2", TokenCategory.DURATION),
        ("rest:measure", TokenCategory.REST),
        ("tremolo:3", TokenCategory.TREMOLO),
        ("time:den:4", TokenCategory.TIME),
        ("harmony:bass:Bb", TokenCategory.HARMONY),
        ("<PAD>", TokenCategory.SPECIAL),
    ],
)
def test_classify_token(token, category):
    assert classify_token(token) == category


def test_filter_tokens():
    tokens = ["measure", "C4", "quarter", "stem:up", "beam:begin"]
    assert filter_tokens(tokens, {"stem", "beam"}) == ["measure", "C4", "quarter"]


def test_reorder_tokens_keeps_group_starts_first():
    tokens = ["measure", "quarter", "C4", "voice:1", "dot", "eighth", "D4"]
    order = {"pitch": 0, "duration": 1, "dot": 2}
    assert reorder_tokens(tokens, order) == [
        "measure", "C4", "quarter", "voice:1", "D4", "eighth", "dot",
    ]  # fmt: skip
