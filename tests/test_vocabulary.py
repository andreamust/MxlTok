import json

import pytest
from conftest import GOLDEN

from mxltok import MXLTokenizer, TokenizerConfig
from mxltok.vocabulary import build_vocabulary

REFERENCE_VOCAB = json.loads((GOLDEN / "vocab.json").read_text())


@pytest.mark.parametrize("pitch_mode", ["absolute", "interval"])
def test_vocabulary_matches_mxlgpt(pitch_mode):
    tokenizer = MXLTokenizer(TokenizerConfig(pitch_mode=pitch_mode))
    assert tokenizer.vocab == REFERENCE_VOCAB[pitch_mode]


def test_vocabulary_sizes():
    special = TokenizerConfig().special_tokens.as_list()
    assert len(build_vocabulary(special, "absolute")) == 720
    assert len(build_vocabulary(special, "interval")) == 975


def test_special_tokens_come_first():
    tokenizer = MXLTokenizer()
    ids = (
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.unk_id,
        tokenizer.mask_id,
    )
    assert ids == (0, 1, 2, 3, 4)


def test_tokens_and_ids_round_trip():
    tokenizer = MXLTokenizer()
    tokens = ["measure", "C4#", "voice:1", "quarter", "dot"]
    assert tokenizer.ids_to_tokens(tokenizer.tokens_to_ids(tokens)) == tokens


def test_unknown_token_is_rejected():
    from mxltok import MxlTokError

    with pytest.raises(MxlTokError, match="voice:99"):
        MXLTokenizer().tokens_to_ids(["voice:99"])
