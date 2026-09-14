import json
import warnings

import pytest
from conftest import ENCODABLE, FIXTURES, GOLDEN

from mxltok import MXLTokenizer, MxlTokError, TokenizerConfig

BPE_IDS = json.loads((GOLDEN / "bpe_ids.json").read_text())
MXLGPT_SAVES = [FIXTURES / "mxlgpt_checkpoint", FIXTURES / "mxlgpt_dataset"]


@pytest.fixture(scope="module")
def bpe_tokenizer() -> MXLTokenizer:
    return MXLTokenizer.from_pretrained(FIXTURES / "mxlgpt_checkpoint")


@pytest.mark.parametrize("path", MXLGPT_SAVES, ids=lambda p: p.name)
def test_mxlgpt_tokenizers_load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # legacy config keys load silently
        tokenizer = MXLTokenizer.from_pretrained(path)
    assert tokenizer.config.arrangement == "measure_first"
    assert not tokenizer.config.include_grace_notes  # as mxlGPT preprocessing did
    assert tokenizer.base_vocab_size == 720
    assert tokenizer.vocab_size == 900


@pytest.mark.parametrize("path", MXLGPT_SAVES, ids=lambda p: p.name)
def test_mxlgpt_bpe_ids_are_reproduced(path):
    tokenizer = MXLTokenizer.from_pretrained(path)
    for score in ENCODABLE:
        assert tokenizer.encode(score).ids == BPE_IDS[score.name], score.name


def test_bpe_ids_decode_to_base_tokens(bpe_tokenizer):
    sequence = bpe_tokenizer.encode(FIXTURES / "mxl" / "860.mxl")
    assert len(sequence.ids) < len(sequence.tokens)
    assert bpe_tokenizer.ids_to_tokens(sequence.ids) == sequence.tokens
    assert bpe_tokenizer.decode(sequence.ids).parts


def test_special_token_ids_with_bpe(bpe_tokenizer):
    special_ids = (bpe_tokenizer.pad_id, bpe_tokenizer.bos_id, bpe_tokenizer.eos_id)
    assert special_ids == (0, 1, 2)


@pytest.mark.parametrize("pretrained", [False, True])
def test_save_and_load(tmp_path, bpe_tokenizer, pretrained):
    tokenizer = (
        bpe_tokenizer
        if pretrained
        else MXLTokenizer(TokenizerConfig(pitch_mode="interval"))
    )
    path = tokenizer.save_pretrained(tmp_path)
    assert path == tmp_path / "tokenizer.json"
    for source in (tmp_path, path):
        restored = MXLTokenizer.from_pretrained(source)
        assert restored.to_dict() == tokenizer.to_dict()


def test_train_bpe():
    tokenizer = MXLTokenizer()
    tokenizer.train_bpe((tokenizer.encode(s).tokens for s in ENCODABLE), vocab_size=800)
    assert tokenizer.vocab_size == 800
    sequence = tokenizer.encode(ENCODABLE[0])
    assert tokenizer.ids_to_tokens(sequence.ids) == sequence.tokens
    assert tokenizer.bos_id == 1


def test_from_pretrained_rejects_other_tokenizers(tmp_path):
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"tokenizer_class": "OctupleTokenizer"})
    )
    with pytest.raises(MxlTokError, match="Octuple"):
        MXLTokenizer.from_pretrained(tmp_path)


def test_from_pretrained_rejects_empty_directories(tmp_path):
    with pytest.raises(MxlTokError, match="No tokenizer"):
        MXLTokenizer.from_pretrained(tmp_path)


def test_decode_accepts_array_likes():
    class Tensor:  # mimics torch.Tensor and numpy.ndarray
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    tokenizer = MXLTokenizer()
    sequence = tokenizer.encode(ENCODABLE[0])
    assert tokenizer.decode(Tensor(sequence.ids)) == tokenizer.decode(sequence.ids)
