"""BPE / Unigram / WordPiece training on top of the base vocabulary."""

from pathlib import Path

import pytest

from mxltok import MXL, TokenizerConfig
from mxltok.utils import TokTrainingIterator
from tests.conftest import FIXTURES

SMALL = [p for p in FIXTURES if p.stat().st_size < 100_000]


def _check_trained(tok: MXL, path: Path) -> None:
    assert tok.is_trained
    base = len(tok.vocab)
    assert len(tok) > base and tok.vocab_model is not None
    seq = tok.encode(path)
    assert seq.are_ids_encoded and max(seq.ids) >= base
    plain = tok.encode(path, encode_ids=False)
    assert not plain.are_ids_encoded and len(seq) < len(plain)
    assert tok.tokens_errors(seq) == 0.0
    decoded = tok.decode(seq, strict=True)
    assert tok.encode(decoded, encode_ids=False).tokens == plain.tokens
    copy = tok.encode(path)
    tok.decode_token_ids(copy)
    assert copy.tokens == plain.tokens and copy.ids == plain.ids


@pytest.mark.parametrize("split", ["bar", "no"])
def test_bpe(tmp_path: Path, split: str):
    tok = MXL(TokenizerConfig(encode_ids_split=split))
    tok.train(vocab_size=len(tok.vocab) + 200, files_paths=SMALL)
    _check_trained(tok, SMALL[0])
    tok.save(tmp_path)
    loaded = MXL(params=tmp_path / "tokenizer.json")
    assert loaded == tok and loaded.is_trained
    assert loaded.encode(SMALL[1]).ids == tok.encode(SMALL[1]).ids
    assert loaded.decode(tok.encode(SMALL[1]).ids, strict=True) == tok.decode(
        tok.encode(SMALL[1]), strict=True
    )
    tok.decode_token_ids(seqs := [tok.encode(p) for p in SMALL[:2]])
    assert all(not s.are_ids_encoded for s in seqs)
    tok.encode_token_ids(seqs)
    assert all(s.are_ids_encoded for s in seqs)


@pytest.mark.parametrize("model", ["Unigram", "WordPiece"])
def test_other_models(model: str):
    tok = MXL()
    tok.train(vocab_size=len(tok.vocab) + 100, model=model, files_paths=SMALL)
    _check_trained(tok, SMALL[0])


def test_retrain_and_guards():
    tok = MXL()
    with pytest.raises(ValueError):
        tok.train(vocab_size=10_000)
    with pytest.warns(UserWarning, match="skipping"):
        tok.train(vocab_size=10, files_paths=SMALL)
    assert not tok.is_trained
    iterator = TokTrainingIterator(tok, SMALL[:3])
    assert len(iterator) == 3 and all(isinstance(c, str) for c in iterator[0])
    assert iterator[0] == iterator.load_file(SMALL[0])
    tok.train(vocab_size=len(tok.vocab) + 50, iterator=iterator)
    first = len(tok)
    tok.train(
        vocab_size=len(tok.vocab) + 80, files_paths=SMALL[:3]
    )  # retrain, same model
    assert len(tok) > first
    with pytest.raises(ValueError):
        MXL().train(vocab_size=1000, model="Foo", files_paths=SMALL)
