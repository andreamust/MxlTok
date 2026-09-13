"""Saving, loading, from_pretrained and dataset tokenization."""

import json
from pathlib import Path

import pytest

from mxltok import MXL, MusicTokenizer, TokenizerConfig
from tests.conftest import FIXTURES

SMALL = [p for p in FIXTURES if p.stat().st_size < 100_000]


def test_save_and_load_params(tmp_path: Path):
    tok = MXL(TokenizerConfig(arrangement="measure_first", use_harmonies=False, foo=1))
    tok.save(tmp_path)
    assert (tmp_path / "tokenizer.json").is_file()
    tok.save(tmp_path / "custom.json", additional_attributes={"note": "x"})
    assert json.loads((tmp_path / "custom.json").read_text())["note"] == "x"
    loaded = MXL(params=tmp_path / "tokenizer.json")
    assert loaded == tok
    assert loaded.config.additional_params == {"foo": 1}
    assert loaded.encode(SMALL[0]) == tok.encode(SMALL[0])
    assert MXL.from_pretrained(tmp_path) == tok
    assert MXL.from_pretrained(tmp_path / "custom.json") == tok
    assert isinstance(MusicTokenizer.from_pretrained(tmp_path), MXL)


def test_from_pretrained_class_mismatch_warns(tmp_path: Path):
    class Other(MXL):
        pass

    MXL().save(tmp_path)
    with pytest.warns(UserWarning, match="returning a MXL"):
        assert type(Other.from_pretrained(tmp_path)) is MXL


def test_save_pretrained_pushes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        MXL, "push_to_hub", lambda self, **kw: calls.append(kw) or "url"
    )
    tok = MXL()
    assert tok.save_pretrained(tmp_path / "repo") is None
    assert (tmp_path / "repo" / "tokenizer.json").is_file()
    assert (
        tok.save_pretrained(tmp_path / "repo", push_to_hub=True, private=True) == "url"
    )
    assert calls == [{"repo_id": "repo", "private": True}]


def test_save_and_load_tokens(tmp_path: Path):
    tok = MXL()
    seq = tok.encode(SMALL[0])
    tok.save_tokens(seq, tmp_path / "t.json", source="x")
    loaded = tok.load_tokens(tmp_path / "t.json")
    assert loaded.ids == seq.ids and not loaded.are_ids_encoded
    assert tok.load_tokens(tmp_path / "t.json", raw=True)["source"] == "x"
    tok.save_tokens(seq.ids, tmp_path / "ids.json")
    assert tok.load_tokens(tmp_path / "ids.json").ids == seq.ids


@pytest.mark.parametrize("workers", [1, 2])
def test_tokenize_dataset(tmp_path: Path, workers: int):
    tok = MXL()
    out = tmp_path / "tokens"
    tok.tokenize_dataset(SMALL[:3], out, parallel_workers_size=workers)
    assert (out / "tokenizer.json").is_file()
    for path in SMALL[:3]:
        assert tok.load_tokens(out / f"{path.stem}.json").ids == tok.encode(path).ids
    tok.tokenize_dataset(SMALL[:1], out, overwrite_mode=False, parallel_workers_size=1)
    assert (out / f"{SMALL[0].stem}_1.json").is_file()
    tok.tokenize_dataset(
        SMALL[:2],
        out / "filtered",
        validation_fn=lambda s: len(s.parts) > 1,
        parallel_workers_size=1,
    )
    assert len(list((out / "filtered").glob("*.json"))) == 1 + sum(
        1 for p in SMALL[:2] if len(tok.decode(tok.encode(p)).parts) > 1
    )


def test_tokenize_dataset_directory(tmp_path: Path):
    src = tmp_path / "src" / "sub"
    src.mkdir(parents=True)
    for path in SMALL[:2]:
        (src / path.name).write_bytes(path.read_bytes())
    MXL().tokenize_dataset(tmp_path / "src", tmp_path / "out", parallel_workers_size=1)
    assert sorted(p.name for p in (tmp_path / "out" / "sub").glob("*.json")) == sorted(
        f"{p.stem}.json" for p in SMALL[:2]
    )
