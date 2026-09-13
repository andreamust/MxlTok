"""Datasets and collator (skipped without torch)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mxltok import MXL  # noqa: E402
from mxltok.pytorch_data import DataCollator, DatasetJSON, DatasetMXL  # noqa: E402
from tests.conftest import FIXTURES  # noqa: E402

SMALL = [p for p in FIXTURES if p.stat().st_size < 60_000][:3]


@pytest.fixture(scope="module")
def tokenizer() -> MXL:
    return MXL()


def test_dataset_mxl(tokenizer: MXL):
    bos, eos = tokenizer["BOS_None"], tokenizer["EOS_None"]
    dataset = DatasetMXL(
        SMALL, tokenizer, max_seq_len=64, bos_token_id=bos, eos_token_id=eos
    )
    assert len(dataset) == 3
    sample = dataset[0]["input_ids"]
    assert sample.dtype == torch.int64 and len(sample) == 64
    assert sample[0] == bos and sample[-1] != eos  # truncated: no EOS
    full = DatasetMXL(
        SMALL, tokenizer, max_seq_len=100_000, bos_token_id=bos, eos_token_id=eos
    )[0]["input_ids"]
    assert full[-1] == eos and len(full) == len(tokenizer.encode(SMALL[0])) + 2
    labelled = DatasetMXL(
        SMALL,
        tokenizer,
        64,
        pre_tokenize=True,
        func_to_get_labels=lambda score, seq, path: len(score.parts),
    )
    assert len(labelled) == 3 and labelled[1]["labels"].item() == len(
        tokenizer.decode(tokenizer.encode(SMALL[1])).parts
    )


def test_dataset_mxl_skips_bad_files(tokenizer: MXL, tmp_path: Path):
    bad = tmp_path / "bad.musicxml"
    bad.write_text("<nope/>")
    with pytest.warns(UserWarning):
        assert DatasetMXL([bad], tokenizer, 16)[0]["input_ids"] is None
    with pytest.warns(UserWarning):
        assert len(DatasetMXL([bad, SMALL[0]], tokenizer, 16, pre_tokenize=True)) == 1


def test_dataset_json_and_collator(tokenizer: MXL, tmp_path: Path):
    tokenizer.tokenize_dataset(SMALL, tmp_path, parallel_workers_size=1)
    files = sorted(p for p in tmp_path.glob("*.json") if p.name != "tokenizer.json")
    dataset = DatasetJSON(files, max_seq_len=50, bos_token_id=tokenizer["BOS_None"])
    assert len(dataset) == 3 and len(dataset[0]["input_ids"]) == 50
    batch = [
        {"input_ids": torch.LongTensor([1, 2, 3])},
        {"input_ids": torch.LongTensor([4, 5])},
    ]
    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    out = collator(batch)
    assert out["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert out["labels"].tolist() == [[1, 2, 3], [4, 5, -100]]
    left = DataCollator(
        tokenizer.pad_token_id,
        pad_on_left=True,
        copy_inputs_as_labels=True,
        shift_labels=True,
    )(batch)
    assert left["input_ids"].tolist() == [[1, 2], [0, 4]]
    assert left["labels"].tolist() == [[2, 3], [4, 5]]
    cls = DataCollator(0)(
        [{"input_ids": torch.LongTensor([1]), "labels": torch.tensor(2)}]
    )
    assert cls["labels"].tolist() == [2]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collator)
    first = next(iter(loader))
    assert first["input_ids"].shape == (2, 50)
