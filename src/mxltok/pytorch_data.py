"""PyTorch ``Dataset``/collator helpers (requires the ``torch`` extra)."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "The `torch` extra is required to use the PyTorch dataset helpers."
    ) from exc
from torch import LongTensor, Tensor
from torch.utils.data import Dataset

from mxltok.classes import TokSequence
from mxltok.musicxml import load
from mxltok.score import Score
from mxltok.tokenizer import MusicTokenizer


def _preprocess_token_ids(
    token_ids: list[int],
    max_seq_len: int,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> list[int]:
    """Truncate to ``max_seq_len`` (BOS/EOS included) and add BOS/EOS if given."""
    if not token_ids:
        return token_ids
    reserved = sum(1 for t in (bos_token_id, eos_token_id) if t is not None)
    if len(token_ids) > max_seq_len - reserved:  # truncate; EOS is then dropped
        token_ids = token_ids[: max_seq_len - (bos_token_id is not None)]
        eos_token_id = None
    if bos_token_id is not None:
        token_ids = [bos_token_id, *token_ids]
    if eos_token_id is not None:
        token_ids = [*token_ids, eos_token_id]
    return token_ids


class DatasetMXL(Dataset[Mapping[str, Any]]):
    """Dataset tokenizing MusicXML files on the fly (or once, with ``pre_tokenize``).

    Each item is ``{"input_ids": LongTensor}`` (``None`` for unreadable files),
    plus ``"labels"`` when ``func_to_get_labels`` is given. BOS/EOS tokens are
    added to every sample when their ids are given.
    """

    def __init__(
        self,
        files_paths: Sequence[str | Path],
        tokenizer: MusicTokenizer,
        max_seq_len: int,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pre_tokenize: bool = False,
        func_to_get_labels: (
            Callable[[Score, TokSequence, Path], int | list[int] | LongTensor] | None
        ) = None,
        sample_key_name: str = "input_ids",
        labels_key_name: str = "labels",
    ) -> None:
        self.files_paths = [Path(p) for p in files_paths]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pre_tokenize = pre_tokenize
        self.func_to_get_labels = func_to_get_labels
        self.sample_key_name = sample_key_name
        self.labels_key_name = labels_key_name
        self.samples: list[Mapping[str, Any]] = []
        if pre_tokenize:
            self.samples = [self._tokenize(path) for path in self.files_paths]
            self.samples = [s for s in self.samples if s[sample_key_name] is not None]

    def _tokenize(self, path: Path) -> Mapping[str, Any]:
        try:
            score = load(path)
        except Exception as exc:
            warnings.warn(f"Skipping {path}: {exc}", stacklevel=2)
            return {self.sample_key_name: None}
        seq = self.tokenizer.encode(score)
        ids = _preprocess_token_ids(
            seq.ids, self.max_seq_len, self.bos_token_id, self.eos_token_id
        )
        item: dict[str, Any] = {self.sample_key_name: LongTensor(ids)}
        if self.func_to_get_labels is not None:
            labels = self.func_to_get_labels(score, seq, path)
            item[self.labels_key_name] = (
                labels
                if isinstance(labels, torch.Tensor)
                else LongTensor([labels]).squeeze()
            )
        return item

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        if self.pre_tokenize:
            return self.samples[index]
        return self._tokenize(self.files_paths[index])

    def __len__(self) -> int:
        return len(self.samples) if self.pre_tokenize else len(self.files_paths)


class DatasetJSON(Dataset[Mapping[str, Any]]):
    """Dataset over JSON token files written by ``tokenizer.tokenize_dataset``."""

    def __init__(
        self,
        files_paths: Sequence[str | Path],
        max_seq_len: int,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self.files_paths = [Path(p) for p in files_paths]
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        with self.files_paths[index].open() as file:
            ids = json.load(file)["ids"]
        ids = _preprocess_token_ids(
            ids, self.max_seq_len, self.bos_token_id, self.eos_token_id
        )
        return {"input_ids": LongTensor(ids)}

    def __len__(self) -> int:
        return len(self.files_paths)


class DataCollator:
    """Pad a batch of samples and build attention masks (and optionally labels).

    :param pad_token_id: id used to pad the inputs.
    :param pad_on_left: pad on the left instead of the right.
    :param copy_inputs_as_labels: add ``labels`` copied from the inputs.
    :param shift_labels: shift inputs/labels by one for teacher forcing.
    :param labels_pad_idx: padding value for labels (ignored by loss functions).
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_on_left: bool = False,
        copy_inputs_as_labels: bool = False,
        shift_labels: bool = False,
        labels_pad_idx: int = -100,
        inputs_kwarg_name: str = "input_ids",
        labels_kwarg_name: str = "labels",
        decoder_inputs_kwarg_name: str = "decoder_input_ids",
    ) -> None:
        self.pad_token = pad_token_id
        self.pad_on_left = pad_on_left
        self.copy_inputs_as_labels = copy_inputs_as_labels
        self.shift_labels = shift_labels
        self.labels_pad_idx = labels_pad_idx
        self.inputs_kwarg_name = inputs_kwarg_name
        self.labels_kwarg_name = labels_kwarg_name
        self.decoder_inputs_kwarg_name = decoder_inputs_kwarg_name

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        columns: list[list[Tensor] | None] = []
        for key in (
            self.inputs_kwarg_name,
            self.decoder_inputs_kwarg_name,
            self.labels_kwarg_name,
        ):
            if key in batch[0]:
                columns.append(
                    [
                        s[key]
                        for s in batch
                        if s[key] is not None and (s[key].dim() == 0 or len(s[key]) > 0)
                    ]
                )
            else:
                columns.append(None)
        x, x_dec, y = columns
        if y is None and self.copy_inputs_as_labels:
            y = deepcopy(x_dec if x_dec is not None else x)
        x_t = _pad_batch(x, self.pad_token, self.pad_on_left) if x is not None else None
        x_dec_t = (
            _pad_batch(x_dec, self.pad_token, self.pad_on_left)
            if x_dec is not None
            else None
        )
        y_t: Tensor | None = None
        if y is not None:
            if y[0].dim() > 0:
                y_t = _pad_batch(y, self.labels_pad_idx, self.pad_on_left)
            else:
                y_t = torch.stack(y)
        if self.shift_labels:
            if x_dec_t is not None:
                x_dec_t = x_dec_t[:, :-1]
            elif x_t is not None:
                x_t = x_t[:, :-1]
            if y_t is not None and y_t.dim() > 1:
                y_t = y_t[:, 1:]
            else:
                warnings.warn(
                    "shift_labels=True but labels are not sequences.", stacklevel=2
                )
        if x_t is not None:
            out[self.inputs_kwarg_name] = x_t
            out["attention_mask"] = (x_t != self.pad_token).int()
        if x_dec_t is not None:
            out[self.decoder_inputs_kwarg_name] = x_dec_t
            out["decoder_attention_mask"] = (x_dec_t != self.pad_token).int()
        if y_t is not None:
            out[self.labels_kwarg_name] = y_t
        return out


def _pad_batch(batch: Sequence[Tensor], pad_token_id: int, pad_on_left: bool) -> Tensor:
    length = max(len(seq) for seq in batch)
    if all(len(seq) == length for seq in batch):
        return torch.stack(list(batch)).long()
    if pad_on_left:
        batch = [torch.flip(seq, dims=(0,)) for seq in batch]
    padded = torch.nn.utils.rnn.pad_sequence(
        list(batch), batch_first=True, padding_value=pad_token_id
    )
    if pad_on_left:
        padded = torch.flip(padded, dims=(1,))
    return padded.long()
