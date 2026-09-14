"""The :class:`MXLTokenizer`: MusicXML scores to LMX tokens and ids, and back."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mxltok.arrangement import join_parts, split_parts
from mxltok.bpe import BPE
from mxltok.categories import filter_tokens, reorder_tokens
from mxltok.config import TokenizerConfig
from mxltok.decoding import PartDecoder
from mxltok.encoding import PartEncoder
from mxltok.errors import MxlTokError
from mxltok.instruments import (
    CATEGORY_TO_MIDI_PROGRAM,
    DEFAULT_MIDI_CHANNEL,
    DRUMS_MIDI_CHANNEL,
    instrument_category,
)
from mxltok.musicxml import read_musicxml
from mxltok.score import Part, Score
from mxltok.vocabulary import INSTRUMENT_PREFIX, NUM_PARTS_PREFIX, build_vocabulary

TOKENIZER_FILE = "tokenizer.json"
_TOKENIZER_CLASS = "MXLTokenizer"


@dataclass
class TokSequence:
    """An encoded score: its tokens and their ids.

    ``ids`` are BPE ids when the tokenizer has a trained BPE model, base
    vocabulary ids otherwise. ``tokens`` are always base tokens.
    """

    tokens: list[str] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)


class MXLTokenizer:
    """Tokenizes MusicXML scores into LMX tokens and decodes them back.

    Example:
        >>> tokenizer = MXLTokenizer()
        >>> sequence = tokenizer.encode("score.mxl")
        >>> sequence.tokens[:4]
        ['<BOS>', 'num_parts:1', 'part:instrument:Piano', 'part:start']
        >>> score = tokenizer.decode(sequence.ids)

    Args:
        config: The tokenization options; the defaults when omitted.
        vocab: The base vocabulary in id order; built from ``config`` when
            omitted. Restored tokenizers pass their saved vocabulary.
        bpe: A trained BPE model over the base vocabulary.
    """

    def __init__(
        self,
        config: TokenizerConfig | None = None,
        *,
        vocab: Sequence[str] | None = None,
        bpe: BPE | None = None,
    ) -> None:
        self.config = config if config is not None else TokenizerConfig()
        special_tokens = self.config.special_tokens.as_list()
        if vocab is None:
            vocab = build_vocabulary(special_tokens, self.config.pitch_mode)
        missing = [token for token in special_tokens if token not in vocab]
        if missing:
            raise MxlTokError(f"Special tokens missing from the vocabulary: {missing}")
        self._vocab = list(vocab)
        self._id_of = {token: id_ for id_, token in enumerate(self._vocab)}
        self.bpe = bpe
        self._encoder = PartEncoder(self.config)

    def __repr__(self) -> str:
        bpe = f", bpe_vocab_size={self.bpe.vocab_size}" if self.bpe else ""
        return f"MXLTokenizer(base_vocab_size={len(self._vocab)}{bpe})"

    # ---------------------------------------------------------- vocabulary

    @property
    def vocab(self) -> list[str]:
        """The base vocabulary, in id order."""
        return list(self._vocab)

    @property
    def base_vocab_size(self) -> int:
        """The number of base tokens."""
        return len(self._vocab)

    @property
    def vocab_size(self) -> int:
        """The number of ids: the size of a model's embedding table."""
        return self.bpe.vocab_size if self.bpe is not None else len(self._vocab)

    @property
    def pad_id(self) -> int:
        """The id of the padding token."""
        return self.token_to_id(self.config.special_tokens.pad)

    @property
    def bos_id(self) -> int:
        """The id of the beginning-of-sequence token."""
        return self.token_to_id(self.config.special_tokens.bos)

    @property
    def eos_id(self) -> int:
        """The id of the end-of-sequence token."""
        return self.token_to_id(self.config.special_tokens.eos)

    @property
    def unk_id(self) -> int:
        """The id of the unknown token."""
        return self.token_to_id(self.config.special_tokens.unk)

    @property
    def mask_id(self) -> int:
        """The id of the mask token."""
        return self.token_to_id(self.config.special_tokens.mask)

    def token_to_id(self, token: str) -> int:
        """Return the id of a single token."""
        if self.bpe is not None:
            return self.bpe.token_to_id(token)
        if token not in self._id_of:
            raise MxlTokError(f"Token {token!r} is not in the vocabulary")
        return self._id_of[token]

    def tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        """Return the ids of a token sequence (BPE-encoded if BPE is trained)."""
        unknown = [token for token in tokens if token not in self._id_of]
        if unknown:
            raise MxlTokError(f"Tokens not in the vocabulary: {sorted(set(unknown))}")
        if self.bpe is not None:
            return self.bpe.encode(tokens)
        return [self._id_of[token] for token in tokens]

    def ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        """Return the base tokens of ids (BPE-decoded if BPE is trained)."""
        if self.bpe is not None:
            return self.bpe.decode(ids)
        try:
            return [self._vocab[id_] for id_ in ids]
        except IndexError as exc:
            raise MxlTokError("Id out of the vocabulary range") from exc

    # ------------------------------------------------------------ encoding

    def encode(self, score: Score | str | Path) -> TokSequence:
        """Tokenize a score or a MusicXML file (``.mxl``, ``.musicxml``, ``.xml``).

        Raises:
            MxlTokError: If the file cannot be read, or the score contains
                content outside the vocabulary.
        """
        if not isinstance(score, Score):
            score = read_musicxml(score)
        tokens = self.encode_tokens(score)
        return TokSequence(tokens, self.tokens_to_ids(tokens))

    def encode_tokens(self, score: Score) -> list[str]:
        """Return the tokens of a score, filtered and wrapped as configured."""
        config = self.config
        parts = score.parts[: config.max_parts] if config.max_parts else score.parts
        encoded = [(instrument_category(p), self._encoder.encode(p)) for p in parts]
        tokens = join_parts(encoded, config.arrangement)
        if parts and config.prepend_instruments:
            tokens = [
                f"{NUM_PARTS_PREFIX}{len(parts)}",
                *(f"{INSTRUMENT_PREFIX}{instrument}" for instrument, _ in encoded),
                *tokens,
            ]
        if config.excluded_categories:
            tokens = filter_tokens(tokens, config.excluded_categories)
        if config.token_order is not None:
            tokens = reorder_tokens(tokens, config.token_order)
        if config.add_bos_eos:
            special = config.special_tokens
            tokens = [special.bos, *tokens, special.eos]
        return tokens

    # ------------------------------------------------------------ decoding

    def decode(
        self,
        sequence: TokSequence | Sequence[int] | Sequence[str],
        strict: bool = False,
    ) -> Score:
        """Rebuild a score from a sequence of ids or tokens.

        Write it to a file with :func:`mxltok.write_musicxml`.

        Args:
            sequence: A :class:`TokSequence`, ids (a list, NumPy array or
                tensor) or tokens. Special tokens are ignored.
            strict: Raise :class:`MxlTokError` on malformed tokens instead of
                skipping them with a warning.
        """
        tokens = self._as_tokens(sequence)
        special = set(self.config.special_tokens.as_list())
        tokens = [token for token in tokens if token not in special]
        parts = split_parts(tokens, self.config.arrangement) or [(None, tokens)]
        decoder = PartDecoder(strict)
        score = Score()
        for number, (category, part_tokens) in enumerate(parts, start=1):
            part = Part(id=f"P{number}", name=category or f"Part {number}")
            if category is not None:
                part.midi_program = CATEGORY_TO_MIDI_PROGRAM.get(category, 1)
                part.midi_channel = (
                    DRUMS_MIDI_CHANNEL if category == "Drums" else DEFAULT_MIDI_CHANNEL
                )
            part.measures = decoder.decode(part_tokens)
            score.parts.append(part)
        return score

    def _as_tokens(
        self, sequence: TokSequence | Sequence[int] | Sequence[str]
    ) -> list[str]:
        if isinstance(sequence, TokSequence):
            return list(sequence.tokens)
        items: list[Any] = (
            sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
        )
        if items and isinstance(items[0], str):
            return list(items)
        return self.ids_to_tokens(int(id_) for id_ in items)

    # ----------------------------------------------------------------- BPE

    def train_bpe(
        self,
        sequences: Iterable[Sequence[str]],
        vocab_size: int,
        length: int | None = None,
        show_progress: bool = False,
    ) -> None:
        """Learn a BPE model from token sequences; ids become BPE ids.

        Example:
            >>> files = sorted(Path("scores").glob("*.mxl"))
            >>> tokenizer.train_bpe(
            ...     (tokenizer.encode(f).tokens for f in files), vocab_size=10_000
            ... )

        Args:
            sequences: Base token sequences, such as ``encode(f).tokens``.
            vocab_size: The size of the BPE vocabulary.
            length: The number of sequences, to display progress.
            show_progress: Display a progress bar.
        """
        self.bpe = BPE.train(
            sequences,
            self._vocab,
            self.config.special_tokens.as_list(),
            vocab_size,
            length=length,
            show_progress=show_progress,
        )

    # --------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        """Return the tokenizer as a JSON-serializable dictionary."""
        from mxltok import __version__

        data: dict[str, Any] = {
            "tokenizer_class": _TOKENIZER_CLASS,
            "mxltok_version": __version__,
            "vocab": self._vocab,
            "config": self.config.to_dict(),
        }
        if self.bpe is not None:
            data["bpe"] = self.bpe.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MXLTokenizer:
        """Restore a tokenizer from :meth:`to_dict` output."""
        tokenizer_class = data.get("tokenizer_class", _TOKENIZER_CLASS)
        if tokenizer_class != _TOKENIZER_CLASS:
            raise MxlTokError(f"Not an LMX tokenizer: {tokenizer_class!r}")
        return cls(
            TokenizerConfig.from_dict(data["config"]),
            vocab=data["vocab"],
            bpe=BPE.from_dict(data["bpe"]) if "bpe" in data else None,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        """Save the tokenizer as ``directory/tokenizer.json`` and return that path."""
        path = Path(directory) / TOKENIZER_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        return path

    @classmethod
    def from_pretrained(
        cls, name_or_path: str | Path, revision: str | None = None
    ) -> MXLTokenizer:
        """Load a saved tokenizer.

        Args:
            name_or_path: A ``tokenizer.json`` file or a directory holding one,
                a dataset directory preprocessed by mxlGPT, or the id of a
                Hugging Face Hub repository (requires ``mxltok[hub]``).
            revision: The Hub revision (branch, tag or commit) to load.
        """
        path = Path(name_or_path)
        if path.is_file():
            return cls.from_dict(_read_json(path))
        if path.is_dir():
            return cls._from_directory(path)
        return cls.from_dict(_read_json(_hub_download(str(name_or_path), revision)))

    @classmethod
    def _from_directory(cls, directory: Path) -> MXLTokenizer:
        if (directory / "manifest.json").is_file():  # mxlGPT preprocessed dataset
            directory = directory / "tokenizer"
        if (directory / TOKENIZER_FILE).is_file():
            data = _read_json(directory / TOKENIZER_FILE)
        elif (directory / "vocab.json").is_file():  # mxlGPT multi-file format
            data = {
                "vocab": _read_json(directory / "vocab.json"),
                "config": _read_json(directory / "config.json"),
            }
        else:
            raise MxlTokError(f"No tokenizer found in {directory}")
        separate_bpe = directory / "bpe_tokenizer.json"
        if "bpe" not in data and separate_bpe.is_file():
            data["bpe"] = _read_json(separate_bpe)
        return cls.from_dict(data)

    def push_to_hub(self, repo_id: str, private: bool = True, **kwargs: Any) -> str:
        """Upload the tokenizer to a Hugging Face Hub repository (``mxltok[hub]``).

        Args:
            repo_id: The repository, such as ``"user/mxltok-bpe10k"``.
            private: Create the repository as private if it does not exist.
            **kwargs: Passed to ``huggingface_hub.HfApi.upload_file``.

        Returns:
            The URL of the uploaded commit.
        """
        hub = _import_hub()
        api = hub.HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)
        content = json.dumps(self.to_dict(), ensure_ascii=False).encode("utf-8")
        commit = api.upload_file(
            path_or_fileobj=content,
            path_in_repo=TOKENIZER_FILE,
            repo_id=repo_id,
            **kwargs,
        )
        return str(commit)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _import_hub() -> Any:
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "Hugging Face Hub support requires: pip install 'mxltok[hub]'"
        ) from exc
    return huggingface_hub


def _hub_download(repo_id: str, revision: str | None) -> Path:
    hub = _import_hub()
    return Path(hub.hf_hub_download(repo_id, TOKENIZER_FILE, revision=revision))
