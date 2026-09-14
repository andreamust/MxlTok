"""Byte-pair encoding (BPE) over LMX tokens, backed by 🤗 ``tokenizers``.

Every base token is mapped to one Unicode character, so that a token sequence
becomes a string the BPE model can merge. BPE ids are a separate id space from
the base vocabulary: a trained model's embedding table has
:attr:`BPE.vocab_size` rows. The mapping is compatible with the BPE models
trained by mxlGPT.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from tokenizers import Tokenizer, models, trainers

from mxltok.errors import MxlTokError

# Base tokens map to CJK ideographs (U+4E00 upwards), which never appear in
# token strings.
_FIRST_CHAR = 0x4E00
_MAX_BASE_VOCAB = 0x6000


class BPE:
    """A trained BPE model over LMX token sequences.

    Train one with :meth:`train`; restore a saved one with :meth:`from_dict`.

    Args:
        tokenizer: The trained 🤗 tokenizer, over characters.
        char_map: The character of every base token.
    """

    def __init__(self, tokenizer: Tokenizer, char_map: dict[str, str]) -> None:
        self._tokenizer = tokenizer
        self._char_map = char_map
        self._token_of_char = {char: token for token, char in char_map.items()}
        self._chars_of_id = {id_: chars for chars, id_ in tokenizer.get_vocab().items()}

    @classmethod
    def train(
        cls,
        sequences: Iterable[Sequence[str]],
        base_vocab: Sequence[str],
        special_tokens: Sequence[str],
        vocab_size: int,
        length: int | None = None,
        show_progress: bool = False,
    ) -> BPE:
        """Learn BPE merges from token sequences.

        Args:
            sequences: The token sequences to learn from (consumed once).
            base_vocab: The base vocabulary, in id order.
            special_tokens: Tokens that are never merged.
            vocab_size: The size of the BPE vocabulary, base tokens included.
            length: The number of sequences, to display progress.
            show_progress: Display a progress bar.
        """
        if vocab_size <= len(base_vocab):
            raise ValueError(
                f"vocab_size ({vocab_size}) must exceed the base vocabulary "
                f"({len(base_vocab)} tokens)"
            )
        if len(base_vocab) > _MAX_BASE_VOCAB:
            raise ValueError(f"The base vocabulary exceeds {_MAX_BASE_VOCAB} tokens")
        char_map = {token: chr(_FIRST_CHAR + i) for i, token in enumerate(base_vocab)}
        trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
            vocab_size=vocab_size,
            special_tokens=[char_map[t] for t in special_tokens if t in char_map],
            initial_alphabet=list(char_map.values()),
            show_progress=show_progress,
        )
        tokenizer = Tokenizer(models.BPE())
        strings = (
            "".join(char_map[token] for token in sequence if token in char_map)
            for sequence in sequences
        )
        tokenizer.train_from_iterator(
            (string for string in strings if string), trainer=trainer, length=length
        )
        return cls(tokenizer, char_map)

    @property
    def vocab_size(self) -> int:
        """The number of BPE ids."""
        return int(self._tokenizer.get_vocab_size())

    def token_to_id(self, token: str) -> int:
        """Return the BPE id of a single base token."""
        id_ = self._tokenizer.token_to_id(self._char_map.get(token, ""))
        if id_ is None:
            raise MxlTokError(f"Token {token!r} is not in the BPE vocabulary")
        return int(id_)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        """Return the BPE ids of a token sequence."""
        return self.encode_batch([tokens])[0]

    def encode_batch(self, sequences: Sequence[Sequence[str]]) -> list[list[int]]:
        """Return the BPE ids of several token sequences, in parallel."""
        strings = [self._to_chars(tokens) for tokens in sequences]
        encodings = self._tokenizer.encode_batch(strings)
        return [
            encoding.ids if s else []
            for encoding, s in zip(encodings, strings, strict=True)
        ]

    def decode(self, ids: Iterable[int]) -> list[str]:
        """Return the base tokens of BPE ids."""
        try:
            return [
                self._token_of_char[char]
                for id_ in ids
                for char in self._chars_of_id[id_]
            ]
        except KeyError as exc:
            raise MxlTokError(f"Id {exc.args[0]!r} is not a BPE id") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the model as a JSON-serializable dictionary."""
        return {
            "char_map": self._char_map,
            "hf_tokenizer": json.loads(self._tokenizer.to_str()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BPE:
        """Restore a model saved with :meth:`to_dict`."""
        tokenizer = Tokenizer.from_str(json.dumps(data["hf_tokenizer"]))
        return cls(tokenizer, dict(data["char_map"]))

    def _to_chars(self, tokens: Sequence[str]) -> str:
        try:
            return "".join(self._char_map[token] for token in tokens)
        except KeyError as exc:
            raise MxlTokError(
                f"Token {exc.args[0]!r} is not in the vocabulary"
            ) from exc
