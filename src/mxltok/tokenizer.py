"""Base tokenizer: vocabulary, ids/tokens/bytes conversions, encode/decode."""

from __future__ import annotations

import json
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence, Sized
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from os import cpu_count
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import ModelHubMixin, hf_hub_download
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

from mxltok.classes import Event, MxlTokError, TokenizerConfig, TokSequence
from mxltok.constants import CHR_ID_START
from mxltok.musicxml import dump, load
from mxltok.score import Score
from mxltok.utils import TokTrainingIterator, deepest_common_dir, get_files_paths

BOS_TOKEN_NAME = "BOS"
EOS_TOKEN_NAME = "EOS"
DEFAULT_TOKENIZER_FILE_NAME = "tokenizer.json"
DEFAULT_TRAINING_MODEL = "BPE"
WORDPIECE_MAX_INPUT_CHARS_PER_WORD = 400
UNIGRAM_MAX_PIECE_LENGTH = 200
TrainingModel = Literal["BPE", "Unigram", "WordPiece"]


class MusicTokenizer(ABC, ModelHubMixin, library_name="MxlTok"):
    """Abstract MusicXML tokenizer with a MidiTok-like interface.

    Subclasses implement the four hooks: :meth:`_create_base_vocabulary`,
    :meth:`_create_token_types_graph`, :meth:`_score_to_tokens` and
    :meth:`_tokens_to_score`.

    :param tokenizer_config: the tokenizer's configuration.
    :param params: path to a saved tokenizer (``tokenizer.json``) to load
        instead; ``tokenizer_config`` is then ignored.
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig | None = None,
        params: str | Path | None = None,
    ) -> None:
        self._vocab_base: dict[str, int] = {}
        self._vocab_base_inv: dict[int, str] = {}
        self._vocab_base_id_to_byte: dict[int, str] = {}
        self._vocab_base_byte_to_token: dict[str, str] = {}
        self._vocab_learned_bytes_to_tokens: dict[str, list[str]] = {}
        self._model: Tokenizer | None = None
        self._token_types_indexes: dict[str, list[int]] = {}
        self.tokens_types_graph: dict[str, set[str]] = {}
        if params is not None:
            self._load_from_json(params)
        else:
            self.config = (
                tokenizer_config if tokenizer_config is not None else TokenizerConfig()
            )
        if not self._vocab_base:
            self._create_vocabulary()
        self._update_token_types_indexes()
        self.tokens_types_graph = self._create_token_types_graph()
        self._add_special_tokens_to_types_graph()

    @abstractmethod
    def _create_base_vocabulary(self) -> list[str]:
        """Return the base tokens (without special tokens), in vocabulary order."""

    @abstractmethod
    def _create_token_types_graph(self) -> dict[str, set[str]]:
        """Return the allowed successions of token types (for ``tokens_errors``)."""

    @abstractmethod
    def _score_to_tokens(self, score: Score, strict: bool) -> TokSequence:
        """Tokenize a :class:`Score` into a sequence with its ``events`` filled."""

    @abstractmethod
    def _tokens_to_score(self, tokens: list[str], strict: bool) -> Score:
        """Rebuild a :class:`Score` from a list of tokens."""

    def _create_vocabulary(self) -> None:
        for token in self.config.special_tokens:
            self.add_to_vocab(token, special_token=True)
        for token in self._create_base_vocabulary():
            self.add_to_vocab(token)

    def add_to_vocab(
        self, token: str | Event, special_token: bool = False, byte_: str | None = None
    ) -> None:
        """Add a token to the base vocabulary (no-op if already present)."""
        token = str(token)
        if token in self._vocab_base:
            return
        id_ = len(self._vocab_base)
        self._vocab_base[token] = id_
        self._vocab_base_inv[id_] = token
        if byte_ is None:
            byte_ = chr(id_ + CHR_ID_START)
        self._vocab_base_id_to_byte[id_] = byte_
        self._vocab_base_byte_to_token[byte_] = token
        if not special_token:
            self._token_types_indexes.setdefault(token.split("_")[0], []).append(id_)

    def _update_token_types_indexes(self) -> None:
        self._token_types_indexes = {}
        for token, id_ in self._vocab_base.items():
            if token not in self.config.special_tokens:
                self._token_types_indexes.setdefault(token.split("_")[0], []).append(
                    id_
                )

    def _add_special_tokens_to_types_graph(self) -> None:
        base_types = set(self.tokens_types_graph)
        for special_token in self.config.special_tokens:
            type_ = special_token.split("_")[0]
            if type_ == EOS_TOKEN_NAME:
                self.tokens_types_graph[type_] = set()
            else:
                self.tokens_types_graph[type_] = base_types | {
                    t.split("_")[0] for t in self.config.special_tokens
                }
            if type_ != BOS_TOKEN_NAME:
                for base_type in base_types:
                    self.tokens_types_graph[base_type].add(type_)

    @property
    def vocab(self) -> dict[str, int]:
        """The base vocabulary, token -> id."""
        return self._vocab_base

    @property
    def vocab_model(self) -> dict[str, int] | None:
        """The trained model's vocabulary (bytes -> id), or ``None`` if untrained."""
        return None if self._model is None else self._model.get_vocab()

    @property
    def vocab_size(self) -> int:
        return len(self)

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return self.config.special_tokens

    @property
    def special_tokens_ids(self) -> list[int]:
        return [self._vocab_base[token] for token in self.config.special_tokens]

    @property
    def pad_token_id(self) -> int:
        return self._vocab_base["PAD_None"]

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def token_ids_of_type(self, token_type: str) -> list[int]:
        """Ids of all base tokens of a type, e.g. ``"Pitch"``."""
        return list(self._token_types_indexes.get(token_type, []))

    def token_id_type(self, id_: int) -> str:
        """Type of a base token id."""
        return self._vocab_base_inv[id_].split("_")[0]

    def encode(
        self, score: Score | str | Path, encode_ids: bool = True, strict: bool = False
    ) -> TokSequence:
        """Tokenize a :class:`Score` or a MusicXML file.

        :param score: score object or path to a ``.musicxml``/``.xml``/``.mxl``.
        :param encode_ids: apply the trained model (BPE...) to the ids, if any.
        :param strict: raise :class:`MxlTokError` on unsupported content instead
            of warning and skipping it.
        """
        if not isinstance(score, Score):
            score = load(score)
        seq = self._score_to_tokens(score, strict)
        self.complete_sequence(seq)
        if encode_ids and self.is_trained:
            self.encode_token_ids(seq)
        return seq

    def complete_sequence(self, seq: TokSequence, complete_bytes: bool = False) -> None:
        """Fill the missing ``tokens``/``ids`` (and ``bytes``) forms of a sequence."""
        if not seq.tokens:
            if seq.events:
                seq.tokens = [str(event) for event in seq.events]
            elif seq.ids:
                seq.tokens = self._ids_to_tokens(seq.ids)
            elif seq.bytes:
                seq.tokens = self._bytes_to_tokens(seq.bytes)
        if not seq.ids:
            seq.ids = self._tokens_to_ids(seq.tokens)
        if complete_bytes and self.is_trained and not seq.bytes:
            seq.bytes = self._ids_to_bytes(seq.ids)

    def _tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        return [self._vocab_base[token] for token in tokens]

    def _ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        return [self._vocab_base_inv[id_] for id_ in ids]

    def _ids_to_bytes(self, ids: Sequence[int]) -> str:
        return "".join(self._vocab_base_id_to_byte[id_] for id_ in ids)

    def _bytes_to_tokens(self, bytes_: str) -> list[str]:
        return [
            token
            for byte_ in bytes_
            for token in self._vocab_learned_bytes_to_tokens[byte_]
        ]

    def train(
        self,
        vocab_size: int,
        model: TrainingModel | models.Model | None = None,
        iterator: Iterable[str | list[str]] | None = None,
        files_paths: Sequence[str | Path] | None = None,
        **kwargs: Any,
    ) -> None:
        """Learn a BPE, Unigram or WordPiece vocabulary with 🤗 tokenizers.

        Token ids are mapped to unique characters (bytes) which the model
        merges; with ``config.encode_ids_split == "bar"`` each bar is a "word".

        :param vocab_size: target vocabulary size (must exceed the base vocab).
        :param model: model name, a ``tokenizers.models.Model`` instance, or
            ``None`` to retrain the current model (``"BPE"`` if untrained).
        :param iterator: sized iterable of byte strings (or lists of them);
            defaults to a :class:`TokTrainingIterator` over ``files_paths``.
        :param files_paths: MusicXML files to train on when no iterator is given.
        :param kwargs: extra arguments for the 🤗 trainer.
        """
        if iterator is None and files_paths is None:
            raise ValueError("Give an iterator or a list of files to train on.")
        if vocab_size <= len(self._vocab_base):
            warnings.warn(
                f"vocab_size ({vocab_size}) must exceed the base vocabulary "
                f"({len(self._vocab_base)}); skipping training.",
                stacklevel=2,
            )
            return
        if iterator is None:
            assert files_paths is not None
            iterator = TokTrainingIterator(self, files_paths)
        initial_alphabet = list(self._vocab_base_id_to_byte.values())
        split = self.config.encode_ids_split != "no"

        if model is None and self._model is not None:
            tokenizer = self._reset_hf_tokenizer(self._model)
        elif isinstance(model, models.Model):
            tokenizer = Tokenizer(model)
        else:
            name = model or DEFAULT_TRAINING_MODEL
            model_kwargs: dict[str, Any] = {}
            if name in ("BPE", "WordPiece"):
                model_kwargs["vocab"] = {b: i for i, b in enumerate(initial_alphabet)}
                if split:
                    model_kwargs["continuing_subword_prefix"] = ""
            if name == "BPE":
                model_kwargs["merges"] = []
                if split:
                    model_kwargs["end_of_word_suffix"] = ""
            elif name == "WordPiece":
                model_kwargs["unk_token"] = kwargs.pop(
                    "unk_token", self._vocab_base_id_to_byte[self.pad_token_id]
                )
                model_kwargs["max_input_chars_per_word"] = kwargs.pop(
                    "max_input_chars_per_word", WORDPIECE_MAX_INPUT_CHARS_PER_WORD
                )
            elif name != "Unigram":
                raise ValueError(f"Unknown training model {model!r}")
            tokenizer = Tokenizer(getattr(models, name)(**model_kwargs))
            if split:
                tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
                tokenizer.decoder = decoders.Metaspace()
        model_json = json.loads(tokenizer.to_str())["model"]
        model_name = model_json["type"]
        if model_name == "WordPiece" and not split:
            warnings.warn(
                "Training WordPiece without splitting ids per bar: sequences longer "
                "than max_input_chars_per_word "
                f"({model_json['max_input_chars_per_word']}) become unknown tokens.",
                stacklevel=2,
            )
        special_tokens = [
            AddedToken(self._vocab_base_id_to_byte[id_])
            for id_ in self.special_tokens_ids
        ]
        if model_name in ("BPE", "WordPiece"):
            kwargs.setdefault(
                "continuing_subword_prefix",
                model_json["continuing_subword_prefix"] or "",
            )
            if model_name == "BPE":
                kwargs.setdefault(
                    "end_of_word_suffix", model_json["end_of_word_suffix"] or ""
                )
        elif model_name == "Unigram":
            kwargs.setdefault("max_piece_length", UNIGRAM_MAX_PIECE_LENGTH)
            if model_json["unk_id"] is not None:
                kwargs.setdefault(
                    "unk_token", model_json["vocab"][model_json["unk_id"]][0]
                )
        trainer_cls = getattr(
            trainers, f"{'Bpe' if model_name == 'BPE' else model_name}Trainer"
        )
        trainer = trainer_cls(
            vocab_size=vocab_size,
            show_progress=kwargs.pop("show_progress", False),
            special_tokens=special_tokens,
            initial_alphabet=initial_alphabet,
            **kwargs,
        )
        length = len(iterator) if isinstance(iterator, Sized) else None
        tokenizer.train_from_iterator(iterator, trainer=trainer, length=length)
        self._model = tokenizer
        self._create_vocab_learned_bytes_to_tokens()

    @staticmethod
    def _reset_hf_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
        """Same model type and settings, empty vocabulary (for retraining)."""
        tokenizer_json = json.loads(tokenizer.to_str())
        model = tokenizer_json["model"]
        if model["type"] == "BPE":
            model["vocab"], model["merges"] = {}, []
        elif model["type"] == "Unigram":
            if model["unk_id"] is not None:
                model["vocab"] = [[model["vocab"][model["unk_id"]][0], 0.0]]
                model["unk_id"] = 0
            else:
                model["vocab"] = []
        elif model["type"] == "WordPiece":
            model["vocab"] = {}
        else:
            raise ValueError(f"Cannot retrain a {model['type']} model")
        return Tokenizer.from_str(json.dumps(tokenizer_json))

    def _create_vocab_learned_bytes_to_tokens(self) -> None:
        assert self._model is not None
        model_json = json.loads(self._model.to_str())
        prefix = model_json["model"].get("continuing_subword_prefix") or ""
        suffix = model_json["model"].get("end_of_word_suffix") or ""
        replacement = ""
        if isinstance(self._model.pre_tokenizer, pre_tokenizers.Metaspace):
            replacement = self._model.pre_tokenizer.replacement
        self._vocab_learned_bytes_to_tokens = {}
        for learned in self._model.get_vocab():
            key = learned.removeprefix(prefix) if prefix else learned
            key = key.removesuffix(suffix) if suffix else key
            key = key.removeprefix(replacement) if replacement else key
            self._vocab_learned_bytes_to_tokens[learned] = [
                self._vocab_base_byte_to_token[byte_] for byte_ in key
            ]

    def encode_token_ids(self, seq: TokSequence | list[TokSequence]) -> None:
        """Encode the ids of the sequence(s) with the trained model (in place)."""
        if self._model is None:
            raise RuntimeError("The tokenizer is not trained.")
        for sequence in seq if isinstance(seq, list) else [seq]:
            chunks = self._split_bytes(sequence)
            sequence.ids = self._model.encode(chunks, is_pretokenized=True).ids
            sequence.are_ids_encoded = True

    def _split_bytes(self, seq: TokSequence) -> list[str]:
        if seq.are_ids_encoded:
            self.decode_token_ids(seq)
        self.complete_sequence(seq, complete_bytes=True)
        if self.config.encode_ids_split == "bar":
            return [chunk.bytes for chunk in seq.split_per_bars() if len(chunk) > 0]
        return [seq.bytes]

    def decode_token_ids(self, seq: TokSequence | list[TokSequence]) -> None:
        """Turn model-encoded ids back into base ids and tokens (in place)."""
        if isinstance(seq, list):
            for sequence in seq:
                self.decode_token_ids(sequence)
            return
        if not seq.are_ids_encoded:
            return
        assert self._model is not None
        tokens: list[str] = []
        for id_ in seq.ids:
            learned = self._model.id_to_token(id_)
            if learned is None:
                raise MxlTokError(f"Id {id_} is not in the trained vocabulary")
            tokens += self._vocab_learned_bytes_to_tokens[learned]
        seq.tokens = tokens
        seq.ids = self._tokens_to_ids(tokens)
        seq.bytes = ""
        seq.are_ids_encoded = False

    def decode(
        self,
        tokens: TokSequence | Sequence[int] | Sequence[str] | Any,
        output_path: str | Path | None = None,
        strict: bool = False,
    ) -> Score:
        """Detokenize a sequence (``TokSequence``, ids, tokens, tensor) to a score.

        :param tokens: the sequence; ids may be a list, numpy array or tensor.
        :param output_path: also write the MusicXML file there (``.mxl`` = zipped).
        :param strict: raise on grammar violations instead of repairing them.
        """
        seq = self._convert_sequence_to_tokseq(tokens)
        if seq.are_ids_encoded:
            self.decode_token_ids(seq)
        self.complete_sequence(seq)
        score = self._tokens_to_score(seq.tokens, strict)
        if output_path is not None:
            dump(score, output_path)
        return score

    def _convert_sequence_to_tokseq(
        self, input_seq: TokSequence | Sequence[int] | Sequence[str] | Any
    ) -> TokSequence:
        if isinstance(input_seq, TokSequence):
            return input_seq
        tolist = getattr(input_seq, "tolist", None)  # numpy arrays, torch tensors
        items: list[Any] = list(tolist() if callable(tolist) else input_seq)
        if not items:
            return TokSequence()
        if isinstance(items[0], str):
            return TokSequence(tokens=items)
        if isinstance(items[0], Event):
            return TokSequence(events=items)
        ids = [int(i) for i in items]
        return TokSequence(ids=ids, are_ids_encoded=self._are_ids_encoded(ids))

    def _are_ids_encoded(self, ids: Sequence[int]) -> bool:
        return self.is_trained and any(id_ >= len(self._vocab_base) for id_ in ids)

    def tokens_errors(
        self, tokens: TokSequence | Sequence[int] | Sequence[str] | Any
    ) -> float:
        """Ratio of tokens whose type cannot follow the previous one (0 = none)."""
        seq = self._convert_sequence_to_tokseq(tokens)
        if len(seq) == 0:
            return 0.0
        num_predicted = len(seq)
        if seq.are_ids_encoded:
            self.decode_token_ids(seq)
        self.complete_sequence(seq)
        return self._tokens_errors(seq.tokens) / num_predicted

    def _tokens_errors(self, tokens: list[str]) -> int:
        errors = 0
        previous_type = tokens[0].split("_")[0]
        if previous_type not in self.tokens_types_graph:
            errors += 1
        for token in tokens[1:]:
            type_ = token.split("_")[0]
            if type_ not in self.tokens_types_graph.get(previous_type, ()):
                errors += 1
            previous_type = type_
        return errors

    def save_tokens(
        self, tokens: TokSequence | Sequence[int] | Any, path: str | Path, **kwargs: Any
    ) -> None:
        """Save token ids (only) as a JSON file, with any extra ``kwargs`` entries."""
        seq = self._convert_sequence_to_tokseq(tokens)
        self.complete_sequence(seq)
        content = {"ids": seq.ids, "ids_encoded": seq.are_ids_encoded, **kwargs}
        with Path(path).open("w") as file:
            json.dump(content, file)

    def load_tokens(
        self, path: str | Path, raw: bool = False
    ) -> TokSequence | dict[str, Any]:
        """Load a JSON file written by :meth:`save_tokens`."""
        with Path(path).open() as file:
            content = json.load(file)
        if raw:
            return content
        return TokSequence(
            ids=content["ids"], are_ids_encoded=content.get("ids_encoded", False)
        )

    def tokenize_dataset(
        self,
        files_paths: str | Path | Sequence[str | Path],
        out_dir: str | Path,
        overwrite_mode: bool = True,
        validation_fn: Callable[[Score], bool] | None = None,
        parallel_workers_size: int = max(1, (cpu_count() or 1) - 1),
    ) -> None:
        """Tokenize files (or a directory of them) into JSON files under ``out_dir``.

        The directory tree below the deepest common directory is reproduced, and
        the tokenizer is saved as ``tokenizer.json`` in ``out_dir``. Files that
        fail to load or that ``validation_fn`` rejects are skipped.
        """
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(files_paths, str | Path):
            root_dir = Path(files_paths).resolve()
            paths = get_files_paths(root_dir)
        else:
            paths = [Path(p) for p in files_paths]
            root_dir = deepest_common_dir(paths) if paths else out_dir
        if not paths:
            warnings.warn("No MusicXML file found to tokenize.", stacklevel=2)
            return
        fn = partial(
            _tokenize_file,
            self,
            root_dir=root_dir,
            out_dir=out_dir,
            overwrite_mode=overwrite_mode,
            validation_fn=validation_fn,
        )
        if parallel_workers_size < 2 or len(paths) < 2:
            for path in paths:
                fn(path)
        else:
            with ProcessPoolExecutor(max_workers=parallel_workers_size) as executor:
                list(
                    executor.map(
                        fn, paths, chunksize=max(1, len(paths) // parallel_workers_size)
                    )
                )
        self.save(out_dir)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form: config, class name, versions and the trained model."""
        from tokenizers import __version__ as tokenizers_version

        from mxltok import __version__

        params: dict[str, Any] = {
            "config": self.config.to_dict(serialize=True),
            "tokenization": self.__class__.__name__,
            "mxltok_version": __version__,
            "hf_tokenizers_version": tokenizers_version,
        }
        if self._model is not None:
            params["_vocab_base"] = self._vocab_base
            params["_model"] = self._model.to_str()
            params["_vocab_base_byte_to_token"] = self._vocab_base_byte_to_token
        return params

    def save(
        self,
        out_path: str | Path,
        additional_attributes: dict[str, Any] | None = None,
        filename: str = DEFAULT_TOKENIZER_FILE_NAME,
    ) -> None:
        """Save the tokenizer as JSON; ``out_path`` may be a directory."""
        params = self.to_dict()
        if additional_attributes:
            params.update(additional_attributes)
        out_path = Path(out_path)
        if out_path.is_dir() or "." not in out_path.name:
            out_path /= filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as file:
            json.dump(params, file, indent=4)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        **push_to_hub_kwargs: Any,
    ) -> str | None:
        """Save into a directory, optionally pushing it to the Hugging Face Hub."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self._save_pretrained(save_directory)
        if push_to_hub:
            push_to_hub_kwargs.pop("config", None)
            push_to_hub_kwargs.pop("model_card_kwargs", None)
            return self.push_to_hub(
                repo_id=repo_id or save_directory.name, **push_to_hub_kwargs
            )
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        self.save(save_directory)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwargs: Any
    ) -> MusicTokenizer:
        """Load a tokenizer from a JSON file, a directory, or a Hugging Face repo."""
        path = Path(pretrained_model_name_or_path)
        if path.is_file():
            return cls._from_pretrained(
                model_id=str(path),
                revision=None,
                cache_dir=None,
                force_download=False,
                local_files_only=True,
                token=None,
                **kwargs,
            )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        **kwargs: Any,
    ) -> MusicTokenizer:
        filename = kwargs.get("filename", DEFAULT_TOKENIZER_FILE_NAME)
        pretrained_path = Path(model_id)
        if pretrained_path.is_file():
            params_path = pretrained_path
        elif (pretrained_path / filename).is_file():
            params_path = pretrained_path / filename
        else:
            params_path = Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    library_name="MxlTok",
                )
            )
        with params_path.open() as file:
            tokenization = json.load(file)["tokenization"]
        if cls.__name__ == tokenization:
            return cls(params=params_path)
        if cls.__name__ != "MusicTokenizer":
            warnings.warn(
                f"{cls.__name__}.from_pretrained called on a {tokenization} tokenizer; "
                f"returning a {tokenization}.",
                stacklevel=2,
            )
        package = sys.modules[__name__.rsplit(".", 1)[0]]
        return getattr(package, tokenization)(params=params_path)

    def _load_from_json(self, file_path: str | Path) -> None:
        with Path(file_path).open() as file:
            params = json.load(file)
        self.config = TokenizerConfig.from_dict(params["config"])
        if "_vocab_base" in params:
            self._vocab_base = params["_vocab_base"]
            self._vocab_base_inv = {
                id_: token for token, id_ in self._vocab_base.items()
            }
        if "_model" in params:
            self._model = Tokenizer.from_str(params["_model"])
        if "_vocab_base_byte_to_token" in params:
            self._vocab_base_byte_to_token = params["_vocab_base_byte_to_token"]
            token_to_byte = {t: b for b, t in self._vocab_base_byte_to_token.items()}
            self._vocab_base_id_to_byte = {
                id_: token_to_byte[token] for token, id_ in self._vocab_base.items()
            }
        if self._model is not None:
            self._create_vocab_learned_bytes_to_tokens()

    def __call__(
        self,
        obj: Score | str | Path | TokSequence | Sequence[int] | Sequence[str] | Any,
        **kwargs: Any,
    ) -> TokSequence | Score:
        """Encode a score/file, or decode a sequence, depending on ``obj``."""
        if isinstance(obj, Score):
            return self.encode(obj, **kwargs)
        if isinstance(obj, str | Path):
            return self.encode(Path(obj), **kwargs)
        return self.decode(obj, **kwargs)

    def __len__(self) -> int:
        if self._model is not None:
            return len(self._model.get_vocab())
        return len(self._vocab_base)

    def __getitem__(self, item: int | str) -> str | int:
        """Token of an id, or id of a token."""
        if isinstance(item, str):
            return self._vocab_base[item]
        return self._vocab_base_inv[item]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MusicTokenizer):
            return False
        return (
            self._vocab_base == other._vocab_base
            and self.vocab_model == other.vocab_model
            and self._vocab_base_byte_to_token == other._vocab_base_byte_to_token
            and self.config == other.config
        )

    def __repr__(self) -> str:
        trained = f", trained ({len(self)} tokens)" if self.is_trained else ""
        return (
            f"{self.__class__.__name__}({len(self._vocab_base)} base tokens{trained})"
        )


def _tokenize_file(
    tokenizer: MusicTokenizer,
    file_path: Path,
    root_dir: Path,
    out_dir: Path,
    overwrite_mode: bool,
    validation_fn: Callable[[Score], bool] | None,
) -> None:
    try:
        score = load(file_path)
    except Exception as exc:
        warnings.warn(f"Skipping {file_path}: {exc}", stacklevel=2)
        return
    if validation_fn is not None and not validation_fn(score):
        return
    out_path = out_dir / file_path.resolve().parent.relative_to(root_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path /= f"{file_path.stem}.json"
    if not overwrite_mode:
        index = 1
        while out_path.is_file():
            out_path = out_path.parent / f"{file_path.stem}_{index}.json"
            index += 1
    tokenizer.save_tokens(tokenizer.encode(score), out_path)
