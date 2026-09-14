# MxlTok

MxlTok tokenizes **MusicXML** scores into **LMX** (Linearized MusicXML) token
sequences for language models, and decodes token sequences back into
MusicXML.

```
<BOS> num_parts:1 part:instrument:Piano part:start part:instrument:Piano
measure key:fifths:-3 time:num:4 time:den:4 clef:G2 dynamics:p
E5b voice:1 whole
measure E5b voice:1 eighth stem:down beam:begin F5 eighth G5 eighth beam:end ...
part:end <EOS>
```

- **Faithful**: notes, rests, chords, grace notes, voices, staves, ties,
  slurs, beams, tuplets, articulations, ornaments, dynamics, hairpins, pedals,
  tempo, chord symbols and repeats.
- **Readable**: every token is a plain word (`C4#`, `quarter`, `stem:up`).
- **Ready for training**: special tokens, BPE with 🤗 `tokenizers`, and
  tokenizers saved as a single JSON file or on the Hugging Face Hub.
- **Light**: one dependency (`tokenizers`); MusicXML is parsed with the
  standard library.

## Installation

```bash
pip install mxltok           # or: uv add mxltok
pip install "mxltok[hub]"    # to load and push tokenizers on the Hugging Face Hub
```

MxlTok requires Python 3.10 or later.

## Quick start

```python
from mxltok import MXLTokenizer, write_musicxml

tokenizer = MXLTokenizer()

sequence = tokenizer.encode("score.mxl")  # .mxl, .musicxml or .xml
sequence.tokens  # ['<BOS>', 'num_parts:1', 'part:instrument:Piano', ...]
sequence.ids  # [1, 10, 26, ...]

score = tokenizer.decode(sequence.ids)  # ids, tokens or a TokSequence
write_musicxml(score, "decoded.musicxml")  # a .mxl path is compressed
```

`decode` accepts lists, NumPy arrays and PyTorch tensors. Sequences generated
by a model need not be well formed: malformed notes are skipped with a
warning, or raise `MxlTokError` with `decode(ids, strict=True)`.

## Configuration

```python
from mxltok import MXLTokenizer, TokenizerConfig

tokenizer = MXLTokenizer(
    TokenizerConfig(
        arrangement="measure_first",  # or "part_first" (default)
        pitch_mode="absolute",  # or "interval"
        include_grace_notes=True,
        include_rests=True,
        prepend_instruments=True,  # num_parts:N and instrument header
        add_bos_eos=True,
        max_parts=0,  # 0 keeps every part
        excluded_categories={"stem"},  # drop whole token categories
        token_order=None,  # reorder tokens within each note
    )
)
```

| Option | Default | Effect |
|---|---|---|
| `arrangement` | `"part_first"` | `part_first` writes each part in turn; `measure_first` writes each measure with all its parts |
| `pitch_mode` | `"absolute"` | `interval` writes pitches as semitones from the previous one (`interval:-3`) |
| `include_rests`, `include_grace_notes` | `True` | encode rests and grace notes |
| `prepend_instruments` | `True` | start with `num_parts:N` and one `part:instrument:*` per part |
| `add_bos_eos` | `True` | wrap the sequence in `<BOS>` … `<EOS>` |
| `max_parts` | `0` | encode only the first parts (0: all) |
| `excluded_categories` | `set()` | remove token categories, such as `{"stem", "beam"}` (see `TokenCategory`) |
| `token_order` | `None` | sort the tokens of each note by category priority; `{}` uses the default order |
| `special_tokens` | `<PAD> <BOS> <EOS> <UNK> <MASK>` | the special tokens, first in the vocabulary |

The token format is specified in [docs/format.md](docs/format.md).

## BPE

Byte-pair encoding merges frequent token successions into new ids, which
shortens sequences. Train it on the token sequences of a corpus; `encode` then
returns BPE ids and `decode` accepts them:

```python
from pathlib import Path

files = sorted(Path("scores").rglob("*.mxl"))
tokenizer.train_bpe((tokenizer.encode(f).tokens for f in files), vocab_size=10_000)

tokenizer.vocab_size  # 10000: the size of the model's embedding table
tokenizer.encode(files[0]).ids
```

## Saving and loading

```python
tokenizer.save_pretrained("my_tokenizer")  # writes my_tokenizer/tokenizer.json
tokenizer = MXLTokenizer.from_pretrained("my_tokenizer")

tokenizer.push_to_hub("user/mxltok-bpe10k")  # requires mxltok[hub]
tokenizer = MXLTokenizer.from_pretrained("user/mxltok-bpe10k")
```

`from_pretrained` also loads the tokenizers saved by mxlGPT, from a
`tokenizer.json` or from a preprocessed dataset directory (`manifest.json`
with a `tokenizer/` folder), BPE included. MxlTok produces exactly the tokens
and ids of mxlGPT, so models trained with it keep working.

## Scores

`read_musicxml` parses a file into a `Score`, a typed model of the MusicXML
that LMX encodes (parts, measures, notes, attributes, directions...), and
`write_musicxml` writes one. Scores can be edited before encoding, for
instance to transpose them for data augmentation:

```python
from mxltok import read_musicxml, transpose

score = read_musicxml("score.mxl")
sequences = [tokenizer.encode(transpose(score, shift)) for shift in range(-3, 4)]
```

## Development

The project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync                  # create the environment
uv run pytest            # run the tests
uv run ruff check        # lint
uv run ruff format       # format
uv run mypy              # type-check
uv build                 # build the wheel and the source distribution
```

The tests compare MxlTok with reference outputs of mxlGPT's tokenizer, stored
under `tests/fixtures`. `scripts/mxlgpt_parity.py` regenerates them and
compares both tokenizers on any corpus of MusicXML files.

## License

MIT
