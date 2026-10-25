# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MxlTok is a PyPI package that tokenizes MusicXML scores into LMX (Linearized
MusicXML) token sequences and decodes them back. It is the clean rewrite of
the tokenizer of the mxlGPT research repo (`../mxlGPT`, messy, kept as the
reference). Training and evaluation code will be added to this repo later;
for now it holds only the tokenizer.

## Commands

```bash
uv sync                                   # environment (Python 3.12 locally, >=3.10 supported)
uv run pytest                             # all tests
uv run pytest tests/test_encoding.py::test_tokens_match_mxlgpt   # one test
uv run pytest -k "860 and measure_first"  # tests of one fixture
uv run ruff check && uv run ruff format   # lint and format
uv run mypy                               # strict type-check of src/
uv build                                  # wheel + sdist
```

Parity with mxlGPT (needs the `../mxlGPT` checkout; runs in its venv):

```bash
PYTHONPATH=../mxlGPT/src:src ../mxlGPT/.venv/bin/python scripts/mxlgpt_parity.py golden tests/fixtures
PYTHONPATH=../mxlGPT/src:src ../mxlGPT/.venv/bin/python scripts/mxlgpt_parity.py compare files.txt --arrangement measure_first --workers 26
```

A large local corpus for `compare` lives in `~/Documents/MuseScore-big-MXL`
(1.5M `.mxl`); the last check on 20k files gave 0 differences in both
arrangements.

## The core invariant: token parity with mxlGPT

Models were trained with mxlGPT's `MXLTokenizer`. MxlTok must produce **the
same tokens and ids on every score**, and load mxlGPT's saved tokenizers
(`tokenizer.json`, or a preprocessed dataset dir with `manifest.json` +
`tokenizer/` + `bpe_tokenizer.json`). Therefore:

- Never change the order or content of `build_vocabulary` (ids depend on it).
- Many odd encoding rules are deliberate compatibility quirks (backward
  repeats before the `measure` token, measures without content emitting no
  `measure` token, first `<notations>`/`<articulations>` only, clef without
  `<sign>` repeating the last clef, state mutated before a note is dropped…).
  They are listed in `docs/format.md` ("Compatibility notes"). Don't "fix"
  encoding behaviour; decoding may be improved (it already fixes mxlGPT's
  backward-repeat bug).
- `tests/fixtures/golden/*` are mxlGPT outputs; regenerate them only with the
  parity script, never by hand from MxlTok's own output.

## Architecture

Pipeline: `musicxml.read_musicxml` → `score.Score` → `encoding.PartEncoder`
(per part) → `arrangement.join_parts` → filtering/reordering
(`categories`) → BOS/EOS → ids (`tokenizer`, optionally `bpe.BPE`). Decoding
runs the reverse: `arrangement.split_parts` → `decoding.PartDecoder` →
`Score` → `musicxml.write_musicxml`.

- `score.py`: typed dataclass model of the MusicXML subset LMX encodes;
  durations are `Fraction` quarters (divisions resolved on read, recomputed on
  write). Also the alteration logic (`resolve_alters` for decoding,
  `assign_accidentals` for `transpose`).
- `musicxml.py`: all XML knowledge. The reader mirrors mxlGPT's extraction
  rules where they affect tokens; integer fields that end up in tokens are
  parsed strictly (bad values raise `MxlTokError`), metadata leniently.
- `encoding.py`: all token-emission knowledge; delta encoding of
  voice/stem/staff per measure. Recoverable problems warn and drop the
  element (`_Unencodable`); out-of-vocabulary content fails the whole score.
- `decoding.py`: groups tokens into prefix/root/suffix groups, then builds
  elements; malformed groups warn (or raise with `strict=True`).
- `vocabulary.py` holds token families and the vocabulary; `instruments.py`
  the General MIDI category resolution; `config.py` the `TokenizerConfig`
  (its `from_dict` silently ignores mxlGPT's legacy keys).
- `tokenizer.py`: the public `MXLTokenizer` facade: vocab/ids (BPE ids when a
  BPE model is attached), encode/decode, save/load, Hugging Face Hub
  (`huggingface_hub` is an optional extra, imported lazily).
- `bpe.py`: maps each base token to a CJK character and runs a 🤗 `tokenizers`
  BPE over the strings; the mapping must stay compatible with mxlGPT's
  `HFTokenBPE`.

## Conventions

- Code clarity is the top priority: small functions, Google-style docstrings
  on public API, comments only for non-obvious reasons.
- Ruff (incl. pydocstyle) and strict mypy must pass; target Python 3.10.
- The only runtime dependency is `tokenizers`; keep it that way.
- Version lives in `src/mxltok/__init__.py` (hatch dynamic version); update
  `CHANGELOG.md`. Publishing runs on GitHub releases via trusted publishing
  (`.github/workflows/publish.yml`).
- Work on feature branches (current: `feat/lmx-compat`, based on `origin/dev`).
