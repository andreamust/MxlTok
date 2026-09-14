# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- `MXLTokenizer`: MusicXML to LMX tokens and ids, and back, in part-first or
  measure-first arrangement, with absolute or interval pitches.
- The LMX format of mxlGPT, token for token: same vocabulary, same ids, same
  tokens on every score. Tokenizers saved by mxlGPT (`tokenizer.json` or
  preprocessed dataset directories, BPE included) load with
  `MXLTokenizer.from_pretrained`.
- BPE training and encoding with 🤗 `tokenizers`, compatible with mxlGPT's
  BPE models.
- A typed `Score` model with `read_musicxml` and `write_musicxml`
  (`.musicxml`, `.xml` and compressed `.mxl`).
- `transpose` for pitch-shift data augmentation.
- Token categories, with category filtering and reordering.
- Saving to and loading from the Hugging Face Hub (`mxltok[hub]`).

### Fixed (compared with mxlGPT)

- Backward repeats are decoded in their own measure; mxlGPT's decoder moved
  them one measure early.
- Malformed model output no longer aborts decoding: malformed notes are
  skipped with a warning (or raise in `strict` mode).
