# MxlTok

MxlTok tokenizes **MusicXML** scores into token sequences, and
turns token sequences back into MusicXML.

## Installation

```bash
pip install mxltok            # core: tokenizers, huggingface_hub, numpy
pip install "mxltok[torch]"   # + PyTorch datasets and collator
```

## Quick start

```python
from mxltok import MXL, TokenizerConfig

config = TokenizerConfig(arrangement="part_first", use_harmonies=True)
tokenizer = MXL(config)

tokens = tokenizer("path/to/score.mxl")   # .mxl, .musicxml or .xml -> TokSequence
tokens.tokens[:8]
# ['Part_Piano', 'Bar_None', 'Key_-1', 'TimeSig_3/4', 'Clef_G2', 'Tempo_120',
#  'Dynamics_mp', 'Pitch_C4']
tokens.ids[:4]
# [23, 20, 30, 62]

score = tokenizer(tokens)                  # TokSequence, ids, tokens or tensors -> Score
tokenizer.decode(tokens, output_path="reconstructed.mxl")
```

Calling the tokenizer dispatches on its argument like MidiTok: a path or `Score` is encoded, anything else is decoded. `encode(..., strict=True)` and `decode(..., strict=True)` raise `MxlTokError` on unsupported content instead of warning and skipping it, which is what you want when validating a dataset.

### Training and data pipeline

```python
from pathlib import Path

from torch.utils.data import DataLoader

from mxltok import MXL, TokenizerConfig, split_files_for_training
from mxltok.pytorch_data import DataCollator, DatasetMXL

tokenizer = MXL(TokenizerConfig(arrangement="measure_first"))
files = list(Path("scores").glob("**/*.mxl"))

tokenizer.train(vocab_size=10_000, files_paths=files)     # BPE (or "Unigram", "WordPiece")
tokenizer.save("tokenizer.json")
tokenizer.push_to_hub("username/mxltok-bpe10k", private=True)
tokenizer = MXL.from_pretrained("username/mxltok-bpe10k")

chunks = split_files_for_training(files, tokenizer, Path("chunks"), max_seq_len=1024)
dataset = DatasetMXL(
    chunks,
    tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
for batch in DataLoader(dataset, batch_size=64, collate_fn=collator):
    ...
```

`tokenizer.tokenize_dataset(files, "tokens/")` writes one JSON file of ids per score, to be read back with `DatasetJSON`. `mxltok.transpose(score, semitones)` transposes a score (pitches, keys, chord symbols, accidentals) for augmentation.

## The MXL tokenization

Each note is a group of *prefix* tokens, one *root* token and *suffix* tokens:

```
Part_Piano Bar_None Key_-1 TimeSig_3/4 Clef_G2 Tempo_120 Dynamics_mp
Pitch_C4 Voice_1 Duration_quarter Stem_up Beam_begin Slur_start Articulation_staccato
Chord_None Pitch_E4b Duration_quarter
Grace_slash Pitch_D4 Duration_16th
Pitch_F4# Duration_eighth TimeMod_3in2 Tuplet_start Tie_start
Rest_None Duration_eighth Dot_1
Backup_None Duration_half
HarmonyRoot_C HarmonyKind_major-seventh HarmonyBass_Eb
RestMeasure_None
Bar_None ...
```

| Type | Values | Notes |
|---|---|---|
| `Part` | General MIDI family (`Piano`, `Strings`, `Voice`, ...) | opens a part |
| `Bar` | `None` | one per measure |
| `Key`, `TimeSig`, `Clef`, `Staff` | `-7..7`, `3/4`, `G2`/`F4`/`percussion`, `1..n` | attributes; `Staff` follows `Clef` in multi-staff parts |
| `Tempo` | BPM on a configurable grid | from `<sound tempo>` |
| `Pitch` | `C4`, `F4#`, `B3b`, `E5n`, `G2##`, `A3bb` | step, octave and *printed* accidental |
| `Interval` | `+3`, `-2b` | diatonic steps from the previous note (`pitch_mode="interval"`) |
| `Rest`, `RestMeasure`, `Chord`, `Grace`, `Hidden`, `Backup`, `Forward` | `None` (`Grace_slash`) | prefixes |
| `Voice`, `Staff`, `Stem` | `1..n`, `1..n`, `up`/`down`/`none` | delta-encoded: only emitted when they change |
| `Duration` | `1024th` ... `maxima` | the root of every note, rest, backup and forward |
| `TimeMod`, `Dot` | `3in2`, `1..3` | tuplet ratio, number of dots |
| `Beam`, `Tie`, `Slur`, `Tuplet` | `begin`/`end`/hooks, `start`/`stop` | |
| `Articulation`, `Ornament`, `Tremolo`, `Notation` | MusicXML names | `staccato`, `trill-mark`, `single-3`, `fermata`, ... |
| `Dynamics`, `Wedge`, `Pedal` | `mp`, `crescendo`, `start`, ... | directions |
| `HarmonyRoot`, `HarmonyKind`, `HarmonyBass` | `C`, `Eb`, `major-seventh` | chord symbols |
| `Repeat` | `forward`, `backward` | repeat barlines |

Alterations are not stored on every note: as in the score, they follow from the key signature, the accidentals printed earlier in the measure and ties. The decoder recomputes them, so enharmonic spelling is preserved exactly.

Two serialization orders are available. **Part-first** (`<score-partwise>`) writes all measures of a part before the next part; **measure-first** (`<score-timewise>`) writes, for each measure, the block of every part:

```
part_first:     Part_A Bar Bar Bar  Part_B Bar Bar Bar
measure_first:  Bar Part_A ... Part_B ...  Bar Part_A ... Part_B ...
```

## Configuration

`TokenizerConfig` has one `use_*` flag per notation family (`use_rests`, `use_grace_notes`, `use_voices`, `use_staves`, `use_stems`, `use_beams`, `use_ties`, `use_slurs`, `use_tuplets`, `use_articulations`, `use_ornaments`, `use_notations`, `use_dynamics`, `use_wedges`, `use_pedals`, `use_tempos`, `use_time_signatures`, `use_key_signatures`, `use_clefs`, `use_harmonies`, `use_repeats`, `use_programs`, `use_print_object`), plus `arrangement`, `pitch_mode` / `max_pitch_interval`, `octave_range`, `max_voices`, `max_staves`, `max_dots`, `max_parts`, `tempo_range` / `num_tempos`, `time_signature_range`, `key_range`, `note_types`, `tuplet_ratios`, `instrument_categories`, `special_tokens` and `encode_ids_split`. Unknown keyword arguments are kept in `config.additional_params` for custom tokenizers. The vocabulary is fixed at construction: encoding never adds tokens.

## Writing your own tokenizer

Subclass `mxltok.MusicTokenizer` and implement `_create_base_vocabulary`, `_create_token_types_graph`, `_score_to_tokens` and `_tokens_to_score`. The base class provides the vocabulary, ids/tokens/bytes conversions, `__call__`, training, saving/loading, the Hugging Face Hub integration and `tokenize_dataset`. `mxltok.encoding.PartEncoder` and `mxltok.decoding.PartDecoder` can be reused for the note grammar.

## Development

```bash
uv sync --group dev          # add --group music21 for the semantic round-trip tests
uv run pytest -n auto        # add --update-golden to refresh tests/fixtures/golden
uv run ruff check src tests
uv run ty check src
```

## Citation

If you use MxlTok, please cite the ISMIR 2026 paper:

```bibtex
@inproceedings{poltronieri2026notation,
  author    = {Poltronieri, Andrea and Spanio, Matteo and Chieppa, Simone and Serra, Xavier and Rocamora, Mart\'{i}n},
  title     = {Does Notation Matter? {MusicXML} Tokenisation for Symbolic Music Generation and Understanding},
  booktitle = {Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026},
  address   = {Abu Dhabi, United Arab Emirates},
}
```
