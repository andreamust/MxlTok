# The LMX token format

LMX (Linearized MusicXML) writes a MusicXML score as a flat sequence of
string tokens. This document specifies the format MxlTok produces. It is
exactly the format of mxlGPT's `MXLTokenizer`, so models trained with mxlGPT
read MxlTok's tokens unchanged.

## Sequence

```
<BOS> num_parts:2 part:instrument:Strings part:instrument:Piano
part:start part:instrument:Strings measure ... measure ... part:end
part:start part:instrument:Piano   measure ... measure ... part:end
<EOS>
```

- `<BOS>` / `<EOS>` wrap the sequence (`add_bos_eos`).
- `num_parts:N` and one `part:instrument:<category>` per part form the header
  (`prepend_instruments`). The vocabulary holds `num_parts:1` to
  `num_parts:16`, so scores with more parts cannot be encoded (or use
  `max_parts`).
- Each part is labelled with one of 19 General MIDI categories: the 16
  program families (`Piano`, `Chromatic-Percussion`, `Organ`, `Guitar`, `Bass`,
  `Strings`, `Ensemble`, `Brass`, `Reed`, `Pipe`, `Synth-Lead`, `Synth-Pad`,
  `Synth-Effects`, `Ethnic`, `Percussive`, `Sound-Effects`) plus `Drums`,
  `Voice` and `Other`. The category comes from the part's MIDI channel
  (10 is `Drums`) or program, else from its name (exact, substring, then
  fuzzy match against a multilingual list of instrument names).

### Arrangements

`part_first` (default) writes all measures of a part before the next part.
`measure_first` writes, for each measure, every part in turn:

```
measure part:start part:instrument:Strings ... part:end
        part:start part:instrument:Piano   ... part:end
measure ...
```

## Measures

Every measure starts with `measure` and lists its elements in document order:
signatures, notes, cursor moves, directions and chord symbols. A measure with
nothing to encode has no `measure` token at all.

### Signatures

| Element | Tokens |
|---|---|
| key signature | `key:fifths:-7` … `key:fifths:7` |
| time signature | `time:num:1` … `time:num:16`, then `time:den:{2,4,8,16}` |
| clef | `clef:G1`…`clef:G5`, `clef:F1`…`clef:F5`, `clef:C1`…`clef:C5`, `clef:PERCUSSION` |
| clef of a staff | the clef, then `staff:N`, when the attributes declare `<staves>` |

A `<clef>` without `<sign>` repeats the last clef of its staff.

### Notes

A note is a group of *prefix* tokens, one *root* token and *suffix* tokens:

```
[print-object:no] [grace [grace:slash]] [chord] <pitch | rest> [voice:N]
<note type | rest:measure>
[ratio] [dot]* [stem:S] [staff:N] [beam:B]* [tied:T]* [tuplet:T]* [marking]* [slur:T]*
```

| Token | Values |
|---|---|
| pitch | `A0` … `B9`, optionally followed by an accidental suffix: `#`, `b`, `##`, `bb`, `n` (`C4#`, `B3bb`, `E5n`) |
| note type | `1024th` `512th` `256th` `128th` `64th` `32nd` `16th` `eighth` `quarter` `half` `whole` `breve` `long` `maxima` |
| ratio | tuplet ratio `<actual>in<normal>` (`3in2`, `5in4`, …; 29 ratios) |
| stem | `stem:up`, `stem:down`, `stem:none` |
| beam | `beam:begin`, `beam:end`, `beam:forward-hook`, `beam:backward-hook` (`continue` is implied) |
| tie, tuplet, slur | `tied:start` `tied:stop`, `tuplet:start` `tuplet:stop`, `slur:start` `slur:stop` |
| marking | `staccato` `accent` `strong-accent` `tenuto` `stress` `spiccato` `breath-mark` `caesura` `fermata` `arpeggiate` `trill-mark` `turn` `inverted-turn` `mordent` `inverted-mordent` `tremolo:<type>` `tremolo:<marks>` `glissando:start` `glissando:stop` |

The accidental suffix of a pitch is the printed accidental when there is one,
else the accidental implied by the sounding alteration: in E♭ major, an
unmarked E♭ is `E5b`.

`voice`, `stem` and `staff` are **delta-encoded**: they are written only when
they differ from the previous note of the measure. A `backup` resets them.
`staff:N` is written only in parts whose clefs announced two staves or more.

A measure rest is `rest rest:measure`. A grace note without a type is an
`eighth`. Unpitched (percussion) notes are not encoded.

### Cursor moves

`<backup>` and `<forward>` are written as `backup <type>` (or `forward <type>`)
pairs whose note types add up to the duration, longest first:
`backup half backup quarter` is a backup of a dotted half note. A duration
that is not a sum of distinct note types (tuplet moves) is encoded as far as
possible, with a warning.

### Directions and chord symbols

| Element | Tokens |
|---|---|
| dynamics | `dynamics:{ppp,pp,p,mp,mf,f,ff,fff,sfz,fp,sf,rfz}` |
| hairpin | `wedge:crescendo`, `wedge:diminuendo`, `wedge:stop` |
| pedal | `pedal:start`, `pedal:stop`, `pedal:change` |
| tempo | `tempo:N`, the BPM quantized to the nearest of 40, 50, 60, 66, 72, 76, 80, …, 200, 208 |
| chord symbol | `harmony:root:<pitch>` `harmony:quality:<quality>` [`harmony:ext:<extension>`] [`harmony:bass:<pitch>`] |

Chord pitches are a step with an optional `b` or `#` (`harmony:root:Bb`). The
MusicXML `<kind>` is split into a quality and an extension:
`major-seventh` is `harmony:quality:major harmony:ext:seventh`.

### Repeats

`repeat:forward` and `repeat:backward` are written right before the `measure`
token of the measure whose barline carries the repeat.

## Interval mode

With `pitch_mode="interval"`, every pitch but the first of a measure is
written as the number of semitones from the previous pitch: `interval:-3`,
`interval:0`, `interval:7`. Decoded pitches are spelled with sharps.

## Vocabulary

The vocabulary is fixed and deterministic: 720 tokens (975 in interval mode),
the special tokens first. Its order defines the token ids, so it never changes.
See `mxltok/vocabulary.py`.

## Compatibility notes

These behaviours are kept for compatibility with mxlGPT's tokens, even where
another choice would be more natural:

- **Backward repeats** precede the `measure` token of their own measure, like
  forward repeats. (mxlGPT's *decoder* misread them as belonging to the
  previous measure; MxlTok decodes them where they belong.)
- **Measures without encodable content** (only unpitched notes, say) produce
  no `measure` token, so in `measure_first` the parts of such a score drift
  apart.
- In `measure_first`, a forward repeat at the start of a part becomes a
  measure of its own.
- **Unsupported content fails the whole score** rather than being dropped:
  more than 16 parts, TAB clefs, voices above 16, staves above 4, tuplet
  ratios outside the list above. Such scores raise `MxlTokError` (mxlGPT
  excluded them from its datasets).
- The tokenizers mxlGPT saved for training **exclude grace notes**
  (`include_grace_notes=False`). The default of MxlTok includes them. Loading
  a saved tokenizer restores its exact configuration.
- Some tempo tokens of the vocabulary (`tempo:42`, `tempo:63`, …) are never
  produced, as the tempo grid lacks them.
- mxlGPT's token-level pitch-shift augmentation (`pitch_shift_range`) is not
  supported: use `mxltok.transpose` on scores instead.
