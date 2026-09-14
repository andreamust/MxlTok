"""Check MxlTok against the reference LMX tokenizer of mxlGPT.

MxlTok must produce exactly the tokens of mxlGPT's ``MXLTokenizer``, so that
the models trained with mxlGPT keep working. This script needs an mxlGPT
checkout and runs in its environment, with both packages importable:

    PYTHONPATH=../mxlGPT/src:src ../mxlGPT/.venv/bin/python \\
        scripts/mxlgpt_parity.py golden tests/fixtures

    PYTHONPATH=../mxlGPT/src:src ../mxlGPT/.venv/bin/python \\
        scripts/mxlgpt_parity.py compare files.txt --arrangement measure_first

``golden`` regenerates the reference files of the test suite; ``compare``
encodes every file listed (one path per line) with both tokenizers and
reports the differences.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tokenizer import MXLTokenizer as ReferenceTokenizer
from tokenizer import TokenizerConfig as ReferenceConfig
from tokenizer.core.hf_bpe import HFTokenBPE

from mxltok import MXLTokenizer, TokenizerConfig

warnings.simplefilter("ignore")

ARRANGEMENTS = ("part_first", "measure_first")


def reference(arrangement: str, **options: object) -> ReferenceTokenizer:
    """Return mxlGPT's tokenizer with rests and grace notes, as MxlTok's default."""
    config = ReferenceConfig(
        include_rests=True,
        include_grace_notes=True,
        arrangement=arrangement,
        add_bos_eos=False,
    )
    for name, value in options.items():
        setattr(config, name, value)
    return ReferenceTokenizer(config)


def reference_roundtrip(tokenizer: ReferenceTokenizer, tokens: list[str]) -> list[str]:
    """Return the tokens of the score mxlGPT decodes from ``tokens``."""
    with tempfile.NamedTemporaryFile("w", suffix=".musicxml", delete=False) as file:
        file.write(tokenizer.decode(tokens))
    try:
        return tokenizer.tokenize(file.name).tokens
    finally:
        Path(file.name).unlink()


def write_tokens(path: Path, tokens: list[str]) -> None:
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# golden
# --------------------------------------------------------------------------- #


def make_golden(fixtures: Path) -> None:
    """Write the reference outputs the test suite compares against."""
    scores = sorted((fixtures / "mxl").iterdir())
    golden = fixtures / "golden"
    golden.mkdir(exist_ok=True)

    for arrangement in ARRANGEMENTS:
        tokenizer = reference(arrangement)
        for score in scores:
            try:
                tokens = tokenizer.tokenize(str(score)).tokens
            except Exception:  # files mxlGPT rejects are tested apart
                continue
            write_tokens(golden / f"{score.stem}.{arrangement}.lmx", tokens)
            write_tokens(
                golden / f"{score.stem}.{arrangement}.roundtrip.lmx",
                reference_roundtrip(tokenizer, tokens),
            )

    interval = reference("part_first", pitch_mode="interval")
    filtered = reference(
        "part_first",
        add_bos_eos=True,
        excluded_categories={"stem", "beam"},
        token_order={},
    )
    for score in scores:
        try:
            write_tokens(
                golden / f"{score.stem}.interval.lmx",
                interval.tokenize(str(score)).tokens,
            )
            write_tokens(
                golden / f"{score.stem}.filtered.lmx",
                filtered.tokenize(str(score)).tokens,
            )
        except Exception:
            continue

    special = ReferenceConfig().special_tokens.as_list()
    vocab = {
        mode: reference("part_first", pitch_mode=mode).vocab
        for mode in ("absolute", "interval")
    }
    (golden / "vocab.json").write_text(json.dumps(vocab, indent=1) + "\n")
    _make_saved_tokenizers(fixtures, scores, special)


def _make_saved_tokenizers(
    fixtures: Path, scores: list[Path], special: list[str]
) -> None:
    """Save a BPE tokenizer as mxlGPT does, with the ids it gives each score."""
    # The configuration mxlGPT's preprocessing used: grace notes excluded.
    tokenizer = ReferenceTokenizer(
        ReferenceConfig(include_rests=True, arrangement="measure_first")
    )
    sequences = {}
    for score in scores:
        try:
            sequences[score.name] = tokenizer.tokenize(str(score)).tokens
        except Exception:
            continue
    bpe = HFTokenBPE()
    bpe.train(
        iterator=list(sequences.values()),
        vocab_size=900,
        base_vocab=tokenizer.token_to_id,
        special_tokens=special,
    )
    tokenizer._hf_bpe = bpe
    tokenizer.save_pretrained(fixtures / "mxlgpt_checkpoint")

    # A preprocessed dataset directory: manifest, multi-file tokenizer, BPE.
    dataset = fixtures / "mxlgpt_dataset"
    tokenizer._hf_bpe = None
    tokenizer.save(dataset / "tokenizer")
    bpe.save(dataset / "tokenizer" / "bpe_tokenizer.json")
    manifest = {"tokenizer_type": "lmx", "has_bpe": True}
    (dataset / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    ids = {name: bpe.encode(tokens) for name, tokens in sequences.items()}
    (fixtures / "golden" / "bpe_ids.json").write_text(json.dumps(ids) + "\n")


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #


def _attempt(encode: Callable[[str], list[str]], path: str) -> list[str] | str:
    try:
        return encode(path)
    except Exception as exc:  # failures are compared too
        return f"{type(exc).__name__}"


def _compare_one(args: tuple[str, str]) -> tuple[str, str, str]:
    path, arrangement = args
    expected = _attempt(lambda p: reference(arrangement).tokenize(p).tokens, path)
    config = TokenizerConfig(arrangement=arrangement, add_bos_eos=False)  # type: ignore[arg-type]
    actual = _attempt(lambda p: MXLTokenizer(config).encode(p).tokens, path)
    if isinstance(expected, str) or isinstance(actual, str):
        if isinstance(expected, str) != isinstance(actual, str):
            return (
                path,
                "one failed",
                f"mxlGPT: {expected!s:.40}, MxlTok: {actual!s:.40}",
            )
        return path, "both failed", f"{expected} / {actual}"
    if expected == actual:
        return path, "identical", ""
    index = next(
        (i for i, (a, b) in enumerate(zip(expected, actual, strict=False)) if a != b),
        min(len(expected), len(actual)),
    )
    window = slice(max(0, index - 5), index + 5)
    return path, "different", f"at {index}: {expected[window]} != {actual[window]}"


def compare(file_list: Path, arrangement: str, workers: int) -> None:
    """Encode every listed file with both tokenizers and report differences."""
    paths = [line for line in file_list.read_text().splitlines() if line]
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(workers) as pool:
        jobs = ((path, arrangement) for path in paths)
        for path, status, detail in pool.map(_compare_one, jobs, chunksize=16):
            counts[status] = counts.get(status, 0) + 1
            if status in ("different", "one failed"):
                print(f"{status.upper()} {path}\n    {detail}")
    print(json.dumps({"files": len(paths), **counts}))
    sys.exit(1 if counts.get("different") or counts.get("one failed") else 0)


def main() -> None:
    """Run the command given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    golden = commands.add_parser("golden", help="regenerate the test references")
    golden.add_argument("fixtures", type=Path)
    comparison = commands.add_parser("compare", help="compare on a list of files")
    comparison.add_argument("file_list", type=Path)
    comparison.add_argument("--arrangement", choices=ARRANGEMENTS, default="part_first")
    comparison.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "golden":
        make_golden(args.fixtures)
    else:
        compare(args.file_list, args.arrangement, args.workers)


if __name__ == "__main__":
    main()
