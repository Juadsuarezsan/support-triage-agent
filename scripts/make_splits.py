"""Deterministic 80/10/10 stratified split of the Bitext CSV by intent.

Writes one file per split with the zero-based row indices of the raw CSV
(``data/splits/{train,val,test}.txt``) plus ``data/splits/SPLIT_INFO.json``
with the seed, the counts per intent and the SHA-256 of the source file, so
the split is reproducible and auditable without shipping the data.

Usage::

    python scripts/make_splits.py                       # data/raw/bitext/*.csv
    python scripts/make_splits.py --csv path/to.csv --out data/splits
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = (
    ROOT
    / "data"
    / "raw"
    / "bitext"
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
DEFAULT_OUT = ROOT / "data" / "splits"
SEED = 20260516
FRACTIONS = (0.8, 0.1, 0.1)


def read_intents(csv_path: Path, column: str = "intent") -> list[str]:
    """Return the intent label of every row, in file order."""
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"column {column!r} not found in {csv_path.name}")
        return [str(row[column]) for row in reader]


def stratified_split(
    labels: list[str], seed: int = SEED, fractions: tuple[float, float, float] = FRACTIONS
) -> dict[str, list[int]]:
    """Split row indices per label into train/val/test with fixed proportions.

    Every label contributes ``round(n * f)`` rows to val and test and the rest
    to train, so small classes are never absent from the held-out sets.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    rng = random.Random(seed)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        idx = by_label[label][:]
        rng.shuffle(idx)
        n = len(idx)
        n_val = round(n * fractions[1])
        n_test = round(n * fractions[2])
        out["test"].extend(idx[:n_test])
        out["val"].extend(idx[n_test : n_test + n_val])
        out["train"].extend(idx[n_test + n_val :])
    for split in out.values():
        split.sort()
    return out


def write_splits(
    splits: dict[str, list[int]], labels: list[str], csv_path: Path, out: Path
) -> None:
    """Persist the index files and the split metadata."""
    out.mkdir(parents=True, exist_ok=True)
    for name, idx in splits.items():
        (out / f"{name}.txt").write_text("\n".join(map(str, idx)) + "\n", encoding="utf-8")
    per_intent: dict[str, dict[str, int]] = {}
    for name, idx in splits.items():
        for i in idx:
            per_intent.setdefault(labels[i], {"train": 0, "val": 0, "test": 0})[name] += 1
    info = {
        "source_file": csv_path.name,
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "seed": SEED,
        "fractions": {"train": FRACTIONS[0], "val": FRACTIONS[1], "test": FRACTIONS[2]},
        "counts": {name: len(idx) for name, idx in splits.items()},
        "per_intent": dict(sorted(per_intent.items())),
    }
    (out / "SPLIT_INFO.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not args.csv.exists():
        print(f"missing {args.csv}; run scripts/download_data.py --bitext first", file=sys.stderr)
        return 1
    labels = read_intents(args.csv)
    splits = stratified_split(labels)
    write_splits(splits, labels, args.csv, args.out)
    print({k: len(v) for k, v in splits.items()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
