#!/usr/bin/env python3
"""
Fit all valid envelope CSVs in data/envelope_exports/ using fn_hz read from
data/examples/**/output/**/modal_results.csv, and write human-inspectable artifacts.

Outputs (default):
  out/verification/<timestamp>/<dataset_name>/<hit_###>/

Progress:
  prints dataset + overall file counters like "Processing 13/125"
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Development import (matches your existing scripts style)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envelope_decay_fit import fit_envelope_decay  # noqa: E402


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def load_envelope_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load envelope data from CSV file.

    Preferred columns:
      - t_s
      - env

    Fallback:
      - col 0 for time
      - try 'env' column if present; else col 3 (historical test data layout)
    """
    df = pd.read_csv(csv_path)

    # Normalize column names a bit (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Time
    if "t_s" in df.columns:
        t = df["t_s"].to_numpy(dtype=float)
    else:
        # fallback to first column
        t = df.iloc[:, 0].to_numpy(dtype=float)

    # Envelope
    if "env" in df.columns:
        env = df["env"].to_numpy(dtype=float)
    else:
        # historical layout used col 3
        if df.shape[1] >= 4:
            env = df.iloc[:, 3].to_numpy(dtype=float)
        else:
            # last resort: second column
            if df.shape[1] >= 2:
                env = df.iloc[:, 1].to_numpy(dtype=float)
            else:
                raise ValueError(f"Not enough columns to infer env in {csv_path}")

    return t, env


def find_modal_results_csv(dataset_name: str, examples_root: Path) -> Path | None:
    """
    Find the modal_results.csv that corresponds to dataset_name.

    Expected pattern (as per repo structure):
      data/examples/*example_dir*/output/*hit_reference*/modal_results.csv

    Where *hit_reference* often matches dataset_name.
    """
    patterns = [
        f"**/output/**/{dataset_name}/modal_results.csv",
        f"**/output/{dataset_name}/modal_results.csv",
    ]

    matches: list[Path] = []
    for pat in patterns:
        matches.extend(sorted(examples_root.glob(pat)))

    if not matches:
        return None

    # Prefer shortest path (usually the most direct match)
    matches.sort(key=lambda p: (len(p.parts), str(p)))
    return matches[0]


def read_fn_hz(modal_csv: Path) -> float:
    """
    Read fn_hz from a modal_results.csv and return a single representative frequency.

    Strategy:
      - take mean of all finite numeric fn_hz entries
      - if column missing or all invalid -> raise
    """
    df = pd.read_csv(modal_csv)
    if "fn_hz" not in df.columns:
        raise ValueError(f"Missing fn_hz column in {modal_csv}")

    fn = pd.to_numeric(df["fn_hz"], errors="coerce")
    fn = fn[np.isfinite(fn)]
    if len(fn) == 0:
        raise ValueError(f"No valid fn_hz values in {modal_csv}")

    return float(fn.mean())


def iter_envelope_csvs(envelope_root: Path) -> Iterable[tuple[str, Path]]:
    """
    Yield (dataset_name, csv_path) for all hit_*.csv under envelope_root/*/
    """
    for dataset_dir in sorted([d for d in envelope_root.iterdir() if d.is_dir()]):
        dataset_name = dataset_dir.name
        for csv_path in sorted(dataset_dir.glob("hit_*.csv")):
            yield dataset_name, csv_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envelope-root",
        default="data/envelope_exports",
        help="Root directory containing envelope exports (default: data/envelope_exports)",
    )
    parser.add_argument(
        "--examples-root",
        default="data/examples",
        help="Root directory containing modal results (default: data/examples)",
    )
    parser.add_argument(
        "--out-root",
        default="out/verification",
        help="Root directory for outputs (default: out/verification)",
    )
    parser.add_argument(
        "--n-pieces",
        type=int,
        default=2,
        help="Number of decay pieces to extract (default: 2)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=300,
        help="Max expanding windows to evaluate per piece (default: 300)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to include (repeatable; default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned work, but do not run fits.",
    )
    args = parser.parse_args()

    envelope_root = Path(args.envelope_root)
    examples_root = Path(args.examples_root)
    out_root = Path(args.out_root) / _timestamp()

    if not envelope_root.exists():
        print(f"ERROR: envelope root not found: {envelope_root}")
        return 2

    if not examples_root.exists():
        print(f"ERROR: examples root not found: {examples_root}")
        return 2

    jobs = list(iter_envelope_csvs(envelope_root))
    total = len(jobs)

    print("envelope-decay-fit — full dataset sweep")
    print("=" * 78)
    print(f"Envelope root : {envelope_root}")
    print(f"Examples root : {examples_root}")
    print(f"Output root   : {out_root}")
    print(f"Total files   : {total}")
    print(f"n_pieces      : {args.n_pieces}")
    print(f"max_windows   : {args.max_windows}")
    print("=" * 78)

    if total == 0:
        print("No hit_*.csv found.")
        return 0

    if args.dry_run:
        for idx, (dataset_name, csv_path) in enumerate(jobs, start=1):
            print(f"[{idx:>4}/{total}] {dataset_name} :: {csv_path.name}")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    skipped: list[str] = []

    # Per-dataset counters (for nicer UX)
    # Build a map dataset -> list of csv paths
    dataset_map: dict[str, list[Path]] = {}
    for dataset_name, csv_path in jobs:
        dataset_map.setdefault(dataset_name, []).append(csv_path)

    if args.dataset:
        requested = set(args.dataset)
        available = set(dataset_map.keys())
        missing = sorted(requested - available)
        if missing:
            print(f"ERROR: dataset(s) not found: {', '.join(missing)}")
            return 2

        dataset_map = {name: dataset_map[name] for name in sorted(requested)}
        total = sum(len(paths) for paths in dataset_map.values())

    overall_i = 0
    for dataset_name in sorted(dataset_map.keys()):
        csv_paths = dataset_map[dataset_name]
        modal_csv = find_modal_results_csv(dataset_name, examples_root)

        if modal_csv is None:
            msg = f"{dataset_name}: no matching modal_results.csv found"
            print(f"\n[SKIP] {msg}")
            skipped.append(msg)
            overall_i += len(csv_paths)
            continue

        try:
            fn_hz = read_fn_hz(modal_csv)
        except Exception as e:
            msg = f"{dataset_name}: failed reading fn_hz from {modal_csv} ({e})"
            print(f"\n[SKIP] {msg}")
            skipped.append(msg)
            overall_i += len(csv_paths)
            continue

        print(f"\n{'=' * 78}")
        print(f"DATASET: {dataset_name}")
        print(f"modal_results.csv: {modal_csv}")
        print(f"fn_hz: {fn_hz:.6f}")
        print(f"files: {len(csv_paths)}")
        print(f"{'=' * 78}")

        for j, csv_path in enumerate(csv_paths, start=1):
            overall_i += 1

            # Progress line (low noise)
            print(
                f"[{overall_i:>4}/{total}] {dataset_name} [{j:>3}/{len(csv_paths)}] :: {csv_path.name}"
            )

            # Human-friendly per-hit output folder
            hit_stem = csv_path.stem  # e.g., hit_001
            hit_out_dir = out_root / dataset_name / hit_stem
            hit_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                t, env = load_envelope_csv(csv_path)

                # Run fit AND write artifacts to hit_out_dir
                _ = fit_envelope_decay(
                    t,
                    env,
                    fn_hz=fn_hz,
                    n_pieces=args.n_pieces,
                    max_windows=args.max_windows,
                    out_dir=str(hit_out_dir),
                )

            except Exception as e:
                msg = f"{dataset_name}/{csv_path.name}: {e}"
                failures.append(msg)

                # Also write a short error marker inside the hit folder
                (hit_out_dir / "ERROR.txt").write_text(msg + "\n", encoding="utf-8")

    # Final summary
    print("\n" + "=" * 78)
    print("DONE")
    print(f"Output root: {out_root}")

    if skipped:
        print(f"Skipped datasets: {len(skipped)}")
        (out_root / "SKIPPED.txt").write_text(
            "\n".join(skipped) + "\n", encoding="utf-8"
        )

    if failures:
        print(f"Failures: {len(failures)}")
        (out_root / "FAILURES.txt").write_text(
            "\n".join(failures) + "\n", encoding="utf-8"
        )
        return 1

    print("All OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
