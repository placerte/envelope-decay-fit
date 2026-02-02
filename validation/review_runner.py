#!/usr/bin/env python3
"""
Run all repo examples / validation scripts and save outputs in a human-friendly folder.

Usage:
  uv run python validation/review_runner.py
  uv run python validation/review_runner.py --skip-all-datasets
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


def run_step(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    out_dir: Path,
    env: dict[str, str] | None = None,
) -> int:
    step_dir = out_dir / name
    step_dir.mkdir(parents=True, exist_ok=True)

    log_path = step_dir / "run.log"
    print(f"\n{'=' * 78}")
    print(f"STEP: {name}")
    print(f"CMD : {' '.join(cmd)}")
    print(f"LOG : {log_path}")
    print(f"{'=' * 78}")

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"STEP: {name}\n")
        f.write(f"CMD : {' '.join(cmd)}\n")
        f.write(f"CWD : {cwd}\n\n")
        f.flush()

        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if p.returncode != 0:
        print(f"❌ FAILED: {name} (exit code {p.returncode})")
    else:
        print(f"✅ OK: {name}")
    return p.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        default="out/run_all_examples",
        help="Root output directory (default: out/run_all_examples)",
    )
    parser.add_argument(
        "--skip-all-datasets",
        action="store_true",
        help="Skip validation/review_all_datasets.py (can be slow/noisy).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = repo_root / args.out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Helpful header
    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                "envelope-decay-fit — run_all_examples output",
                f"timestamp: {ts}",
                f"repo_root: {repo_root}",
                "",
                "Each step has its own folder containing:",
                "  - run.log  (stdout+stderr combined)",
                "",
                "If a step generates figures, they will typically land under ./out/ as well.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Ensure child scripts that write to ./out do so in the repo root.
    # Also keep things deterministic-ish.
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"

    # Use current python (so `uv run python ...` works cleanly)
    py = sys.executable

    failures: list[str] = []

    # 1) Quickstart (creates out/quickstart by default in that script)
    rc = run_step(
        name="01_quickstart",
        cmd=[py, "examples/quickstart.py"],
        cwd=repo_root,
        out_dir=out_dir,
        env=env,
    )
    if rc != 0:
        failures.append("01_quickstart")

    # 2) Expanding windows diagnostic
    rc = run_step(
        name="02_test_windows",
        cmd=[py, "validation/bench_windows.py"],
        cwd=repo_root,
        out_dir=out_dir,
        env=env,
    )
    if rc != 0:
        failures.append("02_test_windows")

    # 3) Fitters sanity on one dataset
    rc = run_step(
        name="03_test_fitters",
        cmd=[py, "validation/bench_fitters.py"],
        cwd=repo_root,
        out_dir=out_dir,
        env=env,
    )
    if rc != 0:
        failures.append("03_test_fitters")

    # 4) Full dataset sweep (optional)
    if not args.skip_all_datasets:
        rc = run_step(
            name="04_test_all_datasets",
            cmd=[py, "validation/review_all_datasets.py"],
            cwd=repo_root,
            out_dir=out_dir,
            env=env,
        )
        if rc != 0:
            failures.append("04_test_all_datasets")
    else:
        (out_dir / "04_test_all_datasets_SKIPPED.txt").write_text(
            "Skipped via --skip-all-datasets\n", encoding="utf-8"
        )

    # Summary
    summary = out_dir / "SUMMARY.txt"
    if failures:
        summary.write_text(
            "FAILURES:\n" + "\n".join(f"- {s}" for s in failures) + "\n",
            encoding="utf-8",
        )
        print(f"\n❌ Some steps failed. See: {summary}")
        print(f"Output folder: {out_dir}")
        return 1

    summary.write_text("ALL OK\n", encoding="utf-8")
    print(f"\n✅ All steps passed.")
    print(f"Output folder: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
