#!/usr/bin/env python3
"""Command-line interface for envelope decay fitting."""

import argparse
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

from .api import fit_envelope_decay
from .manual_segmentation import run_manual_segmentation
from .plots import create_piecewise_fit_plot


def load_envelope_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load envelope data from CSV file.

    Expects columns in order: t_s, (other columns...), env
    Or: t_s, raw, filtered, env

    Args:
        csv_path: path to CSV file

    Returns:
        (t, env): time and envelope arrays
    """
    # Try to detect column structure
    with open(csv_path) as f:
        header = f.readline().strip()

    cols = header.split(",")

    # Find t_s and env columns
    t_idx = None
    env_idx = None

    for i, col in enumerate(cols):
        if col.strip().lower() in ["t_s", "t", "time"]:
            t_idx = i
        if col.strip().lower() == "env":
            env_idx = i

    if t_idx is None or env_idx is None:
        raise ValueError(
            f"Could not find t_s and env columns in {csv_path}. Header: {header}"
        )

    # Load data
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    if data.ndim == 1:
        raise ValueError(f"CSV file has only one column: {csv_path}")

    t = data[:, t_idx]
    env = data[:, env_idx]

    return t, env


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fit piecewise exponential decay to envelope data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  env-decay-fit data.csv --fn-hz 775.0
  
  # Specify output directory and piece count
  env-decay-fit data.csv --fn-hz 775.0 --n-pieces 2 --out-dir ./output
  
  # Use specific max windows for performance
  env-decay-fit data.csv --fn-hz 775.0 --max-windows 300
        """,
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file with columns: t_s, env (and optionally others)",
    )

    parser.add_argument(
        "--fn-hz", type=float, required=True, help="Natural frequency in Hz (required)"
    )

    parser.add_argument(
        "--n-pieces",
        type=int,
        default=2,
        help="Number of decay pieces to extract (default: 2)",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots and results (default: ./out/<timestamp>)",
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=500,
        help="Maximum windows per piece for performance (default: 500)",
    )

    parser.add_argument(
        "--manual-segmentation",
        action="store_true",
        help="Enable manual segmentation with interactive Matplotlib",
    )

    parser.add_argument(
        "--manual-min-points",
        type=int,
        default=10,
        help="Minimum samples per manual segment (default: 10)",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    args = parser.parse_args()

    # Validate input file
    if not args.input_csv.exists():
        print(f"Error: Input file not found: {args.input_csv}", file=sys.stderr)
        return 1

    # Set default output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path("out") / timestamp

    # Load data
    print(f"Loading envelope data from {args.input_csv}...")
    try:
        t, env = load_envelope_csv(args.input_csv)
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(t)} samples, duration: {t[-1] - t[0]:.4f} s")
    print(f"Envelope range: [{env.min():.3e}, {env.max():.3e}]")

    # Run fitting
    if args.manual_segmentation:
        print(
            f"\nManual segmentation mode (fn={args.fn_hz:.2f} Hz, min_points={args.manual_min_points})..."
        )
        try:
            manual_result = run_manual_segmentation(
                t,
                env,
                fn_hz=args.fn_hz,
                min_points=args.manual_min_points,
                out_dir=args.out_dir,
            )
        except Exception as e:
            print(f"Error during manual segmentation: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

        if manual_result is None:
            print("Manual segmentation canceled. No results saved.")
            return 0

        result = manual_result

        if args.out_dir is not None:
            plot_path = create_piecewise_fit_plot(
                result, args.out_dir / "piecewise_fit.png"
            )
            result.artifact_paths["piecewise_fit"] = plot_path
    else:
        print(f"\nFitting with fn={args.fn_hz:.2f} Hz, n_pieces={args.n_pieces}...")
        try:
            result = fit_envelope_decay(
                t,
                env,
                fn_hz=args.fn_hz,
                n_pieces=args.n_pieces,
                out_dir=args.out_dir,
                max_windows=args.max_windows,
            )
        except Exception as e:
            print(f"Error during fitting: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    # Print summary
    print("\n" + result.summary())

    # Report artifacts
    if result.artifact_paths:
        print(f"\nArtifacts written to: {args.out_dir}")
        for name, path in result.artifact_paths.items():
            print(f"  {name}: {path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
