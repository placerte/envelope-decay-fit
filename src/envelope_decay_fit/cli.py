#!/usr/bin/env python3
"""Command-line interface for envelope decay fitting."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

from .api import fit_piecewise_manual, launch_manual_segmentation_ui
from .models import FitResult


def load_envelope_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load envelope data from CSV file.

    Expects columns: t_s and env (case-insensitive).
    """
    with open(csv_path) as f:
        header = f.readline().strip()

    cols = [c.strip().lower() for c in header.split(",")]
    t_idx = None
    env_idx = None

    for i, col in enumerate(cols):
        if col in ["t_s", "t", "time"]:
            t_idx = i
        if col == "env":
            env_idx = i

    if t_idx is None or env_idx is None:
        raise ValueError(
            f"Could not find t_s and env columns in {csv_path}. Header: {header}"
        )

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        raise ValueError(f"CSV file has only one column: {csv_path}")

    t = data[:, t_idx]
    env = data[:, env_idx]
    return t, env


def _parse_breakpoints(value: str) -> list[float]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [float(v) for v in items]


def _default_out_dir(prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("out") / prefix / timestamp


def _write_breakpoints_json(path: Path, breakpoints_t: list[float]) -> None:
    payload = {"breakpoints_t": list(breakpoints_t)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _write_fit_result_json(path: Path, fit: FitResult) -> None:
    pieces_payload: list[dict[str, object]] = []
    for piece in fit.pieces:
        pieces_payload.append(
            {
                "piece_id": int(piece.piece_id),
                "t_start_s": float(piece.t_start_s),
                "t_end_s": float(piece.t_end_s),
                "n_points": int(piece.n_points),
                "params": {k: float(v) for k, v in piece.params.items()},
                "r2": float(piece.r2),
                "flags": list(piece.flags),
            }
        )

    payload = {
        "breakpoints_t": list(fit.breakpoints_t),
        "pieces": pieces_payload,
        "global_metrics": None,
    }
    if fit.global_metrics is not None:
        payload["global_metrics"] = {
            "fn_hz": float(fit.global_metrics.fn_hz),
            "omega_n": float(fit.global_metrics.omega_n),
            "n_pieces": int(fit.global_metrics.n_pieces),
            "n_samples": int(fit.global_metrics.n_samples),
            "duration_s": float(fit.global_metrics.duration_s),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _run_segment(args: argparse.Namespace) -> int:
    t, env = load_envelope_csv(args.input_csv)

    breakpoints = launch_manual_segmentation_ui(
        t,
        env,
        fn_hz=args.fn_hz,
    )

    if not breakpoints:
        print("No breakpoints returned.")
        return 1

    out_dir = args.out_dir or _default_out_dir("segment")
    if args.breakpoints_out is None:
        breakpoints_path = out_dir / "breakpoints.json"
    else:
        breakpoints_path = Path(args.breakpoints_out)

    _write_breakpoints_json(breakpoints_path, breakpoints)
    print(f"Breakpoints saved: {breakpoints_path}")
    return 0


def _run_fit(args: argparse.Namespace) -> int:
    t, env = load_envelope_csv(args.input_csv)

    if args.breakpoints is None and args.breakpoints_file is None:
        raise ValueError("Provide --breakpoints or --breakpoints-file")

    breakpoints_t: list[float]
    if args.breakpoints is not None:
        breakpoints_t = _parse_breakpoints(args.breakpoints)
    else:
        with open(args.breakpoints_file) as handle:
            payload = json.load(handle)
        breakpoints_t = [float(v) for v in payload.get("breakpoints_t", [])]

    fit = fit_piecewise_manual(
        t,
        env,
        breakpoints_t,
        fn_hz=args.fn_hz,
    )

    out_dir = args.out_dir or _default_out_dir("fit")
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "fit_result.json"
    _write_fit_result_json(result_path, fit)

    from .plotting import plot_segmentation_storyboard

    fig = plot_segmentation_storyboard(t, env, fit, yscale=args.yscale)
    plot_path = out_dir / "segmentation_storyboard.png"
    fig.savefig(plot_path, dpi=150)
    fig.clf()

    print(f"Fit results saved: {result_path}")
    print(f"Storyboard plot: {plot_path}")
    return 0


def _run_segment_fit(args: argparse.Namespace) -> int:
    t, env = load_envelope_csv(args.input_csv)

    breakpoints = launch_manual_segmentation_ui(
        t,
        env,
        fn_hz=args.fn_hz,
    )

    if not breakpoints:
        print("No breakpoints returned.")
        return 1

    out_dir = args.out_dir or _default_out_dir("segment_fit")
    breakpoints_path = out_dir / "breakpoints.json"
    _write_breakpoints_json(breakpoints_path, breakpoints)

    fit = fit_piecewise_manual(
        t,
        env,
        breakpoints,
        fn_hz=args.fn_hz,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "fit_result.json"
    _write_fit_result_json(result_path, fit)

    from .plotting import plot_segmentation_storyboard

    fig = plot_segmentation_storyboard(t, env, fit, yscale=args.yscale)
    plot_path = out_dir / "segmentation_storyboard.png"
    fig.savefig(plot_path, dpi=150)
    fig.clf()

    print(f"Breakpoints saved: {breakpoints_path}")
    print(f"Fit results saved: {result_path}")
    print(f"Storyboard plot: {plot_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Envelope decay fitting CLI")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.1")
    subparsers = parser.add_subparsers(dest="command")

    segment = subparsers.add_parser(
        "segment", help="Launch manual segmentation UI and save breakpoints"
    )
    segment.add_argument("input_csv", type=Path)
    segment.add_argument("--fn-hz", type=float, required=True)
    segment.add_argument("--out-dir", type=Path, default=None)
    segment.add_argument("--breakpoints-out", type=Path, default=None)
    segment.set_defaults(func=_run_segment)

    fit = subparsers.add_parser("fit", help="Fit using provided breakpoints")
    fit.add_argument("input_csv", type=Path)
    fit.add_argument("--fn-hz", type=float, required=True)
    fit.add_argument("--breakpoints", type=str, default=None)
    fit.add_argument("--breakpoints-file", type=Path, default=None)
    fit.add_argument("--yscale", type=str, default="linear")
    fit.add_argument("--out-dir", type=Path, default=None)
    fit.set_defaults(func=_run_fit)

    segment_fit = subparsers.add_parser(
        "segment-fit",
        help="Launch manual UI, then fit and save outputs",
    )
    segment_fit.add_argument("input_csv", type=Path)
    segment_fit.add_argument("--fn-hz", type=float, required=True)
    segment_fit.add_argument("--yscale", type=str, default="linear")
    segment_fit.add_argument("--out-dir", type=Path, default=None)
    segment_fit.set_defaults(func=_run_segment_fit)

    argv = sys.argv[1:]
    known_commands = {"segment", "fit", "segment-fit"}
    passthrough_flags = {"-h", "--help", "--version"}
    if argv and argv[0] not in known_commands and argv[0] not in passthrough_flags:
        argv = ["segment-fit", *argv]

    args = parser.parse_args(argv)

    if not args.input_csv.exists():
        print(f"Error: Input file not found: {args.input_csv}", file=sys.stderr)
        return 1

    try:
        return args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
