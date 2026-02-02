#!/usr/bin/env python3
"""Test expanding window generation and trace extraction."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envelope_decay_fit.segmentation.auto.window_scan import (
    extract_param_trace,
    extract_score_trace,
    generate_expanding_windows,
)
from envelope_decay_fit.plotting import export_plot


def load_envelope_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load envelope data from CSV file."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t_s = data[:, 0]
    env = data[:, 3]  # Column 3 is 'env'
    return t_s, env


def test_windows(csv_path: Path, fn_hz: float):
    """Test expanding window generation on a single file."""
    print(f"\n{'=' * 60}")
    print(f"Testing expanding windows: {csv_path.name}")
    print(f"Natural frequency: {fn_hz:.2f} Hz")
    print(f"{'=' * 60}")

    # Load data
    t, env = load_envelope_csv(csv_path)
    print(f"Loaded {len(t)} samples, duration: {t[-1] - t[0]:.4f} s")

    # For testing, use only a subset (first 1000 points) and downsample windows
    max_points = min(1000, len(t))
    t = t[:max_points]
    env = env[:max_points]
    print(f"Using subset: {len(t)} samples for faster testing")

    # Generate expanding windows
    i_end = len(t)
    print(f"\nGenerating expanding windows from end (i_end={i_end})...")
    # Generate fewer windows by using larger step size
    all_windows = generate_expanding_windows(t, env, fn_hz, i_end, min_points=50)
    # Downsample windows for testing
    step = max(1, len(all_windows) // 100)  # Keep ~100 windows
    windows = all_windows[::step]
    print(f"Generated {len(windows)} windows")

    if len(windows) == 0:
        print("No windows generated!")
        return

    print(
        f"\nWindow size range: {windows[0].n_points} to {windows[-1].n_points} points"
    )
    print(f"Window duration range: {windows[0].dt_s:.4f} to {windows[-1].dt_s:.4f} s")

    # Extract traces
    dt_log, r2_log = extract_score_trace(windows, method="log")
    dt_zeta, zeta_log = extract_param_trace(windows, param="zeta", method="log")

    print(f"\nLOG fit R² range: [{r2_log.min():.4f}, {r2_log.max():.4f}]")
    print(f"LOG fit ζ range: [{zeta_log.min():.6f}, {zeta_log.max():.6f}]")

    # Create diagnostic plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: R² vs Δt
    ax = axes[0]
    ax.plot(dt_log, r2_log, "b.-", alpha=0.7, markersize=2)
    ax.axhline(0.99, color="r", linestyle="--", alpha=0.5, label="R²=0.99")
    ax.set_xlabel("Window duration Δt (s)")
    ax.set_ylabel("R² score (LOG fit)")
    ax.set_title(f"{csv_path.stem} - Fit quality vs window size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([max(0, r2_log.min() - 0.1), 1.0])

    # Plot 2: ζ vs Δt
    ax = axes[1]
    ax.plot(dt_zeta, zeta_log, "g.-", alpha=0.7, markersize=2)
    ax.set_xlabel("Window duration Δt (s)")
    ax.set_ylabel("Damping ratio ζ (LOG fit)")
    ax.set_title("Damping ratio vs window size")
    ax.grid(True, alpha=0.3)

    # Plot 3: Envelope with window endpoints
    ax = axes[2]
    ax.semilogy(t, env, "k-", alpha=0.5, linewidth=1, label="Envelope")

    # Highlight some windows
    for i in [0, len(windows) // 4, len(windows) // 2, 3 * len(windows) // 4, -1]:
        if i < len(windows):
            win = windows[i]
            t_win = t[win.i_start : win.i_end]
            env_win = env[win.i_start : win.i_end]
            ax.semilogy(
                t_win,
                env_win,
                alpha=0.7,
                linewidth=2,
                label=f"Win {i}: Δt={win.dt_s:.3f}s",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope amplitude (log scale)")
    ax.set_title("Sample windows on envelope")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    out_dir = Path("out/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{csv_path.stem}_windows_test.png"
    export_plot(
        fig,
        plot_path,
        dpi=100,
        context={
            "plot_kind": "log_decay",
            "fn_hz": float(fn_hz),
            "method": "log",
            "fit_window": [float(t[0]), float(t[-1])] if len(t) else None,
            "zeta": [float(zeta_log.min()), float(zeta_log.max())]
            if len(zeta_log)
            else None,
            "status": None,
        },
    )
    print(f"\nPlot saved: {plot_path}")
    plt.close()

    # Print sample window details
    print(f"\n--- Sample Window Details ---")
    for i in [0, len(windows) // 2, -1]:
        if i < len(windows):
            win = windows[i]
            print(f"\nWindow {win.win_id}:")
            print(f"  Range: i={win.i_start} to i={win.i_end}, n={win.n_points}")
            print(
                f"  Time: t={win.t_start_s:.4f} to {win.t_end_s:.4f}, Δt={win.dt_s:.4f} s"
            )
            print(
                f"  LOG: valid={win.log_fit.valid}, α={win.log_fit.alpha:.4f}, ζ={win.log_fit.zeta:.6f}, R²={win.log_fit.r2:.4f}"
            )
            print(
                f"  LIN0: valid={win.lin0_fit.valid}, α={win.lin0_fit.alpha:.4f}, ζ={win.lin0_fit.zeta:.6f}, R²={win.lin0_fit.r2:.4f}"
            )
            print(f"  LINC: valid={win.linc_fit.valid}", end="")
            if win.linc_fit.valid:
                print(
                    f", α={win.linc_fit.alpha:.4f}, ζ={win.linc_fit.zeta:.6f}, R²={win.linc_fit.r2:.4f}, C={win.linc_fit.params['C']:.3e}"
                )
            else:
                print()


def main():
    """Test expanding windows on sample data."""
    envelope_dir = Path("data/envelope_exports/free_plate_A3H1")

    if not envelope_dir.exists():
        print(f"Error: {envelope_dir} not found")
        return

    envelope_csvs = sorted(envelope_dir.glob("hit_*.csv"))
    print(f"Found {len(envelope_csvs)} envelope files")

    # Get frequency
    modal_csv = Path(
        "data/examples/free_plate_260122/output/free_plate_A3H1/modal_results.csv"
    )
    if modal_csv.exists():
        try:
            modal_df = pd.read_csv(modal_csv)
            fn_hz = float(modal_df["fn_hz"].iloc[0])
            print(f"Using fn = {fn_hz:.2f} Hz from modal_results.csv")
        except Exception:
            fn_hz = 775.0
            print(f"Using default fn = {fn_hz:.2f} Hz")
    else:
        fn_hz = 775.0
        print(f"Using default fn = {fn_hz:.2f} Hz")

    # Test first file
    test_windows(envelope_csvs[0], fn_hz)


if __name__ == "__main__":
    main()
