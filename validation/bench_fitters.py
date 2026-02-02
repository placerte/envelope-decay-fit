#!/usr/bin/env python3
"""Quick validation script to test fitters on real data."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for direct import during development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envelope_decay_fit.fitters import fit_log_domain, fit_lin0_domain, fit_linc_domain
from envelope_decay_fit.plotting import export_plot


def load_envelope_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load envelope data from CSV file.

    Expects columns: t_s, env (other columns ignored).
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t_s = data[:, 0]
    env = data[:, 3]  # Column 3 is 'env' in the test data
    return t_s, env


def test_single_file(csv_path: Path, fn_hz: float, hit_id: int = 1):
    """Test all three fitters on a single envelope file."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {csv_path.name}")
    print(f"Natural frequency: {fn_hz:.2f} Hz")
    print(f"{'=' * 60}")

    # Load data
    t, env = load_envelope_csv(csv_path)
    print(f"Loaded {len(t)} samples, duration: {t[-1] - t[0]:.4f} s")
    print(f"Envelope range: [{env.min():.2e}, {env.max():.2e}]")

    # Test LOG fit
    print("\n--- LOG Domain Fit ---")
    log_result = fit_log_domain(t, env, fn_hz)
    print(f"Valid: {log_result.valid}")
    if log_result.valid:
        print(f"  α = {log_result.alpha:.6f} (1/s)")
        print(f"  ζ = {log_result.zeta:.6f}")
        print(f"  R² = {log_result.r2:.6f}")
        print(f"  RMSE = {log_result.rmse:.6e}")
    else:
        print(f"  Notes: {log_result.notes}")

    # Test LIN0 fit
    print("\n--- LIN0 Domain Fit (no floor) ---")
    lin0_result = fit_lin0_domain(t, env, fn_hz)
    print(f"Valid: {lin0_result.valid}")
    if lin0_result.valid:
        print(f"  α = {lin0_result.alpha:.6f} (1/s)")
        print(f"  ζ = {lin0_result.zeta:.6f}")
        print(f"  R² = {lin0_result.r2:.6f}")
        print(f"  RMSE = {lin0_result.rmse:.6e}")
        print(f"  A = {lin0_result.params['A']:.6e}")
    else:
        print(f"  Notes: {lin0_result.notes}")

    # Test LINC fit
    print("\n--- LINC Domain Fit (with floor) ---")
    linc_result = fit_linc_domain(t, env, fn_hz)
    print(f"Valid: {linc_result.valid}")
    if linc_result.valid:
        print(f"  α = {linc_result.alpha:.6f} (1/s)")
        print(f"  ζ = {linc_result.zeta:.6f}")
        print(f"  R² = {linc_result.r2:.6f}")
        print(f"  RMSE = {linc_result.rmse:.6e}")
        print(f"  A = {linc_result.params['A']:.6e}")
        print(f"  C = {linc_result.params['C']:.6e}")
    else:
        print(f"  Notes: {linc_result.notes}")

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Linear scale
    ax = axes[0]
    ax.plot(t, env, "k-", alpha=0.5, label="Envelope data", linewidth=1)

    t_ref = t[0]
    t_shifted = t - t_ref

    if log_result.valid:
        # LOG fit gives us b (intercept in log space), reconstruct A
        A_log = np.exp(log_result.params["b"])
        env_log = A_log * np.exp(-log_result.alpha * t_shifted)
        ax.plot(
            t, env_log, "b--", label=f"LOG fit (ζ={log_result.zeta:.4f})", linewidth=2
        )

    if lin0_result.valid:
        env_lin0 = lin0_result.params["A"] * np.exp(-lin0_result.alpha * t_shifted)
        ax.plot(
            t,
            env_lin0,
            "g--",
            label=f"LIN0 fit (ζ={lin0_result.zeta:.4f})",
            linewidth=2,
        )

    if linc_result.valid:
        env_linc = (
            linc_result.params["A"] * np.exp(-linc_result.alpha * t_shifted)
            + linc_result.params["C"]
        )
        ax.plot(
            t,
            env_linc,
            "r--",
            label=f"LINC fit (ζ={linc_result.zeta:.4f})",
            linewidth=2,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope amplitude")
    ax.set_title(f"{csv_path.stem} - Linear scale (fn={fn_hz:.1f} Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale
    ax = axes[1]
    ax.semilogy(t, env, "k-", alpha=0.5, label="Envelope data", linewidth=1)

    if log_result.valid:
        ax.semilogy(t, env_log, "b--", label=f"LOG fit", linewidth=2)
    if lin0_result.valid:
        ax.semilogy(t, env_lin0, "g--", label=f"LIN0 fit", linewidth=2)
    if linc_result.valid:
        ax.semilogy(t, env_linc, "r--", label=f"LINC fit", linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope amplitude (log scale)")
    ax.set_title(f"{csv_path.stem} - Log scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    out_dir = Path("out/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{csv_path.stem}_fit_test.png"
    export_plot(
        fig,
        plot_path,
        dpi=100,
        context={
            "plot_kind": "fit",
            "fn_hz": float(fn_hz),
            "method": "log",
            "fit_window": [float(t[0]), float(t[-1])] if len(t) else None,
            "zeta": float(log_result.zeta) if log_result.valid else None,
            "status": None,
        },
    )
    print(f"\nPlot saved: {plot_path}")
    plt.close()


def main():
    """Test fitters on a sample from the example data."""
    # Use the older envelope_exports data with known structure
    envelope_dir = Path("data/envelope_exports/free_plate_A3H1")

    if not envelope_dir.exists():
        print(f"Error: {envelope_dir} not found")
        return

    envelope_csvs = sorted(envelope_dir.glob("hit_*.csv"))
    if not envelope_csvs:
        print(f"Error: No envelope CSVs found in {envelope_dir}")
        return

    print(f"Found {len(envelope_csvs)} envelope files in {envelope_dir}")

    # Get frequency from examples data
    modal_csv = Path(
        "data/examples/free_plate_260122/output/free_plate_A3H1/modal_results.csv"
    )
    if modal_csv.exists():
        try:
            # Read only the columns we need, handling empty values
            import pandas as pd

            modal_df = pd.read_csv(modal_csv)
            fn_hz = float(modal_df["fn_hz"].iloc[0])
            print(f"Using fn = {fn_hz:.2f} Hz from modal_results.csv")
        except Exception as e:
            print(f"Could not read modal_results.csv: {e}")
            # Fallback to a reasonable frequency for free plate
            fn_hz = 775.0
            print(f"Using default fn = {fn_hz:.2f} Hz")
    else:
        # Fallback to a reasonable frequency for free plate
        fn_hz = 775.0
        print(f"Using default fn = {fn_hz:.2f} Hz")

    # Test first file
    test_single_file(envelope_csvs[0], fn_hz)


if __name__ == "__main__":
    main()
