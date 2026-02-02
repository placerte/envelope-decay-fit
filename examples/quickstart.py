#!/usr/bin/env python3
"""Quickstart example for envelope-decay-fit."""

import numpy as np
from pathlib import Path

from envelope_decay_fit import fit_envelope_decay


def example_synthetic():
    """Example with synthetic exponential decay."""
    print("=" * 60)
    print("Example 1: Synthetic Exponential Decay")
    print("=" * 60)

    # Generate synthetic decay
    t = np.linspace(0, 2.0, 2000)
    alpha = 3.0  # decay rate
    fn_hz = 150.0
    env = 0.5 * np.exp(-alpha * t)

    # Add small noise
    np.random.seed(42)
    env += np.random.normal(0, 0.001, len(env))
    env = np.abs(env)

    # Fit
    result = fit_envelope_decay(
        t,
        env,
        fn_hz=fn_hz,
        n_pieces=1,
        max_windows=200,
    )

    # Print results
    print(result.summary())

    print("\nExpected: α ≈ 3.0, ζ ≈ 0.00318")
    print(
        f"Got:      α = {result.pieces[0].log_fit.alpha:.4f}, "
        f"ζ = {result.pieces[0].log_fit.zeta:.5f}"
    )


def example_real_data():
    """Example with real envelope data."""
    print("\n" + "=" * 60)
    print("Example 2: Real Envelope Data")
    print("=" * 60)

    # Load real data
    csv_path = Path("data/envelope_exports/free_plate_A3H1/hit_001.csv")
    if not csv_path.exists():
        print(f"Data file not found: {csv_path}")
        print("Skipping real data example")
        return

    # Load CSV
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    env = data[:, 3]

    print(f"Loaded {len(t)} samples from {csv_path.name}")

    # Fit with 2 pieces
    result = fit_envelope_decay(
        t,
        env,
        fn_hz=775.2,  # Known frequency for this dataset
        n_pieces=2,
        out_dir="out/quickstart",
        max_windows=300,
    )

    # Print results
    print(result.summary())

    if result.artifact_paths:
        print(f"\nPlots saved to: out/quickstart/")


def main():
    """Run quickstart examples."""
    print("envelope-decay-fit Quickstart Examples\n")

    example_synthetic()
    example_real_data()

    print("\n" + "=" * 60)
    print("Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
