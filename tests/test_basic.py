"""Basic tests for envelope-decay-fit package."""

import numpy as np
import pytest

from envelope_decay_fit import fit_piecewise_manual
from envelope_decay_fit.fitters import fit_log_domain


def test_import():
    """Test that package imports correctly."""
    from envelope_decay_fit import (
        FitResult,
        PieceFit,
        fit_piecewise_manual,
        launch_manual_segmentation_ui,
    )

    assert FitResult is not None
    assert PieceFit is not None
    assert fit_piecewise_manual is not None
    assert launch_manual_segmentation_ui is not None


def test_log_fitter_simple():
    """Test LOG domain fitter on synthetic data."""
    # Generate simple exponential decay
    t = np.linspace(0, 1.0, 1000)
    alpha_true = 5.0
    fn_hz = 100.0
    omega_n = 2 * np.pi * fn_hz
    zeta_true = alpha_true / omega_n

    env = np.exp(-alpha_true * t)

    # Fit
    result = fit_log_domain(t, env, fn_hz, t_ref=t[0])

    assert result.valid
    assert abs(result.alpha - alpha_true) < 0.1  # Within 10%
    assert abs(result.zeta - zeta_true) < 0.001
    assert result.r2 > 0.99


def test_fit_piecewise_manual_synthetic():
    """Test manual pipeline on synthetic data."""
    # Generate piecewise decay
    t1 = np.linspace(0, 0.5, 500)
    t2 = np.linspace(0.500001, 1.5, 1000)  # Avoid duplicate at boundary
    t = np.concatenate([t1, t2])

    # Piece 1: fast decay (transient)
    env1 = 0.5 * np.exp(-10.0 * t1)
    # Piece 2: slow decay (established)
    env2 = 0.2 * np.exp(-3.0 * (t2 - 0.5))
    env = np.concatenate([env1, env2])

    # Add small noise
    np.random.seed(42)
    env += np.random.normal(0, 0.001, len(env))
    env = np.abs(env)  # Ensure positive

    fn_hz = 150.0

    breakpoints_t = [float(t[0]), float(t1[-1]), float(t[-1])]

    fit = fit_piecewise_manual(t, env, breakpoints_t, fn_hz=fn_hz)

    assert len(fit.pieces) == 2
    assert fit.pieces[0].n_points > 0
    assert fit.pieces[1].n_points > 0


def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    # Wrong length
    t = np.linspace(0, 1, 200)
    env = np.ones(100)  # Different length

    with pytest.raises(ValueError, match="same length"):
        fit_piecewise_manual(t, env, [0.0, 1.0], fn_hz=100.0)

    # Non-monotonic time (but enough samples)
    t = np.linspace(0, 1, 200)
    t[50] = t[49] - 0.1  # Make non-monotonic
    env = np.exp(-2 * t)

    with pytest.raises(ValueError, match="strictly increasing"):
        fit_piecewise_manual(t, env, [0.0, 1.0], fn_hz=100.0)

    # Too few samples
    t = np.linspace(0, 1, 50)
    env = np.exp(-2 * t)

    with pytest.raises(ValueError, match="at least 2"):
        fit_piecewise_manual(t[:1], env[:1], [0.0, 1.0], fn_hz=100.0)


def test_negative_envelope_warning():
    """Test that negative envelope values generate warnings."""
    t = np.linspace(0, 1, 1000)
    env = np.exp(-2 * t)
    env[500] = -0.1  # Inject negative value

    fn_hz = 100.0
    breakpoints_t = [float(t[0]), float(t[-1])]
    fit = fit_piecewise_manual(t, env, breakpoints_t, fn_hz=fn_hz)

    assert fit.diagnostics is not None
    warning_flags = [
        f for f in fit.diagnostics.flags if f.code == "NEGATIVE_ENV_VALUES"
    ]
    assert len(warning_flags) > 0
