"""Plotting API tests."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from envelope_decay_fit import FitResult, fit_piecewise_manual
from envelope_decay_fit.api import plot_segmentation_storyboard


def _make_fit() -> tuple[np.ndarray, np.ndarray, FitResult]:
    t = np.linspace(0.0, 1.0, 200)
    env = np.exp(-2.0 * t)
    fit = fit_piecewise_manual(t, env, [float(t[0]), float(t[-1])], fn_hz=100.0)
    return t, env, fit


def test_plot_storyboard_defaults_log_scale() -> None:
    """Plot storyboard defaults to log scale."""
    t, env, fit = _make_fit()
    fig = plot_segmentation_storyboard(t, env, fit)

    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert "fn=" in ax.get_title()

    plt.close(fig)


def test_plot_storyboard_title_append_replace() -> None:
    """Plot storyboard supports title append and replace."""
    t, env, fit = _make_fit()

    fig = plot_segmentation_storyboard(
        t, env, fit, title="Batch 1", title_mode="append"
    )
    title = fig.axes[0].get_title()
    assert "Segmentation Storyboard" in title
    assert "Batch 1" in title
    plt.close(fig)

    fig = plot_segmentation_storyboard(
        t, env, fit, title="Batch 1", title_mode="replace"
    )
    title = fig.axes[0].get_title()
    assert title == "Batch 1"
    plt.close(fig)
