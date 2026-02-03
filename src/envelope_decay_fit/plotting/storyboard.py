"""Diagnostic plotting functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Literal, cast

from ..models import FitResult
from ..result import Result


def create_diagnostic_plots(result: Result, out_dir: Path) -> dict[str, Path]:
    """Create diagnostic plots for fit results.

    Args:
        result: Result object from the auto pipeline
        out_dir: directory to write plots

    Returns:
        Dictionary mapping plot names to file paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Plot 1: Piecewise fit visualization
    path = create_piecewise_fit_plot(result, out_dir / "piecewise_fit.png")
    paths["piecewise_fit"] = path

    # Plot 2: Score traces (R² vs Δt)
    path = create_score_traces_plot(result, out_dir / "score_traces.png")
    paths["score_traces"] = path

    # Plot 3: Parameter traces (ζ vs Δt)
    path = create_param_traces_plot(result, out_dir / "param_traces.png")
    paths["param_traces"] = path

    # Plot 4: Segmentation storyboard (R² in global time)
    path = create_segmentation_storyboard_plot(
        result, out_dir / "segmentation_storyboard.png"
    )
    paths["segmentation_storyboard"] = path

    return paths


def plot_segmentation_storyboard(
    t: np.ndarray,
    env: np.ndarray,
    fit: FitResult,
    *,
    ax: Axes | None = None,
    yscale: str = "log",
    title: str | None = None,
    title_mode: Literal["append", "replace"] = "append",
) -> Figure:
    """Plot envelope data with piecewise exponential fits.

    Args:
        t: time array (seconds)
        env: envelope amplitude
        fit: FitResult from the public API
        ax: optional matplotlib Axes to draw on
        yscale: "linear" or "log"
        title: optional title text to append or replace
        title_mode: "append" or "replace" when title is provided

    Returns:
        Matplotlib Figure containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = cast(Figure, ax.figure)

    ax.plot(t, env, "k-", alpha=0.5, linewidth=1, label="Envelope data")

    colors = ["blue", "green", "red", "orange", "purple"]
    for idx, piece in enumerate(fit.pieces):
        params = piece.params
        if "A0" not in params or "alpha" not in params:
            continue

        mask = (t >= piece.t_start_s) & (t <= piece.t_end_s)
        t_piece = t[mask]
        if len(t_piece) == 0:
            continue

        t_shifted = t_piece - piece.t_start_s
        env_fit = params["A0"] * np.exp(-params["alpha"] * t_shifted)

        color = colors[idx % len(colors)]
        ax.plot(
            t_piece,
            env_fit,
            "--",
            color=color,
            linewidth=2,
            label=f"Piece {piece.piece_id}: ζ={params.get('zeta', 0.0):.4f}",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope amplitude")
    base_title = "Segmentation Storyboard"
    if fit.global_metrics is not None:
        base_title = f"Segmentation Storyboard (fn={fit.global_metrics.fn_hz:.1f} Hz)"

    if title:
        if title_mode == "replace":
            full_title = title
        else:
            full_title = f"{base_title} - {title}"
    else:
        full_title = base_title

    ax.set_title(full_title)
    ax.grid(True, alpha=0.3)
    if yscale == "log":
        ax.set_yscale("log", nonpositive="clip")
    else:
        ax.set_yscale("linear")
    ax.legend(fontsize=8, loc="upper right")

    return fig


def create_piecewise_fit_plot(result: Result, out_path: Path) -> Path:
    """Create piecewise fit visualization."""
    from .plot_export import export_plot

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t = result.t
    env = result.env

    # Linear scale plot
    ax = axes[0]
    ax.plot(t, env, "k-", alpha=0.5, linewidth=1, label="Envelope data")

    colors = ["blue", "green", "red", "purple", "orange"]

    for piece in result.pieces:
        color = colors[piece.piece_id % len(colors)]
        t_piece = t[piece.i_start : piece.i_end]
        env_piece = env[piece.i_start : piece.i_end]

        # Plot piece data range
        ax.plot(t_piece, env_piece, "o", color=color, alpha=0.3, markersize=2)

        # Plot LOG fit for this piece
        if piece.log_fit.valid:
            t_ref = t_piece[0]
            t_shifted = t_piece - t_ref
            A = np.exp(piece.log_fit.params["b"])
            env_fit = A * np.exp(-piece.log_fit.alpha * t_shifted)
            ax.plot(
                t_piece,
                env_fit,
                "--",
                color=color,
                linewidth=2,
                label=f"Piece {piece.piece_id}: {piece.label} (ζ={piece.log_fit.zeta:.4f})",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope amplitude")
    ax.set_title(f"Piecewise Decay Fit (fn={result.fn_hz:.1f} Hz)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Log scale plot
    ax = axes[1]
    ax.semilogy(t, env, "k-", alpha=0.5, linewidth=1, label="Envelope data")

    for piece in result.pieces:
        color = colors[piece.piece_id % len(colors)]
        t_piece = t[piece.i_start : piece.i_end]

        if piece.log_fit.valid:
            t_ref = t_piece[0]
            t_shifted = t_piece - t_ref
            A = np.exp(piece.log_fit.params["b"])
            env_fit = A * np.exp(-piece.log_fit.alpha * t_shifted)
            ax.semilogy(
                t_piece,
                env_fit,
                "--",
                color=color,
                linewidth=2,
                label=f"Piece {piece.piece_id} (LOG)",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope amplitude (log scale)")
    ax.set_title("Piecewise Fit - Log Scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    zeta_values: list[float | None] = []
    for piece in result.pieces:
        if piece.log_fit.valid:
            zeta_values.append(float(piece.log_fit.zeta))
        else:
            zeta_values.append(None)

    fig.tight_layout()
    export_plot(
        fig,
        out_path,
        dpi=150,
        context={
            "plot_kind": "fit",
            "fn_hz": float(result.fn_hz),
            "method": "log",
            "fit_window": [float(t[0]), float(t[-1])] if len(t) else None,
            "zeta": zeta_values,
            "status": None,
        },
    )
    plt.close(fig)

    return out_path


def create_score_traces_plot(result: Result, out_path: Path) -> Path:
    """Create R² score traces plot."""
    from .plot_export import export_plot
    from ..segmentation.auto.window_scan import WindowFitRecord, extract_score_trace

    fig, ax = plt.subplots(figsize=(12, 6))

    def split_windows_by_end(
        windows_trace: list[WindowFitRecord],
    ) -> list[list[WindowFitRecord]]:
        if not windows_trace:
            return []

        groups = []
        current = [windows_trace[0]]
        current_i_end = windows_trace[0].i_end

        for win in windows_trace[1:]:
            if win.i_end != current_i_end:
                groups.append(current)
                current = [win]
                current_i_end = win.i_end
            else:
                current.append(win)

        groups.append(current)
        return groups

    # Extract LOG method R² trace per piece
    colors = ["blue", "green", "red", "orange", "purple"]
    window_groups = split_windows_by_end(result.windows_trace)
    non_monotonic_detected = False

    for piece_idx, windows in enumerate(window_groups):
        dt, r2 = extract_score_trace(windows, method="log")
        if len(dt) == 0:
            continue

        if len(dt) > 1 and np.any(np.diff(dt) <= 0):
            non_monotonic_detected = True

        color = colors[piece_idx % len(colors)]
        ax.plot(
            dt,
            r2,
            ".-",
            color=color,
            alpha=0.6,
            markersize=2,
            label=f"LOG fit R² — piece {piece_idx}",
        )

    # Mark breakpoints
    for piece in result.pieces:
        if piece.breakpoint_dt_s is not None:
            ax.axvline(
                piece.breakpoint_dt_s,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Breakpoint {piece.piece_id}",
            )

    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.99, color="gray", linestyle=":", alpha=0.5)

    if non_monotonic_detected:
        ax.text(
            0.01,
            0.02,
            "Non-monotonic window sweep detected - piecewise traces separated",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )

    ax.set_xlabel("Window duration Δt (s)")
    ax.set_ylabel("R² score")
    ax.set_title(f"Fit Quality vs Window Size (fn={result.fn_hz:.1f} Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    export_plot(
        fig,
        out_path,
        dpi=150,
        context={
            "plot_kind": "log_decay",
            "fn_hz": float(result.fn_hz),
            "method": "log",
            "fit_window": None,
            "zeta": None,
            "status": None,
        },
    )
    plt.close(fig)

    return out_path


def create_param_traces_plot(result: Result, out_path: Path) -> Path:
    """Create parameter traces plot (ζ vs Δt)."""
    from .plot_export import export_plot
    from ..segmentation.auto.window_scan import extract_param_trace

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Zeta trace
    ax = axes[0]
    if result.windows_trace:
        dt, zeta = extract_param_trace(result.windows_trace, param="zeta", method="log")
        ax.plot(dt, zeta, "g.-", alpha=0.5, markersize=2, label="ζ (LOG fit)")

    # Mark piece values
    for piece in result.pieces:
        if piece.log_fit.valid:
            ax.axhline(
                piece.log_fit.zeta,
                color="blue",
                linestyle="--",
                alpha=0.5,
                label=f"Piece {piece.piece_id}: ζ={piece.log_fit.zeta:.4f}",
            )

    ax.set_xlabel("Window duration Δt (s)")
    ax.set_ylabel("Damping ratio ζ")
    ax.set_title(f"Damping Ratio vs Window Size (fn={result.fn_hz:.1f} Hz)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Alpha trace
    ax = axes[1]
    if result.windows_trace:
        dt, alpha = extract_param_trace(
            result.windows_trace, param="alpha", method="log"
        )
        ax.plot(dt, alpha, "r.-", alpha=0.5, markersize=2, label="α (LOG fit)")

    for piece in result.pieces:
        if piece.log_fit.valid:
            ax.axhline(
                piece.log_fit.alpha,
                color="purple",
                linestyle="--",
                alpha=0.5,
                label=f"Piece {piece.piece_id}: α={piece.log_fit.alpha:.2f}",
            )

    ax.set_xlabel("Window duration Δt (s)")
    ax.set_ylabel("Decay rate α (1/s)")
    ax.set_title("Decay Rate vs Window Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    zeta_values: list[float | None] = []
    for piece in result.pieces:
        if piece.log_fit.valid:
            zeta_values.append(float(piece.log_fit.zeta))
        else:
            zeta_values.append(None)

    fig.tight_layout()
    export_plot(
        fig,
        out_path,
        dpi=150,
        context={
            "plot_kind": "fit",
            "fn_hz": float(result.fn_hz),
            "method": "log",
            "fit_window": None,
            "zeta": zeta_values,
            "status": None,
        },
    )
    plt.close(fig)

    return out_path


def create_segmentation_storyboard_plot(result: Result, out_path: Path) -> Path:
    """Create segmentation storyboard plot (R² in global time)."""
    from .plot_export import export_plot
    from ..segmentation.auto.window_scan import WindowFitRecord

    fig, ax = plt.subplots(figsize=(12, 6))

    def split_windows_by_end(
        windows_trace: list[WindowFitRecord],
    ) -> list[list[WindowFitRecord]]:
        if not windows_trace:
            return []

        groups = []
        current = [windows_trace[0]]
        current_i_end = windows_trace[0].i_end

        for win in windows_trace[1:]:
            if win.i_end != current_i_end:
                groups.append(current)
                current = [win]
                current_i_end = win.i_end
            else:
                current.append(win)

        groups.append(current)
        return groups

    colors = ["blue", "green", "red", "orange", "purple"]
    window_groups = split_windows_by_end(result.windows_trace)

    for piece_idx, windows in enumerate(window_groups):
        x_list = []
        y_list = []

        for win in windows:
            if win.log_fit.valid:
                x_list.append(win.t_start_s)
                y_list.append(win.log_fit.r2)

        if not x_list:
            continue

        x = np.array(x_list)
        y = np.array(y_list)
        sort_idx = np.argsort(x)

        color = colors[piece_idx % len(colors)]
        ax.plot(
            x[sort_idx],
            y[sort_idx],
            ".-",
            color=color,
            alpha=0.6,
            markersize=2,
            label=f"Storyboard R² — piece {piece_idx}",
        )

    t = result.t
    for piece_idx in range(len(result.pieces) - 1):
        boundary_index = result.pieces[piece_idx].i_start
        if 0 <= boundary_index < len(t):
            boundary_time = t[boundary_index]
            ax.axvline(
                boundary_time,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Handoff {piece_idx}→{piece_idx + 1}",
            )

    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("R² score")
    ax.set_title(
        f"Segmentation Storyboard — Fit Quality vs Time (fn={result.fn_hz:.1f} Hz)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    export_plot(
        fig,
        out_path,
        dpi=150,
        context={
            "plot_kind": "storyboard_r2",
            "fn_hz": float(result.fn_hz),
            "method": "log",
            "fit_window": None,
            "zeta": None,
            "status": None,
        },
    )
    plt.close(fig)

    return out_path
