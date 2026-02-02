"""Public API for envelope decay fitting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .flags import FlagRecord
from .models import FitDiagnostics, FitResult, GlobalFitMetrics, PieceFit
from .result import Result
from .segmentation.auto.pipeline import (
    AutoSegmentationConfig,
    fit_piecewise_auto_result,
)
from .segmentation.manual import (
    ManualUIConfig,
    build_result_from_manual,
    compute_manual_pieces,
    run_manual_segmentation,
    snap_boundary_times_to_indices,
)


LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def fit_piecewise_manual(
    t: np.ndarray,
    env: np.ndarray,
    breakpoints_t: list[float],
    *,
    fn_hz: float,
    fitter: str = "exp",
    weights: str | None = None,
    return_diagnostics: bool = True,
) -> FitResult:
    """Fit decay pieces using explicit manual breakpoints.

    Args:
        t: time array (seconds, strictly increasing)
        env: envelope amplitude
        breakpoints_t: boundary times in seconds (snapped to samples)
        fn_hz: natural frequency in Hz
        fitter: reserved for future expansion (currently only "exp")
        weights: reserved for future expansion
        return_diagnostics: include flags and traces in FitDiagnostics
    """
    _validate_inputs(t, env, fn_hz)

    if fitter != "exp":
        raise ValueError(f"Unsupported fitter: {fitter}")

    if weights is not None:
        LOGGER.info("weights=%s requested but not implemented", weights)

    boundary_indices = snap_boundary_times_to_indices(t, breakpoints_t)
    if len(boundary_indices) < 2:
        raise ValueError("Provide at least two breakpoints")

    env_flags = _collect_env_flags(env)
    pieces, flags = compute_manual_pieces(
        t,
        env,
        fn_hz,
        boundary_indices,
    )

    flags = env_flags + flags
    result = build_result_from_manual(
        t,
        env,
        fn_hz,
        boundary_indices,
        pieces,
        flags,
    )

    return _result_to_fit_result(result, include_diagnostics=return_diagnostics)


def fit_piecewise_auto(
    t: np.ndarray,
    env: np.ndarray,
    *,
    fn_hz: float,
    config: AutoSegmentationConfig | None = None,
    return_diagnostics: bool = True,
) -> FitResult:
    """Fit decay pieces using the experimental auto segmentation pipeline."""
    result = fit_piecewise_auto_result(t, env, fn_hz, config)
    return _result_to_fit_result(result, include_diagnostics=return_diagnostics)


def launch_manual_segmentation_ui(
    t: np.ndarray,
    env: np.ndarray,
    *,
    fn_hz: float,
    initial_breakpoints_t: list[float] | None = None,
    ui_config: ManualUIConfig | None = None,
) -> list[float]:
    """Launch the interactive manual segmentation UI and return breakpoints."""
    if ui_config is None:
        ui_config = ManualUIConfig()

    result = run_manual_segmentation(
        t,
        env,
        fn_hz=fn_hz,
        min_points=ui_config.min_points,
        out_dir=ui_config.out_dir,
        initial_boundaries_time_s=initial_breakpoints_t,
    )

    if result is None:
        return []

    return list(result.manual_boundaries_time_s)


def plot_segmentation_storyboard(
    t: np.ndarray,
    env: np.ndarray,
    fit: FitResult,
    *,
    ax: "Axes | None" = None,
    yscale: str = "linear",
) -> "Figure":
    """Plot envelope data with piecewise exponential fits."""
    from .plotting.storyboard import plot_segmentation_storyboard as _plot

    return _plot(t, env, fit, ax=ax, yscale=yscale)


def _validate_inputs(t: np.ndarray, env: np.ndarray, fn_hz: float) -> None:
    if len(t) != len(env):
        raise ValueError(
            f"t and env must have same length (got {len(t)} vs {len(env)})"
        )

    if len(t) < 2:
        raise ValueError(f"Need at least 2 samples (got {len(t)})")

    if not np.all(np.diff(t) > 0):
        raise ValueError("Time array t must be strictly increasing")

    if fn_hz <= 0:
        raise ValueError(f"Natural frequency must be positive (got {fn_hz})")


def _collect_env_flags(env: np.ndarray) -> list[FlagRecord]:
    flags: list[FlagRecord] = []
    if np.any(env < 0):
        n_neg = int(np.sum(env < 0))
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="input",
                severity="warn",
                code="NEGATIVE_ENV_VALUES",
                message=f"Found {n_neg} negative envelope values",
            )
        )

    if not np.all(np.isfinite(env)):
        n_bad = int(np.sum(~np.isfinite(env)))
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="input",
                severity="warn",
                code="NON_FINITE_ENV_VALUES",
                message=f"Found {n_bad} non-finite envelope values (NaN or Inf)",
            )
        )
    return flags


def _result_to_fit_result(
    result: Result,
    *,
    include_diagnostics: bool,
) -> FitResult:
    pieces: list[PieceFit] = []
    for piece in result.pieces:
        flags: list[str] = []
        if not piece.log_fit.valid:
            flags.append("invalid_log_fit")

        A0 = float(np.exp(piece.log_fit.params["b"])) if piece.log_fit.valid else 0.0
        params = {
            "alpha": float(piece.log_fit.alpha),
            "zeta": float(piece.log_fit.zeta),
            "A0": float(A0),
        }

        pieces.append(
            PieceFit(
                piece_id=int(piece.piece_id),
                t_start_s=float(piece.t_start_s),
                t_end_s=float(piece.t_end_s),
                n_points=int(piece.n_points),
                params=params,
                r2=float(piece.log_fit.r2),
                flags=flags,
            )
        )

    breakpoints = list(result.manual_boundaries_time_s)
    if not breakpoints and result.pieces:
        breakpoints = sorted(
            {
                float(result.pieces[0].t_start_s),
                *[float(p.t_start_s) for p in result.pieces],
                float(result.pieces[-1].t_end_s),
            }
        )

    metrics = GlobalFitMetrics(
        fn_hz=float(result.fn_hz),
        omega_n=float(result.omega_n),
        n_pieces=len(result.pieces),
        n_samples=len(result.t),
        duration_s=float(result.t[-1] - result.t[0]) if len(result.t) else 0.0,
    )

    diagnostics = None
    if include_diagnostics:
        diagnostics = FitDiagnostics(
            flags=list(result.flags),
            windows_trace=list(result.windows_trace) if result.windows_trace else None,
        )

    return FitResult(
        pieces=pieces,
        breakpoints_t=breakpoints,
        global_metrics=metrics,
        diagnostics=diagnostics,
    )
