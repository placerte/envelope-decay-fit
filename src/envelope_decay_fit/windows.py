"""Expanding window generation for piecewise decay fitting."""

from dataclasses import dataclass
import numpy as np
from typing import Iterator

from .fitters import fit_log_domain, fit_lin0_domain, fit_linc_domain, FitResult


@dataclass
class WindowFitRecord:
    """Record for a single expanding window fit.

    Attributes:
        win_id: window identifier (typically integer index)
        i_start: start index in original arrays
        i_end: end index (exclusive)
        t_start_s: start time (seconds)
        t_end_s: end time (seconds)
        dt_s: window duration (seconds)
        n_points: number of samples in window

        log_fit: LOG domain fit result
        lin0_fit: LIN0 domain fit result
        linc_fit: LINC domain fit result
    """

    win_id: int
    i_start: int
    i_end: int
    t_start_s: float
    t_end_s: float
    dt_s: float
    n_points: int

    log_fit: FitResult
    lin0_fit: FitResult
    linc_fit: FitResult


def generate_expanding_windows(
    t: np.ndarray,
    env: np.ndarray,
    fn_hz: float,
    i_end: int,
    i_start_min: int = 0,
    min_points: int = 10,
    max_windows: int | None = 500,
) -> list[WindowFitRecord]:
    """Generate expanding windows by moving start backward from fixed end.

    This implements the backward-expanding window algorithm from specs.md.
    Windows grow from size `min_points` up to the full available range.

    For large datasets, windows are sampled to keep computational cost reasonable
    (max_windows parameter). This is a pragmatic optimization.

    Args:
        t: time array (seconds)
        env: envelope amplitude
        fn_hz: natural frequency (Hz)
        i_end: fixed end index for all windows
        i_start_min: minimum start index (default: 0)
        min_points: minimum window size (default: 10)
        max_windows: maximum number of windows to generate (default: 500, None for all)

    Returns:
        List of WindowFitRecord objects, ordered from smallest to largest window
    """
    windows = []
    win_id = 0

    # Calculate total possible windows
    total_possible = i_end - i_start_min - min_points + 1

    # Determine sampling strategy
    if max_windows is None or total_possible <= max_windows:
        # Generate all windows
        step = 1
        n_windows = total_possible
    else:
        # Sample windows logarithmically (more dense near small windows, sparse for large)
        # This preserves detail where breakpoints are likely while reducing computation
        step = max(1, total_possible // max_windows)
        n_windows = max_windows

    # Start with minimum window size and expand backward
    k_values = range(min_points, i_end - i_start_min + 1, step)

    for k in k_values:
        i_start = i_end - k

        if i_start < i_start_min:
            break

        # Extract window
        t_win = t[i_start:i_end]
        env_win = env[i_start:i_end]

        # Window metadata
        t_start_s = t_win[0]
        t_end_s = t_win[-1]
        dt_s = t_end_s - t_start_s
        n_points = len(t_win)

        # Fit all three methods
        log_fit = fit_log_domain(t_win, env_win, fn_hz, t_ref=t_start_s)
        lin0_fit = fit_lin0_domain(t_win, env_win, fn_hz, t_ref=t_start_s)
        linc_fit = fit_linc_domain(t_win, env_win, fn_hz, t_ref=t_start_s)

        # Create record
        record = WindowFitRecord(
            win_id=win_id,
            i_start=i_start,
            i_end=i_end,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            dt_s=dt_s,
            n_points=n_points,
            log_fit=log_fit,
            lin0_fit=lin0_fit,
            linc_fit=linc_fit,
        )

        windows.append(record)
        win_id += 1

    return windows


def extract_score_trace(
    windows: list[WindowFitRecord],
    method: str = "log",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract score trace (R² vs Δt) from window fits.

    Args:
        windows: list of WindowFitRecord objects
        method: 'log' | 'lin0' | 'linc' (default: 'log')

    Returns:
        (dt_array, r2_array): arrays of window durations and R² scores
    """
    dt_list = []
    r2_list = []

    for win in windows:
        if method == "log":
            fit = win.log_fit
        elif method == "lin0":
            fit = win.lin0_fit
        elif method == "linc":
            fit = win.linc_fit
        else:
            raise ValueError(f"Invalid method: {method}")

        if fit.valid:
            dt_list.append(win.dt_s)
            r2_list.append(fit.r2)

    return np.array(dt_list), np.array(r2_list)


def extract_param_trace(
    windows: list[WindowFitRecord],
    param: str,
    method: str = "log",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract parameter trace (param vs Δt) from window fits.

    Args:
        windows: list of WindowFitRecord objects
        param: 'alpha' | 'zeta' | 'C' (for LINC)
        method: 'log' | 'lin0' | 'linc' (default: 'log')

    Returns:
        (dt_array, param_array): arrays of window durations and parameter values
    """
    dt_list = []
    param_list = []

    for win in windows:
        if method == "log":
            fit = win.log_fit
        elif method == "lin0":
            fit = win.lin0_fit
        elif method == "linc":
            fit = win.linc_fit
        else:
            raise ValueError(f"Invalid method: {method}")

        if fit.valid:
            dt_list.append(win.dt_s)

            if param == "alpha":
                param_list.append(fit.alpha)
            elif param == "zeta":
                param_list.append(fit.zeta)
            elif param == "C":
                if "C" in fit.params:
                    param_list.append(fit.params["C"])
                else:
                    param_list.append(np.nan)
            else:
                raise ValueError(f"Invalid param: {param}")

    return np.array(dt_list), np.array(param_list)
