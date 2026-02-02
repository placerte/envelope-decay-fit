"""Breakpoint detection using two-regime change-point analysis."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Optional

from .flags import FlagRecord


def smooth_trace(
    trace: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Apply light smoothing to a trace using uniform filter.

    Args:
        trace: 1D array to smooth
        window_size: smoothing window size (default: 5)

    Returns:
        Smoothed trace
    """
    if len(trace) < window_size:
        return trace
    return uniform_filter1d(trace, size=window_size, mode="nearest")


def detect_breakpoint_two_regime(
    dt: np.ndarray,
    score: np.ndarray,
    min_established_duration_s: float = 0.1,
    min_established_points: int = 10,
    avoid_tail_fraction: float = 0.05,
) -> tuple[Optional[int], list[FlagRecord]]:
    """Detect breakpoint using two-regime change-point detection.

    Finds the split point that minimizes sum of squared errors when
    fitting two constant values to the left and right segments.

    The algorithm looks for a transition from:
    - Left segment: degraded R² (transient-dominated decay)
    - Right segment: stable high R² (established free decay)

    Args:
        dt: window duration array (seconds), must be monotonically increasing
        score: R² score array (same length as dt)
        min_established_duration_s: minimum duration for established segment (seconds)
        min_established_points: minimum number of points in established segment
        avoid_tail_fraction: avoid selecting breakpoint in last X fraction of data

    Returns:
        (breakpoint_index, flags): breakpoint index in dt/score arrays, or None if not found
    """
    flags = []

    if len(dt) < 2 * min_established_points:
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="breakpoint",
                severity="reject",
                code="INSUFFICIENT_SAMPLES_FOR_BREAKPOINT",
                message=f"Need at least {2 * min_established_points} points, have {len(dt)}",
            )
        )
        return None, flags

    # Find indices that satisfy constraints
    # We want the established segment (right side) to be large enough
    max_split_idx = int(len(dt) * (1.0 - avoid_tail_fraction))

    valid_splits = []
    for i in range(min_established_points, max_split_idx):
        # Check if right segment (established decay) is long enough
        n_right = len(dt) - i
        if n_right < min_established_points:
            continue

        # Check if right segment duration is long enough
        dt_right = dt[-1] - dt[i]
        if dt_right < min_established_duration_s:
            continue

        valid_splits.append(i)

    if len(valid_splits) == 0:
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="breakpoint",
                severity="warn",
                code="NO_VALID_SPLIT_POINTS",
                message=f"No split points satisfy constraints (min_dur={min_established_duration_s}s, min_pts={min_established_points})",
                details=f"max_split_idx={max_split_idx}, n_total={len(dt)}",
            )
        )
        return None, flags

    # For each valid split, compute two-regime SSE
    best_split = None
    best_sse = np.inf

    for split_idx in valid_splits:
        # Left segment: indices 0 to split_idx-1
        # Right segment: indices split_idx to end
        left_scores = score[:split_idx]
        right_scores = score[split_idx:]

        # Fit constant to each segment (mean)
        mean_left = np.mean(left_scores)
        mean_right = np.mean(right_scores)

        # Compute SSE
        sse_left = np.sum((left_scores - mean_left) ** 2)
        sse_right = np.sum((right_scores - mean_right) ** 2)
        sse_total = sse_left + sse_right

        if sse_total < best_sse:
            best_sse = sse_total
            best_split = split_idx

    if best_split is None:
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="breakpoint",
                severity="warn",
                code="NO_BREAKPOINT_FOUND",
                message="Two-regime SSE minimization failed to find breakpoint",
            )
        )
        return None, flags

    # Check quality of detected breakpoint
    left_scores = score[:best_split]
    right_scores = score[best_split:]
    mean_left = np.mean(left_scores)
    mean_right = np.mean(right_scores)

    # If right segment (established) has lower mean than left, that's suspicious
    if mean_right < mean_left:
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="breakpoint",
                severity="warn",
                code="SUSPICIOUS_BREAKPOINT",
                message=f"Established segment has lower R² than transient segment (left={mean_left:.4f}, right={mean_right:.4f})",
                details=f"split_idx={best_split}, dt={dt[best_split]:.4f}s",
            )
        )

    # If established segment has low R², warn
    if mean_right < 0.8:
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="breakpoint",
                severity="warn",
                code="LOW_ESTABLISHED_R2",
                message=f"Established segment has low R² = {mean_right:.4f}",
                details=f"split_idx={best_split}, dt={dt[best_split]:.4f}s",
            )
        )

    # Success
    flags.append(
        FlagRecord(
            scope="global",
            scope_id="breakpoint",
            severity="info",
            code="BREAKPOINT_DETECTED",
            message=f"Breakpoint detected at index {best_split}, dt={dt[best_split]:.4f}s",
            details=f"left_R²={mean_left:.4f}, right_R²={mean_right:.4f}, SSE={best_sse:.4e}",
        )
    )

    return best_split, flags
