"""Public API for envelope decay fitting."""

import numpy as np
from pathlib import Path

from .result import Result, PieceRecord
from .flags import FlagRecord
from .windows import generate_expanding_windows, extract_score_trace
from .breakpoint import detect_breakpoint_two_regime, smooth_trace
from .fitters import fit_log_domain, fit_lin0_domain, fit_linc_domain
from .manual_segmentation import (
    build_result_from_manual,
    compute_manual_pieces,
    snap_boundary_times_to_indices,
    write_manual_segmentation_json,
)


def fit_envelope_decay(
    t: np.ndarray,
    env: np.ndarray,
    fn_hz: float,
    n_pieces: int = 2,
    out_dir: Path | str | None = None,
    max_windows: int = 500,
) -> Result:
    """Fit piecewise exponential decay to envelope data.

    This is the main entry point for the envelope decay fitting package.
    It extracts multiple decay pieces using backward-expanding windows and
    change-point detection.

    Args:
        t: time array (seconds, strictly increasing)
        env: envelope amplitude (non-negative expected)
        fn_hz: natural frequency in Hz (required)
        n_pieces: number of decay pieces to extract (default: 2)
        out_dir: if provided, write plots and CSVs to this directory
        max_windows: maximum windows per piece (default: 500, for performance)

    Returns:
        Result object with pieces, windows, and flags

    Raises:
        ValueError: if inputs are invalid
    """
    # Input validation
    flags = []

    if len(t) != len(env):
        raise ValueError(
            f"t and env must have same length (got {len(t)} vs {len(env)})"
        )

    if len(t) < 100:
        raise ValueError(f"Need at least 100 samples (got {len(t)})")

    if not np.all(np.diff(t) > 0):
        raise ValueError("Time array t must be strictly increasing")

    if fn_hz <= 0:
        raise ValueError(f"Natural frequency must be positive (got {fn_hz})")

    if n_pieces < 1:
        raise ValueError(f"n_pieces must be >= 1 (got {n_pieces})")

    # Warn about negative or non-finite envelope values
    if np.any(env < 0):
        n_neg = np.sum(env < 0)
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
        n_bad = np.sum(~np.isfinite(env))
        flags.append(
            FlagRecord(
                scope="global",
                scope_id="input",
                severity="warn",
                code="NON_FINITE_ENV_VALUES",
                message=f"Found {n_bad} non-finite envelope values (NaN or Inf)",
            )
        )

    omega_n = 2.0 * np.pi * fn_hz

    # Extract pieces
    pieces = []
    all_windows = []
    i_end = len(t)

    for piece_idx in range(n_pieces):
        # Generate expanding windows for this piece
        print(f"Extracting piece {piece_idx + 1}/{n_pieces}...")
        windows = generate_expanding_windows(
            t,
            env,
            fn_hz,
            i_end,
            i_start_min=0,
            min_points=50,
            max_windows=max_windows,
        )

        if len(windows) == 0:
            flags.append(
                FlagRecord(
                    scope="piece",
                    scope_id=str(piece_idx),
                    severity="reject",
                    code="NO_WINDOWS_GENERATED",
                    message=f"No windows generated for piece {piece_idx}",
                )
            )
            break

        all_windows.extend(windows)

        # Extract R² trace
        dt, r2 = extract_score_trace(windows, method="log")

        if len(dt) < 20:
            flags.append(
                FlagRecord(
                    scope="piece",
                    scope_id=str(piece_idx),
                    severity="warn",
                    code="FEW_VALID_WINDOWS",
                    message=f"Only {len(dt)} valid windows for piece {piece_idx}",
                )
            )

        # Smooth trace
        r2_smooth = smooth_trace(r2, window_size=min(5, max(3, len(r2) // 20)))

        # Detect breakpoint
        breakpoint_idx, bp_flags = detect_breakpoint_two_regime(
            dt,
            r2_smooth,
            min_established_duration_s=0.05,  # Pragmatic: 50ms minimum
            min_established_points=10,
        )
        flags.extend(bp_flags)

        # Determine piece boundaries
        if breakpoint_idx is not None:
            # Use detected breakpoint
            piece_window = windows[breakpoint_idx]
            i_start_piece = piece_window.i_start
            breakpoint_dt_s = dt[breakpoint_idx]
        else:
            # Fallback: use full range available
            if len(windows) > 0:
                # Use the largest window
                piece_window = windows[-1]
                i_start_piece = piece_window.i_start
                breakpoint_dt_s = None
            else:
                flags.append(
                    FlagRecord(
                        scope="piece",
                        scope_id=str(piece_idx),
                        severity="reject",
                        code="NO_PIECE_EXTRACTED",
                        message=f"Could not extract piece {piece_idx}",
                    )
                )
                break

        # Extract piece data and fit
        t_piece = t[i_start_piece:i_end]
        env_piece = env[i_start_piece:i_end]

        if len(t_piece) < 10:
            flags.append(
                FlagRecord(
                    scope="piece",
                    scope_id=str(piece_idx),
                    severity="reject",
                    code="PIECE_TOO_SMALL",
                    message=f"Piece {piece_idx} has only {len(t_piece)} samples",
                )
            )
            break

        # Fit piece with all three methods
        t_ref = t_piece[0]
        log_fit = fit_log_domain(t_piece, env_piece, fn_hz, t_ref)
        lin0_fit = fit_lin0_domain(t_piece, env_piece, fn_hz, t_ref)
        linc_fit = fit_linc_domain(t_piece, env_piece, fn_hz, t_ref)

        # Determine label
        if piece_idx == n_pieces - 1:
            label = "transient_dominated_decay"
        elif piece_idx == 0:
            label = "established_free_decay"
        else:
            label = f"intermediate_piece_{piece_idx}"

        piece = PieceRecord(
            piece_id=piece_idx,
            label=label,
            i_start=i_start_piece,
            i_end=i_end,
            t_start_s=t_piece[0],
            t_end_s=t_piece[-1],
            dt_s=t_piece[-1] - t_piece[0],
            n_points=len(t_piece),
            breakpoint_index=breakpoint_idx,
            breakpoint_dt_s=breakpoint_dt_s,
            log_fit=log_fit,
            lin0_fit=lin0_fit,
            linc_fit=linc_fit,
        )

        pieces.append(piece)

        # Update i_end for next piece
        i_end = i_start_piece

        if i_end < 100:
            flags.append(
                FlagRecord(
                    scope="global",
                    scope_id="pieces",
                    severity="info",
                    code="INSUFFICIENT_DATA_FOR_MORE_PIECES",
                    message=f"Only extracted {len(pieces)} pieces (requested {n_pieces}), remaining data too small",
                )
            )
            break

    # Create result
    result = Result(
        t=t,
        env=env,
        fn_hz=fn_hz,
        omega_n=omega_n,
        pieces=pieces,
        windows_trace=all_windows,
        flags=flags,
    )

    # Write artifacts if requested
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Import plotting module (lazy import to avoid matplotlib dependency if not needed)
        from .plots import create_diagnostic_plots

        plot_paths = create_diagnostic_plots(result, out_path)
        result.artifact_paths.update(plot_paths)

    return result


def fit_envelope_decay_manual(
    t: np.ndarray,
    env: np.ndarray,
    fn_hz: float,
    boundaries_time_s: list[float] | None = None,
    boundaries_index: list[int] | None = None,
    min_points: int = 10,
    out_dir: Path | str | None = None,
) -> Result:
    """Fit decay using manually provided boundary points.

    Args:
        t: time array (seconds, strictly increasing)
        env: envelope amplitude
        fn_hz: natural frequency in Hz
        boundaries_time_s: boundary times (seconds) to snap to samples
        boundaries_index: boundary indices (0-based)
        min_points: minimum samples per segment
        out_dir: optional output directory for manual segmentation JSON

    Returns:
        Result with manually segmented pieces
    """
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

    if boundaries_time_s is None and boundaries_index is None:
        raise ValueError("Provide boundaries_time_s or boundaries_index")

    if boundaries_time_s is not None:
        boundary_indices = snap_boundary_times_to_indices(t, boundaries_time_s)
    else:
        boundary_indices = sorted(set(int(i) for i in boundaries_index or []))

    pieces, flags = compute_manual_pieces(
        t,
        env,
        fn_hz,
        boundary_indices,
        min_points=min_points,
    )

    result = build_result_from_manual(
        t,
        env,
        fn_hz,
        boundary_indices,
        pieces,
        flags,
    )

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = write_manual_segmentation_json(
            result, out_path / "manual_segmentation.json"
        )
        result.artifact_paths["manual_segmentation_json"] = json_path

    return result
