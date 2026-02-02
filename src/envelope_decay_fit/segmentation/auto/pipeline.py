"""Experimental automatic segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from ...fitters import fit_linc_domain, fit_lin0_domain, fit_log_domain
from ...flags import FlagRecord
from ...result import PieceRecord, Result
from .breakpoints import detect_breakpoint_two_regime, smooth_trace
from .window_scan import extract_score_trace, generate_expanding_windows

LOGGER = logging.getLogger(__name__)


@dataclass
class AutoSegmentationConfig:
    """Configuration for automatic piecewise segmentation."""

    n_pieces: int = 2
    max_windows: int = 500
    min_points: int = 50
    min_established_duration_s: float = 0.05
    min_established_points: int = 10


def fit_piecewise_auto_result(
    t: np.ndarray,
    env: np.ndarray,
    fn_hz: float,
    config: AutoSegmentationConfig | None = None,
) -> Result:
    """Fit decay pieces using the experimental automatic segmentation pipeline."""
    if config is None:
        config = AutoSegmentationConfig()

    flags: list[FlagRecord] = []

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

    if config.n_pieces < 1:
        raise ValueError(f"n_pieces must be >= 1 (got {config.n_pieces})")

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

    pieces: list[PieceRecord] = []
    all_windows = []
    i_end = len(t)

    for piece_idx in range(config.n_pieces):
        LOGGER.info("Extracting piece %s/%s", piece_idx + 1, config.n_pieces)
        windows = generate_expanding_windows(
            t,
            env,
            fn_hz,
            i_end,
            i_start_min=0,
            min_points=config.min_points,
            max_windows=config.max_windows,
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

        r2_smooth = smooth_trace(r2, window_size=min(5, max(3, len(r2) // 20)))

        breakpoint_idx, bp_flags = detect_breakpoint_two_regime(
            dt,
            r2_smooth,
            min_established_duration_s=config.min_established_duration_s,
            min_established_points=config.min_established_points,
        )
        flags.extend(bp_flags)

        if breakpoint_idx is not None:
            piece_window = windows[breakpoint_idx]
            i_start_piece = piece_window.i_start
            breakpoint_dt_s = dt[breakpoint_idx]
        else:
            if len(windows) > 0:
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

        t_ref = t_piece[0]
        log_fit = fit_log_domain(t_piece, env_piece, fn_hz, t_ref)
        lin0_fit = fit_lin0_domain(t_piece, env_piece, fn_hz, t_ref)
        linc_fit = fit_linc_domain(t_piece, env_piece, fn_hz, t_ref)

        if piece_idx == config.n_pieces - 1:
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

        i_end = i_start_piece

        if i_end < 100:
            flags.append(
                FlagRecord(
                    scope="global",
                    scope_id="pieces",
                    severity="info",
                    code="INSUFFICIENT_DATA_FOR_MORE_PIECES",
                    message=(
                        "Only extracted "
                        + f"{len(pieces)} pieces (requested {config.n_pieces}), "
                        + "remaining data too small"
                    ),
                )
            )
            break

    return Result(
        t=t,
        env=env,
        fn_hz=fn_hz,
        omega_n=omega_n,
        pieces=pieces,
        windows_trace=all_windows,
        flags=flags,
    )
