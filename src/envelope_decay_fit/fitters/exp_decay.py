"""Exponential decay fitting functions (LOG, LIN0, LINC)."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats


@dataclass
class FitResult:
    """Result from a single fit operation.

    Attributes:
        alpha: decay rate (1/s)
        zeta: damping ratio (dimensionless)
        r2: coefficient of determination
        rmse: root mean square error
        valid: whether fit succeeded
        params: additional fit parameters (e.g., A, b, C)
        notes: optional diagnostic notes
    """

    alpha: float
    zeta: float
    r2: float
    rmse: float
    valid: bool
    params: dict
    notes: str = ""


def fit_log_domain(
    t: np.ndarray,
    env: np.ndarray,
    fn_hz: float,
    t_ref: float | None = None,
) -> FitResult:
    """Fit exponential decay in log domain.

    Model: ln(env(t)) ≈ b - α (t - t_ref)

    This is a linear fit in log space. It requires strictly positive envelope
    values and is invalid if any env <= 0.

    Args:
        t: time array (seconds)
        env: envelope amplitude (must be > 0)
        fn_hz: natural frequency in Hz
        t_ref: reference time (default: t[0])

    Returns:
        FitResult with alpha, zeta, r2, and validity
    """
    if t_ref is None:
        t_ref = t[0]

    omega_n = 2.0 * np.pi * fn_hz

    # Check for non-positive values
    if np.any(env <= 0):
        n_bad = np.sum(env <= 0)
        return FitResult(
            alpha=np.nan,
            zeta=np.nan,
            r2=np.nan,
            rmse=np.nan,
            valid=False,
            params={"b": np.nan},
            notes=f"LOG_INVALID: {n_bad} non-positive envelope values",
        )

    # Log transform
    t_shifted = t - t_ref
    ln_env = np.log(env)

    # Linear regression: ln(env) = b - alpha * t_shifted
    # Using scipy.stats.linregress for robustness
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_shifted, ln_env)

    alpha = -slope  # Note: slope is negative in our model
    zeta = alpha / omega_n
    r2 = r_value**2

    # Calculate RMSE in log space
    ln_env_pred = intercept + slope * t_shifted
    rmse = np.sqrt(np.mean((ln_env - ln_env_pred) ** 2))

    return FitResult(
        alpha=alpha,
        zeta=zeta,
        r2=r2,
        rmse=rmse,
        valid=True,
        params={"b": intercept},
        notes="",
    )


def fit_lin0_domain(
    t: np.ndarray,
    env: np.ndarray,
    fn_hz: float,
    t_ref: float | None = None,
) -> FitResult:
    """Fit exponential decay in linear domain (no floor).

    Model: env(t) ≈ A * exp(-α (t - t_ref))

    Uses nonlinear least squares (scipy.optimize.curve_fit).

    Args:
        t: time array (seconds)
        env: envelope amplitude
        fn_hz: natural frequency in Hz
        t_ref: reference time (default: t[0])

    Returns:
        FitResult with alpha, zeta, r2, and validity
    """
    if t_ref is None:
        t_ref = t[0]

    omega_n = 2.0 * np.pi * fn_hz
    t_shifted = t - t_ref

    # Model function
    def exp_decay(t_shift: np.ndarray, A: float, alpha: float) -> np.ndarray:
        return A * np.exp(-alpha * t_shift)

    # Initial guess: A from first point, alpha from log fit if possible
    A_guess = env[0]
    alpha_guess = 0.1  # Default fallback

    # Try to get better alpha guess from log fit
    if np.all(env > 0):
        try:
            log_fit = fit_log_domain(t, env, fn_hz, t_ref)
            if log_fit.valid:
                alpha_guess = log_fit.alpha
        except Exception:
            pass

    try:
        popt, pcov = curve_fit(
            exp_decay,
            t_shifted,
            env,
            p0=[A_guess, alpha_guess],
            maxfev=10000,
        )
        A_fit, alpha_fit = popt

        # Calculate R²
        env_pred = exp_decay(t_shifted, A_fit, alpha_fit)
        ss_res = np.sum((env - env_pred) ** 2)
        ss_tot = np.sum((env - np.mean(env)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Calculate RMSE
        rmse = np.sqrt(np.mean((env - env_pred) ** 2))

        zeta = alpha_fit / omega_n

        return FitResult(
            alpha=alpha_fit,
            zeta=zeta,
            r2=r2,
            rmse=rmse,
            valid=True,
            params={"A": A_fit},
            notes="",
        )

    except Exception as e:
        return FitResult(
            alpha=np.nan,
            zeta=np.nan,
            r2=np.nan,
            rmse=np.nan,
            valid=False,
            params={"A": np.nan},
            notes=f"LIN0_FIT_FAILED: {str(e)}",
        )


def fit_linc_domain(
    t: np.ndarray,
    env: np.ndarray,
    fn_hz: float,
    t_ref: float | None = None,
) -> FitResult:
    """Fit exponential decay with constant floor in linear domain.

    Model: env(t) ≈ A * exp(-α (t - t_ref)) + C

    Uses nonlinear least squares (scipy.optimize.curve_fit).

    Args:
        t: time array (seconds)
        env: envelope amplitude
        fn_hz: natural frequency in Hz
        t_ref: reference time (default: t[0])

    Returns:
        FitResult with alpha, zeta, r2, and validity (includes C in params)
    """
    if t_ref is None:
        t_ref = t[0]

    omega_n = 2.0 * np.pi * fn_hz
    t_shifted = t - t_ref

    # Model function
    def exp_decay_floor(
        t_shift: np.ndarray, A: float, alpha: float, C: float
    ) -> np.ndarray:
        return A * np.exp(-alpha * t_shift) + C

    # Initial guess
    A_guess = env[0] - env[-1]  # Amplitude above floor
    C_guess = env[-1]  # Floor estimate from tail
    alpha_guess = 0.1

    # Try to get better alpha guess from LIN0 fit
    try:
        lin0_fit = fit_lin0_domain(t, env, fn_hz, t_ref)
        if lin0_fit.valid:
            alpha_guess = lin0_fit.alpha
    except Exception:
        pass

    try:
        popt, pcov = curve_fit(
            exp_decay_floor,
            t_shifted,
            env,
            p0=[A_guess, alpha_guess, C_guess],
            maxfev=10000,
        )
        A_fit, alpha_fit, C_fit = popt

        # Calculate R²
        env_pred = exp_decay_floor(t_shifted, A_fit, alpha_fit, C_fit)
        ss_res = np.sum((env - env_pred) ** 2)
        ss_tot = np.sum((env - np.mean(env)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Calculate RMSE
        rmse = np.sqrt(np.mean((env - env_pred) ** 2))

        zeta = alpha_fit / omega_n

        return FitResult(
            alpha=alpha_fit,
            zeta=zeta,
            r2=r2,
            rmse=rmse,
            valid=True,
            params={"A": A_fit, "C": C_fit},
            notes="",
        )

    except Exception as e:
        return FitResult(
            alpha=np.nan,
            zeta=np.nan,
            r2=np.nan,
            rmse=np.nan,
            valid=False,
            params={"A": np.nan, "C": np.nan},
            notes=f"LINC_FIT_FAILED: {str(e)}",
        )
