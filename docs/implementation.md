# envelope-decay-fit — implementation.md

## Purpose

This document describes the concrete implementation plan for the `envelope-decay-fit` package, based on the locked specifications in `specs.md`.

The goals of this implementation are:

* clean separation between **core computation** and **endpoints** (CLI / file outputs),
* fast iteration and debuggability (CSV + plots),
* easy integration into a larger application (`wav-to-freq`),
* minimal architectural regret for v1.

---

## High-level architecture

The package is split into three conceptual layers:

```
CLI (argparse)
   │
   ▼
api.fit_piecewise_auto()    ← pure computation, no I/O by default (experimental)
   │
   ├── tail trimming
   ├── expanding-window fitting
   ├── breakpoint detection
   ├── piece extraction
   └── flag generation
   │
   ▼
Result (dataclass)
   │
   └── artifacts writers (CSV / plots) [optional side-effects]
```

Key principles:

* **Core engine never writes to disk unless explicitly requested**.
* **CLI always writes artifacts by default** (for inspection/debugging).
* All heavy logic lives in importable modules (no CLI-only logic).

---

## Package layout

Standard `src/` layout, compatible with `uv build` and PyPI:

```
envelope-decay-fit/
├── pyproject.toml
├── README.md
├── LICENSE
├── specs.md
├── implementation.md
├── src/envelope_decay_fit/
│   ├── __init__.py
│   ├── api.py              # public API entrypoint
│   ├── models.py           # FitResult / PieceFit models
│   ├── fitters/            # LOG / LIN0 / LINC fits (SciPy)
│   ├── segmentation/
│   │   ├── manual.py
│   │   └── auto/            # experimental auto pipeline
│   ├── plotting/
│   ├── flags.py
│   ├── result.py           # internal Result dataclasses
│   └── cli.py              # argparse-based CLI
├── tests/
├── examples/
└── validation/
```

---

## Dependencies (v1)

Required:

* `numpy`
* `scipy`

Optional (CLI / artifacts):

* `matplotlib`

Notes:

* SciPy is allowed in v1 for robustness and speed of development.
* Portability optimizations (NumPy-only backend) are explicitly deferred.

---

## Public API

### Function entrypoint (locked)

```python
from envelope_decay_fit.api import fit_piecewise_manual

result = fit_piecewise_manual(
    t,
    env,
    breakpoints_t=[t[0], t[-1]],
    fn_hz=fn_hz,
)
```

Arguments:

* `t: ndarray[float]` — time array (seconds, strictly increasing)
* `env: ndarray[float]` — envelope amplitude (non-negative expected)
* `fn_hz: float` — natural frequency in Hz (required)
* `breakpoints_t: list[float]` — manual boundary times (seconds)
* `fn_hz: float` — natural frequency in Hz (required)

Returns:

* `Result` dataclass (see below)

---

## Core dataclasses

### Config

Holds all tunable parameters with sane defaults.

Examples:

* `tail_trim_mode: "off" | "auto" | "fixed"`
* `tail_probe_s`
* `tail_k_mad`
* `tail_min_run_s`
* `min_points_for_selection = 32`
* smoothing parameters for score traces

### WindowFitRecord

One per expanding window:

* window bounds (`i_start`, `i_end`, `t_start`, `t_end`, `dt`, `n_points`)
* LOG fit results
* LIN0 fit results
* LINC fit results
* validity flags and notes

### PieceRecord

One per extracted piece:

* piece bounds and label
* breakpoint metadata
* representative fit results per method (LOG / LIN0 / LINC)

### FlagRecord

Structured flags:

* `scope`: `global | window | piece`
* `scope_id`
* `severity`: `info | warn | reject`
* `code`
* `message`

### Result

Top-level return object:

* input metadata
* trimming diagnostics
* list of `WindowFitRecord`
* list of `PieceRecord`
* list of `FlagRecord`
* artifact file paths (if written)

---

## Core algorithm (step-by-step)

### 1. Input validation

* check shapes and lengths
* ensure `t` is strictly increasing
* warn if `env < 0` or contains NaN/Inf

### 2. Tail trimming (optional, auto)

* estimate floor from tail using median + MAD
* compute cut level
* determine effective end index
* emit trimming flags and diagnostics

### 3. Piece extraction loop

For each requested piece:

1. Set current `i_end`.
2. Generate expanding windows by moving `i_start` backward.
3. For each window:

   * compute LOG fit
   * compute LIN0 fit
   * compute LINC fit
   * record scores and validity
4. Build score trace `R²_log(Δt)`.
5. Smooth trace lightly.
6. Apply change-point detection (two-regime SSE minimization).
7. Enforce constraints:

   * minimum window size for selection
   * avoid tail-noise region
8. Extract piece using detected breakpoint.
9. Record representative parameters from the extracted window.
10. Update `i_end` and continue.

Stop early if insufficient data remains.

---

## Fitting backends

### LOG fit

* linear regression on `ln(env)` vs `t`
* invalid if any `env <= 0` in window

### LIN0 fit

* nonlinear least squares (`curve_fit`)
* model: `A * exp(-alpha * (t - t_ref))`

### LINC fit

* nonlinear least squares with floor
* model: `A * exp(-alpha * (t - t_ref)) + C`

All fits return:

* `alpha`
* derived `zeta = alpha / (2π fn)`
* `R²`
* `RMSE`

---

## Breakpoint detection

* Primary trace: `R²_log(Δt)`
* Light smoothing (moving average or Savitzky–Golay)
* Split index chosen by minimizing SSE of a two-regime model
* Late regime assumed stable/high quality
* Emit flags if breakpoint is weak or ambiguous

---

## Artifacts (optional side-effects)

### CSV writers

* `envelope.csv`
* `windows_trace.csv`
* `pieces.csv`
* `flags.csv`
* `summary.csv`

All CSVs use **wide format** for spreadsheet debugging.

### Plots

* `piecewise_fit.png`: envelope + fitted pieces
* `trace_scores.png`: R² traces vs Δt
* `trace_params.png`: ζ traces (and C if relevant)

Plots include light annotation boxes with top flags and values.

---

## CLI implementation

### Framework (locked)

* `argparse` (stdlib)

### Example usage

```bash
env-decay-fit input.csv \
  --fn-hz 150.0 \
  --n-pieces 2
```

Defaults:

* input CSV must contain columns `t_s`, `env`
* output written to `./out/<timestamp>/`

CLI responsibilities:

* CSV parsing + validation
* invoking core API
* artifact writing
* short console summary

---

## Non-goals (v1)

* envelope extraction from raw waveform
* NumPy-only fitting backend
* automatic selection of a single “best” ζ
* GUI / TUI

---

## Notes for future versions

* plug-in fit backends (SciPy vs NumPy)
* AIC/BIC scoring
* minimum cycles constraint (in addition to min points)
* JSON / NPZ artifact export
* tighter integration into `wav-to-freq`
