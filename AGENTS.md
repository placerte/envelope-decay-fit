# AGENTS.md

Agent guidelines for working in the `envelope-decay-fit` repository.

---

## Project Overview

**Type:** Python 3.13+ scientific utility for piecewise exponential decay fitting  
**Status:** Early development (v0.1.1) — core implementation not yet built  
**Purpose:** Fit time-domain response envelopes to extract damping ratio (ζ) with rich diagnostics  
**Package Manager:** `uv` (required)

---

## Build / Test / Lint Commands

### Environment Setup
```bash
uv venv                    # Create virtual environment
uv pip install -e .        # Install in editable mode
```

### Testing
**Status:** Test framework not yet configured (likely pytest when added)

Expected commands (once tests exist):
```bash
pytest tests/              # Run all tests
pytest tests/test_fitters.py  # Run specific test file
pytest tests/test_fitters.py::test_log_fit  # Run single test
pytest -v                  # Verbose output
pytest -k "pattern"        # Run tests matching pattern
```

### Linting / Formatting
**Status:** No linters or formatters configured yet

When added, likely commands:
```bash
ruff check .               # Lint (if ruff is added)
ruff format .              # Format (if ruff is added)
mypy src/                  # Type checking (if mypy is added)
```

### Build
```bash
uv build                   # Build package (when ready)
```

---

## Code Style Guidelines

### Philosophy (CRITICAL — from docs/persona.md)

**User is NOT a software developer** → keep code simple, readable, and explicit.

**Core principles:**
- Explicit over clever
- Clarity over elegance
- Simple control flow (loops preferred over dense comprehensions)
- Avoid frameworks, patterns, and "clever" architectures
- No string-based magic or reflection (`getattr`, dynamic wiring)
- Fast iteration over perfect design

### Type Hints (REQUIRED)
- Use type hints throughout
- Be explicit with types (no `Any` unless truly justified)
- Example:
```python
def fit_log_domain(t: np.ndarray, env: np.ndarray, t_ref: float) -> tuple[float, float, float]:
    """Returns (alpha, b, r2)"""
    ...
```

### Imports

**Standard library:**
```python
from pathlib import Path
from dataclasses import dataclass
import argparse
```

**Scientific stack:**
```python
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
```

**Optional (CLI/plotting):**
```python
import matplotlib.pyplot as plt
```

**Package imports:**
```python
from envelope_decay_fit.api import fit_piecewise_manual
from envelope_decay_fit.config import Config
from envelope_decay_fit.flags import FlagRecord
```

**Style:**
- Explicit imports (no `import *`)
- Group: stdlib → third-party → local
- Sort alphabetically within groups

### Naming Conventions

**Variables:**
- `snake_case` for all variables and functions
- Physics/math symbols allowed: `α`, `ζ`, `ω_n`
- Descriptive suffixes: `_s` (seconds), `_hz` (Hertz), `_rad` (radians)

**Standard variable names (from specs.md):**
- `t` — time array (seconds)
- `env` — envelope amplitude
- `fn_hz` or `f_n` — natural frequency in Hz
- `omega_n` or `ω_n` — angular frequency (rad/s)
- `alpha` or `α` — decay rate (1/s)
- `zeta` or `ζ` — damping ratio (dimensionless)
- `dt` or `Δt` — time duration/window length
- `i_start`, `i_end` — array indices
- `t_start`, `t_end` — time values (prefer `_s` suffix: `t_start_s`, `t_end_s`)
- `n_points`, `n_samples` — counts

**Classes:**
- `CamelCase` for classes and dataclasses
- Examples: `Config`, `FlagRecord`, `PieceResult`, `WindowFit`

**Segment labels:**
- `established_free_decay` — late, stable decay region
- `transient_dominated_decay` — early, coupled decay region

**Method suffixes:**
- `_log` — log-domain fit
- `_lin0` — linear domain, no floor
- `_linc` — linear domain, with floor constant C

### Control Flow

**Prefer explicit loops:**
```python
# Good
results = []
for i, window in enumerate(windows):
    fit = compute_fit(window)
    results.append(fit)
```

**Avoid dense comprehensions when they hurt readability:**
```python
# Avoid (unless very simple)
results = [compute_fit(w) for i, w in enumerate(windows) if is_valid(w) and len(w.data) > 10]
```

### Error Handling

**Philosophy:** "Compute everything, report everything, decide later"

**Use flag-based reporting for data issues (NOT exceptions):**
```python
@dataclass
class FlagRecord:
    scope: str        # 'global' | 'window' | 'piece'
    scope_id: str     # identifier
    severity: str     # 'info' | 'warn' | 'reject'
    code: str         # flag code (e.g., 'TAIL_TRIM_APPLIED')
    message: str      # human readable
    details: str = "" # optional extra info
```

**Flag examples:**
- `TAIL_TRIM_APPLIED` (info)
- `TAIL_FLOOR_DOMINANT` (warn)
- `INSUFFICIENT_SAMPLES` (reject)
- `NO_BREAKPOINT_FOUND` (warn/reject)
- `LOG_INVALID_NONPOSITIVE_ENV` (warn)
- `ZETA_UNSTABLE_IN_ESTABLISHED_SEGMENT` (warn)

**DO use exceptions for:**
- Input validation failures (bad array shapes, non-monotonic time)
- Programming errors (assertions, invalid state)

**Pattern:**
```python
# Validate inputs strictly
if not np.all(np.diff(t) > 0):
    raise ValueError("Time array 't' must be strictly increasing")

# Flag data quality issues, but continue
if np.any(env < 0):
    flags.append(FlagRecord(
        scope="global",
        scope_id="input",
        severity="warn",
        code="NEGATIVE_ENV_VALUES",
        message=f"Found {np.sum(env < 0)} negative envelope values"
    ))
```

### Data Structures

**Prefer dataclasses over dictionaries:**
```python
@dataclass
class WindowFit:
    i_start: int
    i_end: int
    t_start_s: float
    t_end_s: float
    dt_s: float
    alpha_log: float
    zeta_log: float
    r2_log: float
    valid_log: bool
```

**NOT:**
```python
window_fit = {"i_start": 10, "i_end": 100, ...}  # Avoid
```

---

## Architecture Principles

### Separation of Concerns

**Core engine never writes to disk unless explicitly requested:**
```python
# Core API (no I/O by default)
result = fit_piecewise_manual(t, env, [t[0], t[-1]], fn_hz=150.0)

# Explicit artifact writing
result = fit_piecewise_manual(t, env, [t[0], t[-1]], fn_hz=150.0)
```

**CLI always writes artifacts by default** (for debugging).

### No Magic

- Avoid dynamic attribute access (`getattr`, `setattr`)
- Avoid string-based dispatch
- Prefer explicit calls over reflection
- Keep module dependencies explicit

---

## Testing Guidelines (When Tests Exist)

- Test files: `tests/test_*.py`
- Test data: Use CSV files in `data/envelope_exports/` (Git LFS tracked)
- Test naming: `test_<function_name>_<scenario>`
- Use fixtures for common test data setup
- Assert on dataclass fields explicitly (no magic comparisons)

---

## File I/O Conventions

### Input CSV format:
Required columns: `t_s` (seconds), `env` (amplitude)

### Output CSV format:
- **Wide format** (spreadsheet-friendly)
- One row per record
- Clear column names with units (`t_start_s`, `alpha_log`, `r2_log`)

### Paths:
- Always use `pathlib.Path` (never string concatenation)
- Check existence before reading
- Create parent directories before writing

---

## Documentation

- Docstrings for all public functions/classes
- Focus on **what** and **why**, not just **how**
- Include units in parameter descriptions
- Reference equation forms when applicable

Example:
```python
def fit_log_domain(t: np.ndarray, env: np.ndarray, t_ref: float) -> tuple[float, float, float]:
    """
    Fit exponential decay in log domain.
    
    Model: ln(env(t)) ≈ b - α (t - t_ref)
    
    Args:
        t: Time array (seconds)
        env: Envelope amplitude (must be > 0)
        t_ref: Reference time (typically window start)
    
    Returns:
        (alpha, b, r2): Decay rate (1/s), intercept, R² score
    """
```

---

## Git / Version Control

- Use `uv` exclusively for package management
- Large data files (*.png, *.csv) tracked with Git LFS
- Commit messages: concise, imperative mood
- Do NOT commit:
  - `__pycache__/`
  - `.venv/`
  - `*.pyc`
  - temporary output directories

---

## Key References

**MUST READ before making changes:**
- `docs/persona.md` — User preferences and interaction style
- `docs/specs.md` — Locked technical specifications
- `docs/implementation.md` — Implementation architecture

**Philosophy:**
- Keep it simple
- Make it inspectable
- Report everything, hide nothing
- Let the user decide what's "truth"

---

## Questions / Uncertainty

If you encounter ambiguity:
1. Check `docs/specs.md` first (specifications are locked)
2. Check `docs/persona.md` for style preferences
3. Ask the user directly — "I don't know" is acceptable
4. Show assumptions and tradeoffs in your reasoning
