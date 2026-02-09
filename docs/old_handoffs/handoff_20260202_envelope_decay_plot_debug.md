# handoff_20260202_envelope_decay_plot_debug.md

## Objective

Introduce **deterministic figure debugging** into the `envelope-decay-fit` project
by integrating the `mpl-plot-report` package and refactoring all figure exports
to produce structured sidecars alongside PNGs.

This work is strictly about **plot export unification and metadata capture**.
No changes to:
- DSP logic
- fitting math
- plot design / aesthetics

---

## Background

A lightweight plotting sidecar tool was recently created and validated:

- Package: **mpl-plot-report**
- PyPI: https://pypi.org/project/mpl-plot-report/
- Status: **Already successfully integrated in `wav-to-freq`**

That integration validated:
- API shape
- JSON + Markdown sidecar format
- CI / agent friendliness
- Debug value for Matplotlib-heavy workflows

The *original intended target*, however, was **`envelope-decay-fit`**.
This handoff exists to correct that and port the workflow cleanly.

---

## envelope-decay-fit context

### Project goal
- Fit exponential decay envelopes to vibration responses
- Extract damping ratio ζ from decay curves

### Typical figures
- Raw response vs time
- Filtered response
- Envelope (Hilbert / peak-picking)
- Log decrement / linearized decay
- Fit diagnostics (residuals, goodness of fit)

### Current pain point
- Many ad-hoc `plt.figure()` / `plt.subplots()`
- Direct `plt.savefig()` / `fig.savefig()`
- Figures emitted from:
  - dev scripts
  - tests / experiments
  - CLI runs

Plot debugging is becoming difficult because:
- intent is implicit
- pixels are not diffable
- metadata is scattered or lost

---

## Integration strategy (locked)

Mirror the **wav-to-freq** solution, adapted to this repo.

### 1. Dependency

Add:

    uv add mpl-plot-report

No vendoring, no copy-paste of internals.

---

### 2. Single export wrapper

Introduce **one canonical export helper**, e.g.:

    envelope_decay_fit/
    └── plotting/
        └── plot_export.py

Responsibilities:
- accept a `matplotlib.figure.Figure`
- write:
  - PNG
  - `.plot.json`
  - `.plot.md`
- attach a structured `context` dict

No plotting logic inside this file.

---

### 3. Context schema (initial)

Context is intentionally lightweight and extensible.

Typical fields:

    context = {
        "plot_kind": "raw | filtered | envelope | log_decay | fit | residuals",
        "fs_hz": float,
        "method": "hilbert | peak_picking | log_dec",
        "fit_window": [t_start, t_end],
        "zeta": float | None,
        "status": "OK | WARNING | REJECTED",
    }

Notes:
- Values may be `None` when unavailable
- Schema is descriptive, not validated (yet)

---

### 4. Refactor rule (hard rule)

**Every figure save must go through the wrapper.**

Replace:
- `plt.savefig(...)`
- `fig.savefig(...)`

With:
- `export_plot(fig, path=..., context=...)`

Applies to:
- dev scripts
- tests
- CLI outputs

Visual PNG output **must remain identical**.

---

## What is explicitly out of scope

❌ Plot redesign  
❌ Label/style normalization  
❌ Fitting algorithm changes  
❌ DSP refactors  
❌ Rule engines / linting (can be added later)

This is *infrastructure only*.

---

## Acceptance criteria

The integration is considered complete when:

1. `mpl-plot-report` is a declared dependency
2. A single plot export wrapper exists
3. No direct `savefig()` calls remain
4. Each generated PNG has:
   - a matching `.plot.json`
   - a matching `.plot.md`
5. Existing plots render unchanged visually
6. Sidecars contain sufficient context to:
   - diff plots
   - debug failures
   - reason about ζ extraction

---

## Notes for future work (not now)

- Optional rule sets (missing labels, axis sanity, etc.)
- CI diffing of `.plot.json`
- Auto-collection of plot diagnostics
- Agent-based plot review

None of the above is required for this handoff.

---

## Reference

Use the `wav-to-freq` integration as the canonical example for:
- wrapper shape
- naming conventions
- directory placement

Do not reinvent unless envelope-decay-fit imposes a hard constraint.

