# envelope-decay-fit

Piecewise exponential decay fitting for time-domain response envelopes, with explicit damping ratio (ζ) extraction and rich diagnostics.

This package is designed as a **small, focused utility** that can be:

* used standalone (CLI + CSV + plots) for development and debugging,
* embedded programmatically into larger workflows (e.g. `wav-to-freq`).

---

## What this does

Given a **time-domain envelope** `env(t)` (e.g. Hilbert envelope) and a known natural frequency `f_n`, this tool:

* fits exponential decay models of the form:

  [ env(t) \approx A e^{-\alpha t} (+ C) ]

* derives the damping ratio:

  [ \zeta = \alpha / (2 \pi f_n) ]

* handles **real-world data** where decay is *not* purely exponential:

  * early hit artifacts,
  * modal coupling,
  * noise-floor dominated tails,

* supports **manual, human-in-the-loop breakpoints** as the primary workflow,
* keeps the automatic segmentation pipeline as experimental/secondary,

* computes **multiple fits** per window (log-domain, linear w/ and w/o floor),

* offers **span-based Tx measurement** (T10/T20/T30/T60) as a second method,

* reports **all results**, plus quality metrics and flags.

The philosophy is: **compute everything, report everything, decide later**.

---

## Installation

### From PyPI (once published)

```bash
pip install envelope-decay-fit
```

or with `uv`:

```bash
uv pip install envelope-decay-fit
```

### From source (development)

```bash
git clone https://github.com/<your-org>/envelope-decay-fit.git
cd envelope-decay-fit
uv venv
uv pip install -e .
```

---

## CLI usage

The CLI is intended mainly for **debugging, exploration, and small workflows**.

### Input format

Input is a CSV file with **at least** the following columns:

* `t_s` — time in seconds (strictly increasing)
* `env` — envelope amplitude

Example:

```csv
t_s,env
0.0000,1.23
0.0005,1.18
0.0010,1.12
...
```

### Basic example (manual workflow)

Default mode runs manual segmentation and then fits immediately:

```bash
env-decay-fit input.csv --fn-hz 150.0
```

Explicit combined command (same behavior):

```bash
env-decay-fit segment-fit input.csv --fn-hz 150.0
```

Or run the steps separately:

```bash
env-decay-fit segment input.csv --fn-hz 150.0 \
  --breakpoints-out out/breakpoints.json

env-decay-fit fit input.csv --fn-hz 150.0 \
  --breakpoints-file out/breakpoints.json
```

### Manual segmentation controls

The UI is keyboard-driven. Mouse movement updates the cursor position.

* `a`: add boundary at cursor (snapped to nearest sample)
* `x`: delete nearest boundary
* `c`: clear all boundaries
* `l`: toggle y-scale (lin/log)
* `h`: toggle help panel
* `q`: quit and save current state

### Other notes

* The CLI writes outputs to `out/` by default.
* Run `env-decay-fit --help` for the full list of options.

---

## Outputs

The CLI writes to `out/` by default.

* `breakpoints.json` — manual breakpoints from the UI (segment command)
* `fit_result.json` — piecewise fit summary (fit command)
* `segmentation_storyboard.png` — envelope with fitted pieces

---

## Programmatic usage

The core API performs **no file I/O by default** and returns a structured result.

```python
import numpy as np
from envelope_decay_fit import (
    fit_piecewise_manual,
    launch_manual_segmentation_ui,
    plot_segmentation_storyboard,
)

t = np.array([...])
env = np.array([...])
fn_hz = 150.0

# Manual breakpoints supplied explicitly
breakpoints_t = [t[0], t[-1]]
fit = fit_piecewise_manual(t, env, breakpoints_t, fn_hz=fn_hz)

# Optional interactive UI for breakpoint selection
breakpoints_t = launch_manual_segmentation_ui(t, env, fn_hz=fn_hz)

# Plotting (no file I/O unless you save the figure)
fig = plot_segmentation_storyboard(t, env, fit)
fig.savefig("out/storyboard.png", dpi=150)
```

Experimental auto segmentation is available as `fit_piecewise_auto(...)`, but it
is intentionally not the default workflow.

---

## Design notes

* The decay start time `t0` is **not fitted as a free parameter**.
  Amplitude is anchored to the window start to keep fits well-posed.
* Breakpoints are detected using **change-point detection** on the log-fit R² trace.
* Tail trimming is automatic and robust (median + MAD), and applied before fitting.
* The tool never silently discards results: questionable regions are **flagged**, not hidden.

See `specs.md` and `implementation.md` for full technical details.

---

## Testing and validation

Unit tests live in `tests/` and are fast by default:

```bash
pytest -q
```

Slow, human-review workflows live under `validation/`:

```bash
uv run python validation/review_runner.py
```

---

## License

MIT License.

You are free to use, modify, and embed this code in other projects.

---

## Status

**Version 0.2.0** - Working prototype

The package is functional and has been tested on real datasets. Key features:
- ✅ Three fitting methods (LOG, LIN0, LINC)
- ✅ Piecewise decay extraction with breakpoint detection
- ✅ Diagnostic plots and flag system
- ✅ CLI and programmatic API
- ✅ Span-based Tx measurement (interactive, opt-in)

### Known Limitations (v0.2.0)

1. **Window Sampling**: For performance, expanding windows are sampled (max 500 by default) rather than generating all possible windows. This is a pragmatic optimization that provides good results while keeping computation tractable for large datasets.

2. **No Tail Trimming**: Automatic tail trimming (median + MAD floor estimation) is not yet implemented. Users should pre-process data to remove noise-dominated tails if needed.

3. **Beating Detection**: The package will flag unusual patterns (low R², negative damping) but does not explicitly detect or segment beating phenomena. This is deferred to future versions.

4. **CSV Output**: The package currently generates plots but does not write CSV files for windows_trace, pieces, or flags. This can be added if needed.

5. **Limited Validation**: The breakpoint detection works well for clean exponential decays but may struggle with complex multi-modal responses or heavy beating.

### Performance

- Typical runtime: ~5-10 seconds for 68K samples (1.5s duration) with `max_windows=300`
- Scales linearly with `max_windows` parameter
- For very large datasets, consider downsampling or reducing `max_windows`

### Future Work (v0.3.0+)

- Implement tail trimming
- Add CSV output writers
- Improve breakpoint detection for complex signals
- Add beating detection
- Optimize window generation (Option B from specs)
- Add comprehensive test suite

Feedback and experiments are welcome!
