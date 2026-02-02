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

* automatically segments the response into **piecewise decay regions** (default: 2 pieces),

* computes **multiple fits** per window (log-domain, linear w/ and w/o floor),

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

### Basic example

```bash
env-decay-fit input.csv \
  --fn-hz 150.0
```

This will:

* run the full piecewise decay analysis,
* create an output directory:

```
./out/2026-01-27_161530/
```

* write CSV summaries and diagnostic plots.

### Common options

```bash
env-decay-fit input.csv \
  --fn-hz 150.0 \
  --n-pieces 2 \
  --out-dir ./out/test_run
```

Key arguments:

* `--fn-hz` *(required)*: natural frequency in Hz
* `--n-pieces`: number of decay pieces to extract (default: 2)
* `--out-dir`: output directory (default: `./out/<timestamp>/`)

Run `env-decay-fit --help` for the full list.

### Manual segmentation (interactive)

Use the interactive Matplotlib workflow to select boundary points manually:

```bash
env-decay-fit input.csv \
  --fn-hz 150.0 \
  --manual-segmentation
```

Controls:

* Left click: add boundary point (snapped to nearest sample)
* `c`: clear all points
* `d`: delete nearest point to mouse
* `l`: toggle y-scale (lin/log)
* `u`: undo last point
* `enter`: commit and close
* `q`: quit without saving

Manual segmentation results are written to `manual_segmentation.json` in the output
directory (if `--out-dir` is provided).

---

## Outputs

When run via the CLI, the following artifacts are generated:

### CSV files

* `envelope.csv` — original input data
* `windows_trace.csv` — all expanding-window fits ("print everything")
* `pieces.csv` — extracted decay pieces and representative parameters
* `flags.csv` — warnings, rejections, and diagnostics
* `summary.csv` — single-row quick overview

All CSVs use a **wide, spreadsheet-friendly format**.

### Plots

* `piecewise_fit.png` — envelope with piecewise fitted curves
* `trace_scores.png` — R² vs window duration
* `trace_params.png` — ζ (and floor C) vs window duration

Plots include light annotation boxes with key values and flags.

---

## Programmatic usage

The core API performs **no file I/O by default** and returns a structured result.

```python
import numpy as np
from envelope_decay_fit.api import fit_envelope_decay

# t and env are numpy arrays
t = np.array([...])
env = np.array([...])

result = fit_envelope_decay(
    t,
    env,
    fn_hz=150.0,
    n_pieces=2,
)

# Access results
result.pieces
result.windows_trace
result.flags
```

If you want artifacts written programmatically:

```python
result = fit_envelope_decay(
    t,
    env,
    fn_hz=150.0,
    out_dir="./out/run_001",
)
```

---

## Design notes

* The decay start time `t0` is **not fitted as a free parameter**.
  Amplitude is anchored to the window start to keep fits well-posed.
* Breakpoints are detected using **change-point detection** on the log-fit R² trace.
* Tail trimming is automatic and robust (median + MAD), and applied before fitting.
* The tool never silently discards results: questionable regions are **flagged**, not hidden.

See `specs.md` and `implementation.md` for full technical details.

---

## License

MIT License.

You are free to use, modify, and embed this code in other projects.

---

## Status

**Version 0.1.0** - Working prototype

The package is functional and has been tested on real datasets. Key features:
- ✅ Three fitting methods (LOG, LIN0, LINC)
- ✅ Piecewise decay extraction with breakpoint detection
- ✅ Diagnostic plots and flag system
- ✅ CLI and programmatic API

### Known Limitations (v0.1.0)

1. **Window Sampling**: For performance, expanding windows are sampled (max 500 by default) rather than generating all possible windows. This is a pragmatic optimization that provides good results while keeping computation tractable for large datasets.

2. **No Tail Trimming**: Automatic tail trimming (median + MAD floor estimation) is not yet implemented. Users should pre-process data to remove noise-dominated tails if needed.

3. **Beating Detection**: The package will flag unusual patterns (low R², negative damping) but does not explicitly detect or segment beating phenomena. This is deferred to future versions.

4. **CSV Output**: The package currently generates plots but does not write CSV files for windows_trace, pieces, or flags. This can be added if needed.

5. **Limited Validation**: The breakpoint detection works well for clean exponential decays but may struggle with complex multi-modal responses or heavy beating.

### Performance

- Typical runtime: ~5-10 seconds for 68K samples (1.5s duration) with `max_windows=300`
- Scales linearly with `max_windows` parameter
- For very large datasets, consider downsampling or reducing `max_windows`

### Future Work (v0.2.0+)

- Implement tail trimming
- Add CSV output writers
- Improve breakpoint detection for complex signals
- Add beating detection
- Optimize window generation (Option B from specs)
- Add comprehensive test suite

Feedback and experiments are welcome!
