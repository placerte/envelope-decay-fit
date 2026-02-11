# code\_style.md

> Living document: conventions for humans + agents contributing to this repo.
> Goal: maximize readability, correctness, and agent reliability while minimizing noise.

## 0) Scope

Applies to:

- Python packages, CLIs, TUIs (Textual), scripts
- Tests (pytest)
- Documentation adjacent code (examples, snippets)

Non-goals:

- Enforce a single “perfect” style everywhere.
- Replace tooling (ruff/black/mypy) with prose rules.

---

## 1) Guiding principles

1. **Type hints first.** Prefer explicit types and dataclasses/pydantic-style models where it improves clarity.
2. **Avoid stringly-typed wiring.** No `getattr()` / attribute lookup by string for core logic; prefer explicit attributes and typed objects. 
3. **Boundaries are documented.** Public APIs and domain/math functions must explain assumptions and units.
4. **Small, testable units.** Functions should be easy to test in isolation.
5. **Readable over clever.** Prefer explicit loops over dense comprehensions when clarity is improved.
6. **Stable interfaces.** Don’t break existing workflows; add new methods behind feature flags or new entry points.

---

## 2) Python formatting + linting expectations

- Use `black`-compatible formatting.
- Use `ruff` for linting; fix warnings unless explicitly justified.
- Prefer `pathlib.Path` over string paths.
- Prefer `logging` over `print` (except quick scripts / debug paths).

---

## 3) Naming

- **Functions:** verbs, explicit (e.g., `extract_natural_frequencies`, `fit_decay_envelope`).
- **Private helpers:** prefix with `_`.
- **Constants:** `UPPER_SNAKE_CASE`.
- **Units in names** when ambiguity exists (e.g., `time_s`, `freq_hz`).

---

## 4) Docstrings policy (selective, high-signal)

### 4.1 Required docstrings

Docstrings are **required** for:

- Public API functions/classes (anything imported by users)
- Domain/math functions (anything encoding physical assumptions)
- File/IO boundaries (read/write, parsing, serialization)
- Any function with non-obvious units, conventions, or edge cases

### 4.2 Optional docstrings

Docstrings are optional (often omitted) for:

- Trivial wrappers with obvious behavior
- Tiny internal helpers where name + typing + body are self-evident

### 4.3 Docstring contents (minimum)

- What it does (1–2 lines)
- Parameters (including **units** where relevant)
- Returns (shape/type, units)
- Key assumptions (e.g., linear damping, single-mode, log-linear fit)
- Failure modes / raised exceptions if meaningful

Docstring style: Google or NumPy style are both OK; be consistent within a module.

---

## 5) Domain assumptions + units

Any function dealing with measurements MUST state:

- Time base (seconds)
- Frequency base (Hz)
- Amplitude representation (linear, dB, log amplitude)
- Damping ratio convention (ζ, log decrement, T60 mapping, etc.)

If assumptions differ across methods (e.g., piecewise fit vs box fit), document explicitly.

---

## 6) Error handling

- Prefer **explicit exceptions** with actionable messages.
- Validate inputs early (types, ranges, shapes).
- For CLI/TUI: convert internal exceptions into user-friendly messages.

---

## 7) Project structure conventions

- Keep **pure logic** separate from **UI** (Textual/Tkinter) and from **IO**.
- Prefer a `core/` or `domain/` module for math/physics logic.
- Keep CLI entry points thin: parse args → call core → format output.

---

## 8) Testing conventions

- Use `pytest`.
- Prefer deterministic tests (seed randomness).
- Test:
  - core math correctness
  - edge cases (empty spans, invalid windows, NaNs)
  - regression tests for previously fixed bugs

For UI (Textual):

- Prefer **Pilot** tests first, then snapshots/visual inspection, then asciinema review as last resort.

---

## 9) Agent instructions (how agents should edit code)

Agents MUST:

1. Preserve existing public APIs unless explicitly instructed.
2. Add docstrings **only** where required by this policy.
3. Keep changes small and localized.
4. Add or update tests for any behavior change.
5. Avoid refactors that change behavior “for style”.

##
