# Contributing

## Dev setup

```bash
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on Linux/macOS
pip install -e ".[dev,ocr,dashboard]"
```

## Running tests

```bash
pytest tests/ -v
```

## Test data policy

**Never add real patient data, or real "de-identified" exports from another
source, to this repository or to any test fixture.** All fixtures must be
synthetically generated (see `tests/conftest.py`). This applies to CI
artifacts, example notebooks, and issue attachments alike — if you're
reporting a bug found on real data, describe or synthesize a reproduction
instead of attaching the original file.

## Code style

- `ruff check .` must pass.
- Every `Transform` that modifies a `Sample` must append an `AuditEntry`
  rather than mutating silently.
- New detection logic must document, in its docstring, which failure mode it
  biases toward under uncertainty (false negative vs. false positive) — see
  `BurnedInTextRedact` for the pattern.
