# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FiQCI EMS (Error Mitigation Service) is a Python library that wraps IQM quantum backends (via `iqm-client[qiskit]`) and applies quantum error mitigation transparently. Users pick a `mitigation_level` (0–3) and get improved results through standard Qiskit interfaces. Requires Python 3.11 or 3.12. Package name `fiqci-ems`; importable as `fiqci.ems`. Version is derived from git tags via `setuptools_scm`.

## Commands

All Python work goes through `uv` (see global instructions).

```bash
uv sync                              # install with dev deps
uv sync --group docs                 # add docs deps
uv run pytest                        # run tests
uv run pytest --cov                  # tests with coverage
uv run pytest tests/test_rem.py      # single test file
uv run pytest tests/test_rem.py::test_name   # single test
uv run ruff check --fix              # lint + autofix
uv run ruff format                   # format
uv run pyrefly check                 # type check (config in pyproject [tool.pyrefly])
uv run sphinx-build docs/ docs/_build  # build docs
```

Formatting note: ruff uses **tab indentation** (`indent-style = "tab"`), `line-length = 120`, and `skip-magic-trailing-comma`. Source files use tabs — match that. `pre-commit` is configured (ruff, pyupgrade `--py311-plus`, whitespace hooks).

Tests use mocked backends and `qiskit-aer` (`AerSimulator`) as a stand-in for real hardware — no network/hardware access needed. `pyrefly` only type-checks `src/`; tests and notebooks are excluded.

## Architecture

Three public interfaces, all sharing the same `mitigation_level` / `calibration_shots` / `calibration_file` constructor args and exported from `fiqci.ems` (`src/fiqci/ems/__init__.py`):

- **`FiQCIBackend`** (`fiqci_backend.py`) — the core engine. Wraps an `IQMBackendBase`, delegates unknown attributes to it via `__getattr__`, and is what the other two interfaces build on. Sampler-style mitigation levels.
- **`FiQCISampler`** (`primitives/fiqci_sampler.py`) — thin wrapper over `FiQCIBackend` returning mitigated measurement counts. Levels: 0 none, 1 REM (M3), 2 +DD, 3 +Pauli twirling.
- **`FiQCIEstimator`** (`primitives/fiqci_estimator.py`) — computes expectation values of `SparsePauliOp` observables. Levels: 0 none, 1 REM, 2 +DD, 3 +ZNE (the estimator builds a `FiQCIBackend` at level 2 and adds ZNE on top). Note level 3 differs from Sampler/Backend (ZNE instead of Pauli twirling).

### Mitigation techniques (composable, configured per-run)

- **REM** — `mitigators/rem.py`: `M3IQM` subclasses mthree's `M3Mitigation`, adapted for IQM (balanced calibration default, IQM `calibration_set_id` handling, custom cals file format with validation). Calibration is kicked off **non-blocking** (`async_cal`) so it runs in parallel with circuit jobs.
- **DD (Dynamical Decoupling)** — `mitigators/dd.py`: builds IQM `CircuitCompilationOptions` from `(threshold_length, sequence, strategy)` tuples; applied at submission via `circuit_compilation_options` kwarg.
- **ZNE (Zero-Noise Extrapolation)** — `mitigators/zne.py` (extrapolation fns) + `transpiler_passes/zne_circuits.py` (local/global gate folding). Estimator-only.
- **Pauli twirling** — `transpiler_passes/pauli_twirl.py`: expands each circuit into `num_twirls + 1` twirled variants; counts are averaged back per group.
- **Basis measurement** — `transpiler_passes/basis_measurement.py`: `ModifyMeasurementBasis` transpiler pass + helpers that turn observables into measurement-basis subcircuits (X/Y/Z), used by the estimator to compute expectation values.

### Lazy execution model (important)

`run()` does **not** block. It submits circuits and returns a lazy handle immediately; mitigation and result combination are deferred to the first `result()` call (via a `post_process` closure that snapshots settings at submission time, so later setting mutations don't corrupt in-flight results).

- `BatchedJob` — splits the (post-twirl/post-ZNE) flat circuit list into batches of `max_batch_size` (default 100), one backend job each. `result()` waits for all batches, concatenates their `results` in submission order so combined `get_counts(idx)` matches the original circuit index, then runs `post_process`. Exposes `job_ids()`, `status()`/`statuses()`, `done()`, `partial_results()` immediately.
- `MitigatedJob` — wraps a `BatchedJob` for level ≥1; marks that mitigation was configured.
- `FiQCIEstimatorJob` — wraps the job and a deferred `_compute` fn; `expectation_values()` / `raw_expectation_values()` compute lazily and cache.
- **Non-atomic submission**: if the backend rejects a batch mid-stream, `run()` does not throw — it logs a warning, stops submitting, and returns a handle covering every intended batch. The rejected batch reports `ERROR`, skipped ones `CANCELLED` (via `_UnsubmittedBatch` placeholders), and `result()` raises `BatchFailedError` identifying the failing batch.

Raw (pre-mitigation) counts are available via `FiQCIBackend.raw_counts` but only **after** `result()` has been called (post-processing is lazy). Mitigated results carry `fiqci_ems` metadata in each result header.

### Count-key handling subtlety

M3 expects spaceless bitstrings with one bit per measured qubit, but Qiskit result keys include all classical bits, register spaces, and zero-filled unmeasured bits. `FiQCIBackend._key_layout` / `_reduce_counts` / `_expand_counts` strip these down for M3 correction and restore the original key structure afterward. With twirling, `_trim_result_to_groups` keeps the representative (original-circuit) entry per group at the correct stride so result headers stay aligned.
