## [0.3.0] - 30.3.2026

### Added
- Zero Noise Extrapolation (ZNE) for `FiQCIEstimator` as `mitigation_level=3` (readout error mitigation + ZNE)
- `FiQCIEstimator.zne()` method for manual ZNE configuration with the following options:
  - `fold_gates`: list of gate names to fold, or `None` to fold all two-qubit gates
  - `scale_factors`: list of at least two odd integers (default `[1, 3, 5]`)
  - `folding_method`: `"local"` (per-gate) or `"global"` (whole circuit)
  - `extrapolation_method`: `"exponential"`, `"richardson"`, `"linear"`, or `"polynomial"`
  - `extrapolation_degree`: degree for polynomial extrapolation
- `transpiler_passes/zne_circuits.py`: transpiler pass for generating noise-scaled circuits global or local folding
- `mitigators/zne.py`: extrapolation methods for ZNE post-processing

### Changed
- Restructured package layout: mitigation methods moved to `mitigators/`, execution primitives to `primitives/`, circuit modification passes to `transpiler_passes/`

[https://github.com/FiQCI/fiqci-ems/pull/8](https://github.com/FiQCI/fiqci-ems/pull/8)

## [0.2.0] - 24.3.2026

### Added
- `FIQCIEstimator` class: wraps `FIQCIBackend` to calculate observable expectation values from `SparsePauliOp` observables
- `FIQCISampler` class: wraps `FIQCIBackend` for circuit sampling
- Basis measurement utilities for generating multi-basis observable circuits
- Direct control over individual mitigation tools (e.g. `.rem()`) on `FIQCIBackend`, `FIQCISampler`, and `FIQCIEstimator`
- `FiQCIEstimatorJobCollection` for managing multiple jobs from estimator runs
- Circuit utility for removing idle qubits

### Changed
- Refactored REM-related attributes into a single `REMSettings` typed dict
- `FIQCIBackend.run()` now executes based on enabled mitigators rather than `mitigation_level`
- Top-level imports for `FIQCISampler` and `FIQCIEstimator` via `fiqci.ems`

### Note
- All changes are backwards compatible; using `FIQCIBackend` directly still works as before

[https://github.com/FiQCI/fiqci-ems/pull/7](https://github.com/FiQCI/fiqci-ems/pull/7)


## [0.1.1] - 24.02.2026

- Bump to `iqm-client[qiskit]==33.0.5` for new IQM OS version.
- Support python 3.12

## [0.1.0] - 05.12.2025

- Enable publishing to PyPi
- Fix: M3 was incorrectly calculating the Calibration matrices
- Add: FiQCI Backend that allows configurable mitigation levels.

## [0.0.3] - 04.12.2025

- Fix publishing to testPyPI
- Manually trigger the `publish.yml` workflow from `tag_and_release.yml`
  - Trigger workflow with `gh`

## [0.0.2] - 04.12.2025

- Fix CI github action workflow
- Fix publish workflow
- Fix `pyproject.toml` metadata

## [0.0.1] - 04.12.2025

Initial version of FiQCI Error Mitigation service. Features:
- Readout Error Mitigation with Qiskit's M3
