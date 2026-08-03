## [WIP] [1.0.0] 31.7.2026

### Breaking changes

Importing from `fiqci.ems` is unaffected — `FiQCIBackend`, `FiQCISampler`, `FiQCIEstimator` and `BatchFailedError` are still exported from there. The rest only affects code reaching into submodules or mutating settings dicts directly.

- the `fiqci.ems.fiqci_backend` module was split into the `fiqci.ems.backend` package and no longer exists. `BatchedJob`, `MitigatedJob`, `PartialBatch` and `BatchFailedError` now come from `fiqci.ems.backend`
- `get_obs_subcircuits(subcircuits, measurement_settings, ops)` now takes the observable instead of the measurement settings: `get_obs_subcircuits(subcircuits, observable, ops)`, and derives the settings itself
- `_combine_pauli_ops()` is renamed to the now-public `get_measurement_settings()`
- `mitigator_options` returns a copy, so mutating it no longer reconfigures a run. Use `rem()` / `dd()` / `pauli_twirl()` / `zne()`, which validate their input
- ZNE extrapolation now fits against the **achieved** scale factors rather than the requested ones, so expectation values can differ for any scale factor folding cannot reach exactly (anything other than odd integers). Inspect them with `job.achieved_scale_factors()`

See the [docs](https://fiqci.fi/fiqci-ems/docs/) for the current interfaces and behaviour.

### Changed
- added a `standard_errors()` method to `FiQCIEstimator` that lazily calculates and returns shot noise and ZNE fit errors
- https://github.com/FiQCI/fiqci-ems/pull/41

---

- allow passing a callable `extrapolation_fn(scales, values) -> float` to `.zne()` in addition to using the built-in methods
  - a callable that accepts a `sigmas` keyword argument is given the per-scale shot standard errors and may return `(values, standard_errors)`, which are reported via `standard_errors()` like the built-in extrapolators' propagated errors
  - fixes `expectation_values()` raising `UnboundLocalError` when a callable extrapolation method was used
- https://github.com/FiQCI/fiqci-ems/pull/40

---

- migrate `combine_pauli_ops` to use Qiskit's `Paulilist.group_qubit_wise_commuting`
- `get_obs_subcircuits` now directly takes observables as an argument and handles calling `get_measurement_settings` inside the function
- https://github.com/FiQCI/fiqci-ems/pull/38

---

- refactor monolithic `fiqci_backend` into a `backend` module
  - split into `core.py`, `counts.py`, and `jobs.py`
- https://github.com/FiQCI/fiqci-ems/pull/37

---

- Zero noise extrapolation now supports any >=1 floats as scale factors
  - If exact given value cannot be reached, and approximation is used (fractional folding). If exact values is not achieved log a warning
- `zne()` now accepts either a list of floats or a list of list of floats
  - this allows the user to pass circuit specific scale factors
- fix bug in folding method handling
- expose `requested_scale_factors()` and `achieved_scale_factors` in `FiQCIEstimatorJob`
  - allow the user to view per job used scale factors even if estimator config changed
- expose `mitigator_options` on the returned job handles (`BatchedJob`/`MitigatedJob`, and `FiQCIEstimatorJob`)
  - a frozen snapshot of the mitigation settings (`mitigation_level`, `rem`, `dd`, `pauli_twirl`, plus `zne` on the estimator job) in effect at submission, so the user can inspect what a run actually used even after the backend/estimator config is mutated
  - unlike the live, mutable `mitigator_options` property on the backend/estimator, the job's snapshot never changes; the live M3 mitigator object is omitted from the REM entry
- https://github.com/FiQCI/fiqci-ems/pull/39

---

- warn at `run()` when discrete folding collapses distinct ZNE scale factors onto the same achieved value
- `mitigator_options` on `FiQCIBackend`, `FiQCISampler` and `FiQCIEstimator` now returns a copy
- a custom `extrapolation_method` may now return `(values, None)` to report values without standard errors
- `_calculate_expectation_values` no longer raises `ZeroDivisionError` for a measurement circuit with no recorded shots. It reports `0.0`, matching `_calculate_shot_errors`
- ship a PEP 561 `py.typed` marker so the package's type annotations are visible to downstream type checkers
- add a `seed` argument to `pauli_twirl()` for reproducible twirling, and fix `gates_to_twirl` being silently ignored when passed as a generator
- fix `total_circuits_generated()` under-reporting when the observables differ per circuit, and raise `ValueError` if a list of observables or per-circuit `scale_factors` does not have one entry per base circuit
- `FiQCIEstimator` now raises `ValueError` instead of `NotImplementedError` for an unsupported `mitigation_level`, matching `FiQCIBackend` and `FiQCISampler`
- `FiQCIBackend.run()`'s default `shots` is now 2048 instead of 1024, matching both primitives
- a malformed calibration file now raises `M3Error` naming the file and the problem instead of a bare `KeyError` or `JSONDecodeError`
- https://github.com/FiQCI/fiqci-ems/pull/42

---

- fix `FiQCIEstimator` silently returning `1.0` for every observable when the input circuit has classical bits not consumed by a final measurement (e.g. `QuantumCircuit(n, m)` without measurements, or a mid-circuit measurement)
- mid-circuit measurements are no longer overwritten, and a circuit with an existing `meas` register no longer raises `DAGCircuitError`
- https://github.com/FiQCI/fiqci-ems/pull/43

---

- ZNE now folds the circuit before the measurement-basis rotations are appended, so every measurement group of a pair shares one folded core and one achieved scale factor
- fixes `folding_method="global"` folding X/Y groups by a different amount than the achieved scale factor used for extrapolation
- https://github.com/FiQCI/fiqci-ems/pull/44

## [0.8.1] 12.6.2026

### Changed
- fix zero noise extrapolation accepting invalid scale factors
- fix handling of `extrapolation_degree` when manually setting ZNE options using `.zne()`
- fix pauli twirling crash on non IQM backends
- fix folding of non self inverse gates when using zne

## [0.8.0] 5.6.2026

### Changed
- `run()` on the backend, sampler, and estimator now returns a lazy job handle immediately instead of blocking until all batches complete and mitigation is applied. Error mitigation, twirl averaging, batch-result combination, and estimator expectation values are computed on the first `result()`/`expectation_values()` call and cached.
- The handle exposes per-batch `job_id()`s immediately, an aggregated `status()`/`done()`, `job_ids()`, and `partial_results()` for batch-granular access while other batches are still running.
- `result()` now raises `BatchFailedError` naming the failing batch and the original circuit indices it covered when any batch fails, instead of an opaque error during result combination.
- If the backend rejects a batch partway through submission, `run()` no longer throws and no longer loses the already-submitted jobs. It logs a warning, stops submitting further batches, and returns a handle covering every intended batch: submitted batches keep their job ids and status, the rejected batch reports `ERROR`, and batches skipped afterwards report `CANCELLED` (with `job_id()` `None`). Use `statuses()`/`status()`/`partial_results()` to inspect the outcome; `result()` still raises `BatchFailedError` because a complete combined result cannot be formed. `BatchFailedError` is exported from `fiqci.ems`.
- A single batch is now always wrapped in `BatchedJob` (previously the underlying job was returned as-is) so the polling/partial-result API is uniform.
- `backend.raw_counts` is now populated only after the run's `result()` has been retrieved (post-processing is lazy).
- Mitigation settings (ZNE configuration, the M3 mitigator, mitigation level, calibration shots) are snapshotted at submission time, so deferred post-processing stays consistent with the configuration used at `run()` even if settings are changed before results are accessed.
- Internal: `MitigatedJob(handle)` and `FiQCIEstimatorJob(job, compute_fn, observables)` constructor signatures changed (these classes are not part of the public API).

## [0.7.2] - 4.6.2026

### Changed
- Fix `max_batch_size` not being passed down from ems primitives to mthree for calibration job execution.

## [0.7.1] - 3.6.2026

## Changed
- Fix a rare divide by zero in `exponential_extrapolation()` causing `nan` expectation values.

[https://github.com/FiQCI/fiqci-ems/pull/16](https://github.com/FiQCI/fiqci-ems/pull/16)


## [0.7.0] - 29.5.2026

### Changed
- Add support for applying readout error mitigation on circuits with measurements on multiple classical registers
- Pauli twirling no longer drops the transpiled circuit's `TranspileLayout`; twirled circuits previously lost their layout and could place CZ gates on non-adjacent physical qubits, raising `CircuitValidationError` (more likely with higher `num_twirls`)
- Fixed result ordering when Pauli twirling is enabled: `result.get_counts()` for a list of circuits is now correctly aligned with the input circuit order (previously a mis-trim could put another circuit's counts in circuit `i`'s slot)

[https://github.com/FiQCI/fiqci-ems/pull/14](https://github.com/FiQCI/fiqci-ems/pull/14)

## [0.6.1] - 8.5.2026

### Changed
- Fix for a bug in expectation value calculation

[https://github.com/FiQCI/fiqci-ems/pull/13](https://github.com/FiQCI/fiqci-ems/pull/13)

## [0.6.0] - 4.5.2026

### Added
- add an optional `max_batch_size` argument to primitives `.run()` methods.
    - `FiQCIBackend` now batches passed circuits into max `max_batch_size` sets before execution
    - By default value is `100`
    - Leads to more efficient use of QPU resources and less wait time in queue for user
- add `BatchedJob` class:
    - Wrapper to combine multiple jobs into a single result for easy use elsewhere
- add `total_circuits_generated()` method to all primitives
  - prints out the number of circuits to be generated using the active mitigation options
- add some useful logging for batching and number of circuits being generated

### Changed
- `FiQCIEstimator` now always receives a single result from `FiQCIBackend` so `FiQCIEstimatorJobCollection` only takes one job as argument -> rename to `FiQCIEstimatorJob`
- Raw counts from individual twirled circuits now included in the raw_results(). Also returned from the per circuit Result headers.
- Update `IQMClient` minimum version to 34.0.2

[https://github.com/FiQCI/fiqci-ems/pull/12](https://github.com/FiQCI/fiqci-ems/pull/12)


## [0.5.0] - 28.4.2026

### Added
- Pauli Twirling for for `FiQCISampler` and `FiQCIBackend` as `mitigation_level=3` (REM + DD + Pauli Twirling). Can be manually enabled for `FiQCIEstimator`
- Manual configuration using the `pauli_twirl()` method of the primitive
  - `num_twirls`: number of twirled variant to generate per circuit
  - `gates_to_twirl`: gates to be twirled, by default all two qubit gates
- `transpiler_passes/pauli_twirl.py`: `get_twirled_circuits()` for generating twirled circuits

### Changed
- Bugfix: ZNE expectation values in FiQCIEstimator now correctly scoped per observable group

[https://github.com/FiQCI/fiqci-ems/pull/11](https://github.com/FiQCI/fiqci-ems/pull/11)

## [0.4.0] - 3.4.2026

### Added
- Dynamical decoupling (DD) for `FiQCISampler`, `FiQCIEstimator`, and `FiQCIBackend` as `mitigation_level=2` (REM + DD)
- Manual configuration using the `.dd()` method of the primitive
    - `gate_sequences`: list of DD sequences to apply
- `mitigators/dd.py`: `build_dd_options()` for getting options for DD

[https://github.com/FiQCI/fiqci-ems/pull/10](https://github.com/FiQCI/fiqci-ems/pull/10)


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
