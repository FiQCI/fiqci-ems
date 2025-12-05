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
