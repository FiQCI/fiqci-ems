# FiQCI EMS

FiQCI Error Mitigation Service (EMS) is a Python library for quantum error mitigation as part of the [Finnish Quantum Computing Infrastructure (FiQCI)](https://fiqci.fi). It wraps IQM quantum backends and applies error mitigation transparently, allowing users to run circuits with improved accuracy by specifying a mitigation level.

This python package can be pre-installed on a HPC system or installed by the user. The main goal of the project is to allow users using FiQCI quantum computers to easily add flags to run error mitigated quantum jobs. 

## Mitigation Levels

| Level | Mitigation Applied | Technique |
|:-----:|:------------------:|:---------:|
| 0 | None | Raw results |
| 1 | Readout Error Mitigation | M3 (matrix-free measurement mitigation) |
| 2 | Level 1 + additional | TBD |
| 3 | Level 2 + additional | TBD |

The default is level 1, which applies M3 readout error mitigation.

## Installation

[Uv](https://docs.astral.sh/uv/getting-started/installation/) is recommended for installation

```bash
uv pip install fiqci-ems
#or
uv add fiqci-ems
```

Requires Python 3.11 or 3.12.

## Usage

Start by initialising your IQM backend.

```python
from iqm.qiskit_iqm import IQMProvider

provider = IQMProvider()
backend = provider.get_backend()
```

EMS provides three interfaces depending on your use case.

### FiQCISampler - sampling interface

For users who need measurement counts with built-in mitigation:

```python
from fiqci.ems import FiQCISampler

# Using mitigation_level
sampler = FiQCISampler(backend, mitigation_level=1)
job = sampler.run(circuits, shots=2048)
result = job.result()

# Or manually set mitigation options
sampler.rem(enabled=True, calibration_shots=2000, calibration_file="cals.json")

# See applied and available options
sampler.mitigation_options()
```

### FiQCIEstimator - expectation values

Computes expectation values of Pauli observables directly from circuits:

```python
from fiqci.ems import FiQCIEstimator
from qiskit.quantum_info import SparsePauliOp

# Using mitigation_level
estimator = FiQCIEstimator(backend, mitigation_level=1)
job_collection = estimator.run(circuits, observables=[SparsePauliOp("ZZ")])
evs = job_collection.expectation_values()

# Or manually set mitigation options
estimator.rem(enabled=True, calibration_shots=2000, calibration_file="cals.json")

# See applied and available options
estimator.mitigation_options()
```

### FiQCIBackend - drop-in backend replacement

FiQCIBackend is used under the hood by both sampler and estimator. Wraps any IQM backend and applies error mitigation to `run()` calls:

```python
from fiqci.ems import FiQCIBackend

# Using mitigation_level
backend = FiQCIBackend(backend, mitigation_level=1)
job = backend.run(circuit, shots=1024)
result = job.result()

# Or manually set mitigation options
backend.rem(enabled=True, calibration_shots=2000, calibration_file="cals.json")

# See applied and available options
backend.mitigation_options()
```

Access raw (pre-mitigation) counts via `backend.raw_counts`.

### Advanced Usage

It is also possible to manually configure and directly use the M3 mitigator without the wrapper classes above. Consult the docs for how this is done.

## Configuration

All three interfaces accept the same core options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mitigation_level` | `1` | Mitigation level (0-3) |
| `calibration_shots` | `1000` | Shots used for M3 calibration circuits |
| `calibration_file` | `None` | Path to save/load calibration data (JSON) |

Mitigation can also be configured directly. See the docs for `FiQCISampler`, `FiQCIEstimator`, and `FiQCIBackend` to see all available options.

## Documentation

Full documentation including API reference, guides, and Jupyter notebook examples is available at [docs](link)
## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Lint and format
uv run ruff check --fix
uv run ruff format

# Type check
uv run pyrefly check 
```

## Building docs
```bash
#Install docs dependencies
uv sync --group docs

#Build docs
uv run sphinx-build docs/ docs/_build
```

## License

Apache 2.0, see [LICENSE](LICENSE) for details.

## Having trouble?

Contact [servicedesk@csc.fi](mailto:servicedesk@csc.fi) or raise an issue here.
