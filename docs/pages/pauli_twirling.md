# Pauli Twirling with FiQCI EMS

## What is Pauli Twirling

Pauli Twirling is an error mitigation (or tailoring) technique that aims to transform gate noise towards a stochastic Pauli channel which can be more efficiently mitigated by other methods. Pauli twirling generates multiple variants of a circuit by sandwchiching gates in between random sets of Pauli gates. These random Pauli sets are chosen so that the action of the sandwhiched gate is equal to just the original gate. The results of all the variant circuits can then be averaged to get the mitigated/tailored results.

The steps are then:

1. **Generate variant circuits** by sandwhiching two-qubit gates between Pauli channels that cancel out.
2. **Execute** all variant circuits
3. **Average** the results from execution

## Usage

Using the {class}`~fiqci.ems.FiQCISampler` or {class}`~fiqci.ems.FiQCIBackend` Pauli Twirling can be enabled by setting `mitigation_level=3`. With {class}`~fiqci.ems.FiQCIEstimator` it needs to be manually configured with  {meth}`~fiqci.ems.FiQCIEstimator.pauli_twirl`.

```python
from fiqci.ems import FiQCISampler

sampler = FiQCIEstimator(backend=backend, mitigation_level=3)
```

### Manual Configuration

For fine-grained control, enable Pauli Twirling explicitly via the {meth}`~fiqci.ems.FiQCIEstimator.pauli_twirl` method of the primitive:

```python
from qiskit.circuit.library import CZGate

estimator = FiQCIEstimator(backend=backend, mitigation_level=1)

estimator.pauli_twirl(
    enabled=True,
    num_twirls=5,
    gates_to_twirl=[CZGate()]
)
```