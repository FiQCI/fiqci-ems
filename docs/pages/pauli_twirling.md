# Pauli Twirling with FiQCI EMS

## What is Pauli Twirling

Pauli Twirling is an error mitigation (or tailoring) technique that aims to transform gate noise towards a stochastic Pauli channel which can be more efficiently mitigated by other methods. Pauli twirling generates multiple variants of a circuit by sandwiching gates in between random sets of Pauli gates. These random Pauli sets are chosen so that the action of the sandwiched gate is equal to just the original gate. The results of all the variant circuits can then be averaged to get the mitigated/tailored results.

The steps are then:

1. **Generate variant circuits** by sandwiching two-qubit gates between Pauli channels that cancel out.
2. **Execute** all variant circuits
3. **Average** the results from execution

## Usage

Using the {class}`~fiqci.ems.FiQCISampler` or {class}`~fiqci.ems.FiQCIBackend` Pauli Twirling can be enabled by setting `mitigation_level=3`. With {class}`~fiqci.ems.FiQCIEstimator` it needs to be manually configured with  {meth}`~fiqci.ems.FiQCIEstimator.pauli_twirl`.

```python
from fiqci.ems import FiQCISampler

sampler = FiQCISampler(backend=backend, mitigation_level=3)
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

## Devices with computational resonators

On Deneb-class devices, `transpile_to_IQM` routes two-qubit interactions through a computational
resonator: a MOVE gate transfers a qubit's state into the resonator, several CZ gates act on
`(partner, resonator)` pairs, and a second MOVE brings the state back. Twirling those CZ gates is
not obviously possible, because twirling wraps a gate in single-qubit Paulis and a resonator accepts
no single-qubit gates.

FiQCI EMS twirls them anyway, by moving the resonator half of each twirl pair out of the MOVE
sandwich. The resonator is holding the moved qubit's state, so a Pauli on the resonator is the same
Pauli applied to that qubit before the state was moved in (or after it comes back out). Getting it
there means commuting it past the sandwich's other gates:

- `I` and `Z` commute with CZ and pass through unchanged.
- `X` and `Y` pick up a `Z` on the far qubit of every CZ they cross.

The qubit half of each pair stays where it always was, next to the gate. The resulting circuit is
exactly equivalent to the untwirled one and passes IQM's MOVE-sandwich validation in strict mode,
with no single-qubit gate on the resonator and none on the moved qubit between the MOVE pair.

Two limits are worth knowing:

- **What gets twirled is the sandwich, not the bare CZ.** The Paulis no longer sit adjacent to the
  gate, so the MOVE transfers and the neighbouring CZs lie between a twirl Pauli and the gate it
  belongs to. This still tailors coherent error into stochastic Pauli error, and it tailors the MOVE
  error too, but it is a coarser claim than twirling an isolated CZ.
- **A sandwich holding an operation the Paulis cannot be commuted through is left alone.** Only the
  gates in `gates_to_twirl` are handled; anything else on the resonator wire between the MOVEs
  disqualifies that sandwich, and its gates are skipped rather than guessed at.

MOVE itself has no unitary representation and is dropped from `gates_to_twirl` with a warning if you
pass it. And whenever twirling matches no gates at all, a warning is raised at submission. That
also catches a circuit containing none of the gates being twirled, such as a `cx`-based circuit
under the default `gates_to_twirl=[CZGate()]`.

## References

- J. J. Wallman, J. Emerson, "Noise Tailoring for Scalable Quantum Computation via Randomized Compiling", [https://arxiv.org/pdf/1512.01098](https://arxiv.org/pdf/1512.01098)
- The QEM Zoo, "Pauli Twirling", [https://qemzoo.com/technique.html?id=pauli-twirling](https://qemzoo.com/technique.html?id=pauli-twirling)
