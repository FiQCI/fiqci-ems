# Readout Error Mitigation with FiQCI EMS


## What is Readout Error Mitigation?

Readout errors occur when quantum measurements incorrectly identify the state of a qubit. For example, a qubit prepared in state |0⟩ might be measured as |1⟩, and vice versa. These errors are typically characterized by:

- **P(0|0)**: Probability of correctly measuring 0 when qubit is in state |0⟩
- **P(1|0)**: Probability of incorrectly measuring 1 when qubit is in state |0⟩
- **P(0|1)**: Probability of incorrectly measuring 0 when qubit is in state |1⟩
- **P(1|1)**: Probability of correctly measuring 1 when qubit is in state |1⟩

## The M3 (Matrix-free Measurement Mitigation) Method

M3 is a readout error mitigation technique that:

1. **Calibrates** by running circuits that prepare known computational basis states
2. **Characterizes** the confusion matrix describing measurement errors
3. **Corrects** measured distributions by inverting the error model

Unlike traditional methods that explicitly compute and invert large matrices, M3 uses tensor network methods to efficiently handle multi-qubit systems, making it scalable to larger quantum computers.

**Key advantages of M3:**
- Scales efficiently to many qubits
- Handles correlated and uncorrelated readout errors
- Provides quasi-probability distributions (can have negative values)
- Can convert to nearest valid probability distribution

## Quasi-probabilities, counts and error bars

M3's correction produces a **quasi-probability** distribution, whose entries may be negative. That is
an unbiased estimate but not a physical distribution, so EMS uses it two different ways depending on
what you asked for:

- **{class}`~fiqci.ems.primitives.fiqci_sampler.FiQCISampler` counts** are the quasi-probabilities
  projected onto the nearest physical distribution (negative entries clipped to zero) and scaled back
  to the shot count, because counts have to be non-negative integers for the rest of Qiskit.
- **{class}`~fiqci.ems.primitives.fiqci_estimator.FiQCIEstimator` expectation values** are computed
  from the *unprojected* quasi-probabilities. Clipping removes exactly the low-probability outcomes an
  expectation value is most sensitive to, which biases ⟨P⟩ towards ±1 and lands it exactly on ±1
  whenever the clipped outcomes are the only ones contributing the opposite sign, as for a two-qubit
  ZZ observable. A mitigated expectation value can therefore fall slightly outside `[-1, 1]`; that is
  the honest estimate for an over-subtracting calibration, and a value far outside means the readout
  calibration is too noisy (raise `calibration_shots`).

Mitigation trades bias for variance, so `standard_errors()` reports the shot noise of the **raw**
counts inflated by `sqrt(mitigation_overhead)`, mthree's own error bound. A mitigated value's error
bar is therefore always larger than the unmitigated one it came from.

Both quantities, and how many outcomes the projection clipped, travel in each result's header:

```python
result = FiQCISampler(backend, mitigation_level=1).run(circuit, shots=4096).result()
metadata = result.results[0].header["fiqci_ems"]

metadata["quasi_probabilities"]   # unprojected M3 output, keyed like the counts
metadata["mitigation_overhead"]   # M3's variance amplification factor (>= 1)
metadata["clipped_outcomes"]      # outcomes the projection zeroed
metadata["raw_counts"]            # pre-mitigation counts
```

A warning is raised when the projection had to clip anything, since the counts from those circuits are
biased towards the physical boundary.

## Advanced: Direct M3IQM Control

For fine-grained control over the mitigation process, you can use the {class}`~fiqci.ems.mitigators.rem.M3IQM` class directly. This allows you to:

- Choose calibration strategies (`"balanced"`, `"independent"`, `"marginal"`)
- Inspect per-qubit calibration matrices
- Apply correction manually and access quasi-probability distributions
- Calculate expectation values and standard deviations from quasi-distributions

See the [Advanced Readout Error Mitigation example](../notebooks/advanced_readout_error_mitigation_m3) for a full walkthrough.

## Examples

- [Advanced Readout Error Mitigation](../notebooks/advanced_readout_error_mitigation_m3). Includes direct {class}`~fiqci.ems.mitigators.rem.M3IQM` usage for fine-grained control over calibration and correction.

## References:
- Nation, P., Kang, H., Sundaresen N., Gambetta J., "Scalable Mitigation of Measurement Errors on Quantum Computers" PRX Quantum 2, 040326 (2021). [https://doi.org/10.1103/PRXQuantum.2.040326](https://doi.org/10.1103/PRXQuantum.2.040326)
- [https://github.com/Qiskit/qiskit-addon-mthree](https://github.com/Qiskit/qiskit-addon-mthree)
