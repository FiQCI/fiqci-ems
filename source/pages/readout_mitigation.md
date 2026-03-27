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

**References:**
- Nation, P., Kang, H., Sundaresen N., Gambetta J., "Scalable Mitigation of Measurement Errors on Quantum Computers" PRX Quantum 2, 040326 (2021). https://doi.org/10.1103/PRXQuantum.2.040326
- https://github.com/Qiskit/qiskit-addon-mthree
