from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister, Gate
from qiskit.circuit.library import CZGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import Operator, pauli_basis
 
import numpy as np

from copy import deepcopy

from typing import Iterable, Optional


class ZNECircuits(TransformationPass):
    """A pass to generate circuits for zero-noise extrapolation (ZNE) by folding gates."""

    def __init__(
        self,
        fold_gates: Optional[Iterable[str]] = None,
        scale_factor: int = None,
    ):
        """
        Initialize the ZNECircuits pass.

        Args:
            fold_gates: An optional iterable of gate names to fold. If None, all gates will be folded.
        """
        super().__init__()
        self.fold_gates = set(fold_gates) if fold_gates is not None else None
        self.scale_factor = scale_factor

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the ZNECircuits pass on the given DAGCircuit.

        Args:
            dag: The input DAGCircuit to transform.

        Returns:
            A new DAGCircuit with folded gates for ZNE.
        """
        cloned_dag = deepcopy(dag)

        for node in cloned_dag.op_nodes():
            if node.num_qubits != 2 or node.op.name == "barrier":
                continue  # Skip gates with no qubits (e.g., barriers)
            if self.fold_gates is None or node.name in self.fold_gates:
                if self.scale_factor == 1:
                    continue  # Skip the original circuit
                
                mini_dag = DAGCircuit()
                register = QuantumRegister(2)
                mini_dag.add_qreg(register)

                for _ in range(self.scale_factor):
                    mini_dag.apply_operation_back(node.op, [register[0], register[1]])
                
                cloned_dag.substitute_node_with_dag(node, mini_dag)


        return cloned_dag
    
def exponential_extrapolation(expectation_values, scale_factors):
    """
    Perform exponential extrapolation to estimate the zero-noise value.

    Args:
        expectation_values: A list of expectation values corresponding to different noise levels.

    Returns:
        The extrapolated zero-noise expectation value.
    """
    if len(expectation_values) < 2:
        raise ValueError("At least two expectation values are required for exponential extrapolation.")
    
    x = np.array(scale_factors)  # Noise scale factors
    y = np.array(expectation_values)

    negative_index = []
    for i, val in enumerate(y[0]):
        if val < 0:
            negative_index.append(i)

    y = np.abs(y)

    # Fit an exponential curve to the data points
    coeffs = np.polyfit(x, np.log(y), 1)
    a, b = coeffs

    # Extrapolate to zero noise (x=0)
    zero_noise_value = np.exp(b)

    zero_noise_value = [-v if i in negative_index else v for i, v in enumerate(zero_noise_value)]

    return zero_noise_value

def richardson_extrapolation(expectation_values, scales, degree=None):
    """
    Polynomial (Richardson) extrapolation to estimate the zero-noise value.

    Args:
        expectation_values: Array-like of shape (n_scales, n_obs) or (n_scales,)
        scales: Noise scale factors used (e.g., [1, 3, 5])
        degree: Optional polynomial degree; defaults to min(n_scales-1, 2)

    Returns:
        Zero-noise estimate(s) per observable.
    """
    import numpy as np

    y = np.asarray(expectation_values, dtype=float)
    x = np.asarray(scales, dtype=float)

    if y.ndim == 1:
        y = y[:, None]

    if len(x) != y.shape[0]:
        raise ValueError("Length mismatch between scales and expectation_values.")

    deg = degree if degree is not None else min(y.shape[0] - 1, 2)
    out = np.empty(y.shape[1])

    for j in range(y.shape[1]):
        mask = np.isfinite(y[:, j])
        if mask.sum() < 2:
            out[j] = np.nan
            continue
        coeffs = np.polyfit(x[mask], y[mask, j], deg)
        out[j] = np.polyval(coeffs, 0.0)

    return out if out.size > 1 else out[0]
