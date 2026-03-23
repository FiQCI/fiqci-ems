from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import PassManager

from copy import deepcopy

from typing import Iterable, Optional


class ZNECircuits(TransformationPass):
	"""A pass to generate circuits for zero-noise extrapolation (ZNE) by folding gates."""

	def __init__(self, fold_gates: Optional[Iterable[str]] = None, scale_factor: int = None):
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


def _get_zne_circuits(
	circuits: DAGCircuit, fold_gates: Optional[Iterable[str]] = None, scale_factors: Optional[Iterable[int]] = [1, 3, 5]
) -> list[DAGCircuit]:
	"""Generate ZNE circuits by folding gates in the input DAGCircuit.

	Args:
	    circuits: The input DAGCircuit to transform.
	    fold_gates: An optional iterable of gate names to fold. If None, all gates will be folded.
	    scale_factors: An optional iterable of scale factors for folding. If None, defaults to [1, 3, 5].
	Returns:
	    A list of DAGCircuits with folded gates for ZNE.
	"""
	zne_circuits = []
	for circuit in circuits:
		for scale in scale_factors:
			pm = PassManager(ZNECircuits(fold_gates=fold_gates, scale_factor=scale))
			zne_circuit = pm.run(circuit)
			zne_circuits.append(zne_circuit)

	return zne_circuits
