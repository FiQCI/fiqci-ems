from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import PassManager

from copy import deepcopy

from typing import Iterable, Optional


class ZNECircuits(TransformationPass):
	"""A pass to generate circuits for zero-noise extrapolation (ZNE) by folding gates."""

	def __init__(
		self, fold_gates: Optional[Iterable[str]] = None, scale_factor: int = 1, folding_method: str = "local"
	):
		"""
		Initialize the ZNECircuits pass.

		Args:
		    fold_gates: An optional iterable of gate names to fold. If None, all gates will be folded.
		    scale_factor: The factor by which to scale the circuit.
		    folding_method: The method to use for folding gates ("local" or "global").
		"""
		super().__init__()
		self.fold_gates = set(fold_gates) if fold_gates is not None else None
		self.scale_factor = scale_factor
		self.folding_method = folding_method

	def run(self, dag: DAGCircuit) -> DAGCircuit:
		"""
		Run the ZNECircuits pass on the given DAGCircuit.

		Args:
		    dag: The input DAGCircuit to transform.

		Returns:
		    A new DAGCircuit with folded gates for ZNE.
		"""
		cloned_dag = deepcopy(dag)

		if self.folding_method == "local":
			for node in cloned_dag.op_nodes():
				if node.num_qubits != 2 or node.op.name == "barrier":
					continue  # Skip gates with no qubits (e.g., barriers)
				if self.fold_gates is None or node.name in self.fold_gates:
					if self.scale_factor == 1:
						continue  # Skip the original circuit

					mini_dag = DAGCircuit()
					register = QuantumRegister(2)
					mini_dag.add_qreg(register)

					for ind in range(self.scale_factor):
						if ind % 2 == 0:
							mini_dag.apply_operation_back(node.op, [register[0], register[1]])
						else:
							mini_dag.apply_operation_back(node.op.inverse(), [register[0], register[1]])

					cloned_dag.substitute_node_with_dag(node, mini_dag)

		elif self.folding_method == "global":
			original_dag = deepcopy(cloned_dag)
			original_no_meas = deepcopy(cloned_dag)
			original_no_meas.remove_all_ops_named("measure")
			adjoint_dag = circuit_to_dag(dag_to_circuit(original_no_meas).inverse())

			cloned_dag.remove_all_ops_named("measure")

			for i in range(self.scale_factor - 1):
				is_last = i == self.scale_factor - 2
				if i % 2 == 0:
					cloned_dag.compose(adjoint_dag)
				else:
					cloned_dag.compose(original_dag if is_last else original_no_meas)

		return cloned_dag


def _get_zne_circuits(
	circuits: list[QuantumCircuit],  # list of QuantumCircuits to generate ZNE circuits from
	fold_gates: Optional[
		Iterable[str]
	] = None,  # list of gate names to fold, if None, all gates two qubit gates will be folded
	scale_factors: Optional[Iterable[int]] = [1, 3, 5],  # list of atleast two odd ints
	folding_method: str = "local",  # "local" or "global"
) -> list[QuantumCircuit]:
	"""Generate ZNE circuits by folding gates in the input QuantumCircuit.

	Args:
	    circuits: The input QuantumCircuit to transform.
	    fold_gates: An optional iterable of gate names to fold. If None, all gates will be folded.
	    scale_factors: An optional iterable of odd integer scale factors for folding. If None, defaults to [1, 3, 5].
		folding_method: The method to use for folding gates ("local" or "global").
	Returns:
	    A list of QuantumCircuits with folded gates for ZNE.
	"""
	zne_circuits = []
	if scale_factors is None:
		scale_factors = [1, 3, 5]

	if not all(isinstance(s, int) and s > 0 and s % 2 == 1 for s in scale_factors):
		raise ValueError("Scale factors must be positive odd integers.")

	for scale in scale_factors:
		for circuit in circuits:
			pm = PassManager(ZNECircuits(fold_gates=fold_gates, scale_factor=scale, folding_method=folding_method))

			zne_circuit = pm.run(circuit)

			zne_circuits.append(zne_circuit)

	return zne_circuits
