from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister, Gate, StandardEquivalenceLibrary
from qiskit.circuit.library import CZGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import BasisTranslator, Decompose
from qiskit.quantum_info import Operator, pauli_basis

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase

import numpy as np

from typing import Iterable, Optional

# Module-level cache: gate name -> list of (pauli_left, pauli_right) pairs.
# Computed once per gate type and reused across all PauliTwirl instances.
_twirl_set_cache: dict[str, list] = {}


def _get_twirl_set(gate: Gate) -> list:
	"""Get or compute the twirl pair set for a gate, using the module-level cache."""
	if gate.name not in _twirl_set_cache:
		twirl_list = []
		for pauli_left in pauli_basis(2):
			for pauli_right in pauli_basis(2):
				if (Operator(pauli_left) @ Operator(gate)).equiv(Operator(gate) @ pauli_right):
					twirl_list.append((pauli_left, pauli_right))
		_twirl_set_cache[gate.name] = twirl_list
	return _twirl_set_cache[gate.name]


class PauliTwirl(TransformationPass):
	"""Add Pauli twirls to two-qubit gates."""

	def __init__(self, gates_to_twirl: Optional[Iterable[Gate]] = None, seed=None):
		"""
		Args:
		    gates_to_twirl: Gates to twirl. The default behavior is to twirl all
		        two-qubit basis gates.
		    seed: Seed or numpy Generator for the random twirl selection. Passing an existing
		        Generator shares its state (and is returned unchanged by ``np.random.default_rng``),
		        so a single Generator can drive many twirls reproducibly.
		"""
		if gates_to_twirl is None:
			gates_to_twirl = [CZGate()]
		# Materialised so the gates can be iterated more than once; a generator would be consumed
		# building twirl_set below, leaving nothing to match against in run().
		self.gates_to_twirl = list(gates_to_twirl)
		self.twirl_set = {gate.name: _get_twirl_set(gate) for gate in self.gates_to_twirl}
		self._twirling_gate_classes = tuple(gate.base_class for gate in self.gates_to_twirl)
		self._rng = np.random.default_rng(seed)
		super().__init__()

	def run(self, dag: DAGCircuit) -> DAGCircuit:
		# collect all nodes in DAG and proceed if it is to be twirled
		for node in dag.op_nodes():
			if not isinstance(node.op, self._twirling_gate_classes):
				continue

			# random integer to select Pauli twirl pair
			pauli_index = self._rng.integers(0, len(self.twirl_set[node.op.name]))
			twirl_pair = self.twirl_set[node.op.name][pauli_index]

			# instantiate mini_dag and attach quantum register
			mini_dag = DAGCircuit()
			register = QuantumRegister(2)
			mini_dag.add_qreg(register)

			# apply left Pauli, gate to twirl, and right Pauli to empty mini-DAG
			mini_dag.apply_operation_back(twirl_pair[0].to_instruction(), [register[0], register[1]])
			mini_dag.apply_operation_back(node.op, [register[0], register[1]])
			mini_dag.apply_operation_back(twirl_pair[1].to_instruction(), [register[0], register[1]])

			# substitute gate to twirl node with twirling mini-DAG
			dag.substitute_node_with_dag(node, mini_dag)

		return dag


def get_twirled_circuits(
	circuits: list[QuantumCircuit],
	num_twirls: int,
	gates_to_twirl: Optional[Iterable[Gate]] = None,
	backend: Optional[IQMBackendBase] = None,
	seed=None,
) -> list[QuantumCircuit]:
	"""
	Generate twirled circuits from input circuits.

	For each input circuit, produces the original circuit followed by num_twirls
	twirled copies, giving groups of (num_twirls + 1) circuits in a flat list.

	Args:
	    circuits: List of QuantumCircuits to generate twirled circuits from.
	    num_twirls: Number of twirled circuits to generate per input circuit.
	    gates_to_twirl: Optional list of gates to twirl, if None, all two-qubit basis gates will be twirled.
		backend: The backend to transpile the circuits for.
	    seed: Seed for the random twirl selection. One Generator is shared across every generated
	        circuit, so each twirl differs while the run as a whole is reproducible.
	Returns:
	    Flat list of circuits: [orig_0, twirl_0_1, ..., twirl_0_T, orig_1, twirl_1_1, ..., twirl_1_T, ...].
	"""
	twirled_circuits = []

	# One pass instance for every circuit and twirl, so its Generator advances across them all.
	twirl_pass = PauliTwirl(gates_to_twirl=gates_to_twirl, seed=seed)

	if backend is not None:
		names = {getattr(i[0], "name", None) for i in backend.target.instructions}
		basis_gates = [n for n in names if isinstance(n, str)]
		pm = PassManager(
			[
				twirl_pass,
				Decompose(gates_to_decompose=["pauli"]),
				BasisTranslator(target_basis=basis_gates, equivalence_library=StandardEquivalenceLibrary),
			]
		)
	else:
		pm = PassManager([twirl_pass])

	for circuit in circuits:
		twirled_circuits.append(circuit)
		for _ in range(num_twirls):
			twirled = pm.run(circuit)
			# PassManager.run returns a fresh circuit that drops the input's TranspileLayout.
			# The IQM backend maps virtual qubits to physical ones via circuit.layout, so without
			# it the twirled circuit is validated/run against an identity layout and its CZ gates
			# can land on non-adjacent physical qubits, raising CircuitValidationError. The twirl
			# passes preserve qubit ordering, so the original layout stays valid; reattach it.
			twirled._layout = circuit._layout
			twirled_circuits.append(twirled)

	return twirled_circuits
