import logging
import warnings

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister, Gate, StandardEquivalenceLibrary
from qiskit.circuit.library import CZGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import BasisTranslator, Decompose
from qiskit.quantum_info import Operator, pauli_basis

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.move_gate import MoveGate

import numpy as np

from typing import Iterable, Optional

logger: logging.Logger = logging.getLogger(__name__)

# MOVE transfers a qubit state to/from a computational resonator. It has no unitary representation
# (``Operator(MoveGate())`` raises), so it can be neither twirled nor translated to the backend's
# Qiskit target basis, which does not list it. It is passed through untouched.
_MOVE_GATE_NAME = MoveGate().name

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

	def __init__(
		self, gates_to_twirl: Optional[Iterable[Gate]] = None, seed=None, skip_wires: Optional[Iterable[int]] = None
	):
		"""
		Args:
		    gates_to_twirl: Gates to twirl. The default behavior is to twirl all
		        two-qubit basis gates.
		    seed: Seed or numpy Generator for the random twirl selection. Passing an existing
		        Generator shares its state (and is returned unchanged by ``np.random.default_rng``),
		        so a single Generator can drive many twirls reproducibly.
		    skip_wires: Circuit qubit indices that must not receive twirling gates. Twirling wraps a
		        gate in single-qubit Paulis, so a gate acting on a computational resonator cannot be
		        twirled: the resonator accepts no single-qubit gates. Gates touching these wires are
		        left untouched.
		"""
		if gates_to_twirl is None:
			gates_to_twirl = [CZGate()]
		# Materialised so the gates can be iterated more than once; a generator would be consumed
		# building twirl_set below, leaving nothing to match against in run().
		requested = list(gates_to_twirl)
		self.gates_to_twirl = [gate for gate in requested if gate.name != _MOVE_GATE_NAME]
		if len(self.gates_to_twirl) != len(requested):
			warnings.warn(
				"MOVE gates cannot be twirled (they have no unitary representation) and were dropped "
				"from gates_to_twirl; they are left untouched in the circuit."
			)
		self.twirl_set = {gate.name: _get_twirl_set(gate) for gate in self.gates_to_twirl}
		self._twirling_gate_classes = tuple(gate.base_class for gate in self.gates_to_twirl)
		self._rng = np.random.default_rng(seed)
		self._skip_wires = frozenset(skip_wires) if skip_wires is not None else frozenset()
		#: Number of gates twirled by the most recent :meth:`run`.
		self.twirled_gate_count = 0
		super().__init__()

	def run(self, dag: DAGCircuit) -> DAGCircuit:
		# collect all nodes in DAG and proceed if it is to be twirled
		self.twirled_gate_count = 0
		for node in dag.op_nodes():
			if not isinstance(node.op, self._twirling_gate_classes):
				continue

			if self._skip_wires and any(dag.find_bit(qubit).index in self._skip_wires for qubit in node.qargs):
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
			self.twirled_gate_count += 1

		return dag


def _resonator_wires(backend: Optional[IQMBackendBase], circuits: list[QuantumCircuit]) -> frozenset[int]:
	"""Circuit qubit indices that are computational resonators rather than qubits.

	After MOVE routing a resonator occupies a wire of the circuit, and two-qubit gates act on
	``(qubit, resonator)`` pairs. Such gates cannot be twirled, because a resonator accepts no
	single-qubit gates. Returns an empty set for backends without resonators.
	"""
	architecture = getattr(backend, "architecture", None)
	declared = getattr(architecture, "computational_resonators", None)
	# Duck-typed and mocked backends may expose anything here, so only trust a real collection.
	if not isinstance(declared, (list, tuple, set, frozenset)):
		return frozenset()
	resonator_names = {name for name in declared if isinstance(name, str)}
	if not resonator_names:
		return frozenset()

	width = max((circuit.num_qubits for circuit in circuits), default=0)
	wires = set()
	for index in range(width):
		try:
			name = backend.index_to_qubit_name(index)  # type: ignore[union-attr]
		except Exception:  # pragma: no cover - defensive: index outside the backend's mapping
			continue
		if name in resonator_names:
			wires.add(index)
	return frozenset(wires)


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
	twirl_pass = PauliTwirl(gates_to_twirl=gates_to_twirl, seed=seed, skip_wires=_resonator_wires(backend, circuits))

	if backend is not None:
		names = {getattr(i[0], "name", None) for i in backend.target.instructions}
		basis_gates = {n for n in names if isinstance(n, str)}
		# Resonator devices route through MOVE, which the backend accepts but does not list in its
		# Qiskit target. Declaring it here keeps BasisTranslator from trying (and failing) to
		# rewrite the MOVE gates already in the circuit.
		if any(instruction.operation.name == _MOVE_GATE_NAME for circuit in circuits for instruction in circuit.data):
			logger.debug("MOVE gates present; passing them through Pauli twirling untouched")
			basis_gates.add(_MOVE_GATE_NAME)
		pm = PassManager(
			[
				twirl_pass,
				Decompose(gates_to_decompose=["pauli"]),
				BasisTranslator(target_basis=sorted(basis_gates), equivalence_library=StandardEquivalenceLibrary),
			]
		)
	else:
		pm = PassManager([twirl_pass])

	twirled_any = False
	for circuit in circuits:
		twirled_circuits.append(circuit)
		for _ in range(num_twirls):
			twirled = pm.run(circuit)
			twirled_any = twirled_any or twirl_pass.twirled_gate_count > 0
			# PassManager.run returns a fresh circuit that drops the input's TranspileLayout.
			# The IQM backend maps virtual qubits to physical ones via circuit.layout, so without
			# it the twirled circuit is validated/run against an identity layout and its CZ gates
			# can land on non-adjacent physical qubits, raising CircuitValidationError. The twirl
			# passes preserve qubit ordering, so the original layout stays valid; reattach it.
			twirled._layout = circuit._layout
			twirled_circuits.append(twirled)

	if num_twirls and not twirled_any:
		warnings.warn(
			f"Pauli twirling matched no gates, so {num_twirls} extra circuit(s) per input will be "
			"submitted with no mitigation applied. Either the circuit contains none of the gates "
			f"being twirled ({[gate.name for gate in twirl_pass.gates_to_twirl]}), or every candidate "
			"gate acts on a computational resonator, which cannot be twirled. Disable Pauli twirling "
			"to avoid spending the extra shots."
		)

	return twirled_circuits
