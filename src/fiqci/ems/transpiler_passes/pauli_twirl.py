import logging
import warnings

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Qubit, QuantumRegister, Gate, StandardEquivalenceLibrary
from qiskit.circuit.library import CZGate, XGate, YGate, ZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import BasisTranslator, Decompose, Optimize1qGatesDecomposition
from qiskit.quantum_info import Operator, pauli_basis

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.move_gate import MoveGate

import numpy as np

from collections import defaultdict
from collections.abc import Callable, Iterable

logger: logging.Logger = logging.getLogger(__name__)

# MOVE transfers a qubit state to/from a computational resonator. It has no unitary representation
# (``Operator(MoveGate())`` raises), so it can be neither twirled nor translated to the backend's
# Qiskit target basis, which does not list it. It is passed through untouched.
_MOVE_GATE_NAME = MoveGate().name

_PAULI_GATES: dict[str, Callable[[], Gate]] = {"X": XGate, "Y": YGate, "Z": ZGate}

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

	def __init__(self, gates_to_twirl: Iterable[Gate] | None = None, seed=None):
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
		#: Number of gates twirled by the most recent :meth:`run`.
		self.twirled_gate_count = 0
		super().__init__()

	def _draw_twirl_pair(self, gate_name: str) -> tuple:
		twirl_set = self.twirl_set[gate_name]
		return twirl_set[self._rng.integers(0, len(twirl_set))]

	def run(self, dag: DAGCircuit) -> DAGCircuit:
		self.twirled_gate_count = 0
		if dag.named_nodes(_MOVE_GATE_NAME):
			return self._twirl_through_resonators(dag)

		# collect all nodes in DAG and proceed if it is to be twirled
		for node in dag.op_nodes():
			if not isinstance(node.op, self._twirling_gate_classes):
				continue

			twirl_pair = self._draw_twirl_pair(node.op.name)

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

	def _find_move_sandwiches(self, circuit: QuantumCircuit, resonators: set[Qubit]) -> list[dict]:
		"""Locate each MOVE pair and the twirlable gates it encloses.

		A sandwich is usable only if every operation on the resonator wire between the two MOVEs is a
		gate we know how to commute a Pauli through, i.e. one of ``gates_to_twirl``. Anything else
		(another two-qubit gate, a stray single-qubit gate) disqualifies the sandwich and its gates
		are left untwirled rather than guessed at.
		"""
		sandwiches: list[dict] = []
		for resonator in resonators:
			opened_at: int | None = None
			moved_qubit: Qubit | None = None
			enclosed: list[tuple[int, Qubit]] = []
			usable = True
			for index, instruction in enumerate(circuit.data):
				if resonator not in instruction.qubits:
					continue
				name = instruction.operation.name
				if name == _MOVE_GATE_NAME:
					if opened_at is None:
						partners = [qubit for qubit in instruction.qubits if qubit is not resonator]
						opened_at, moved_qubit, enclosed, usable = index, partners[0] if partners else None, [], True
					else:
						if usable and enclosed and moved_qubit is not None:
							sandwiches.append(
								{"open": opened_at, "close": index, "moved": moved_qubit, "gates": enclosed}
							)
						opened_at = None
				elif opened_at is None or name == "barrier":
					continue
				elif isinstance(instruction.operation, self._twirling_gate_classes) and len(instruction.qubits) == 2:
					partners = [qubit for qubit in instruction.qubits if qubit is not resonator]
					enclosed.append((index, partners[0]))
				else:
					usable = False
		return sandwiches

	def _twirl_through_resonators(self, dag: DAGCircuit) -> DAGCircuit:
		"""Twirl a MOVE-routed circuit by propagating resonator-side Paulis out of the sandwich.

		A resonator holds the state of the qubit that was moved into it, so a Pauli applied to the
		resonator is the same Pauli applied to that qubit before the MOVE (or after the closing MOVE
		for the trailing half of the pair). Getting it there means commuting it past the other gates
		of the sandwich: ``I``/``Z`` pass freely, while ``X``/``Y`` pick up a ``Z`` on the far qubit of
		every gate they cross. The qubit half of each twirl pair stays where it always was, next to
		the gate.
		"""
		circuit = dag_to_circuit(dag)
		# IQM requires MOVE to be applied as [qubit, resonator], so the circuit itself identifies the
		# resonator wires. Deriving them here rather than from the backend's component list avoids
		# having to map circuit wires to physical qubits, which only agree when the layout is trivial.
		resonators = {
			instruction.qubits[1]
			for instruction in circuit.data
			if instruction.operation.name == _MOVE_GATE_NAME and len(instruction.qubits) == 2
		}

		enclosed_at: dict[int, tuple[dict, int]] = {}
		for sandwich in self._find_move_sandwiches(circuit, resonators):
			for position, (index, _partner) in enumerate(sandwich["gates"]):
				enclosed_at[index] = (sandwich, position)

		# Paulis to emit around instruction i, keyed by the index of the instruction they attach to.
		before: dict[int, list[tuple[Qubit, str]]] = defaultdict(list)
		after: dict[int, list[tuple[Qubit, str]]] = defaultdict(list)

		for index, instruction in enumerate(circuit.data):
			if not isinstance(instruction.operation, self._twirling_gate_classes):
				continue
			touches_resonator = any(qubit in resonators for qubit in instruction.qubits)
			if touches_resonator and index not in enclosed_at:
				continue  # nowhere safe to put the resonator-side Paulis; leave the gate alone

			left, right = self._draw_twirl_pair(instruction.operation.name)
			# Qiskit Pauli labels are little-endian, so the last character belongs to qargs[0].
			left_labels = {instruction.qubits[0]: left.to_label()[-1], instruction.qubits[1]: left.to_label()[-2]}
			right_labels = {instruction.qubits[0]: right.to_label()[-1], instruction.qubits[1]: right.to_label()[-2]}

			if not touches_resonator:
				for qubit in instruction.qubits:
					before[index].append((qubit, left_labels[qubit]))
					after[index].append((qubit, right_labels[qubit]))
				self.twirled_gate_count += 1
				continue

			sandwich, position = enclosed_at[index]
			resonator = next(qubit for qubit in instruction.qubits if qubit in resonators)
			partner = next(qubit for qubit in instruction.qubits if qubit is not resonator)
			before[index].append((partner, left_labels[partner]))
			after[index].append((partner, right_labels[partner]))

			# The left Pauli travels backwards to the opening MOVE, the right one forwards to the
			# closing MOVE, each landing on the moved qubit just outside the sandwich.
			left_resonator, right_resonator = left_labels[resonator], right_labels[resonator]
			if left_resonator != "I":
				before[sandwich["open"]].append((sandwich["moved"], left_resonator))
				if left_resonator in ("X", "Y"):
					for crossed_index, crossed_partner in sandwich["gates"][:position]:
						before[crossed_index].append((crossed_partner, "Z"))
			if right_resonator != "I":
				after[sandwich["close"]].append((sandwich["moved"], right_resonator))
				if right_resonator in ("X", "Y"):
					for crossed_index, crossed_partner in sandwich["gates"][position + 1 :]:
						after[crossed_index].append((crossed_partner, "Z"))
			self.twirled_gate_count += 1

		twirled = circuit.copy_empty_like()
		for index, instruction in enumerate(circuit.data):
			for qubit, label in before[index]:
				if label != "I":
					twirled.append(_PAULI_GATES[label](), [qubit])
			twirled.append(instruction)
			for qubit, label in after[index]:
				if label != "I":
					twirled.append(_PAULI_GATES[label](), [qubit])
		return circuit_to_dag(twirled)


def get_twirled_circuits(
	circuits: list[QuantumCircuit],
	num_twirls: int,
	gates_to_twirl: Iterable[Gate] | None = None,
	backend: IQMBackendBase | None = None,
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
				Optimize1qGatesDecomposition(basis=basis_gates),
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
			"gate sits in a MOVE sandwich that also holds operations a twirl Pauli cannot be commuted "
			"through. Disable Pauli twirling to avoid spending the extra shots."
		)

	return twirled_circuits
