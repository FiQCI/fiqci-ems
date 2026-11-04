import numpy as np
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import CircuitInstruction, ControlFlowOp, Operation, QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import PassManager

from iqm.qiskit_iqm.move_gate import MoveGate

from copy import deepcopy

from collections.abc import Iterable

_MOVE_GATE_NAME = MoveGate().name


class SelfInverseMoveGate(MoveGate):
	"""A :class:`MoveGate` that reports itself as its own inverse.

	``MoveGate`` is deliberately left without a definition so the transpiler cannot decompose it,
	which also makes ``inverse()`` raise. Folding needs an inverse, and MOVE is an involution on its
	invariant subspace: its off-diagonal entries are an undefined phase and that phase's reciprocal,
	so ``MOVE MOVE`` is the identity whatever the phase happens to be. Subclassing keeps the gate's
	name and type, so IQM serialisation and MOVE-sandwich validation see an ordinary MOVE.
	"""

	def inverse(self, annotated: bool = False) -> "SelfInverseMoveGate":
		return SelfInverseMoveGate()


def _is_locally_foldable(name: str, num_qubits: int, fold_gates: set[str] | None) -> bool:
	"""Whether local folding acts on a gate. Shared so counting and folding cannot drift apart."""
	if num_qubits != 2 or name == "barrier":
		return False
	if fold_gates is not None:
		return name in fold_gates
	# Folding MOVE adds state transfers but does not scale how long the state sits in the resonator,
	# where the dominant error is, so the noise would not scale by the requested factor. It is only
	# folded when named explicitly. Global folding inverts the whole circuit and so always folds it.
	return name != _MOVE_GATE_NAME


def _is_fold_boundary(operation: Operation) -> bool:
	"""Whether global folding cuts the circuit at ``operation`` instead of inverting it.

	Measurements, resets and classical control flow have no inverse. Shared so counting and folding
	cannot drift apart.
	"""
	return operation.name in ("measure", "reset") or isinstance(operation, ControlFlowOp)


def _replace_moves_with_self_inverse(dag: DAGCircuit) -> None:
	"""Swap MOVE gates in place for :class:`SelfInverseMoveGate` so folding can invert them."""
	for node in dag.op_nodes():
		if node.op.name == _MOVE_GATE_NAME and not isinstance(node.op, SelfInverseMoveGate):
			dag.substitute_node(node, SelfInverseMoveGate(), inplace=True)


def _num_folds(num_gates: int, scale_factor: float) -> int:
	"""Number of single-gate folds needed to approximate ``scale_factor``.

	Each fold of a gate (``G -> G G† G``) adds two gates, so folding ``num_folds`` gates across a
	circuit of ``num_gates`` foldable gates yields an effective noise scaling of
	``(num_gates + 2 * num_folds) / num_gates``. Solving for the requested scale factor and rounding
	gives the closest achievable folding.
	"""
	return round((scale_factor - 1.0) * num_gates / 2.0)


def _achieved_scale_factor(num_gates: int, scale_factor: float, folding_method: str = "local") -> float:
	"""Effective noise scaling actually realised when targeting ``scale_factor``.

	Because folding is discrete, the requested scale factor can only be approximated. This returns the
	scaling the folding actually produces so callers (e.g. the estimator) can extrapolate against the
	real x-values rather than the requested ones. For local folding this is ``(N + 2·num_folds)/N``;
	for global folding it is ``1 + 2·num_global_folds + 2·num_to_fold/N`` (see ``ZNECircuits.run``).
	"""
	if num_gates == 0:
		return 1.0
	if folding_method == "global":
		num_global_folds = int((scale_factor - 1) // 2)
		frac = (scale_factor - 1) / 2 - num_global_folds
		num_to_fold = round(frac * num_gates)
		return 1 + 2 * num_global_folds + 2 * num_to_fold / num_gates
	return (num_gates + 2 * _num_folds(num_gates, scale_factor)) / num_gates


def _count_foldable_gates(
	circuit: QuantumCircuit, folding_method: str = "local", fold_gates: Iterable[str] | None = None
) -> int:
	"""Number of gates the folding will act on: all invertible non-barrier gates (global) or foldable 2-qubit gates (local)."""
	if folding_method == "global":
		return sum(
			1 for instr in circuit.data if instr.operation.name != "barrier" and not _is_fold_boundary(instr.operation)
		)
	fold_set = set(fold_gates) if fold_gates is not None else None
	return sum(1 for instr in circuit.data if _is_locally_foldable(instr.operation.name, len(instr.qubits), fold_set))


def _achieved_scale_factors(
	circuit: QuantumCircuit,
	scale_factors: Iterable[float],
	folding_method: str = "local",
	fold_gates: Iterable[str] | None = None,
) -> list[float]:
	"""Scale factors actually realised for ``circuit`` when targeting each of ``scale_factors``."""
	num_gates = _count_foldable_gates(circuit, folding_method, fold_gates)
	return [_achieved_scale_factor(num_gates, s, folding_method) for s in scale_factors]


class ZNECircuits(TransformationPass):
	"""A pass to generate circuits for zero-noise extrapolation (ZNE) by folding gates."""

	def __init__(
		self, fold_gates: Iterable[str] | None = None, scale_factor: float = 1, folding_method: str = "local", seed=None
	):
		"""
		Initialize the ZNECircuits pass.

		Args:
		    fold_gates: An optional iterable of gate names to fold. If None, every two-qubit gate is
		        folded except MOVE, which is only folded when named explicitly. Ignored by global
		        folding, which folds the whole circuit including MOVE.
		    scale_factor: The factor by which to scale the noise. Any real number >= 1. Non-odd-integer
		        values are approximated by partially folding a randomly-sampled subset of gates.
		    folding_method: The method to use for folding gates ("local" or "global"). Global folding cuts
		        the circuit at measurements, resets and control flow, which have no inverse, and folds each
		        segment between them in place.
		    seed: Seed or numpy Generator for the random gate sampling used to approximate non-odd-integer
		        scale factors. Passing an existing Generator shares its state (and is returned unchanged
		        by ``np.random.default_rng``), so a single Generator can drive many passes reproducibly.
		"""
		super().__init__()
		self.fold_gates = set(fold_gates) if fold_gates is not None else None
		self.scale_factor = scale_factor
		self.folding_method = folding_method
		self._rng = np.random.default_rng(seed)

	def run(self, dag: DAGCircuit) -> DAGCircuit:
		"""
		Run the ZNECircuits pass on the given DAGCircuit.

		Args:
		    dag: The input DAGCircuit to transform.

		Returns:
		    A new DAGCircuit with folded gates for ZNE.
		"""
		cloned_dag = deepcopy(dag)

		if self.scale_factor <= 1:
			return cloned_dag  # Original circuit, nothing to fold

		_replace_moves_with_self_inverse(cloned_dag)

		if self.folding_method == "local":
			return self._run_local(cloned_dag)
		elif self.folding_method == "global":
			return self._run_global(cloned_dag)

		return cloned_dag

	def _run_local(self, cloned_dag: DAGCircuit) -> DAGCircuit:
		"""Fold individual two-qubit gates in place to approximate ``self.scale_factor``."""
		# Collect foldable nodes up front so we can sample a subset and avoid mutating while iterating.
		foldable = [
			node
			for node in cloned_dag.op_nodes()
			if _is_locally_foldable(node.op.name, node.num_qubits, self.fold_gates)
		]
		num_gates = len(foldable)
		if num_gates == 0:
			return cloned_dag

		num_folds = _num_folds(num_gates, self.scale_factor)
		if num_folds == 0:
			return cloned_dag

		base, extra = divmod(num_folds, num_gates)
		# A random subset gets one extra fold so the average matches the requested scale factor.
		extra_indices = set(self._rng.choice(num_gates, size=extra, replace=False).tolist()) if extra > 0 else set()

		for idx, node in enumerate(foldable):
			folds = base + (1 if idx in extra_indices else 0)
			if folds == 0:
				continue

			mini_dag = DAGCircuit()
			register = QuantumRegister(2)
			mini_dag.add_qreg(register)

			# 2 * folds + 1 instances: the original plus `folds` G† G pairs (alternating).
			for ind in range(2 * folds + 1):
				if ind % 2 == 0:
					mini_dag.apply_operation_back(node.op, [register[0], register[1]])
				else:
					mini_dag.apply_operation_back(node.op.inverse(), [register[0], register[1]])

			cloned_dag.substitute_node_with_dag(node, mini_dag)

		return cloned_dag

	def _run_global(self, cloned_dag: DAGCircuit) -> DAGCircuit:
		"""Fold the whole circuit (``C -> C C† C ...``), partially folding a suffix for the remainder.

		Measurements, resets and control flow cannot be inverted, so the circuit is cut at each of them
		and every segment in between is folded in place. ``circuit.data`` is a topological order, and a
		run of consecutive instructions in it has no path that leaves the run and re-enters it, so
		replacing a segment ``S`` by ``S S† S`` leaves the circuit unchanged. Without such operations
		there is one segment, the whole circuit.
		"""
		circuit = dag_to_circuit(cloned_dag)

		segments: list[list[CircuitInstruction]] = [[]]
		boundaries: list[CircuitInstruction] = []
		for instr in circuit.data:
			if _is_fold_boundary(instr.operation):
				boundaries.append(instr)
				segments.append([])
			else:
				segments[-1].append(instr)

		# Integer part: full global folds. Fractional part: fold a suffix of the gates.
		num_global_folds = int((self.scale_factor - 1) // 2)
		frac = (self.scale_factor - 1) / 2 - num_global_folds

		gate_positions = [
			(seg, i)
			for seg, segment in enumerate(segments)
			for i, instr in enumerate(segment)
			if instr.operation.name != "barrier"
		]
		num_to_fold = round(frac * len(gate_positions))

		# The suffix is the last `num_to_fold` gates of the circuit (rather than a random subset), so the
		# appended `suffix† suffix` cannot change the overall unitary. Segments are consecutive, so the
		# suffix is a tail of every segment it reaches; record where each of those tails starts.
		tail_starts: dict[int, int] = {}
		for seg, i in gate_positions[len(gate_positions) - num_to_fold :]:
			tail_starts.setdefault(seg, i)

		folded = circuit.copy_empty_like()
		for seg, segment in enumerate(segments):
			block = self._sub_circuit(circuit, segment)
			folded.compose(block, inplace=True)
			for _ in range(num_global_folds):
				folded.compose(block.inverse(), inplace=True)
				folded.compose(block, inplace=True)

			if seg in tail_starts:
				tail = self._sub_circuit(
					circuit, [instr for instr in segment[tail_starts[seg] :] if instr.operation.name != "barrier"]
				)
				folded.compose(tail.inverse(), inplace=True)
				folded.compose(tail, inplace=True)

			if seg < len(boundaries):
				folded.append(boundaries[seg])

		return circuit_to_dag(folded)

	@staticmethod
	def _sub_circuit(circuit: QuantumCircuit, instructions: list[CircuitInstruction]) -> QuantumCircuit:
		"""A circuit over ``circuit``'s wires holding ``instructions``, without its global phase.

		``folded`` already carries the phase from ``copy_empty_like``, and ``compose`` adds the composed
		circuit's phase, so a block that kept it would add the phase once more per segment.
		"""
		sub = circuit.copy_empty_like()
		sub.global_phase = 0
		for instr in instructions:
			sub.append(instr)
		return sub


def _get_zne_circuits(
	circuits: list[QuantumCircuit],  # list of QuantumCircuits to generate ZNE circuits from
	fold_gates: None
	| (Iterable[str]) = None,  # list of gate names to fold, if None, all gates two qubit gates will be folded
	scale_factors: Iterable[float] | None = [1, 3, 5],  # list of at least two real numbers >= 1
	folding_method: str = "local",  # "local" or "global"
	seed=None,  # seed for the random gate sampling used to approximate non-odd-integer scale factors
) -> list[QuantumCircuit]:
	"""Generate ZNE circuits by folding gates in the input QuantumCircuit.

	Args:
	    circuits: The input QuantumCircuit to transform.
	    fold_gates: An optional iterable of gate names to fold. If None, every two-qubit gate is folded
	        except MOVE, which is only folded when named explicitly. Ignored by global folding, which
	        folds the whole circuit including MOVE.
	    scale_factors: An optional iterable of real scale factors (>= 1) for folding. If None, defaults
	        to [1, 3, 5]. Non-odd-integer values are approximated by partial/random folding.
	    folding_method: The method to use for folding gates ("local" or "global"). Global folding cuts the
	        circuit at measurements, resets and control flow and folds each segment between them in place.
	    seed: Seed for the random gate sampling. A single Generator is shared across all generated
	        circuits so the sampling is distinct per (scale factor, circuit) but reproducible overall.
	Returns:
	    A list of QuantumCircuits with folded gates for ZNE.
	"""
	zne_circuits = []
	if scale_factors is None:
		scale_factors = [1, 3, 5]

	if not all(isinstance(s, (int, float)) and not isinstance(s, bool) and s >= 1 for s in scale_factors):
		raise ValueError("Scale factors must be real numbers >= 1.")

	rng = np.random.default_rng(seed)

	for scale in scale_factors:
		for circuit in circuits:
			pm = PassManager(
				ZNECircuits(fold_gates=fold_gates, scale_factor=scale, folding_method=folding_method, seed=rng)
			)

			zne_circuit = pm.run(circuit)
			zne_circuit._layout = circuit.layout

			zne_circuits.append(zne_circuit)

	return zne_circuits
