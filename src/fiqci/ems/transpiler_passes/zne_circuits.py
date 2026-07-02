import warnings

import numpy as np
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import PassManager

from copy import deepcopy

from typing import Iterable, Optional


def _num_folds(num_gates: int, scale_factor: float) -> int:
	"""Number of single-gate folds needed to approximate ``scale_factor``.

	Each fold of a gate (``G -> G G† G``) adds two gates, so folding ``num_folds`` gates across a
	circuit of ``num_gates`` foldable gates yields an effective noise scaling of
	``(num_gates + 2 * num_folds) / num_gates``. Solving for the requested scale factor and rounding
	gives the closest achievable folding.
	"""
	return round((scale_factor - 1.0) * num_gates / 2.0)


def _achieved_scale_factor(num_gates: int, scale_factor: float) -> float:
	"""Effective noise scaling actually realised when targeting ``scale_factor`` (see ``_num_folds``)."""
	if num_gates == 0:
		return 1.0
	return (num_gates + 2 * _num_folds(num_gates, scale_factor)) / num_gates


class ZNECircuits(TransformationPass):
	"""A pass to generate circuits for zero-noise extrapolation (ZNE) by folding gates."""

	def __init__(
		self,
		fold_gates: Optional[Iterable[str]] = None,
		scale_factor: float = 1,
		folding_method: str = "local",
		seed=None,
	):
		"""
		Initialize the ZNECircuits pass.

		Args:
		    fold_gates: An optional iterable of gate names to fold. If None, all gates will be folded.
		    scale_factor: The factor by which to scale the noise. Any real number >= 1. Non-odd-integer
		        values are approximated by partially folding a randomly-sampled subset of gates.
		    folding_method: The method to use for folding gates ("local" or "global").
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
			if node.num_qubits == 2
			and node.op.name != "barrier"
			and (self.fold_gates is None or node.name in self.fold_gates)
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
		"""Fold the whole circuit (``C -> C C† C ...``), partially folding a suffix for the remainder."""
		circuit = dag_to_circuit(cloned_dag)

		# Separate measurements; everything else forms the invertible `core` we fold.
		measurements = [instr for instr in circuit.data if instr.operation.name == "measure"]
		core = circuit.copy_empty_like()
		for instr in circuit.data:
			if instr.operation.name != "measure":
				core.append(instr)

		# Integer part: full global folds. Fractional part: fold a suffix of the gates.
		num_global_folds = int((self.scale_factor - 1) // 2)
		frac = (self.scale_factor - 1) / 2 - num_global_folds

		gate_indices = [i for i, instr in enumerate(core.data) if instr.operation.name != "barrier"]
		num_to_fold = round(frac * len(gate_indices))

		folded = core.copy()
		for _ in range(num_global_folds):
			folded.compose(core.inverse(), inplace=True)
			folded.compose(core, inplace=True)

		if num_to_fold > 0:
			# Build the sub-circuit of the last `num_to_fold` gates; a suffix (rather than a random
			# subset) keeps the appended `suffix† suffix` from changing the overall unitary.
			suffix = core.copy_empty_like()
			for i in gate_indices[-num_to_fold:]:
				suffix.append(core.data[i])
			folded.compose(suffix.inverse(), inplace=True)
			folded.compose(suffix, inplace=True)

		for instr in measurements:
			folded.append(instr)

		return circuit_to_dag(folded)


def _get_zne_circuits(
	circuits: list[QuantumCircuit],  # list of QuantumCircuits to generate ZNE circuits from
	fold_gates: Optional[
		Iterable[str]
	] = None,  # list of gate names to fold, if None, all gates two qubit gates will be folded
	scale_factors: Optional[Iterable[float]] = [1, 3, 5],  # list of at least two real numbers >= 1
	folding_method: str = "local",  # "local" or "global"
	seed=None,  # seed for the random gate sampling used to approximate non-odd-integer scale factors
) -> list[QuantumCircuit]:
	"""Generate ZNE circuits by folding gates in the input QuantumCircuit.

	Args:
	    circuits: The input QuantumCircuit to transform.
	    fold_gates: An optional iterable of gate names to fold. If None, all gates will be folded.
	    scale_factors: An optional iterable of real scale factors (>= 1) for folding. If None, defaults
	        to [1, 3, 5]. Non-odd-integer values are approximated by partial/random folding.
	    folding_method: The method to use for folding gates ("local" or "global").
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
			# Warn when the discretisation of folding can't reach the requested scale (small circuits).
			num_gates = (
				sum(1 for instr in circuit.data if instr.operation.name not in ("measure", "barrier"))
				if folding_method == "global"
				else sum(
					1
					for instr in circuit.data
					if len(instr.qubits) == 2
					and instr.operation.name != "barrier"
					and (fold_gates is None or instr.operation.name in set(fold_gates))
				)
			)
			achieved = _achieved_scale_factor(num_gates, scale)
			if abs(achieved - scale) > 0.1:
				warnings.warn(
					f"Requested scale factor {scale} could only be approximated as {achieved:.3f} for a "
					f"circuit with {num_gates} foldable gate(s); extrapolation still uses the requested value."
				)

			pm = PassManager(
				ZNECircuits(fold_gates=fold_gates, scale_factor=scale, folding_method=folding_method, seed=rng)
			)

			zne_circuit = pm.run(circuit)

			zne_circuits.append(zne_circuit)

	return zne_circuits
