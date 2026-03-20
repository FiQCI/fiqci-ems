from __future__ import annotations

from copy import deepcopy

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Instruction
from qiskit.circuit.library import HGate, SdgGate, Measure
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass

from qiskit.transpiler.passes import RemoveFinalMeasurements
from qiskit.quantum_info import SparsePauliOp


class _ModifyMeasurementBasis(TransformationPass):
	def __init__(self, measurement_settings: list[dict[int, str]], ops: dict[str, Instruction] | None = None):
		self.measurement_settings = measurement_settings
		self.ops = ops
		super().__init__()

	def run(self, dag: DAGCircuit) -> DAGCircuit:
		cloned_dag = deepcopy(dag)

		cloned_dag.add_creg(ClassicalRegister(len(self.measurement_settings[0]), name="meas"))

		clbit_index = 0

		for setting in self.measurement_settings:
			for qubit, basis in setting.items():
				if basis == "X":
					if self.ops and "X-meas" in self.ops:
						cloned_dag.apply_operation_back(self.ops["X-meas"], [cloned_dag.qubits[qubit]])
					else:
						cloned_dag.apply_operation_back(HGate(), [cloned_dag.qubits[qubit]])
					cloned_dag.apply_operation_back(
						Measure(), [cloned_dag.qubits[qubit]], [cloned_dag.clbits[clbit_index]]
					)
					clbit_index += 1
				elif basis == "Y":
					if self.ops and "Y-meas" in self.ops:
						cloned_dag.apply_operation_back(self.ops["Y-meas"], [cloned_dag.qubits[qubit]])
					else:
						cloned_dag.apply_operation_back(SdgGate(), [cloned_dag.qubits[qubit]])
						cloned_dag.apply_operation_back(HGate(), [cloned_dag.qubits[qubit]])
					cloned_dag.apply_operation_back(
						Measure(), [cloned_dag.qubits[qubit]], [cloned_dag.clbits[clbit_index]]
					)
					clbit_index += 1
				elif basis == "Z":
					cloned_dag.apply_operation_back(
						Measure(), [cloned_dag.qubits[qubit]], [cloned_dag.clbits[clbit_index]]
					)
					clbit_index += 1
					# No change needed for Z-basis measurement
					pass
				else:
					raise ValueError(f"Unsupported measurement basis: {basis}")

		return cloned_dag


def _get_obs_subcircuits(
	subcircuits: list[QuantumCircuit],
	measurement_settings: list[dict[int, str]],
	ops: dict[str, Instruction] | None = None,
) -> list[dict[int, QuantumCircuit]]:
	pms = [PassManager([_ModifyMeasurementBasis([setting], ops)]) for setting in measurement_settings]

	remove_meas_pm = PassManager([RemoveFinalMeasurements()])

	obs_subcircuits = []
	for pm in pms:
		pm_circs = {}
		for ind, subcircuit in enumerate(subcircuits):
			modified_circuit = pm.run(remove_meas_pm.run(subcircuit)).decompose(
				gates_to_decompose=["X-meas", "Y-meas"]
			)  # Decompose custom measurement
			if modified_circuit.num_qubits == 0:
				continue
			pm_circs[ind] = modified_circuit
		obs_subcircuits.append(pm_circs)
	return obs_subcircuits


def _get_observable_circuit_index(pauli, combined: list[dict[int, str]]):
	"""Find which measurement setting covers the non-identity letters of `pauli`,
	and return the indices of the qubits involved."""
	label = pauli
	non_identity = {i: p for i, p in enumerate(label) if p.to_label() != "I"}

	for idx, setting in enumerate(combined):
		# All non-identity qubits must be measured in the matching basis
		if all(setting.get(q) == p.to_label() for q, p in non_identity.items()):
			return {"circuit_index": idx, "obs_indices": list(range(len(non_identity))), "num_meas": len(non_identity)}

	return {"circuit_index": None, "obs_indices": [], "num_meas": 0}


def _combine_pauli_ops(op: SparsePauliOp) -> list[dict[int, str]]:  # noqa: C901
	"""Combine Pauli operators that have no conflicting non-identity components.

	Args:
	    op (SparsePauliOp): The SparsePauliOp to analyze.

	Returns:
	    list[dict[int, str]]: A list of combined measurement settings, where each dict
	                        maps qubit indices to Pauli basis measurements.
	"""

	pauli_strings = [pauli.to_label()[::-1] for pauli in op.paulis]

	combined_settings = []
	used = [False] * len(pauli_strings)

	for i, pauli_string in enumerate(pauli_strings):
		if used[i]:
			continue

		# Start a new combined setting with the current Pauli string
		combined = {}
		for qubit_index, pauli in enumerate(pauli_string):
			if pauli != "I":
				combined[qubit_index] = pauli

		used[i] = True

		# Try to combine with remaining Pauli strings
		for j in range(i + 1, len(pauli_strings)):
			if used[j]:
				continue

			# Check if pauli_strings[j] can be combined with current combined setting
			can_combine = True
			for qubit_index, pauli in enumerate(pauli_strings[j]):
				if pauli != "I":
					if qubit_index in combined and combined[qubit_index] != pauli:
						can_combine = False
						break

			# If compatible, add to combined setting
			if can_combine:
				for qubit_index, pauli in enumerate(pauli_strings[j]):
					if pauli != "I":
						combined[qubit_index] = pauli
				used[j] = True

		combined_settings.append(combined)

	return combined_settings
