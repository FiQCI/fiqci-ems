"""Utility functions"""

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def probabilities_to_counts(probabilities, shots) -> list[dict]:
	"""Convert probabilities to counts"""
	try:
		probabilities[0]
	except KeyError:
		# If probabilities is not iterable, treat it as a single set of probabilities
		probabilities = [probabilities]

	counts_list = []
	for probs in probabilities:
		counts = {}
		for k, prob in probs.items():
			counts[k] = int(prob * shots)
		counts_list.append(counts)

	return counts_list


def _count_gates(circuit: QuantumCircuit) -> dict[Qubit, int]:
	"""Count the number of gates acting on each qubit in a QuantumCircuit.

	Args:
	    circuit (QuantumCircuit): The input quantum circuit.

	Returns:
	    dict[Qubit, int]: A dictionary mapping each qubit to the number of gates
	    acting on it.
	"""
	gate_count = dict.fromkeys(circuit.qubits, 0)
	for instruction in circuit.data:
		for qubit in instruction.qubits:
			gate_count[qubit] += 1

	return gate_count


def _remove_idle_wires(circuit: QuantumCircuit) -> QuantumCircuit:
	"""Remove idle wires from a QuantumCircuit.

	Args:
	    circuit (QuantumCircuit): The input quantum circuit.

	Returns:
	    QuantumCircuit: A new quantum circuit with idle wires removed.
	"""
	gate_count = _count_gates(circuit)
	active_qubits = [q for q in circuit.qubits if gate_count[q] > 0]

	new_circuit = QuantumCircuit(len(active_qubits))
	qubit_map = {old: new_circuit.qubits[i] for i, old in enumerate(active_qubits)}

	for instruction in circuit.data:
		new_qubits = [qubit_map[q] for q in instruction.qubits]
		new_circuit.append(instruction.operation, new_qubits, instruction.clbits)

	return new_circuit
