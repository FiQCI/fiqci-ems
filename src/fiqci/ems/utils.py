"""Utility functions"""

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def probabilities_to_counts(probabilities, shots) -> list[dict]:
	"""Convert probabilities to counts.

	Uses the largest-remainder method so the counts sum to exactly ``shots`` (for a normalised input).
	Truncating each outcome instead drops up to one count per outcome, which leaves the realised total
	below the requested shot count and so understates the ``N`` of any shot-error estimate taken from
	these counts.
	"""
	try:
		probabilities[0]
	except KeyError:
		# If probabilities is not iterable, treat it as a single set of probabilities
		probabilities = [probabilities]

	counts_list = []
	for probs in probabilities:
		scaled = {k: prob * shots for k, prob in probs.items()}
		counts = {k: int(value) for k, value in scaled.items()}
		deficit = shots - sum(counts.values())
		if deficit > 0:
			# Only the truncated fractions are handed back out, so an input summing below 1 stays
			# below `shots` instead of being padded up to it.
			by_remainder = sorted(
				(k for k in scaled if scaled[k] > counts[k]),
				key=lambda k: (scaled[k] - counts[k], scaled[k]),
				reverse=True,
			)
			for k in by_remainder[:deficit]:
				counts[k] += 1
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
