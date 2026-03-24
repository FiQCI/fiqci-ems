"""Tests for utility functions."""

import numpy as np
from qiskit import QuantumCircuit

from fiqci.ems.utils import _count_gates, _remove_idle_wires, probabilities_to_counts


def test_probabilities_to_counts():
	"""Test probabilities_to_counts with various inputs."""
	# Test single dict
	probs = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
	result = probabilities_to_counts(probs, 1000)
	assert isinstance(result, list)
	assert len(result) == 1
	assert all(isinstance(v, int) for v in result[0].values())

	# Test list of dicts
	probs_list = [{"00": 0.5, "11": 0.5}, {"00": 0.75, "11": 0.25}]
	result = probabilities_to_counts(probs_list, 100)
	assert len(result) == 2
	assert all(isinstance(v, int) for v in result[0].values())

	# Test numpy float types
	probs_numpy = {
		"10": np.float32(0.23291016),
		"11": np.float32(0.23828125),
		"00": np.float32(0.25146484),
		"01": np.float32(0.27734375),
	}
	result = probabilities_to_counts(probs_numpy, 2048)
	assert len(result) == 1
	assert all(isinstance(v, int) for v in result[0].values())


def test_count_gates_simple():
	"""Test _count_gates with a simple circuit."""
	qc = QuantumCircuit(3)
	qc.h(0)
	qc.cx(0, 1)
	gate_count = _count_gates(qc)
	assert gate_count[qc.qubits[0]] == 2  # h + cx
	assert gate_count[qc.qubits[1]] == 1  # cx
	assert gate_count[qc.qubits[2]] == 0  # idle


def test_count_gates_empty_circuit():
	"""Test _count_gates with an empty circuit."""
	qc = QuantumCircuit(2)
	gate_count = _count_gates(qc)
	assert all(v == 0 for v in gate_count.values())


def test_remove_idle_wires():
	"""Test _remove_idle_wires removes qubits with no gates."""
	qc = QuantumCircuit(4)
	qc.h(0)
	qc.cx(0, 2)
	# qubits 1 and 3 are idle
	result = _remove_idle_wires(qc)
	assert result.num_qubits == 2
	assert len(result.data) == 2


def test_remove_idle_wires_no_idle():
	"""Test _remove_idle_wires when all qubits are active."""
	qc = QuantumCircuit(2)
	qc.h(0)
	qc.x(1)
	result = _remove_idle_wires(qc)
	assert result.num_qubits == 2
	assert len(result.data) == 2


def test_remove_idle_wires_all_idle():
	"""Test _remove_idle_wires when all qubits are idle."""
	qc = QuantumCircuit(3)
	result = _remove_idle_wires(qc)
	assert result.num_qubits == 0
	assert len(result.data) == 0
