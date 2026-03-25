"""Unit tests for basis_measurement module."""

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from fiqci.ems.basis_measurement import get_obs_subcircuits, _combine_pauli_ops, _get_observable_circuit_index


class TestCombinePauliOps:
	"""Tests for _combine_pauli_ops."""

	def test_single_pauli(self) -> None:
		"""Test combining a single Pauli operator."""
		op = SparsePauliOp.from_list([("ZZ", 1.0)])
		result = _combine_pauli_ops(op)

		assert len(result) == 1
		assert result[0] == {0: "Z", 1: "Z"}

	def test_compatible_paulis_are_combined(self) -> None:
		"""Test that compatible Pauli operators are combined into one setting."""
		# ZI and IZ have no conflicts -> should combine
		op = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0)])
		result = _combine_pauli_ops(op)

		assert len(result) == 1
		# Pauli labels are reversed internally, so ZI -> qubit 1: Z, IZ -> qubit 0: Z
		assert result[0] == {0: "Z", 1: "Z"}

	def test_conflicting_paulis_are_separate(self) -> None:
		"""Test that conflicting Pauli operators get separate settings."""
		# ZI and XI conflict on qubit 1
		op = SparsePauliOp.from_list([("ZI", 1.0), ("XI", 1.0)])
		result = _combine_pauli_ops(op)

		assert len(result) == 2

	def test_identity_only(self) -> None:
		"""Test a Pauli with only identity terms."""
		op = SparsePauliOp.from_list([("II", 1.0)])
		result = _combine_pauli_ops(op)

		assert len(result) == 1
		assert result[0] == {}  # No non-identity qubits

	def test_mixed_compatible_and_conflicting(self) -> None:
		"""Test a mix of compatible and conflicting operators."""
		# ZZ and ZI are compatible (both Z on qubit 1), XI conflicts with Z on qubit 1
		op = SparsePauliOp.from_list([("ZZ", 1.0), ("ZI", 1.0), ("XI", 1.0)])
		result = _combine_pauli_ops(op)

		# ZZ and ZI should combine, XI separate
		assert len(result) == 2

	def test_three_qubit_operators(self) -> None:
		"""Test with 3-qubit Pauli operators."""
		op = SparsePauliOp.from_list([("ZZI", 1.0), ("IIZ", 1.0)])
		result = _combine_pauli_ops(op)

		# No conflicts, should combine
		assert len(result) == 1
		assert result[0] == {0: "Z", 1: "Z", 2: "Z"}

	def test_all_bases(self) -> None:
		"""Test with X, Y, and Z bases."""
		op = SparsePauliOp.from_list([("XYZ", 1.0)])
		result = _combine_pauli_ops(op)

		assert len(result) == 1
		# XYZ reversed: Z on qubit 0, Y on qubit 1, X on qubit 2
		assert result[0] == {0: "Z", 1: "Y", 2: "X"}

	def test_same_basis_different_qubits_combine(self) -> None:
		"""Test that same basis on different qubits combines."""
		op = SparsePauliOp.from_list([("XI", 1.0), ("IX", 1.0)])
		result = _combine_pauli_ops(op)

		assert len(result) == 1
		assert result[0] == {0: "X", 1: "X"}


class TestGetObservableCircuitIndex:
	"""Tests for _get_observable_circuit_index."""

	def test_single_z_pauli_matches(self) -> None:
		"""Test matching a single Z Pauli to a Z measurement setting."""
		op = SparsePauliOp.from_list([("IZ", 1.0)])
		pauli = op.paulis[0]
		combined = [{0: "Z"}]

		result = _get_observable_circuit_index(pauli, combined)

		assert result["circuit_index"] == 0
		assert result["num_meas"] == 1

	def test_no_matching_setting(self) -> None:
		"""Test when no measurement setting covers the observable."""
		op = SparsePauliOp.from_list([("IX", 1.0)])
		pauli = op.paulis[0]
		combined = [{0: "Z"}]

		result = _get_observable_circuit_index(pauli, combined)

		assert result["circuit_index"] is None
		assert result["obs_indices"] == []
		assert result["num_meas"] == 0

	def test_matches_second_setting(self) -> None:
		"""Test matching to the second measurement setting."""
		op = SparsePauliOp.from_list([("IX", 1.0)])
		pauli = op.paulis[0]
		combined = [{0: "Z"}, {0: "X"}]

		result = _get_observable_circuit_index(pauli, combined)

		assert result["circuit_index"] == 1

	def test_multi_qubit_match(self) -> None:
		"""Test matching a multi-qubit observable."""
		op = SparsePauliOp.from_list([("ZZ", 1.0)])
		pauli = op.paulis[0]
		combined = [{0: "Z", 1: "Z"}]

		result = _get_observable_circuit_index(pauli, combined)

		assert result["circuit_index"] == 0
		assert result["num_meas"] == 2

	def test_identity_pauli(self) -> None:
		"""Test with an all-identity Pauli (no non-identity qubits)."""
		op = SparsePauliOp.from_list([("II", 1.0)])
		pauli = op.paulis[0]
		combined = [{0: "Z"}]

		result = _get_observable_circuit_index(pauli, combined)

		# All-identity has no constraints, should match first setting
		assert result["circuit_index"] == 0
		assert result["num_meas"] == 0

	def test_partial_match_fails(self) -> None:
		"""Test that a partial basis match does not succeed."""
		# ZX needs Z on qubit 0 and X on qubit 1
		op = SparsePauliOp.from_list([("ZX", 1.0)])
		pauli = op.paulis[0]
		# Setting only has Z on both qubits
		combined = [{0: "Z", 1: "Z"}]

		result = _get_observable_circuit_index(pauli, combined)

		# X on qubit 0 (reversed) doesn't match Z
		assert result["circuit_index"] is None


class TestGetObsSubcircuits:
	"""Tests for get_obs_subcircuits."""

	def test_z_basis_measurement(self) -> None:
		"""Test that Z-basis measurement adds only a measurement gate."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)

		settings = [{0: "Z", 1: "Z"}]
		result = get_obs_subcircuits([qc], settings)

		assert len(result) == 1
		circuit = result[0][0]
		assert circuit.num_qubits == 2
		assert circuit.num_clbits == 2

	def test_x_basis_adds_hadamard(self) -> None:
		"""Test that X-basis measurement adds H gate before measurement."""
		qc = QuantumCircuit(1)
		qc.x(0)

		settings = [{0: "X"}]
		result = get_obs_subcircuits([qc], settings)

		circuit = result[0][0]
		# Should have H gate before measurement
		op_names = [inst.operation.name for inst in circuit]
		assert "h" in op_names
		assert "measure" in op_names

	def test_y_basis_adds_sdg_and_hadamard(self) -> None:
		"""Test that Y-basis measurement adds Sdg and H gates before measurement."""
		qc = QuantumCircuit(1)
		qc.x(0)

		settings = [{0: "Y"}]
		result = get_obs_subcircuits([qc], settings)

		circuit = result[0][0]
		op_names = [inst.operation.name for inst in circuit]
		assert "sdg" in op_names
		assert "h" in op_names
		assert "measure" in op_names

	def test_existing_measurements_removed(self) -> None:
		"""Test that existing final measurements are removed before adding new ones."""
		qc = QuantumCircuit(2, 2)
		qc.h(0)
		qc.cx(0, 1)
		qc.measure([0, 1], [0, 1])

		settings = [{0: "Z"}]
		result = get_obs_subcircuits([qc], settings)

		circuit = result[0][0]
		# Should only have 1 measurement (from the Z setting), not the original 2
		measure_count = sum(1 for inst in circuit if inst.operation.name == "measure")
		assert measure_count == 1

	def test_multiple_settings_produce_multiple_circuits(self) -> None:
		"""Test that multiple measurement settings produce multiple circuit groups."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)

		settings = [{0: "Z"}, {0: "X"}]
		result = get_obs_subcircuits([qc], settings)

		assert len(result) == 2

	def test_multiple_subcircuits(self) -> None:
		"""Test with multiple input subcircuits."""
		qc1 = QuantumCircuit(2)
		qc1.h(0)
		qc2 = QuantumCircuit(2)
		qc2.x(0)

		settings = [{0: "Z", 1: "Z"}]
		result = get_obs_subcircuits([qc1, qc2], settings)

		assert len(result) == 1
		# Both subcircuits should be present
		assert 0 in result[0]
		assert 1 in result[0]

	def test_custom_ops_x_measurement(self) -> None:
		"""Test that custom X-meas instruction is used when provided."""
		qc = QuantumCircuit(1)
		qc.x(0)

		x_meas = QuantumCircuit(1)
		x_meas.h(0)
		x_meas_inst = x_meas.to_instruction(label="X-meas")

		settings = [{0: "X"}]
		result = get_obs_subcircuits([qc], settings, ops={"X-meas": x_meas_inst})

		circuit = result[0][0]
		assert circuit.num_clbits >= 1

	def test_custom_ops_y_measurement(self) -> None:
		"""Test that custom Y-meas instruction is used when provided."""
		qc = QuantumCircuit(1)
		qc.x(0)

		y_meas = QuantumCircuit(1)
		y_meas.sdg(0)
		y_meas.h(0)
		y_meas_inst = y_meas.to_instruction(label="Y-meas")

		settings = [{0: "Y"}]
		result = get_obs_subcircuits([qc], settings, ops={"Y-meas": y_meas_inst})

		circuit = result[0][0]
		assert circuit.num_clbits >= 1

	def test_unsupported_basis_raises_error(self) -> None:
		"""Test that unsupported measurement basis raises ValueError."""
		qc = QuantumCircuit(1)
		qc.x(0)

		settings = [{0: "W"}]

		with pytest.raises(ValueError, match="Unsupported measurement basis: W"):
			get_obs_subcircuits([qc], settings)
