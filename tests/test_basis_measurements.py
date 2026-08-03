"""Unit tests for basis_measurement module."""

import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit.converters import circuit_to_dag

from fiqci.ems.transpiler_passes.basis_measurement import (
	ModifyMeasurementBasis,
	get_obs_subcircuits,
	get_measurement_settings,
	_get_observable_circuit_index,
)


class TestGetMeasurementSettings:
	"""Tests for get_measurement_settings."""

	def test_single_pauli(self) -> None:
		"""Test combining a single Pauli operator."""
		op = SparsePauliOp.from_list([("ZZ", 1.0)])
		result = get_measurement_settings(op)

		assert len(result) == 1
		assert result[0] == {0: "Z", 1: "Z"}

	def test_compatible_paulis_are_combined(self) -> None:
		"""Test that compatible Pauli operators are combined into one setting."""
		# ZI and IZ have no conflicts -> should combine
		op = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0)])
		result = get_measurement_settings(op)

		assert len(result) == 1
		# Pauli labels are reversed internally, so ZI -> qubit 1: Z, IZ -> qubit 0: Z
		assert result[0] == {0: "Z", 1: "Z"}

	def test_conflicting_paulis_are_separate(self) -> None:
		"""Test that conflicting Pauli operators get separate settings."""
		# ZI and XI conflict on qubit 1
		op = SparsePauliOp.from_list([("ZI", 1.0), ("XI", 1.0)])
		result = get_measurement_settings(op)

		assert len(result) == 2

	def test_identity_only(self) -> None:
		"""Test a Pauli with only identity terms."""
		op = SparsePauliOp.from_list([("II", 1.0)])
		result = get_measurement_settings(op)

		assert len(result) == 1
		assert result[0] == {}  # No non-identity qubits

	def test_mixed_compatible_and_conflicting(self) -> None:
		"""Test a mix of compatible and conflicting operators."""
		# ZZ and ZI are compatible (both Z on qubit 1), XI conflicts with Z on qubit 1
		op = SparsePauliOp.from_list([("ZZ", 1.0), ("ZI", 1.0), ("XI", 1.0)])
		result = get_measurement_settings(op)

		# ZZ and ZI should combine, XI separate
		assert len(result) == 2

	def test_three_qubit_operators(self) -> None:
		"""Test with 3-qubit Pauli operators."""
		op = SparsePauliOp.from_list([("ZZI", 1.0), ("IIZ", 1.0)])
		result = get_measurement_settings(op)

		# No conflicts, should combine
		assert len(result) == 1
		assert result[0] == {0: "Z", 1: "Z", 2: "Z"}

	def test_all_bases(self) -> None:
		"""Test with X, Y, and Z bases."""
		op = SparsePauliOp.from_list([("XYZ", 1.0)])
		result = get_measurement_settings(op)

		assert len(result) == 1
		# XYZ reversed: Z on qubit 0, Y on qubit 1, X on qubit 2
		assert result[0] == {0: "Z", 1: "Y", 2: "X"}

	def test_same_basis_different_qubits_combine(self) -> None:
		"""Test that same basis on different qubits combines."""
		op = SparsePauliOp.from_list([("XI", 1.0), ("IX", 1.0)])
		result = get_measurement_settings(op)

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

		result = get_obs_subcircuits([qc], SparsePauliOp("ZZ"))

		assert len(result) == 1
		circuit = result[0][0]
		assert circuit.num_qubits == 2
		assert circuit.num_clbits == 2

	def test_x_basis_adds_hadamard(self) -> None:
		"""Test that X-basis measurement adds H gate before measurement."""
		qc = QuantumCircuit(1)
		qc.x(0)

		result = get_obs_subcircuits([qc], SparsePauliOp("X"))

		circuit = result[0][0]
		op_names = [inst.operation.name for inst in circuit]
		assert "h" in op_names
		assert "measure" in op_names

	def test_y_basis_adds_sdg_and_hadamard(self) -> None:
		"""Test that Y-basis measurement adds Sdg and H gates before measurement."""
		qc = QuantumCircuit(1)
		qc.x(0)

		result = get_obs_subcircuits([qc], SparsePauliOp("Y"))

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

		# IZ: only qubit 0 measured in Z — should produce exactly 1 measurement
		result = get_obs_subcircuits([qc], SparsePauliOp("IZ"))

		circuit = result[0][0]
		measure_count = sum(1 for inst in circuit if inst.operation.name == "measure")
		assert measure_count == 1

	def test_multiple_settings_produce_multiple_circuits(self) -> None:
		"""Test that conflicting observables produce multiple circuit groups."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)

		# IZ and IX conflict on qubit 0 → two groups
		result = get_obs_subcircuits([qc], SparsePauliOp(["IZ", "IX"]))

		assert len(result) == 2

	def test_multiple_subcircuits(self) -> None:
		"""Test with multiple input subcircuits."""
		qc1 = QuantumCircuit(2)
		qc1.h(0)
		qc2 = QuantumCircuit(2)
		qc2.x(0)

		result = get_obs_subcircuits([qc1, qc2], SparsePauliOp("ZZ"))

		assert len(result) == 1
		assert 0 in result[0]
		assert 1 in result[0]

	def test_custom_ops_x_measurement(self) -> None:
		"""Test that custom X-meas instruction is used when provided."""
		qc = QuantumCircuit(1)
		qc.x(0)

		x_meas = QuantumCircuit(1)
		x_meas.h(0)
		x_meas_inst = x_meas.to_instruction(label="X-meas")

		result = get_obs_subcircuits([qc], SparsePauliOp("X"), ops={"X-meas": x_meas_inst})

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

		result = get_obs_subcircuits([qc], SparsePauliOp("Y"), ops={"Y-meas": y_meas_inst})

		circuit = result[0][0]
		assert circuit.num_clbits >= 1

	def test_unsupported_basis_raises_error(self) -> None:
		"""Test that ModifyMeasurementBasis raises ValueError for unsupported basis."""
		qc = QuantumCircuit(1)
		qc.x(0)

		pass_obj = ModifyMeasurementBasis([{0: "W"}])
		with pytest.raises(ValueError, match="Unsupported measurement basis: W"):
			pass_obj.run(circuit_to_dag(qc))


class TestBasisMeasurementRegisterWiring:
	"""The basis measurements must land in the pass's own register, not the input circuit's.

	``dag.clbits`` starts with any classical bits the input circuit already had, so indexing it
	from 0 wrote the measurements into the user's register and left the new one empty, which made
	every parity read ``+1``.
	"""

	@staticmethod
	def _measure_targets(circuit) -> list[tuple[str, int]]:
		"""``(register_name, index)`` each measurement in ``circuit`` writes to."""
		targets = []
		for instruction in circuit.data:
			if instruction.operation.name == "measure":
				register, index = circuit.find_bit(instruction.clbits[0]).registers[0]
				targets.append((register.name, index))
		return targets

	def _bell_with_creg(self, num_clbits: int) -> QuantumCircuit:
		qc = QuantumCircuit(2, num_clbits)
		qc.h(0)
		qc.cx(0, 1)
		return qc

	@pytest.mark.parametrize("num_clbits", [1, 2, 5])
	def test_measures_into_own_register_not_the_input_circuits(self, num_clbits: int) -> None:
		"""An idle pre-existing register must not absorb the basis measurements."""
		circuit = get_obs_subcircuits([self._bell_with_creg(num_clbits)], SparsePauliOp(["ZZ"]))[0][0]

		assert self._measure_targets(circuit) == [("meas", 0), ("meas", 1)]

	def test_own_register_is_added_last(self) -> None:
		"""Being last makes its bits leftmost in the count keys, which the bit positions assume."""
		circuit = get_obs_subcircuits([self._bell_with_creg(2)], SparsePauliOp(["ZZ"]))[0][0]

		assert [register.name for register in circuit.cregs] == ["c", "meas"]

	def test_mid_circuit_measurement_is_preserved(self) -> None:
		"""A non-final measurement keeps its own target instead of being overwritten."""
		qc = QuantumCircuit(2)
		aux = ClassicalRegister(1, "aux")
		qc.add_register(aux)
		qc.h(0)
		qc.measure(0, aux[0])
		qc.cx(0, 1)

		circuit = get_obs_subcircuits([qc], SparsePauliOp(["ZZ"]))[0][0]

		assert self._measure_targets(circuit) == [("aux", 0), ("meas", 0), ("meas", 1)]

	def test_existing_meas_register_does_not_collide(self) -> None:
		"""A surviving register already named 'meas' previously raised DAGCircuitError."""
		qc = QuantumCircuit(2)
		qc.add_register(ClassicalRegister(2, "meas"))
		qc.h(0)
		qc.cx(0, 1)

		circuit = get_obs_subcircuits([qc], SparsePauliOp(["ZZ"]))[0][0]

		assert self._measure_targets(circuit) == [("meas1", 0), ("meas1", 1)]

	def test_unique_name_skips_every_taken_suffix(self) -> None:
		"""Naming walks past all taken names rather than stopping at the first suffix."""
		qc = QuantumCircuit(2)
		for name in ("meas", "meas1", "meas2"):
			qc.add_register(ClassicalRegister(1, name))
		qc.h(0)
		qc.cx(0, 1)

		circuit = get_obs_subcircuits([qc], SparsePauliOp(["ZZ"]))[0][0]

		assert self._measure_targets(circuit) == [("meas3", 0), ("meas3", 1)]
