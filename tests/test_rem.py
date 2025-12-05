"""Unit tests for Readout Error Mitigation (REM) functionality."""

import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from mthree.exceptions import M3Error
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2

from fiqci.ems.rem import M3IQM, balanced_cal_strings


class TestBalancedCalStrings:
	"""Tests for balanced_cal_strings function."""

	def test_balanced_cal_strings_single_qubit_generates_correct_strings(self) -> None:
		"""Test that single qubit generates ['0', '1']."""
		result = balanced_cal_strings(1)
		assert result == ["0", "1"]

	def test_balanced_cal_strings_two_qubits_generates_correct_strings(self) -> None:
		"""Test that two qubits generate all 4 combinations."""
		result = balanced_cal_strings(2)
		assert result == ["00", "01", "10", "11"]

	def test_balanced_cal_strings_three_qubits_generates_correct_strings(self) -> None:
		"""Test that three qubits generate all 8 combinations."""
		result = balanced_cal_strings(3)
		assert result == ["000", "001", "010", "011", "100", "101", "110", "111"]

	def test_balanced_cal_strings_zero_qubits_raises_error(self) -> None:
		"""Test that zero qubits raises ValueError."""
		with pytest.raises(ValueError, match="Number of qubits must be at least 1"):
			balanced_cal_strings(0)

	def test_balanced_cal_strings_negative_qubits_raises_error(self) -> None:
		"""Test that negative qubits raises ValueError."""
		with pytest.raises(ValueError, match="Number of qubits must be at least 1"):
			balanced_cal_strings(-1)

	def test_balanced_cal_strings_length_is_power_of_two(self) -> None:
		"""Test that result length is 2^num_qubits."""
		for num_qubits in [1, 2, 3, 4, 5]:
			result = balanced_cal_strings(num_qubits)
			assert len(result) == 2**num_qubits

	def test_balanced_cal_strings_all_unique(self) -> None:
		"""Test that all strings are unique."""
		result = balanced_cal_strings(4)
		assert len(result) == len(set(result))


class TestM3IQM:
	"""Tests for M3IQM class."""

	@pytest.fixture
	def mock_backend(self) -> Iterator[Mock]:
		"""Create a mock IQM backend."""
		backend = Mock(spec=BackendV2)
		backend.num_qubits = 5
		backend.max_circuits = 300
		backend.run = Mock(return_value=Mock())
		yield backend

	@pytest.fixture
	def mock_system_info(self) -> dict[str, Any]:
		"""Create mock system info."""
		return {"max_shots": 10000, "max_circuits": 100, "inoperable_qubits": [], "num_qubits": 5}

	@pytest.fixture
	def m3iqm_instance(self, mock_backend: Mock, mock_system_info: dict[str, Any]) -> Iterator[M3IQM]:
		"""Create an M3IQM instance with mocked backend."""
		with (
			patch.object(M3IQM, "__init__", return_value=None),
			patch.object(M3IQM, "_grab_additional_cals", return_value=None),
		):
			instance = M3IQM.__new__(M3IQM)
			instance.system = mock_backend
			instance.system_info = mock_system_info
			instance.num_qubits = 5
			instance._thread = None
			instance.single_qubit_cals = None
			instance.cal_shots = None
			instance.rep_delay = None
			instance.cal_method = None
			instance.cals_file = None
			instance.cal_timestamp = None
			yield instance

	@pytest.fixture
	def m3iqm_real_grab_cals(self, mock_backend: Mock, mock_system_info: dict[str, Any]) -> Iterator[M3IQM]:
		"""Create an M3IQM instance with real _grab_additional_cals for testing."""
		with patch.object(M3IQM, "__init__", return_value=None):
			instance = M3IQM.__new__(M3IQM)
			instance.system = mock_backend
			instance.system_info = mock_system_info
			instance.num_qubits = 5
			instance._thread = None
			instance._job_error = None
			instance.single_qubit_cals = None
			instance.cal_shots = None
			instance.rep_delay = None
			instance.cal_method = None
			instance.cals_file = None
			instance.cal_timestamp = None
			yield instance

	def test_cals_from_system_with_thread_running_raises_error(self, m3iqm_instance: M3IQM) -> None:
		"""Test that calling cals_from_system while calibration is running raises error."""
		m3iqm_instance._thread = Mock(spec=threading.Thread)

		with pytest.raises(M3Error, match="Calibration currently in progress"):
			m3iqm_instance.cals_from_system()

	def test_cals_from_system_with_no_qubits_uses_all_qubits(self, m3iqm_instance: M3IQM) -> None:
		"""Test that None qubits parameter uses all available qubits."""
		with patch.object(m3iqm_instance, "_grab_additional_cals") as mock_grab:
			m3iqm_instance.cals_from_system(qubits=None)
			# Should be called with range(5)
			called_qubits = mock_grab.call_args[0][0]
			assert list(called_qubits) == list(range(5))

	def test_cals_from_system_skips_inoperable_qubits(self, m3iqm_instance: M3IQM) -> None:
		"""Test that inoperable qubits are filtered out."""
		m3iqm_instance.system_info["inoperable_qubits"] = [1, 3]

		with (
			patch.object(m3iqm_instance, "_grab_additional_cals") as mock_grab,
			pytest.warns(UserWarning, match="inoperable qubits"),
		):
			m3iqm_instance.cals_from_system(qubits=None)
			called_qubits = mock_grab.call_args[0][0]
			assert 1 not in called_qubits
			assert 3 not in called_qubits
			assert set(called_qubits) == {0, 2, 4}

	def test_cals_from_system_sets_default_method_to_balanced(self, m3iqm_instance: M3IQM) -> None:
		"""Test that default calibration method is 'balanced' for IQM."""
		with patch.object(m3iqm_instance, "_grab_additional_cals"):
			m3iqm_instance.cals_from_system(qubits=[0, 1])
			assert m3iqm_instance.cal_method == "balanced"

	def test_cals_from_system_sets_specified_method(self, m3iqm_instance: M3IQM) -> None:
		"""Test that specified calibration method is used."""
		with patch.object(m3iqm_instance, "_grab_additional_cals"):
			m3iqm_instance.cals_from_system(qubits=[0, 1], method="independent")
			assert m3iqm_instance.cal_method == "independent"

	def test_cals_from_system_resets_cal_timestamp(self, m3iqm_instance: M3IQM) -> None:
		"""Test that calibration timestamp is reset."""
		m3iqm_instance.cal_timestamp = "old_timestamp"
		with patch.object(m3iqm_instance, "_grab_additional_cals"):
			m3iqm_instance.cals_from_system(qubits=[0, 1])
			assert m3iqm_instance.cal_timestamp is None

	def test_grab_additional_cals_without_system_raises_error(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that _grab_additional_cals without system raises error."""
		m3iqm_real_grab_cals.system = None

		with pytest.raises(M3Error, match="System is not set"):
			m3iqm_real_grab_cals._grab_additional_cals([0, 1])

	def test_grab_additional_cals_with_invalid_method_raises_error(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that invalid calibration method raises error."""
		with pytest.raises(M3Error, match="Invalid calibration method"):
			m3iqm_real_grab_cals._grab_additional_cals([0, 1], method="invalid_method")

	def test_grab_additional_cals_with_inoperable_qubits_raises_error(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that calibrating inoperable qubits raises error."""
		m3iqm_real_grab_cals.system_info["inoperable_qubits"] = [1, 2]
		m3iqm_real_grab_cals.single_qubit_cals = [None] * 5
		m3iqm_real_grab_cals.cal_shots = 1000

		with pytest.raises(M3Error, match="Attempting to calibrate inoperable qubits"):
			m3iqm_real_grab_cals._grab_additional_cals([1, 3])

	def test_grab_additional_cals_initializes_single_qubit_cals_if_none(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that single_qubit_cals is initialized if None."""
		m3iqm_real_grab_cals.single_qubit_cals = None
		m3iqm_real_grab_cals.cal_shots = 1000

		mock_circuit = Mock(spec=QuantumCircuit)
		with (
			patch("mthree.circuits._marg_meas_states", return_value=[mock_circuit]),
			patch("mthree.mitigation._job_thread"),
		):
			m3iqm_real_grab_cals._grab_additional_cals([0, 1], method="marginal")
			assert m3iqm_real_grab_cals.single_qubit_cals == [None] * 5

	def test_grab_additional_cals_sets_default_shots(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that default shots is set to min(max_shots, 10000)."""
		m3iqm_real_grab_cals.single_qubit_cals = [None] * 5
		m3iqm_real_grab_cals.cal_shots = None

		mock_circuit = Mock(spec=QuantumCircuit)
		with (
			patch("mthree.circuits._marg_meas_states", return_value=[mock_circuit]),
			patch("mthree.mitigation._job_thread"),
		):
			m3iqm_real_grab_cals._grab_additional_cals([0, 1], method="marginal")
			assert m3iqm_real_grab_cals.cal_shots == 10000

	def test_grab_additional_cals_respects_max_shots_limit(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that shots respects backend max_shots limit."""
		m3iqm_real_grab_cals.single_qubit_cals = [None] * 5
		m3iqm_real_grab_cals.cal_shots = None
		m3iqm_real_grab_cals.system_info["max_shots"] = 5000

		mock_circuit = Mock(spec=QuantumCircuit)
		with (
			patch("mthree.circuits._marg_meas_states", return_value=[mock_circuit]),
			patch("mthree.mitigation._job_thread"),
		):
			m3iqm_real_grab_cals._grab_additional_cals([0, 1], method="marginal")
			assert m3iqm_real_grab_cals.cal_shots == 5000

	@pytest.mark.parametrize("method", ["marginal", "balanced", "independent"])
	def test_grab_additional_cals_accepts_valid_methods(self, m3iqm_real_grab_cals: M3IQM, method: str) -> None:
		"""Test that all valid calibration methods are accepted."""
		m3iqm_real_grab_cals.single_qubit_cals = [None] * 5
		m3iqm_real_grab_cals.cal_shots = 1000

		mock_circuit = Mock(spec=QuantumCircuit)
		with (
			patch("mthree.circuits._marg_meas_states", return_value=[mock_circuit]),
			patch("fiqci.ems.rem.balanced_cal_strings", return_value=["00", "01", "10", "11"]),
			patch("mthree.circuits.balanced_cal_circuits", return_value=[mock_circuit]),
			patch("mthree.circuits._tensor_meas_states", return_value=[mock_circuit]),
			patch("mthree.mitigation._job_thread"),
		):
			# Should not raise
			m3iqm_real_grab_cals._grab_additional_cals([0, 1], method=method)


class TestIntegration:
	"""Integration tests for REM module."""

	def test_balanced_cal_strings_used_in_m3iqm_workflow(self) -> None:
		"""Test that balanced_cal_strings integrates correctly with M3IQM."""
		# This tests that the function signature and return type are compatible
		strings = balanced_cal_strings(2)
		assert all(isinstance(s, str) for s in strings)
		assert all(len(s) == 2 for s in strings)
		assert all(c in "01" for s in strings for c in s)

	def test_mitigation_reduces_errors_on_bell_state(self) -> None:
		"""Test that mitigation actually reduces errors on a Bell state.

		Simulates a Bell state experiment with realistic readout errors and
		verifies that mitigation increases the correct state populations.

		This test manually sets calibration matrices and verifies that applying
		mitigation to noisy counts improves the fidelity toward the ideal state.
		"""
		# Create a simple mock backend
		backend = Mock()
		backend.version = 2
		backend.name = "mock_iqm_backend"
		backend.num_qubits = 20
		config = Mock()
		config.num_qubits = 20
		config.max_shots = 10000
		config.simulator = False
		backend.configuration.return_value = config

		# Create M3 mitigator
		m3 = M3IQM(backend)

		# Initialize calibration list
		m3.single_qubit_cals = [None] * 20

		# Manually set calibration matrices (realistic 5% readout error)
		# P(measure i | prepared j) for each qubit
		m3.single_qubit_cals[8] = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)
		m3.single_qubit_cals[16] = np.array([[0.94, 0.04], [0.06, 0.96]], dtype=np.float32)

		# Simulate noisy Bell state: ideal is 50% |00⟩ + 50% |11⟩
		# With ~5% error per qubit, we get ~150 error counts out of 2000
		raw_counts = {
			"00": 900,  # Should be ~1000 (correct state)
			"01": 75,  # Error: measured 01 instead of 00 or 11
			"10": 75,  # Error: measured 10 instead of 00 or 11
			"11": 950,  # Should be ~1000 (correct state)
		}

		# Apply mitigation
		quasi_dist = m3.apply_correction(raw_counts, qubits=[8, 16])

		# Get nearest probability distribution and convert to counts
		mitigated_probs = quasi_dist.nearest_probability_distribution()
		mitigated_counts = {state: int(prob * 2000) for state, prob in mitigated_probs.items()}

		# Verify that mitigation improved the results
		# Count correct states (00 and 11)
		raw_correct = raw_counts.get("00", 0) + raw_counts.get("11", 0)
		mitigated_correct = mitigated_counts.get("00", 0) + mitigated_counts.get("11", 0)

		# Mitigation should increase correct states significantly
		# From 1850 raw to at least 1950 mitigated (>100 count improvement)
		assert mitigated_correct > raw_correct, (
			f"Mitigation should increase correct states: {raw_correct} → {mitigated_correct}"
		)
		assert mitigated_correct >= 1950, f"Mitigated should be >=1950 correct counts, got {mitigated_correct}"
