"""Unit tests for Readout Error Mitigation (REM) functionality."""

import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock, patch

import pytest
from mthree.classes import QuasiDistribution
from mthree.exceptions import M3Error
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2

from fiqci.ems.rem import M3IQM, apply_readout_error_mitigation, balanced_cal_strings, readout_error_m3


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
		return {"max_shots": 10000, "inoperable_qubits": [], "num_qubits": 5}

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
		"""Test that default calibration method is 'balanced'."""
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

	def test_grab_additional_cals_converts_dict_qubits_to_list(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that dict of qubits is converted to list of unique values."""
		m3iqm_real_grab_cals.single_qubit_cals = [None] * 5
		m3iqm_real_grab_cals.cal_shots = 1000

		mock_circuit = Mock(spec=QuantumCircuit)
		with patch("fiqci.ems.rem._marg_meas_states", return_value=[mock_circuit]) as mock_marg:
			with patch("fiqci.ems.rem.transpile", return_value=mock_circuit):
				with patch("fiqci.ems.rem._job_thread"):
					m3iqm_real_grab_cals._grab_additional_cals({"q0": 0, "q1": 1, "q2": 2}, method="marginal")
					# Check that qubit list contains the unique values
					called_qubits = mock_marg.call_args[0][0]
					assert set(called_qubits) == {0, 1, 2}

	def test_grab_additional_cals_handles_list_of_dicts(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that list of dict mappings is converted correctly."""
		m3iqm_real_grab_cals.single_qubit_cals = [None] * 5
		m3iqm_real_grab_cals.cal_shots = 1000

		mock_circuit = Mock(spec=QuantumCircuit)
		with patch("fiqci.ems.rem._marg_meas_states", return_value=[mock_circuit]) as mock_marg:
			with patch("fiqci.ems.rem.transpile", return_value=mock_circuit):
				with patch("fiqci.ems.rem._job_thread"):
					qubits = [{"q0": 0, "q1": 1}, {"q0": 2, "q1": 3}]
					m3iqm_real_grab_cals._grab_additional_cals(qubits, method="marginal")
					# Should extract unique qubits from all dicts
					called_qubits = mock_marg.call_args[0][0]
					assert set(called_qubits) == {0, 1, 2, 3}

	def test_grab_additional_cals_initializes_single_qubit_cals_if_none(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that single_qubit_cals is initialized if None."""
		m3iqm_real_grab_cals.single_qubit_cals = None
		m3iqm_real_grab_cals.cal_shots = 1000

		mock_circuit = Mock(spec=QuantumCircuit)
		with patch("fiqci.ems.rem._marg_meas_states", return_value=[mock_circuit]):
			with patch("fiqci.ems.rem.transpile", return_value=mock_circuit):
				with patch("fiqci.ems.rem._job_thread"):
					m3iqm_real_grab_cals._grab_additional_cals([0, 1], method="marginal")
					assert m3iqm_real_grab_cals.single_qubit_cals == [None] * 5

	def test_grab_additional_cals_sets_default_shots(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that default shots is set to min(max_shots, 10000)."""
		m3iqm_real_grab_cals.single_qubit_cals = [None] * 5
		m3iqm_real_grab_cals.cal_shots = None

		mock_circuit = Mock(spec=QuantumCircuit)
		with patch("fiqci.ems.rem._marg_meas_states", return_value=[mock_circuit]):
			with patch("fiqci.ems.rem.transpile", return_value=mock_circuit):
				with patch("fiqci.ems.rem._job_thread"):
					m3iqm_real_grab_cals._grab_additional_cals([0, 1], method="marginal")
					assert m3iqm_real_grab_cals.cal_shots == 10000

	def test_grab_additional_cals_respects_max_shots_limit(self, m3iqm_real_grab_cals: M3IQM) -> None:
		"""Test that shots respects backend max_shots limit."""
		m3iqm_real_grab_cals.single_qubit_cals = [None] * 5
		m3iqm_real_grab_cals.cal_shots = None
		m3iqm_real_grab_cals.system_info["max_shots"] = 5000

		mock_circuit = Mock(spec=QuantumCircuit)
		with patch("fiqci.ems.rem._marg_meas_states", return_value=[mock_circuit]):
			with patch("fiqci.ems.rem.transpile", return_value=mock_circuit):
				with patch("fiqci.ems.rem._job_thread"):
					m3iqm_real_grab_cals._grab_additional_cals([0, 1], method="marginal")
					assert m3iqm_real_grab_cals.cal_shots == 5000

	@pytest.mark.parametrize("method", ["marginal", "balanced", "independent"])
	def test_grab_additional_cals_accepts_valid_methods(self, m3iqm_real_grab_cals: M3IQM, method: str) -> None:
		"""Test that all valid calibration methods are accepted."""
		m3iqm_real_grab_cals.single_qubit_cals = [None] * 5
		m3iqm_real_grab_cals.cal_shots = 1000

		mock_circuit = Mock(spec=QuantumCircuit)
		with (
			patch("fiqci.ems.rem._marg_meas_states", return_value=[mock_circuit]),
			patch("fiqci.ems.rem.balanced_cal_strings", return_value=["00", "01", "10", "11"]),
			patch("fiqci.ems.rem.balanced_cal_circuits", return_value=[mock_circuit]),
			patch("fiqci.ems.rem._tensor_meas_states", return_value=[mock_circuit]),
			patch("fiqci.ems.rem.transpile", return_value=mock_circuit),
			patch("fiqci.ems.rem._job_thread"),
		):
			# Should not raise
			m3iqm_real_grab_cals._grab_additional_cals([0, 1], method=method)


class TestReadoutErrorM3:
	"""Tests for readout_error_m3 function."""

	def test_readout_error_m3_returns_quasidistribution(self) -> None:
		"""Test that readout_error_m3 returns a QuasiDistribution object."""
		mock_mit = Mock(spec=M3IQM)
		# Create a proper QuasiDistribution mock return value
		quasi_dist = QuasiDistribution({"00": 0.95, "11": 0.05})
		mock_mit.apply_correction = Mock(return_value=quasi_dist)

		counts = {"00": 900, "11": 100}
		qubits = [0, 1]

		result = readout_error_m3(counts, mock_mit, qubits)

		mock_mit.apply_correction.assert_called_once_with(counts, qubits)
		# Verify the result is a dict-like object (QuasiDistribution inherits from dict)
		assert isinstance(result, dict)
		assert "00" in result
		assert "11" in result

	def test_readout_error_m3_with_dict_qubits(self) -> None:
		"""Test that readout_error_m3 works with dict qubits mapping."""
		mock_mit = Mock(spec=M3IQM)
		quasi_dist = QuasiDistribution({"00": 0.98, "01": 0.02})
		mock_mit.apply_correction = Mock(return_value=quasi_dist)

		counts = {"00": 980, "01": 20}
		qubits = {"q0": 0, "q1": 1}

		result = readout_error_m3(counts, mock_mit, qubits)

		mock_mit.apply_correction.assert_called_once_with(counts, qubits)
		assert isinstance(result, dict)
		assert "00" in result
		assert "01" in result


class TestApplyReadoutErrorMitigation:
	"""Tests for apply_readout_error_mitigation function."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		"""Create a mock IQM backend."""
		backend = Mock()
		backend.num_qubits = 3
		return backend

	@pytest.fixture
	def mock_circuits(self) -> list[QuantumCircuit]:
		"""Create mock quantum circuits."""
		circuit1 = QuantumCircuit(2)
		circuit1.h(0)
		circuit1.cx(0, 1)
		circuit1.measure_all()

		circuit2 = QuantumCircuit(2)
		circuit2.h(0)
		circuit2.measure_all()

		return [circuit1, circuit2]

	@pytest.fixture
	def mock_counts(self) -> list[dict[str, int]]:
		"""Create mock measurement counts."""
		return [{"00": 500, "11": 500}, {"0": 600, "1": 400}]

	def test_apply_readout_error_mitigation_accepts_iqm_backend(
		self, mock_backend: Mock, mock_circuits: list[QuantumCircuit], mock_counts: list[dict[str, int]]
	) -> None:
		"""Test that IQMBackendBase instance is accepted."""
		with (
			patch("fiqci.ems.rem.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.rem.M3IQM") as mock_m3iqm_class,
		):
			mock_mit = Mock()
			mock_mit.apply_correction = Mock(return_value=Mock())
			mock_m3iqm_class.return_value = mock_mit

			# Should not raise
			apply_readout_error_mitigation(mock_backend, mock_circuits, mock_counts)

	def test_apply_readout_error_mitigation_creates_m3iqm_instance(
		self, mock_backend: Mock, mock_circuits: list[QuantumCircuit], mock_counts: list[dict[str, int]]
	) -> None:
		"""Test that M3IQM instance is created with backend."""
		with (
			patch("fiqci.ems.rem.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.rem.M3IQM") as mock_m3iqm_class,
		):
			mock_mit = Mock()
			mock_mit.apply_correction = Mock(return_value=Mock())
			mock_m3iqm_class.return_value = mock_mit

			apply_readout_error_mitigation(mock_backend, mock_circuits, mock_counts, mit_shots=2000)

			mock_m3iqm_class.assert_called_once_with(mock_backend)
			mock_mit.cals_from_system.assert_called_once()

	def test_apply_readout_error_mitigation_calls_cals_from_system(
		self, mock_backend: Mock, mock_circuits: list[QuantumCircuit], mock_counts: list[dict[str, int]]
	) -> None:
		"""Test that calibration is performed with correct parameters."""
		with (
			patch("fiqci.ems.rem.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.rem.M3IQM") as mock_m3iqm_class,
		):
			mock_mit = Mock()
			mock_mit.apply_correction = Mock(return_value=Mock())
			mock_m3iqm_class.return_value = mock_mit

			apply_readout_error_mitigation(mock_backend, mock_circuits, mock_counts, mit_shots=3000)

			# Check that cals_from_system was called with correct shots
			call_kwargs = mock_mit.cals_from_system.call_args[1]
			assert call_kwargs["shots"] == 3000

	def test_apply_readout_error_mitigation_applies_correction_to_all_circuits(
		self, mock_backend: Mock, mock_circuits: list[QuantumCircuit], mock_counts: list[dict[str, int]]
	) -> None:
		"""Test that correction is applied to all circuits."""
		with (
			patch("fiqci.ems.rem.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.rem.M3IQM") as mock_m3iqm_class,
		):
			mock_mit = Mock()
			mock_mit.apply_correction = Mock(return_value=Mock())
			mock_m3iqm_class.return_value = mock_mit

			result = apply_readout_error_mitigation(mock_backend, mock_circuits, mock_counts)

			# Should call apply_correction for each circuit/counts pair
			assert mock_mit.apply_correction.call_count == len(mock_circuits)
			assert len(result) == len(mock_circuits)

	def test_apply_readout_error_mitigation_uses_default_shots(
		self, mock_backend: Mock, mock_circuits: list[QuantumCircuit], mock_counts: list[dict[str, int]]
	) -> None:
		"""Test that default mit_shots value is used."""
		with (
			patch("fiqci.ems.rem.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.rem.M3IQM") as mock_m3iqm_class,
		):
			mock_mit = Mock()
			mock_mit.apply_correction = Mock(return_value=Mock())
			mock_m3iqm_class.return_value = mock_mit

			apply_readout_error_mitigation(mock_backend, mock_circuits, mock_counts)

			# Check that default 1000 shots is used
			call_kwargs = mock_mit.cals_from_system.call_args[1]
			assert call_kwargs["shots"] == 1000


class TestIntegration:
	"""Integration tests for REM module."""

	def test_balanced_cal_strings_used_in_m3iqm_workflow(self) -> None:
		"""Test that balanced_cal_strings integrates correctly with M3IQM."""
		# This tests that the function signature and return type are compatible
		strings = balanced_cal_strings(2)
		assert all(isinstance(s, str) for s in strings)
		assert all(len(s) == 2 for s in strings)
		assert all(c in "01" for s in strings for c in s)
