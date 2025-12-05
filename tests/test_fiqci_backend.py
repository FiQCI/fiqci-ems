"""Unit tests for FiQCIBackend class."""

from unittest.mock import Mock, patch

import pytest
from qiskit import QuantumCircuit

from fiqci.ems.fiqci_backend import FiQCIBackend, MitigatedJob


class TestFiQCIBackend:
	"""Tests for FiQCIBackend class."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		"""Create a mock IQM backend."""
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	@pytest.fixture
	def mock_circuit(self) -> QuantumCircuit:
		"""Create a simple quantum circuit."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)
		qc.measure_all()
		return qc

	def test_init_with_valid_mitigation_level(self, mock_backend: Mock) -> None:
		"""Test initialization with valid mitigation level."""
		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1)
		assert mitigated_backend.mitigation_level == 1
		assert mitigated_backend.backend == mock_backend

	def test_init_with_invalid_mitigation_level_raises_error(self, mock_backend: Mock) -> None:
		"""Test initialization with invalid mitigation level raises ValueError."""
		with pytest.raises(ValueError, match="mitigation_level must be 0-3"):
			FiQCIBackend(mock_backend, mitigation_level=4)

	def test_init_creates_m3iqm_for_level_1(self, mock_backend: Mock) -> None:
		"""Test that M3IQM mitigator is created for level 1."""
		with patch("fiqci.ems.fiqci_backend.M3IQM") as mock_m3iqm:
			mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1)
			mock_m3iqm.assert_called_once_with(mock_backend)
			assert mitigated_backend._mitigator is not None

	def test_init_no_mitigator_for_level_0(self, mock_backend: Mock) -> None:
		"""Test that no mitigator is created for level 0."""
		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		assert mitigated_backend._mitigator is None

	def test_run_with_level_0_passes_through(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that level 0 passes through to backend without mitigation."""
		mock_job = Mock()
		mock_backend.run.return_value = mock_job

		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		result = mitigated_backend.run(mock_circuit, shots=1024)

		assert result == mock_job
		mock_backend.run.assert_called_once()

	def test_run_with_level_1_applies_mitigation(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that level 1 applies M3 mitigation."""
		# Setup mocks
		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.return_value = {"00": 500, "11": 500}
		mock_result.to_dict.return_value = {
			"results": [{"data": {"counts": {"00": 500, "11": 500}}, "shots": 1024, "success": True}],
			"backend_name": "mock",
			"job_id": "test-job-id",
			"qobj_id": "test-qobj-id",
			"success": True,
			"status": "COMPLETED",
		}
		mock_job.result.return_value = mock_result
		mock_backend.run.return_value = mock_job

		with (
			patch("fiqci.ems.fiqci_backend.M3IQM") as mock_m3iqm_class,
			patch("fiqci.ems.fiqci_backend.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.fiqci_backend.probabilities_to_counts", return_value=[{"00": 480, "11": 520}]),
		):
			mock_mitigator = Mock()
			mock_quasi_dist = Mock()
			mock_quasi_dist.nearest_probability_distribution.return_value = {"00": 0.48, "11": 0.52}
			mock_mitigator.apply_correction.return_value = mock_quasi_dist
			mock_mitigator.single_qubit_cals = None
			mock_m3iqm_class.return_value = mock_mitigator

			mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1, calibration_shots=1000)
			result = mitigated_backend.run(mock_circuit, shots=1024)

			# Verify calibration was called
			mock_mitigator.cals_from_system.assert_called_once()
			# Verify mitigation was applied
			mock_mitigator.apply_correction.assert_called_once()
			# Verify result is MitigatedJob
			assert isinstance(result, MitigatedJob)

	def test_run_with_circuit_list(self, mock_backend: Mock) -> None:
		"""Test running with list of circuits."""
		circuits = [QuantumCircuit(2), QuantumCircuit(2)]
		mock_backend.run.return_value = Mock()

		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		mitigated_backend.run(circuits, shots=1024)

		mock_backend.run.assert_called_once()
		# Verify circuits list was passed
		args = mock_backend.run.call_args[0]
		assert args[0] == circuits

	def test_run_with_empty_circuits_raises_error(self, mock_backend: Mock) -> None:
		"""Test that empty circuit list raises ValueError."""
		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=0)

		with pytest.raises(ValueError, match="No circuits provided"):
			mitigated_backend.run([], shots=1024)

	def test_run_with_level_2_raises_not_implemented(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that level 2 raises NotImplementedError."""
		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=2)

		with pytest.raises(NotImplementedError, match="Mitigation level 2 not yet implemented"):
			mitigated_backend.run(mock_circuit, shots=1024)

	def test_getattr_delegates_to_backend(self, mock_backend: Mock) -> None:
		"""Test that attribute access is delegated to underlying backend."""
		mock_backend.custom_attribute = "test_value"

		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=0)

		assert mitigated_backend.custom_attribute == "test_value"

	def test_calibration_shots_parameter(self, mock_backend: Mock) -> None:
		"""Test that calibration_shots parameter is stored."""
		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1, calibration_shots=2048)

		assert mitigated_backend._calibration_shots == 2048

	def test_run_calibrates_only_once(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that M3 calibration happens only once, even for multiple runs."""
		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.return_value = {"00": 500, "11": 500}
		mock_result.to_dict.return_value = {
			"results": [{"data": {"counts": {"00": 500, "11": 500}}, "shots": 1024, "success": True}],
			"backend_name": "mock",
			"job_id": "test-job-id",
			"qobj_id": "test-qobj-id",
			"success": True,
			"status": "COMPLETED",
		}
		mock_job.result.return_value = mock_result
		mock_backend.run.return_value = mock_job

		with (
			patch("fiqci.ems.fiqci_backend.M3IQM") as mock_m3iqm_class,
			patch("fiqci.ems.fiqci_backend.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.fiqci_backend.probabilities_to_counts", return_value=[{"00": 480, "11": 520}]),
		):
			mock_mitigator = Mock()
			mock_quasi_dist = Mock()
			mock_quasi_dist.nearest_probability_distribution.return_value = {"00": 0.48, "11": 0.52}
			mock_mitigator.apply_correction.return_value = mock_quasi_dist
			# First run: no calibration yet
			mock_mitigator.single_qubit_cals = None
			mock_m3iqm_class.return_value = mock_mitigator

			mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1)

			# First run should calibrate
			mitigated_backend.run(mock_circuit, shots=1024)
			assert mock_mitigator.cals_from_system.call_count == 1

			# Second run should NOT calibrate again
			mock_mitigator.single_qubit_cals = [Mock()]  # Simulate already calibrated
			mitigated_backend.run(mock_circuit, shots=1024)
			# Still only 1 call from first run
			assert mock_mitigator.cals_from_system.call_count == 1


class TestMitigatedJob:
	"""Tests for MitigatedJob class."""

	def test_result_returns_mitigated_result(self) -> None:
		"""Test that result() returns the mitigated result."""
		mock_original_job = Mock()
		mock_mitigated_result = Mock()

		mitigated_job = MitigatedJob(mock_original_job, mock_mitigated_result)

		assert mitigated_job.result() == mock_mitigated_result

	def test_getattr_delegates_to_original_job(self) -> None:
		"""Test that attribute access is delegated to original job."""
		mock_original_job = Mock()
		mock_original_job.job_id = "test-job-123"
		mock_mitigated_result = Mock()

		mitigated_job = MitigatedJob(mock_original_job, mock_mitigated_result)

		assert mitigated_job.job_id == "test-job-123"

	def test_result_ignores_timeout_parameter(self) -> None:
		"""Test that result() accepts but ignores timeout parameter."""
		mock_original_job = Mock()
		mock_mitigated_result = Mock()

		mitigated_job = MitigatedJob(mock_original_job, mock_mitigated_result)

		# Should not raise error
		result = mitigated_job.result(timeout=10.0)
		assert result == mock_mitigated_result
