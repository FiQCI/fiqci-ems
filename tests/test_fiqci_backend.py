"""Unit tests for FiQCIBackend class."""

from unittest.mock import Mock, patch

import pytest
from qiskit import QuantumCircuit
from qiskit.providers import JobStatus

from fiqci.ems.backend import BatchFailedError, BatchedJob, FiQCIBackend, MitigatedJob


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
		with patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm:
			mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1)
			mock_m3iqm.assert_called_once_with(mock_backend)
			assert mitigated_backend._rem["mitigator"] is not None

	def test_init_no_mitigator_for_level_0(self, mock_backend: Mock) -> None:
		"""Test that no mitigator is created for level 0."""
		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		assert mitigated_backend._rem["mitigator"] is None

	def test_run_with_level_0_passes_through(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that level 0 wraps the backend job in a lazy BatchedJob without mitigation."""
		mock_job = _make_result_mock([{"00": 1024}])
		mock_backend.run.return_value = mock_job

		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		result = mitigated_backend.run(mock_circuit, shots=1024)

		assert isinstance(result, BatchedJob)
		assert result.job_ids() == [mock_job.job_id()]
		mock_backend.run.assert_called_once()
		# run() must not block on results.
		mock_job.result.assert_not_called()

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
			patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm_class,
			patch("fiqci.ems.backend.core.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.backend.core.probabilities_to_counts", return_value=[{"00": 480, "11": 520}]),
		):
			mock_mitigator = Mock()
			mock_quasi_dist = Mock()
			mock_quasi_dist.nearest_probability_distribution.return_value = {"00": 0.48, "11": 0.52}
			mock_mitigator.apply_correction.return_value = mock_quasi_dist
			mock_mitigator.single_qubit_cals = None
			mock_m3iqm_class.return_value = mock_mitigator

			mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1, calibration_shots=1000)
			result = mitigated_backend.run(mock_circuit, shots=1024)

			# Calibration is kicked off eagerly in run() so it runs in parallel with the jobs.
			mock_mitigator.cals_from_system.assert_called_once()
			# Verify result is MitigatedJob
			assert isinstance(result, MitigatedJob)
			# Mitigation is lazy: not applied until result() is requested.
			mock_mitigator.apply_correction.assert_not_called()

			result.result()
			# Now mitigation has been applied.
			mock_mitigator.apply_correction.assert_called_once()

	def test_disabling_rem_after_run_does_not_break_deferred_mitigation(
		self, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		"""Settings snapshot: disabling REM between run() and result() must not break post-processing.

		``rem(enabled=False)`` clears ``self._rem["mitigator"]``; because the mitigator is snapshotted
		at submission time, the deferred M3 correction still runs against the originally-configured
		mitigator instead of raising.
		"""
		mock_job = _make_result_mock([{"00": 500, "11": 500}])
		mock_backend.run.return_value = mock_job

		with (
			patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm_class,
			patch("fiqci.ems.backend.core.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.backend.core.probabilities_to_counts", return_value=[{"00": 480, "11": 520}]),
		):
			mock_mitigator = Mock()
			mock_quasi_dist = Mock()
			mock_quasi_dist.nearest_probability_distribution.return_value = {"00": 0.48, "11": 0.52}
			mock_mitigator.apply_correction.return_value = mock_quasi_dist
			mock_mitigator.single_qubit_cals = None
			mock_m3iqm_class.return_value = mock_mitigator

			mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1)
			result = mitigated_backend.run(mock_circuit, shots=1024)

			# User disables REM after submission (clears the mitigator on the backend).
			mitigated_backend.rem(enabled=False)
			assert mitigated_backend._rem["mitigator"] is None

			# Deferred correction still runs against the snapshotted mitigator, without raising.
			result.result()
			mock_mitigator.apply_correction.assert_called_once()

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

	def test_run_with_level_4_raises_value_error(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that level 4 raises ValueError."""
		with pytest.raises(ValueError, match="mitigation_level must be 0-3, got 4"):
			_mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=4)

	def test_getattr_delegates_to_backend(self, mock_backend: Mock) -> None:
		"""Test that attribute access is delegated to underlying backend."""
		mock_backend.custom_attribute = "test_value"

		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=0)

		assert mitigated_backend.custom_attribute == "test_value"

	def test_calibration_shots_parameter(self, mock_backend: Mock) -> None:
		"""Test that calibration_shots parameter is stored."""
		mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1, calibration_shots=2048)

		assert mitigated_backend._rem["calibration_shots"] == 2048

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
			patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm_class,
			patch("fiqci.ems.backend.core.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.backend.core.probabilities_to_counts", return_value=[{"00": 480, "11": 520}]),
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


class TestREMSettings:
	"""Tests for changing REM settings via the rem() method."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		"""Create a mock IQM backend."""
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	@pytest.fixture
	def backend_level0(self, mock_backend: Mock) -> FiQCIBackend:
		"""Create a FiQCIBackend with mitigation level 0 (no mitigation)."""
		return FiQCIBackend(mock_backend, mitigation_level=0)

	@pytest.fixture
	def backend_level1(self, mock_backend: Mock) -> FiQCIBackend:
		"""Create a FiQCIBackend with mitigation level 1."""
		with patch("fiqci.ems.backend.core.M3IQM"):
			return FiQCIBackend(mock_backend, mitigation_level=1)

	def test_rem_enable_on_level0(self, mock_backend: Mock, backend_level0: FiQCIBackend) -> None:
		"""Test enabling REM on a level 0 backend."""
		with patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm:
			backend_level0.rem(enabled=True, calibration_shots=500)
			assert backend_level0._rem["enabled"] is True
			assert backend_level0._rem["calibration_shots"] == 500
			mock_m3iqm.assert_called_once_with(mock_backend)

	def test_rem_disable(self, backend_level1: FiQCIBackend) -> None:
		"""Test disabling REM clears mitigator."""
		assert backend_level1._rem["enabled"] is True
		backend_level1.rem(enabled=False)
		assert backend_level1._rem["enabled"] is False
		assert backend_level1._rem["mitigator"] is None

	def test_rem_enable_after_disable(self, mock_backend: Mock, backend_level1: FiQCIBackend) -> None:
		"""Test re-enabling REM after disabling it."""
		backend_level1.rem(enabled=False)
		assert backend_level1._rem["enabled"] is False

		with patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm:
			backend_level1.rem(enabled=True, calibration_shots=2048)
			assert backend_level1._rem["enabled"] is True
			assert backend_level1._rem["calibration_shots"] == 2048
			mock_m3iqm.assert_called_once_with(mock_backend)

	def test_rem_change_calibration_shots_reinitializes(self, mock_backend: Mock, backend_level1: FiQCIBackend) -> None:
		"""Test that changing calibration_shots triggers reinitialization."""
		with patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm:
			backend_level1.rem(enabled=True, calibration_shots=4096)
			assert backend_level1._rem["calibration_shots"] == 4096
			mock_m3iqm.assert_called_once_with(mock_backend)

	def test_rem_change_calibration_file_reinitializes(self, mock_backend: Mock, backend_level1: FiQCIBackend) -> None:
		"""Test that changing calibration_file triggers reinitialization."""
		with patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm:
			backend_level1.rem(enabled=True, calibration_file="/tmp/new_cal.json")
			assert backend_level1._rem["calibration_file"] == "/tmp/new_cal.json"
			mock_m3iqm.assert_called_once_with(mock_backend)

	def test_rem_same_settings_does_not_reinitialize(self, backend_level1: FiQCIBackend) -> None:
		"""Test that calling rem() with same settings does not reinitialize."""
		original_mitigator = backend_level1._rem["mitigator"]
		with patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm:
			backend_level1.rem(enabled=True)
			mock_m3iqm.assert_not_called()
			assert backend_level1._rem["mitigator"] is original_mitigator

	def test_rem_disable_preserves_settings(self, backend_level1: FiQCIBackend) -> None:
		"""Test that disabling REM preserves calibration settings."""
		with patch("fiqci.ems.backend.core.M3IQM"):
			backend_level1.rem(enabled=True, calibration_shots=2048, calibration_file="/tmp/cal.json")
		backend_level1.rem(enabled=False)
		assert backend_level1._rem["calibration_shots"] == 2048
		assert backend_level1._rem["calibration_file"] == "/tmp/cal.json"


class TestMitigatedJob:
	"""Tests for MitigatedJob class (a lazy view over the BatchedJob handle)."""

	def test_result_delegates_to_handle(self) -> None:
		"""Test that result() returns the handle's (mitigated) result."""
		mock_handle = Mock()
		mock_mitigated_result = Mock()
		mock_handle.result.return_value = mock_mitigated_result

		mitigated_job = MitigatedJob(mock_handle)

		assert mitigated_job.result() is mock_mitigated_result

	def test_getattr_delegates_to_handle(self) -> None:
		"""Test that attribute/polling access is delegated to the handle."""
		mock_handle = Mock()
		mock_handle.job_ids.return_value = ["test-job-123"]

		mitigated_job = MitigatedJob(mock_handle)

		assert mitigated_job.job_ids() == ["test-job-123"]

	def test_result_passes_timeout_to_handle(self) -> None:
		"""Test that result() forwards the timeout parameter to the handle."""
		mock_handle = Mock()
		mock_mitigated_result = Mock()
		mock_handle.result.return_value = mock_mitigated_result

		mitigated_job = MitigatedJob(mock_handle)

		result = mitigated_job.result(timeout=10.0)
		assert result is mock_mitigated_result
		mock_handle.result.assert_called_once_with(10.0)


class TestJobMitigatorOptionsSnapshot:
	"""Tests that a returned job carries a frozen snapshot of the mitigation settings used."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	@pytest.fixture
	def mock_circuit(self) -> QuantumCircuit:
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)
		qc.measure_all()
		return qc

	def test_level0_job_reports_all_disabled(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""A level-0 handle reports a snapshot with every technique disabled."""
		mock_backend.run.return_value = _make_result_mock([{"00": 1024}])

		job = FiQCIBackend(mock_backend, mitigation_level=0).run(mock_circuit, shots=1024)

		options = job.mitigator_options
		assert options["mitigation_level"] == 0
		assert options["rem"]["enabled"] is False
		assert options["dd"]["enabled"] is False
		assert options["pauli_twirl"]["enabled"] is False

	def test_snapshot_omits_live_m3_object(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""The snapshot carries REM settings but not the heavyweight M3 mitigator instance."""
		mock_backend.run.return_value = _make_result_mock([{"00": 1024}])

		with patch("fiqci.ems.backend.core.M3IQM"):
			job = FiQCIBackend(mock_backend, mitigation_level=1).run(mock_circuit, shots=1024)

		rem = job.mitigator_options["rem"]
		assert rem["enabled"] is True
		assert "mitigator" not in rem
		assert set(rem) == {"enabled", "calibration_shots", "calibration_file"}

	def test_snapshot_frozen_against_later_backend_mutation(
		self, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		"""Mutating the backend's settings after run() does not change the job's snapshot."""
		mock_backend.run.return_value = _make_result_mock([{"00": 1024}])

		backend = FiQCIBackend(mock_backend, mitigation_level=0)
		job = backend.run(mock_circuit, shots=1024)

		# Later reconfigure the backend; the already-submitted job must be unaffected.
		backend.dd(enabled=True)
		backend.pauli_twirl(enabled=True, num_twirls=7)

		assert job.mitigator_options["dd"]["enabled"] is False
		assert job.mitigator_options["pauli_twirl"]["enabled"] is False
		# The backend's own live view does reflect the mutation.
		assert backend.mitigator_options["dd"]["enabled"] is True

	def test_snapshot_captures_dd_config(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""DD settings in effect at submission are recorded in the snapshot."""
		mock_backend.run.return_value = _make_result_mock([{"00": 1024}])

		backend = FiQCIBackend(mock_backend, mitigation_level=0)
		backend.dd(enabled=True)
		job = backend.run(mock_circuit, shots=1024)

		assert job.mitigator_options["dd"]["enabled"] is True

	def test_snapshot_captures_twirl_config(self) -> None:
		"""Pauli-twirl settings in effect at submission are recorded in the snapshot."""
		from qiskit_aer import AerSimulator

		qc = QuantumCircuit(2)
		qc.cx(0, 1)
		qc.measure_all()

		backend = FiQCIBackend(AerSimulator(), mitigation_level=0)
		backend.pauli_twirl(enabled=True, num_twirls=3)
		job = backend.run(qc, shots=1024)

		options = job.mitigator_options
		assert options["pauli_twirl"]["enabled"] is True
		assert options["pauli_twirl"]["num_twirls"] == 3


def _make_result_mock(counts_per_circuit: list[dict[str, int]]) -> Mock:
	"""Build a Mock job whose result.to_dict()/get_counts behave like a real backend job."""
	mock_job = Mock()
	mock_result = Mock()
	mock_result.to_dict.return_value = {
		"results": [
			{"data": {"counts": counts}, "shots": sum(counts.values()), "success": True}
			for counts in counts_per_circuit
		],
		"backend_name": "mock",
		"job_id": "test-job-id",
		"qobj_id": "test-qobj-id",
		"success": True,
		"status": "COMPLETED",
	}
	mock_result.get_counts.return_value = counts_per_circuit if len(counts_per_circuit) > 1 else counts_per_circuit[0]
	mock_job.result.return_value = mock_result
	return mock_job


class TestBackendBatching:
	"""Tests for FiQCIBackend.run circuit batching."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	def test_run_single_batch_is_wrapped_in_batched_job(self, mock_backend: Mock) -> None:
		"""Input <= max_batch_size: backend.run is called once and the job is wrapped in a BatchedJob.

		Even a single batch is wrapped so the polling/partial-result API is uniform; the wrapper's
		combined result round-trips the underlying counts.
		"""
		circuits = [QuantumCircuit(2) for _ in range(3)]
		mock_job = _make_result_mock([{"00": 1024}] * 3)
		mock_backend.run.return_value = mock_job

		fiqci_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		result = fiqci_backend.run(circuits, shots=1024, max_batch_size=10)

		assert isinstance(result, BatchedJob)
		assert mock_backend.run.call_count == 1
		assert result.job_ids() == [mock_job.job_id()]
		# Lazy: results not fetched until result() is called.
		mock_job.result.assert_not_called()
		assert result.result().get_counts(0) == {"00": 1024}

	def test_run_multiple_batches_returns_batched_job(self, mock_backend: Mock) -> None:
		"""Input > max_batch_size: input is split and a BatchedJob wraps the per-batch jobs."""
		circuits = [QuantumCircuit(2) for _ in range(7)]
		mock_backend.run.side_effect = [
			_make_result_mock([{"00": 1024}] * 3),
			_make_result_mock([{"00": 1024}] * 3),
			_make_result_mock([{"00": 1024}]),
		]

		fiqci_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		result = fiqci_backend.run(circuits, shots=1024, max_batch_size=3)

		assert isinstance(result, BatchedJob)
		assert mock_backend.run.call_count == 3

	def test_partial_submission_returns_handle_and_stops_submitting(self, mock_backend: Mock) -> None:
		"""A mid-stream rejection stops submission and returns a handle covering every batch.

		Submitted batches keep their job ids and status; the rejected batch is ERROR and the
		batches skipped afterwards are CANCELLED. run() does not raise.
		"""
		circuits = [QuantumCircuit(2) for _ in range(5)]
		good_a = _make_result_mock([{"00": 1024}])
		good_a.job_id.return_value = "job-0"
		good_a.status.return_value = JobStatus.DONE
		good_b = _make_result_mock([{"00": 1024}])
		good_b.job_id.return_value = "job-1"
		good_b.status.return_value = JobStatus.DONE
		# Two batches submit, the third is rejected; the 4th/5th must never be attempted.
		mock_backend.run.side_effect = [good_a, good_b, ValueError("cx not native")]

		fiqci_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		result = fiqci_backend.run(circuits, shots=1024, max_batch_size=1)

		# No further submissions were attempted after the failure (only 3 backend.run calls).
		assert mock_backend.run.call_count == 3
		assert isinstance(result, BatchedJob)
		# Submitted batches keep their ids; unsubmitted ones are None (index-aligned).
		assert result.job_ids() == ["job-0", "job-1", None, None, None]
		# Submitted -> DONE, rejected -> ERROR, skipped -> CANCELLED.
		assert result.statuses() == [
			JobStatus.DONE,
			JobStatus.DONE,
			JobStatus.ERROR,
			JobStatus.CANCELLED,
			JobStatus.CANCELLED,
		]
		# Aggregated status surfaces the failure; the handle is terminal.
		assert result.status() == JobStatus.ERROR
		assert result.done() is True

	def test_partial_submission_partial_results_expose_submitted_batches(self, mock_backend: Mock) -> None:
		"""partial_results() returns results for the submitted batches and None for the rest."""
		circuits = [QuantumCircuit(2) for _ in range(3)]
		good = _make_result_mock([{"00": 1024}])
		good.job_id.return_value = "job-0"
		good.status.return_value = JobStatus.DONE
		mock_backend.run.side_effect = [good, ValueError("rejected")]

		fiqci_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		result = fiqci_backend.run(circuits, shots=1024, max_batch_size=1)

		partials = result.partial_results()
		assert [p.circuit_range for p in partials] == [(0, 1), (1, 2), (2, 3)]
		assert [p.status for p in partials] == [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]
		assert partials[0].result is not None
		assert partials[1].result is None and partials[2].result is None
		# The full combined result cannot be formed with missing batches.
		with pytest.raises(BatchFailedError):
			result.result()

	def test_first_batch_submission_failure_returns_all_placeholder_handle(self, mock_backend: Mock) -> None:
		"""If the very first batch is rejected, the handle is all placeholders (no real jobs)."""
		circuits = [QuantumCircuit(2) for _ in range(3)]
		mock_backend.run.side_effect = ValueError("rejected")

		fiqci_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		result = fiqci_backend.run(circuits, shots=1024, max_batch_size=1)

		assert mock_backend.run.call_count == 1
		assert result.job_ids() == [None, None, None]
		assert result.statuses() == [JobStatus.ERROR, JobStatus.CANCELLED, JobStatus.CANCELLED]
		assert result.status() == JobStatus.ERROR

	def test_run_passes_correct_circuits_per_batch(self, mock_backend: Mock) -> None:
		"""Each batch sent to the underlying backend is a contiguous slice of the input list."""
		circuits = [QuantumCircuit(2) for _ in range(5)]
		mock_backend.run.side_effect = [
			_make_result_mock([{"00": 1024}] * 2),
			_make_result_mock([{"00": 1024}] * 2),
			_make_result_mock([{"00": 1024}]),
		]

		fiqci_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		fiqci_backend.run(circuits, shots=1024, max_batch_size=2)

		batches_sent = [call.args[0] for call in mock_backend.run.call_args_list]
		assert batches_sent[0] == circuits[0:2]
		assert batches_sent[1] == circuits[2:4]
		assert batches_sent[2] == circuits[4:5]

	def test_run_default_max_batch_size_is_100(self, mock_backend: Mock) -> None:
		"""Default max_batch_size of 100 keeps a 50-circuit input as a single backend job."""
		circuits = [QuantumCircuit(2) for _ in range(50)]
		mock_backend.run.return_value = _make_result_mock([{"00": 1024}] * 50)

		fiqci_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		fiqci_backend.run(circuits, shots=1024)

		assert mock_backend.run.call_count == 1

	def test_run_default_max_batch_size_splits_above_100(self, mock_backend: Mock) -> None:
		"""Default max_batch_size of 100 splits a 250-circuit input into 3 batches."""
		circuits = [QuantumCircuit(2) for _ in range(250)]
		mock_backend.run.side_effect = [
			_make_result_mock([{"00": 1024}] * 100),
			_make_result_mock([{"00": 1024}] * 100),
			_make_result_mock([{"00": 1024}] * 50),
		]

		fiqci_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		fiqci_backend.run(circuits, shots=1024)

		assert mock_backend.run.call_count == 3
		batch_sizes = [len(call.args[0]) for call in mock_backend.run.call_args_list]
		assert batch_sizes == [100, 100, 50]

	def test_run_combined_result_indices_match_submission_order(self, mock_backend: Mock) -> None:
		"""Counts retrieved from the BatchedJob's result match the original circuit submission order."""
		circuits = [QuantumCircuit(2) for _ in range(5)]
		mock_backend.run.side_effect = [
			_make_result_mock([{"a": 1}, {"b": 2}]),
			_make_result_mock([{"c": 3}, {"d": 4}]),
			_make_result_mock([{"e": 5}]),
		]

		fiqci_backend = FiQCIBackend(mock_backend, mitigation_level=0)
		result = fiqci_backend.run(circuits, shots=1024, max_batch_size=2)

		assert isinstance(result, BatchedJob)
		combined = result.result()
		assert combined.get_counts(0) == {"a": 1}
		assert combined.get_counts(1) == {"b": 2}
		assert combined.get_counts(2) == {"c": 3}
		assert combined.get_counts(3) == {"d": 4}
		assert combined.get_counts(4) == {"e": 5}


class TestBatchedJob:
	"""Tests for BatchedJob class."""

	def test_result_combines_results_from_all_jobs(self) -> None:
		"""result() concatenates each underlying job's results list in submission order."""
		job_a = _make_result_mock([{"00": 1}, {"11": 2}])
		job_b = _make_result_mock([{"01": 3}])

		batched = BatchedJob([job_a, job_b])
		combined = batched.result()

		assert combined.get_counts(0) == {"00": 1}
		assert combined.get_counts(1) == {"11": 2}
		assert combined.get_counts(2) == {"01": 3}

	def test_result_is_cached(self) -> None:
		"""Calling result() twice does not call result() on the underlying jobs again."""
		job_a = _make_result_mock([{"00": 1}])
		job_b = _make_result_mock([{"11": 2}])

		batched = BatchedJob([job_a, job_b])
		first = batched.result()
		second = batched.result()

		assert first is second
		assert job_a.result.call_count == 1
		assert job_b.result.call_count == 1

	def test_getattr_delegates_to_first_job(self) -> None:
		"""Attribute access (for names not defined on BatchedJob) delegates to the first job."""
		job_a = Mock()
		job_a.custom_attribute = "first-job"
		job_b = Mock()
		job_b.custom_attribute = "second-job"

		batched = BatchedJob([job_a, job_b])

		assert batched.custom_attribute == "first-job"

	def test_job_id_and_job_ids(self) -> None:
		"""job_id() returns the first job's id; job_ids() returns all in submission order."""
		job_a = Mock()
		job_a.job_id.return_value = "first-job"
		job_b = Mock()
		job_b.job_id.return_value = "second-job"

		batched = BatchedJob([job_a, job_b])

		assert batched.job_id() == "first-job"
		assert batched.job_ids() == ["first-job", "second-job"]


class TestCalibrationMaxBatchSize:
	"""Regression tests for max_batch_size parameter passed to calibration."""

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

	def test_run_passes_default_max_batch_size_to_calibration(
		self, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		"""Test that default max_batch_size=100 is passed to calibration."""
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
			patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm_class,
			patch("fiqci.ems.backend.core.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.backend.core.probabilities_to_counts", return_value=[{"00": 480, "11": 520}]),
		):
			mock_mitigator = Mock()
			mock_quasi_dist = Mock()
			mock_quasi_dist.nearest_probability_distribution.return_value = {"00": 0.48, "11": 0.52}
			mock_mitigator.apply_correction.return_value = mock_quasi_dist
			mock_mitigator.single_qubit_cals = None
			mock_m3iqm_class.return_value = mock_mitigator

			mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1)
			mitigated_backend.run(mock_circuit, shots=1024)

			# Verify cals_from_system was called with max_batch_size=100 (default)
			mock_mitigator.cals_from_system.assert_called_once()
			call_kwargs = mock_mitigator.cals_from_system.call_args[1]
			assert call_kwargs["max_batch_size"] == 100

	def test_run_passes_custom_max_batch_size_to_calibration(
		self, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		"""Test that custom max_batch_size is passed to calibration."""
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
			patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm_class,
			patch("fiqci.ems.backend.core.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.backend.core.probabilities_to_counts", return_value=[{"00": 480, "11": 520}]),
		):
			mock_mitigator = Mock()
			mock_quasi_dist = Mock()
			mock_quasi_dist.nearest_probability_distribution.return_value = {"00": 0.48, "11": 0.52}
			mock_mitigator.apply_correction.return_value = mock_quasi_dist
			mock_mitigator.single_qubit_cals = None
			mock_m3iqm_class.return_value = mock_mitigator

			mitigated_backend = FiQCIBackend(mock_backend, mitigation_level=1)
			# Use custom max_batch_size
			mitigated_backend.run(mock_circuit, shots=1024, max_batch_size=50)

			# Verify cals_from_system was called with max_batch_size=50
			mock_mitigator.cals_from_system.assert_called_once()
			call_kwargs = mock_mitigator.cals_from_system.call_args[1]
			assert call_kwargs["max_batch_size"] == 50


def _make_job_with_status(status: JobStatus, counts: list[dict[str, int]] | None = None, job_id: str = "job") -> Mock:
	"""Build a Mock batch job with an explicit status and (optionally) a usable result."""
	job = _make_result_mock(counts) if counts is not None else Mock()
	job.status.return_value = status
	job.job_id.return_value = job_id
	return job


class TestBatchedJobStatus:
	"""Tests for aggregated status, polling, and partial results on the lazy handle."""

	@pytest.mark.parametrize(
		"statuses, expected",
		[
			([JobStatus.DONE, JobStatus.DONE], JobStatus.DONE),
			([JobStatus.DONE, JobStatus.RUNNING], JobStatus.RUNNING),
			([JobStatus.QUEUED, JobStatus.RUNNING], JobStatus.RUNNING),
			([JobStatus.QUEUED, JobStatus.INITIALIZING], JobStatus.QUEUED),
			([JobStatus.DONE, JobStatus.ERROR], JobStatus.ERROR),
			([JobStatus.ERROR, JobStatus.CANCELLED], JobStatus.ERROR),
			([JobStatus.DONE, JobStatus.CANCELLED], JobStatus.CANCELLED),
			([JobStatus.VALIDATING, JobStatus.QUEUED], JobStatus.VALIDATING),
		],
	)
	def test_aggregate_status_priority(self, statuses: list[JobStatus], expected: JobStatus) -> None:
		"""ERROR/CANCELLED dominate; otherwise DONE only if all done, else least-advanced active."""
		assert BatchedJob._aggregate_status(statuses) == expected

	def test_status_and_done_poll_live(self) -> None:
		"""status()/done() reflect the live per-batch statuses without fetching results."""
		job_a = _make_job_with_status(JobStatus.DONE, job_id="a")
		job_b = _make_job_with_status(JobStatus.RUNNING, job_id="b")

		batched = BatchedJob([job_a, job_b], [(0, 1), (1, 2)])

		assert batched.status() == JobStatus.RUNNING
		assert batched.done() is False
		# Polling must not block on results.
		job_a.result.assert_not_called()
		job_b.result.assert_not_called()

		job_b.status.return_value = JobStatus.DONE
		assert batched.status() == JobStatus.DONE
		assert batched.done() is True

	def test_job_ids_available_without_results(self) -> None:
		"""All batch job ids are available immediately, before any result is fetched."""
		job_a = _make_job_with_status(JobStatus.RUNNING, job_id="aaa")
		job_b = _make_job_with_status(JobStatus.QUEUED, job_id="bbb")

		batched = BatchedJob([job_a, job_b], [(0, 2), (2, 3)])

		assert batched.job_ids() == ["aaa", "bbb"]
		job_a.result.assert_not_called()
		job_b.result.assert_not_called()

	def test_partial_results_exposes_completed_batches_only(self) -> None:
		"""partial_results() returns results for DONE batches and None for in-flight ones."""
		done_job = _make_job_with_status(JobStatus.DONE, counts=[{"00": 10}], job_id="done")
		running_job = _make_job_with_status(JobStatus.RUNNING, job_id="running")

		batched = BatchedJob([done_job, running_job], [(0, 1), (1, 2)])
		partials = batched.partial_results()

		assert [p.index for p in partials] == [0, 1]
		assert partials[0].circuit_range == (0, 1)
		assert partials[0].status == JobStatus.DONE
		assert partials[0].result is not None
		assert partials[1].circuit_range == (1, 2)
		assert partials[1].status == JobStatus.RUNNING
		assert partials[1].result is None

	def test_result_raises_batch_failed_error_naming_batch_and_range(self) -> None:
		"""A failed batch makes result() raise BatchFailedError identifying the batch and circuits."""
		good_job = _make_job_with_status(JobStatus.DONE, counts=[{"00": 10}, {"11": 5}], job_id="good")
		bad_job = _make_job_with_status(JobStatus.ERROR, job_id="bad-123")

		batched = BatchedJob([good_job, bad_job], [(0, 2), (2, 4)])

		with pytest.raises(BatchFailedError) as exc_info:
			batched.result()

		message = str(exc_info.value)
		assert "batch 1" in message
		assert "circuits 2-3" in message
		assert "bad-123" in message

	def test_result_runs_post_process_once_and_caches(self) -> None:
		"""result() applies post_process to the combined result exactly once, then caches."""
		job = _make_job_with_status(JobStatus.DONE, counts=[{"00": 4}], job_id="j")
		calls: list[int] = []

		def post(result):
			calls.append(1)
			return result

		batched = BatchedJob([job], [(0, 1)], post_process=post)
		first = batched.result()
		second = batched.result()

		assert first is second
		assert len(calls) == 1
		assert job.result.call_count == 1


class TestMitigatorOptionsIsolation:
	"""``mitigator_options`` hands out a copy, so callers cannot bypass the setter validation."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	def test_mutating_returned_dict_does_not_change_settings(self, mock_backend: Mock) -> None:
		"""Reassigning entries of the returned dict leaves the backend's own settings intact."""
		with patch("fiqci.ems.backend.core.M3IQM"):
			backend = FiQCIBackend(mock_backend, mitigation_level=3)

		options = backend.mitigator_options
		options["rem"]["enabled"] = "clobbered"
		options["dd"]["enabled"] = "clobbered"
		options["pauli_twirl"]["num_twirls"] = 9999

		assert backend._rem["enabled"] is True
		assert backend._dd["enabled"] is True
		assert backend._pauli_twirl["num_twirls"] == 10

	def test_mutating_nested_gate_sequences_list_does_not_change_settings(self, mock_backend: Mock) -> None:
		"""The nested ``dd.gate_sequences`` list is copied too, not just the outer dicts."""
		with patch("fiqci.ems.backend.core.M3IQM"):
			backend = FiQCIBackend(mock_backend, mitigation_level=2)

		original_length = len(backend._dd["gate_sequences"])
		backend.mitigator_options["dd"]["gate_sequences"].append(("x", "y", "z"))

		assert len(backend._dd["gate_sequences"]) == original_length

	def test_mutating_nested_gates_to_twirl_list_does_not_change_settings(self, mock_backend: Mock) -> None:
		"""A materialised ``gates_to_twirl`` collection is copied as well."""
		backend = FiQCIBackend(mock_backend, mitigation_level=0)
		backend.pauli_twirl(True, num_twirls=2, gates_to_twirl=["cz"])

		backend.mitigator_options["pauli_twirl"]["gates_to_twirl"].append("clobbered")

		assert backend._pauli_twirl["gates_to_twirl"] == ["cz"]

	def test_iterator_gates_to_twirl_is_not_consumed_by_reading_options(self, mock_backend: Mock) -> None:
		"""Copying must not drain a user-supplied iterator; it is passed through by reference."""
		backend = FiQCIBackend(mock_backend, mitigation_level=0)
		backend.pauli_twirl(True, num_twirls=2, gates_to_twirl=(g for g in ["cz"]))

		backend.mitigator_options
		backend.mitigator_options

		assert list(backend._pauli_twirl["gates_to_twirl"]) == ["cz"]

	def test_live_m3_mitigator_is_shared_by_reference(self, mock_backend: Mock) -> None:
		"""The mitigator owns the calibration data, so it is deliberately not copied."""
		with patch("fiqci.ems.backend.core.M3IQM"):
			backend = FiQCIBackend(mock_backend, mitigation_level=1)

		assert backend.mitigator_options["rem"]["mitigator"] is backend._rem["mitigator"]
