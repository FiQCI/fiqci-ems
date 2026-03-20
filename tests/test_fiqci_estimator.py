"""Unit tests for FiQCIEstimator class."""

from unittest.mock import Mock, patch

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from fiqci.ems.fiqci_estimator import FiQCIEstimator, FiQCIEstimatorJobCollection


class TestFiQCIEstimator:
	"""Tests for FiQCIEstimator class."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		"""Create a mock IQM backend."""
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		backend.target.operation_names = ["h", "cx", "rz", "sx", "x", "sdg"]
		return backend

	@pytest.fixture
	def mock_circuit(self) -> QuantumCircuit:
		"""Create a simple quantum circuit."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)
		return qc

	@pytest.fixture
	def single_observable(self) -> SparsePauliOp:
		"""Create a single SparsePauliOp observable."""
		return SparsePauliOp.from_list([("ZZ", 1.0)])

	@pytest.fixture
	def multi_observable(self) -> SparsePauliOp:
		"""Create a multi-term SparsePauliOp observable."""
		return SparsePauliOp.from_list([("ZZ", 0.5), ("XX", 0.3), ("ZI", 0.2)])

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_init_creates_fiqci_backend(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that FiQCIEstimator creates a FiQCIBackend on init."""
		_estimator = FiQCIEstimator(mock_backend, mitigation_level=1, calibration_shots=2000)
		mock_fiqci_backend_class.assert_called_once_with(mock_backend, 1, 2000, None)

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_init_default_parameters(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that default parameters are passed to FiQCIBackend."""
		_estimator = FiQCIEstimator(mock_backend)
		mock_fiqci_backend_class.assert_called_once_with(mock_backend, 1, 1000, None)

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_init_with_calibration_files(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that calibration_files parameter is forwarded."""
		_estimator = FiQCIEstimator(mock_backend, calibration_files="cal.json")
		mock_fiqci_backend_class.assert_called_once_with(mock_backend, 1, 1000, "cal.json")

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_run_delegates_to_internal_run(self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit, single_observable: SparsePauliOp) -> None:
		"""Test that run() delegates to _run()."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target.operation_names = ["h", "cx", "rz", "sx", "x", "sdg"]
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend)
		with patch.object(estimator, "_run") as mock_internal_run:
			mock_internal_run.return_value = Mock()
			estimator.run(mock_circuit, single_observable, shots=512)
			mock_internal_run.assert_called_once_with(mock_circuit, single_observable, shots=512)

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_run_mismatched_list_lengths_raises_error(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that mismatched list lengths raise ValueError."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target.operation_names = ["h", "cx", "rz", "sx", "x", "sdg"]
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend)

		circuits = [QuantumCircuit(2), QuantumCircuit(2)]
		observables = [SparsePauliOp.from_list([("ZZ", 1.0)])]

		with pytest.raises(ValueError, match="Length of observables and circuits lists must match"):
			estimator.run(circuits, observables)

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_run_single_circuit_single_observable(self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test run with a single circuit and single observable."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target.operation_names = ["h", "cx", "rz", "sx", "x", "sdg"]
		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.return_value = {"00": 500, "11": 500}
		mock_job.result.return_value = mock_result
		mock_fiqci_backend.run.return_value = mock_job
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend)
		obs = SparsePauliOp.from_list([("ZZ", 1.0)])
		result = estimator.run(mock_circuit, obs)

		assert isinstance(result, FiQCIEstimatorJobCollection)
		mock_fiqci_backend.run.assert_called_once()

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_run_list_circuits_single_observable(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test run with list of circuits and a single observable."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target.operation_names = ["h", "cx", "rz", "sx", "x", "sdg"]
		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.return_value = {"00": 500, "11": 500}
		mock_job.result.return_value = mock_result
		mock_fiqci_backend.run.return_value = mock_job
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		circuits = [QuantumCircuit(2), QuantumCircuit(2)]
		for qc in circuits:
			qc.h(0)
			qc.cx(0, 1)

		obs = SparsePauliOp.from_list([("ZZ", 1.0)])
		estimator = FiQCIEstimator(mock_backend)
		result = estimator.run(circuits, obs)

		assert isinstance(result, FiQCIEstimatorJobCollection)
		assert mock_fiqci_backend.run.call_count == 2

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_run_paired_lists(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test run with paired lists of circuits and observables."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target.operation_names = ["h", "cx", "rz", "sx", "x", "sdg"]
		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.return_value = {"00": 500, "11": 500}
		mock_job.result.return_value = mock_result
		mock_fiqci_backend.run.return_value = mock_job
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		circuits = [QuantumCircuit(2), QuantumCircuit(2)]
		for qc in circuits:
			qc.h(0)
			qc.cx(0, 1)

		observables = [
			SparsePauliOp.from_list([("ZZ", 1.0)]),
			SparsePauliOp.from_list([("XX", 1.0)]),
		]

		estimator = FiQCIEstimator(mock_backend)
		result = estimator.run(circuits, observables)

		assert isinstance(result, FiQCIEstimatorJobCollection)
		assert mock_fiqci_backend.run.call_count == 2

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_run_default_shots(self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that default shots is 2048."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target.operation_names = ["h", "cx", "rz", "sx", "x", "sdg"]
		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.return_value = {"00": 1024, "11": 1024}
		mock_job.result.return_value = mock_result
		mock_fiqci_backend.run.return_value = mock_job
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend)
		obs = SparsePauliOp.from_list([("ZZ", 1.0)])
		estimator.run(mock_circuit, obs)

		call_kwargs = mock_fiqci_backend.run.call_args[1]
		assert call_kwargs["shots"] == 2048


class TestCalculateExpectationValues:
	"""Tests for calculate_expectation_values method."""

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_single_z_observable(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test expectation value for a single Z observable."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("IZ", 1.0)])
		# measurement_settings: [{0: 'Z'}]
		measurement_settings = [{0: "Z"}]
		# counts where qubit 0 is measured: '0' 700 times, '1' 300 times
		counts = [{"0": 700, "1": 300}]

		exp_vals = estimator.calculate_expectation_values(counts, obs, measurement_settings)

		# parity: '0' -> +1, '1' -> -1
		# exp_val = (700 * 1 + 300 * (-1)) / 1000 = 0.4
		assert len(exp_vals) == 1
		assert exp_vals[0] == pytest.approx(0.4)

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_zz_observable(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test expectation value for ZZ observable."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("ZZ", 1.0)])
		measurement_settings = [{0: "Z", 1: "Z"}]
		# '00' -> parity +1, '01' -> -1, '10' -> -1, '11' -> +1
		counts = [{"00": 400, "01": 100, "10": 100, "11": 400}]

		exp_vals = estimator.calculate_expectation_values(counts, obs, measurement_settings)

		# exp_val = (400 + 400 - 100 - 100) / 1000 = 0.6
		assert len(exp_vals) == 1
		assert exp_vals[0] == pytest.approx(0.6)

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_single_counts_dict_wrapped_in_list(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that a single counts dict is automatically wrapped in a list."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("IZ", 1.0)])
		measurement_settings = [{0: "Z"}]
		# Pass counts as a single dict, not a list
		counts = {"0": 500, "1": 500}

		exp_vals = estimator.calculate_expectation_values(counts, obs, measurement_settings)

		assert len(exp_vals) == 1
		assert exp_vals[0] == pytest.approx(0.0)

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_no_matching_measurement_setting(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that observables with no matching setting return 0."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("IX", 1.0)])
		# Only Z measurements available, not X
		measurement_settings = [{0: "Z"}]
		counts = [{"0": 500, "1": 500}]

		exp_vals = estimator.calculate_expectation_values(counts, obs, measurement_settings)

		assert len(exp_vals) == 1
		assert exp_vals[0] == 0

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_all_zeros_counts(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test expectation value when all counts are in the 0 state."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("IZ", 1.0)])
		measurement_settings = [{0: "Z"}]
		counts = [{"0": 1000}]

		exp_vals = estimator.calculate_expectation_values(counts, obs, measurement_settings)

		assert len(exp_vals) == 1
		assert exp_vals[0] == pytest.approx(1.0)

	@patch("fiqci.ems.fiqci_estimator.FiQCIBackend")
	def test_all_ones_counts(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test expectation value when all counts are in the 1 state."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("IZ", 1.0)])
		measurement_settings = [{0: "Z"}]
		counts = [{"1": 1000}]

		exp_vals = estimator.calculate_expectation_values(counts, obs, measurement_settings)

		assert len(exp_vals) == 1
		assert exp_vals[0] == pytest.approx(-1.0)


class TestFiQCIEstimatorJobCollection:
	"""Tests for FiQCIEstimatorJobCollection class."""

	def test_jobs_returns_all_jobs(self) -> None:
		"""Test that jobs() returns the list of mitigated jobs."""
		mock_jobs = [Mock(), Mock()]
		collection = FiQCIEstimatorJobCollection(mock_jobs, [[0.5], [0.3]], Mock())

		assert collection.jobs() == mock_jobs

	def test_expectation_values_returns_all(self) -> None:
		"""Test that expectation_values() returns all values when no index given."""
		exp_vals = [[0.5, 0.3], [0.1, -0.2]]
		collection = FiQCIEstimatorJobCollection([Mock()], exp_vals, Mock())

		assert collection.expectation_values() == exp_vals

	def test_expectation_values_by_index(self) -> None:
		"""Test that expectation_values(index) returns values for specific circuit."""
		exp_vals = [[0.5, 0.3], [0.1, -0.2]]
		collection = FiQCIEstimatorJobCollection([Mock()], exp_vals, Mock())

		assert collection.expectation_values(0) == [0.5, 0.3]
		assert collection.expectation_values(1) == [0.1, -0.2]

	def test_observables_returns_all(self) -> None:
		"""Test that observables() returns all observables when no index given."""
		obs = SparsePauliOp.from_list([("ZZ", 1.0)])
		collection = FiQCIEstimatorJobCollection([Mock()], [[0.5]], obs)

		assert collection.observables() is obs

	def test_observables_by_index(self) -> None:
		"""Test that observables(index) returns specific observable."""
		obs_list = [
			SparsePauliOp.from_list([("ZZ", 1.0)]),
			SparsePauliOp.from_list([("XX", 1.0)]),
		]
		collection = FiQCIEstimatorJobCollection([Mock()], [[0.5], [0.3]], obs_list)

		assert collection.observables(0) == obs_list[0]
		assert collection.observables(1) == obs_list[1]
