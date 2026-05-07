"""Unit tests for FiQCIEstimator class."""

from unittest.mock import Mock, patch

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, CXGate, RZGate, SXGate, XGate, SdgGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import Target

from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator, FiQCIEstimatorJob


def _make_target(num_qubits=5):
	"""Create a minimal real Qiskit Target for use in tests."""
	target = Target(num_qubits=num_qubits)
	target.add_instruction(HGate())
	target.add_instruction(CXGate())
	target.add_instruction(RZGate(0.0))
	target.add_instruction(SXGate())
	target.add_instruction(XGate())
	target.add_instruction(SdgGate())
	return target


class TestFiQCIEstimator:
	"""Tests for FiQCIEstimator class."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		"""Create a mock IQM backend."""
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		backend.target = _make_target()
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

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_init_creates_fiqci_backend(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that FiQCIEstimator creates a FiQCIBackend on init."""
		_estimator = FiQCIEstimator(mock_backend, mitigation_level=1, calibration_shots=2000)
		mock_fiqci_backend_class.assert_called_once_with(mock_backend, 1, 2000, None)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_init_default_parameters(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that default parameters are passed to FiQCIBackend."""
		_estimator = FiQCIEstimator(mock_backend)
		mock_fiqci_backend_class.assert_called_once_with(mock_backend, 1, 1000, None)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_init_with_calibration_file(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that calibration_file parameter is forwarded."""
		_estimator = FiQCIEstimator(mock_backend, calibration_file="cal.json")
		mock_fiqci_backend_class.assert_called_once_with(mock_backend, 1, 1000, "cal.json")

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_run_delegates_to_internal_run(
		self,
		mock_fiqci_backend_class: Mock,
		mock_backend: Mock,
		mock_circuit: QuantumCircuit,
		single_observable: SparsePauliOp,
	) -> None:
		"""Test that run() delegates to _run()."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target = _make_target()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend)
		with patch.object(estimator, "_run") as mock_internal_run:
			mock_internal_run.return_value = Mock()
			estimator.run(mock_circuit, single_observable, shots=512)
			mock_internal_run.assert_called_once_with(mock_circuit, single_observable, shots=512, max_batch_size=100)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_run_mismatched_list_lengths_raises_error(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that mismatched list lengths raise ValueError."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target = _make_target()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend)

		circuits = [QuantumCircuit(2), QuantumCircuit(2)]
		observables = [SparsePauliOp.from_list([("ZZ", 1.0)])]

		with pytest.raises(ValueError, match="Length of observables and circuits lists must match"):
			estimator.run(circuits, observables)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_run_single_circuit_single_observable(
		self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		"""Test run with a single circuit and single observable."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target = _make_target()
		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.return_value = {"00": 500, "11": 500}
		mock_job.result.return_value = mock_result
		mock_fiqci_backend.run.return_value = mock_job
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend)
		obs = SparsePauliOp.from_list([("ZZ", 1.0)])
		result = estimator.run(mock_circuit, obs)

		assert isinstance(result, FiQCIEstimatorJob)
		mock_fiqci_backend.run.assert_called_once()

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_run_list_circuits_single_observable(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test run with list of circuits and a single observable."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target = _make_target()
		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.return_value = [{"00": 500, "11": 500}, {"00": 500, "11": 500}]
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

		assert isinstance(result, FiQCIEstimatorJob)
		assert mock_fiqci_backend.run.call_count == 1

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_run_paired_lists(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test run with paired lists of circuits and observables."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target = _make_target()
		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.return_value = [{"00": 500, "11": 500}, {"00": 500, "11": 500}]
		mock_job.result.return_value = mock_result
		mock_fiqci_backend.run.return_value = mock_job
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		circuits = [QuantumCircuit(2), QuantumCircuit(2)]
		for qc in circuits:
			qc.h(0)
			qc.cx(0, 1)

		observables = [SparsePauliOp.from_list([("ZZ", 1.0)]), SparsePauliOp.from_list([("XX", 1.0)])]

		estimator = FiQCIEstimator(mock_backend)
		result = estimator.run(circuits, observables)

		assert isinstance(result, FiQCIEstimatorJob)
		assert mock_fiqci_backend.run.call_count == 1

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_run_default_shots(
		self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		"""Test that default shots is 2048."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target = _make_target()
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


class TestEstimatorBatching:
	"""Tests for FiQCIEstimator measurement-circuit flattening and batching."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		backend.target = _make_target()
		return backend

	@pytest.fixture
	def mock_circuit(self) -> QuantumCircuit:
		"""Create a simple quantum circuit for batching tests."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)
		return qc

	@staticmethod
	def _make_job(counts_per_circuit: list[dict[str, int]]) -> Mock:
		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.return_value = counts_per_circuit
		mock_job.result.return_value = mock_result
		return mock_job

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_run_flattens_pairs_into_single(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Multiple pairs that should be flattened for batching by FiQCIBackend."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target = _make_target()
		mock_fiqci_backend.run.return_value = self._make_job([{"00": 500, "11": 500}] * 3)
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		circuits = [QuantumCircuit(2) for _ in range(3)]
		for qc in circuits:
			qc.h(0)
			qc.cx(0, 1)
		obs = SparsePauliOp.from_list([("ZZ", 1.0)])

		estimator = FiQCIEstimator(mock_backend)
		estimator.run(circuits, obs, max_batch_size=10)

		assert mock_fiqci_backend.run.call_count == 1
		# All 3 measurement circuits sent in one batch
		assert len(mock_fiqci_backend.run.call_args_list[0].args[0]) == 3

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_run_default_max_batch_size_is_100(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Default max_batch_size is 100; 50 pairs (50 flat circuits) fit in a single batch."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target = _make_target()
		mock_fiqci_backend.run.return_value = self._make_job([{"00": 500, "11": 500}] * 50)
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		circuits = [QuantumCircuit(2) for _ in range(50)]
		for qc in circuits:
			qc.h(0)
			qc.cx(0, 1)
		obs = SparsePauliOp.from_list([("ZZ", 1.0)])

		estimator = FiQCIEstimator(mock_backend)
		estimator.run(circuits, obs)

		assert mock_fiqci_backend.run.call_count == 1

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_run_per_pair_counts_assigned_correctly(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Counts returned across batches are sliced back to the correct pair using pair_lengths."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target = _make_target()
		# Two pairs: pair 0 measures Z (all-zero counts -> +1), pair 1 measures Z (all-one counts -> -1)
		mock_fiqci_backend.run.return_value = self._make_job([{"00": 1000}, {"11": 1000}])
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		circuits = [QuantumCircuit(2) for _ in range(2)]
		for qc in circuits:
			qc.h(0)
			qc.cx(0, 1)
		obs = SparsePauliOp.from_list([("ZZ", 1.0)])

		estimator = FiQCIEstimator(mock_backend)
		result = estimator.run(circuits, obs)

		# Pair 0: '00' -> +1 parity for ZZ -> +1.0
		# Pair 1: '11' -> +1 parity for ZZ ('11' has even number of 1s) -> +1.0
		assert result.expectation_values(0) == pytest.approx([1.0])
		assert result.expectation_values(1) == pytest.approx([1.0])

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_run_max_batch_size_forwarded_through_run(
		self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		"""The public run() forwards max_batch_size to _run()."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.target = _make_target()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend)
		obs = SparsePauliOp.from_list([("ZZ", 1.0)])
		with patch.object(estimator, "_run") as mock_internal_run:
			mock_internal_run.return_value = Mock()
			estimator.run(mock_circuit, obs, max_batch_size=42)
			mock_internal_run.assert_called_once_with(mock_circuit, obs, shots=2048, max_batch_size=42)


class TestCalculateExpectationValues:
	"""Tests for calculate_expectation_values method."""

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
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

		exp_vals = estimator._calculate_expectation_values(counts, obs, measurement_settings)

		# parity: '0' -> +1, '1' -> -1
		# exp_val = (700 * 1 + 300 * (-1)) / 1000 = 0.4
		assert len(exp_vals) == 1
		assert exp_vals[0] == pytest.approx(0.4)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zz_observable(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test expectation value for ZZ observable."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("ZZ", 1.0)])
		measurement_settings = [{0: "Z", 1: "Z"}]
		# '00' -> parity +1, '01' -> -1, '10' -> -1, '11' -> +1
		counts = [{"00": 400, "01": 100, "10": 100, "11": 400}]

		exp_vals = estimator._calculate_expectation_values(counts, obs, measurement_settings)

		# exp_val = (400 + 400 - 100 - 100) / 1000 = 0.6
		assert len(exp_vals) == 1
		assert exp_vals[0] == pytest.approx(0.6)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_single_counts_dict_wrapped_in_list(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that a single counts dict is automatically wrapped in a list."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("IZ", 1.0)])
		measurement_settings = [{0: "Z"}]
		# Pass counts as a single dict, not a list
		counts = {"0": 500, "1": 500}

		exp_vals = estimator._calculate_expectation_values(counts, obs, measurement_settings)

		assert len(exp_vals) == 1
		assert exp_vals[0] == pytest.approx(0.0)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_no_matching_measurement_setting(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that observables with no matching setting return 0."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("IX", 1.0)])
		# Only Z measurements available, not X
		measurement_settings = [{0: "Z"}]
		counts = [{"0": 500, "1": 500}]

		exp_vals = estimator._calculate_expectation_values(counts, obs, measurement_settings)

		assert len(exp_vals) == 1
		assert exp_vals[0] == 0

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_all_zeros_counts(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test expectation value when all counts are in the 0 state."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("IZ", 1.0)])
		measurement_settings = [{0: "Z"}]
		counts = [{"0": 1000}]

		exp_vals = estimator._calculate_expectation_values(counts, obs, measurement_settings)

		assert len(exp_vals) == 1
		assert exp_vals[0] == pytest.approx(1.0)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_all_ones_counts(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test expectation value when all counts are in the 1 state."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("IZ", 1.0)])
		measurement_settings = [{0: "Z"}]
		counts = [{"1": 1000}]

		exp_vals = estimator._calculate_expectation_values(counts, obs, measurement_settings)

		assert len(exp_vals) == 1
		assert exp_vals[0] == pytest.approx(-1.0)


class TestFiQCIEstimatorJob:
	"""Tests for FiQCIEstimatorJob class."""

	def test_expectation_values_returns_all(self) -> None:
		"""Test that expectation_values() returns all values when no index given."""
		exp_vals = [[0.5, 0.3], [0.1, -0.2]]
		collection = FiQCIEstimatorJob([Mock()], exp_vals, Mock(), exp_vals)

		assert collection.expectation_values() == exp_vals

	def test_expectation_values_by_index(self) -> None:
		"""Test that expectation_values(index) returns values for specific circuit."""
		exp_vals = [[0.5, 0.3], [0.1, -0.2]]
		collection = FiQCIEstimatorJob([Mock()], exp_vals, Mock(), exp_vals)

		assert collection.expectation_values(0) == [0.5, 0.3]
		assert collection.expectation_values(1) == [0.1, -0.2]

	def test_observables_returns_all(self) -> None:
		"""Test that observables() returns all observables when no index given."""
		obs = SparsePauliOp.from_list([("ZZ", 1.0)])
		collection = FiQCIEstimatorJob([Mock()], [[0.5]], obs, [[0.5]])

		assert collection.observables() is obs

	def test_observables_by_index(self) -> None:
		"""Test that observables(index) returns specific observable."""
		obs_list = [SparsePauliOp.from_list([("ZZ", 1.0)]), SparsePauliOp.from_list([("XX", 1.0)])]
		collection = FiQCIEstimatorJob([Mock()], [[0.5], [0.3]], obs_list, [[0.5], [0.3]])

		assert collection.observables(0) == obs_list[0]
		assert collection.observables(1) == obs_list[1]
