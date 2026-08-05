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
	def test_zne_setting_change_after_run_does_not_affect_results(
		self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		"""Settings snapshot: toggling ZNE between run() and value access must not change results.

		ZNE settings are snapshotted at submission, so a run submitted with ZNE off keeps computing
		plain expectation values even if the user enables ZNE before accessing them (which would
		otherwise mis-split the counts by scale factors that were never submitted).
		"""

		def make_backend() -> Mock:
			backend = Mock()
			backend.target = _make_target()
			job = Mock()
			res = Mock()
			res.get_counts.return_value = {"00": 500, "11": 500}
			job.result.return_value = res
			backend.run.return_value = job
			return backend

		obs = SparsePauliOp.from_list([("ZZ", 1.0)])

		# Reference: ZNE stays off for the whole lifecycle.
		mock_fiqci_backend_class.return_value = make_backend()
		reference = FiQCIEstimator(mock_backend).run(mock_circuit, obs)
		expected = reference.expectation_values()

		# Submit with ZNE off, then enable ZNE before accessing the values.
		mock_fiqci_backend_class.return_value = make_backend()
		estimator = FiQCIEstimator(mock_backend)
		job = estimator.run(mock_circuit, obs)
		estimator.zne(enabled=True, scale_factors=[1, 3, 5], extrapolation_method="exponential")

		# Snapshot wins: still the plain (non-ZNE) result, and no crash from mis-split counts.
		assert job.expectation_values() == expected

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

	@pytest.mark.parametrize("counts", [pytest.param([{}], id="empty-dict"), pytest.param([{"0": 0}], id="all-zero")])
	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zero_total_counts_report_zero_instead_of_dividing_by_zero(
		self, mock_fiqci_backend_class: Mock, counts: list[dict[str, int]]
	) -> None:
		"""A measurement circuit with no recorded shots reports 0.0, matching _calculate_shot_errors."""
		mock_backend = Mock()
		mock_fiqci_backend_class.return_value = Mock()

		estimator = FiQCIEstimator(mock_backend)

		obs = SparsePauliOp.from_list([("IZ", 1.0)])
		measurement_settings = [{0: "Z"}]

		assert estimator._calculate_expectation_values(counts, obs, measurement_settings) == [0.0]
		assert estimator._calculate_shot_errors(counts, obs, measurement_settings) == [0.0]


class TestEstimatorMitigatorOptionsIsolation:
	"""``FiQCIEstimator.mitigator_options`` hands out a copy of the live ZNE settings."""

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_mutating_returned_zne_dict_does_not_change_settings(self, mock_fiqci_backend_class: Mock) -> None:
		"""Reassigning entries of the returned ``zne`` dict leaves the estimator's settings intact."""
		mock_fiqci_backend_class.return_value = Mock(mitigator_options={})

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, scale_factors=[1, 3], fold_gates=["cz"], folding_method="local")

		options = estimator.mitigator_options
		options["zne"]["enabled"] = "clobbered"
		options["zne"]["folding_method"] = "clobbered"

		assert estimator._zne["enabled"] is True
		assert estimator._zne["folding_method"] == "local"

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_mutating_nested_zne_lists_does_not_change_settings(self, mock_fiqci_backend_class: Mock) -> None:
		"""``scale_factors`` and ``fold_gates`` are copied too, not just the outer dict."""
		mock_fiqci_backend_class.return_value = Mock(mitigator_options={})

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, scale_factors=[1, 3], fold_gates=["cz"])

		options = estimator.mitigator_options
		options["zne"]["scale_factors"].append(99)
		options["zne"]["fold_gates"].append("clobbered")

		assert estimator._zne["scale_factors"] == [1, 3]
		assert estimator._zne["fold_gates"] == ["cz"]

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_nested_per_circuit_scale_factor_sublists_are_copied(self, mock_fiqci_backend_class: Mock) -> None:
		"""Per-circuit scale factors are a list of lists; the sublists must be copied as well."""
		mock_fiqci_backend_class.return_value = Mock(mitigator_options={})

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, scale_factors=[[1, 3], [1, 3, 5]])

		estimator.mitigator_options["zne"]["scale_factors"][0].append(99)

		assert estimator._zne["scale_factors"] == [[1, 3], [1, 3, 5]]


def _const_compute(exp_vals, raw=None, errors=None):
	"""Build a compute_fn returning fixed (expectation_values, raw_expectation_values, standard_errors)."""
	raw = exp_vals if raw is None else raw
	errors = (
		[{"shot_error": v, "zne_extrapolation_error": None, "total": v} for v in exp_vals] if errors is None else errors
	)
	return lambda: (exp_vals, raw, errors)


class TestFiQCIEstimatorJob:
	"""Tests for FiQCIEstimatorJob class (lazy expectation-value computation)."""

	def test_expectation_values_returns_all(self) -> None:
		"""Test that expectation_values() returns all values when no index given."""
		exp_vals = [[0.5, 0.3], [0.1, -0.2]]
		collection = FiQCIEstimatorJob(Mock(), _const_compute(exp_vals), Mock())

		assert collection.expectation_values() == exp_vals

	def test_expectation_values_by_index(self) -> None:
		"""Test that expectation_values(index) returns values for specific circuit."""
		exp_vals = [[0.5, 0.3], [0.1, -0.2]]
		collection = FiQCIEstimatorJob(Mock(), _const_compute(exp_vals), Mock())

		assert collection.expectation_values(0) == [0.5, 0.3]
		assert collection.expectation_values(1) == [0.1, -0.2]

	def test_observables_returns_all(self) -> None:
		"""Test that observables() returns all observables when no index given."""
		obs = SparsePauliOp.from_list([("ZZ", 1.0)])
		collection = FiQCIEstimatorJob(Mock(), _const_compute([[0.5]]), obs)

		assert collection.observables() is obs

	def test_observables_by_index(self) -> None:
		"""Test that observables(index) returns specific observable."""
		obs_list = [SparsePauliOp.from_list([("ZZ", 1.0)]), SparsePauliOp.from_list([("XX", 1.0)])]
		collection = FiQCIEstimatorJob(Mock(), _const_compute([[0.5], [0.3]]), obs_list)

		assert collection.observables(0) == obs_list[0]
		assert collection.observables(1) == obs_list[1]

	def test_standard_errors_returns_all_and_by_index(self) -> None:
		"""standard_errors() returns the per-pair list; standard_errors(i) returns one pair's dict."""
		errors = [
			{"shot_error": [0.01, 0.02], "zne_extrapolation_error": None, "total": [0.01, 0.02]},
			{"shot_error": [0.03], "zne_extrapolation_error": [0.05], "total": [0.05]},
		]
		collection = FiQCIEstimatorJob(Mock(), _const_compute([[0.5, 0.3], [0.1]], errors=errors), Mock())

		assert collection.standard_errors() == errors
		assert collection.standard_errors(0) == errors[0]
		assert collection.standard_errors(1) == errors[1]

	def test_computation_is_lazy_and_cached(self) -> None:
		"""compute_fn runs only on first value access, and exactly once."""
		calls: list[int] = []

		def compute():
			calls.append(1)
			return [[0.5]], [[0.5]], [{"shot_error": [0.01], "zne_extrapolation_error": None, "total": [0.01]}]

		collection = FiQCIEstimatorJob(Mock(), compute, Mock())
		# Not computed just by constructing.
		assert calls == []

		collection.expectation_values()
		collection.expectation_values()
		collection.raw_expectation_values()
		collection.standard_errors()
		assert len(calls) == 1

	def test_polling_delegates_to_underlying_job(self) -> None:
		"""status()/job_ids() delegate to the underlying job without computing values."""
		job = Mock()
		job.status.return_value = "RUNNING"
		job.job_ids.return_value = ["x", "y"]
		calls: list[int] = []

		def compute():
			calls.append(1)
			return [[0.5]], [[0.5]], [{"shot_error": [0.01], "zne_extrapolation_error": None, "total": [0.01]}]

		collection = FiQCIEstimatorJob(job, compute, Mock())

		assert collection.status() == "RUNNING"
		assert collection.job_ids() == ["x", "y"]
		assert calls == []


def _bell() -> QuantumCircuit:
	qc = QuantumCircuit(2)
	qc.h(0)
	qc.cx(0, 1)
	return qc


class TestTotalCircuitsGenerated:
	"""The advisory count must match what ``run`` actually submits.

	The measurement-group count depends on the observable, so taking it from ``observables[0]`` and
	applying it to every circuit under-reported whenever the observables differed per circuit.
	"""

	def _estimator(self, **zne_kwargs):
		from qiskit_aer import AerSimulator

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		if zne_kwargs:
			estimator.zne(enabled=True, extrapolation_method="linear", **zne_kwargs)
		return estimator

	def _submitted_count(self, estimator, circuits, observables) -> int:
		"""How many circuits actually reach the backend."""
		import warnings

		from qiskit_aer import AerSimulator

		counts: list[int] = []
		unpatched = AerSimulator.run

		def spy(backend, submitted, **kwargs):
			counts.append(len(submitted) if isinstance(submitted, list) else 1)
			return unpatched(backend, submitted, **kwargs)

		with patch.object(AerSimulator, "run", spy), warnings.catch_warnings():
			warnings.simplefilter("ignore")
			estimator.run(circuits, observables, shots=64).expectation_values(0)
		return sum(counts)

	def test_single_observable_scales_with_circuit_count(self) -> None:
		estimator = self._estimator()
		obs = SparsePauliOp(["ZZ", "XX"])  # two measurement groups

		assert estimator.total_circuits_generated(1, obs) == 2
		assert estimator.total_circuits_generated(3, obs) == 6

	def test_differing_observables_are_summed_not_multiplied(self) -> None:
		"""One group for the first pair plus three for the second is 4, not 2 x 1."""
		estimator = self._estimator()
		observables = [SparsePauliOp(["ZZ"]), SparsePauliOp(["ZZ", "XX", "YY"])]

		assert estimator.total_circuits_generated(2, observables) == 4

	def test_prediction_matches_submission_for_differing_observables(self) -> None:
		estimator = self._estimator()
		observables = [SparsePauliOp(["ZZ"]), SparsePauliOp(["ZZ", "XX", "YY"])]

		predicted = estimator.total_circuits_generated(2, observables)
		actual = self._submitted_count(estimator, [_bell(), _bell()], observables)

		assert predicted == actual == 4

	def test_prediction_matches_submission_with_nested_scale_factors(self) -> None:
		estimator = self._estimator(scale_factors=[[1, 3], [1, 3, 5]])
		obs = SparsePauliOp(["ZZ", "XX"])

		predicted = estimator.total_circuits_generated(2, obs)
		actual = self._submitted_count(estimator, [_bell(), _bell()], obs)

		assert predicted == actual == 10  # 2 groups * (2 + 3) scales

	def test_pauli_twirl_multiplies_the_total(self) -> None:
		estimator = self._estimator()
		estimator.pauli_twirl(True, num_twirls=2, seed=1)
		observables = [SparsePauliOp(["ZZ"]), SparsePauliOp(["ZZ", "XX"])]

		predicted = estimator.total_circuits_generated(2, observables)
		actual = self._submitted_count(estimator, [_bell(), _bell()], observables)

		assert predicted == actual == 9  # (1 + 2 groups) * (2 twirls + 1)

	def test_observable_count_mismatch_raises(self) -> None:
		estimator = self._estimator()

		with pytest.raises(ValueError, match="observable"):
			estimator.total_circuits_generated(3, [SparsePauliOp(["ZZ"]), SparsePauliOp(["XX"])])

	def test_nested_scale_factor_count_mismatch_raises(self) -> None:
		estimator = self._estimator(scale_factors=[[1, 3], [1, 3, 5]])

		with pytest.raises(ValueError, match="scale_factors"):
			estimator.total_circuits_generated(3, SparsePauliOp(["ZZ"]))

	def test_detailed_collapses_uniform_values_and_lists_varying_ones(self) -> None:
		estimator = self._estimator()

		uniform = estimator.total_circuits_generated(2, SparsePauliOp(["ZZ", "XX"]), detailed=True)
		assert uniform["measurement_circuits_per_basis"] == 2
		assert uniform["total_circuits"] == 4

		varying = estimator.total_circuits_generated(
			2, [SparsePauliOp(["ZZ"]), SparsePauliOp(["ZZ", "XX", "YY"])], detailed=True
		)
		assert varying["measurement_circuits_per_basis"] == [1, 3]
		assert varying["total_circuits"] == 4


class TestMitigationLevelValidation:
	"""All three interfaces reject a bad level the same way, so callers catch one exception type."""

	@pytest.mark.parametrize("level", [-1, 4, 7])
	def test_estimator_raises_value_error(self, level: int) -> None:
		from qiskit_aer import AerSimulator

		with pytest.raises(ValueError, match="mitigation_level must be 0-3"):
			FiQCIEstimator(AerSimulator(), mitigation_level=level)

	def test_all_interfaces_agree(self) -> None:
		from qiskit_aer import AerSimulator

		from fiqci.ems import FiQCIBackend, FiQCISampler

		for cls in (FiQCIBackend, FiQCISampler, FiQCIEstimator):
			with pytest.raises(ValueError, match="mitigation_level must be 0-3"):
				cls(AerSimulator(), mitigation_level=7)

	def test_default_shots_match_across_interfaces(self) -> None:
		"""A user moving between interfaces should not silently change shot count."""
		import inspect

		from fiqci.ems import FiQCIBackend, FiQCISampler

		defaults = {
			cls.__name__: inspect.signature(cls.run).parameters["shots"].default
			for cls in (FiQCIBackend, FiQCISampler, FiQCIEstimator)
		}

		assert len(set(defaults.values())) == 1, defaults


class TestFinalMeasurementRejection:
	"""A measured circuit transpiled for IQM has lost the RZ frame the X/Y bases depend on."""

	def _estimator(self):
		from qiskit_aer import AerSimulator

		return FiQCIEstimator(AerSimulator(), mitigation_level=0)

	def test_measure_all_is_rejected(self) -> None:
		circuit = _bell()
		circuit.measure_all()

		with pytest.raises(ValueError, match="ends in measurement"):
			self._estimator().run(circuit, SparsePauliOp(["ZZ"]), shots=64)

	def test_partially_measured_circuit_is_rejected(self) -> None:
		circuit = QuantumCircuit(2, 1)
		circuit.h(0)
		circuit.cx(0, 1)
		circuit.measure(0, 0)

		with pytest.raises(ValueError, match="ends in measurement"):
			self._estimator().run(circuit, SparsePauliOp(["ZZ"]), shots=64)

	def test_the_offending_circuit_index_is_reported(self) -> None:
		measured = _bell()
		measured.measure_all()

		with pytest.raises(ValueError, match="Circuit 1 ends in measurement"):
			self._estimator().run([_bell(), measured], SparsePauliOp(["ZZ"]), shots=64)

	def test_mid_circuit_measurement_is_still_accepted(self) -> None:
		circuit = QuantumCircuit(2, 1)
		circuit.h(0)
		circuit.measure(0, 0)
		circuit.cx(0, 1)

		values = self._estimator().run(circuit, SparsePauliOp(["ZZ"]), shots=1024).expectation_values(0)

		assert values[0] == pytest.approx(1.0, abs=0.05)

	def test_unmeasured_circuit_is_accepted(self) -> None:
		values = self._estimator().run(_bell(), SparsePauliOp(["ZZ"]), shots=1024).expectation_values(0)

		assert values[0] == pytest.approx(1.0, abs=0.05)
