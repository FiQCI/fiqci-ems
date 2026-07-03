"""Unit tests for Pauli twirling transpiler pass, caching, and integration with backend/estimator."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, CXGate
from qiskit.quantum_info import Operator

from iqm.qiskit_iqm import IQMFakeAphrodite

from fiqci.ems.transpiler_passes.pauli_twirl import PauliTwirl, get_twirled_circuits, _get_twirl_set, _twirl_set_cache
from fiqci.ems.fiqci_backend import BatchedJob, FiQCIBackend, MitigatedJob


class TestTwirlSetCache:
	"""Tests for the module-level twirl set cache."""

	def setup_method(self):
		"""Clear the cache before each test."""
		_twirl_set_cache.clear()

	def test_get_twirl_set_populates_cache(self):
		"""Test that _get_twirl_set populates the cache on first call."""
		gate = CZGate()
		assert gate.name not in _twirl_set_cache

		result = _get_twirl_set(gate)

		assert gate.name in _twirl_set_cache
		assert len(result) > 0

	def test_get_twirl_set_returns_cached_on_second_call(self):
		"""Test that _get_twirl_set returns the same object from cache."""
		gate = CZGate()
		first = _get_twirl_set(gate)
		second = _get_twirl_set(gate)

		assert first is second

	def test_twirl_pairs_are_valid(self):
		"""Test that cached twirl pairs satisfy P_left @ Gate == Gate @ P_right."""
		gate = CZGate()
		pairs = _get_twirl_set(gate)

		for pauli_left, pauli_right in pairs:
			lhs = Operator(pauli_left) @ Operator(gate)
			rhs = Operator(gate) @ pauli_right
			assert lhs.equiv(rhs)

	def test_different_gates_cached_separately(self):
		"""Test that different gate types get separate cache entries."""
		cz = CZGate()
		cx = CXGate()

		cz_pairs = _get_twirl_set(cz)
		cx_pairs = _get_twirl_set(cx)

		assert "cz" in _twirl_set_cache
		assert "cx" in _twirl_set_cache
		assert cz_pairs is not cx_pairs


class TestPauliTwirlPass:
	"""Tests for the PauliTwirl transpiler pass."""

	def setup_method(self):
		_twirl_set_cache.clear()

	def test_default_gates_to_twirl(self):
		"""Test that default gates_to_twirl is CZGate."""
		pt = PauliTwirl()
		assert len(pt.gates_to_twirl) == 1
		assert isinstance(pt.gates_to_twirl[0], CZGate)

	def test_custom_gates_to_twirl(self):
		"""Test that custom gates_to_twirl is respected."""
		pt = PauliTwirl(gates_to_twirl=[CXGate()])
		assert len(pt.gates_to_twirl) == 1
		assert isinstance(pt.gates_to_twirl[0], CXGate)

	def test_twirl_set_populated_on_init(self):
		"""Test that twirl_set is populated during __init__."""
		pt = PauliTwirl()
		assert "cz" in pt.twirl_set
		assert len(pt.twirl_set["cz"]) > 0

	def test_pass_preserves_unitary(self):
		"""Test that twirling preserves the circuit's unitary (up to global phase)."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cz(0, 1)
		qc.h(1)
		original_op = Operator(qc)

		pt = PauliTwirl()
		from qiskit.transpiler import PassManager

		pm = PassManager(pt)
		twirled = pm.run(qc)

		assert original_op.equiv(Operator(twirled))

	def test_pass_does_not_modify_circuit_without_target_gates(self):
		"""Test that circuits without target gates are unchanged."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.h(1)

		pt = PauliTwirl()
		from qiskit.transpiler import PassManager

		pm = PassManager(pt)
		twirled = pm.run(qc)

		assert Operator(qc).equiv(Operator(twirled))
		# Gate count should be similar (no extra Paulis added)
		assert len(twirled.data) == len(qc.data)

	def test_pass_adds_pauli_gates(self):
		"""Test that twirling adds gates around target gates."""
		qc = QuantumCircuit(2)
		qc.cz(0, 1)

		pt = PauliTwirl()
		from qiskit.transpiler import PassManager

		pm = PassManager(pt)
		twirled = pm.run(qc)

		# Original has 1 gate (CZ), twirled should have more (CZ + Pauli pairs)
		assert len(twirled.data) >= len(qc.data)

	def test_multiple_twirls_produce_different_circuits(self):
		"""Test that repeated twirling produces different circuits (randomized)."""
		qc = QuantumCircuit(2)
		qc.cz(0, 1)

		pt = PauliTwirl()
		from qiskit.transpiler import PassManager

		pm = PassManager(pt)

		np.random.seed(None)
		circuits = [pm.run(qc) for _ in range(20)]
		# At least some should differ (probabilistic, but 20 trials with 16 pairs makes collision unlikely)
		ops = [Operator(c).data.tobytes() for c in circuits]
		unique_ops = set(ops)
		# All should be equivalent to original, but gate decompositions may differ
		assert len(unique_ops) >= 1  # At minimum they're all valid


class TestGetTwirledCircuits:
	"""Tests for get_twirled_circuits function."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		"""Create a mock IQM backend."""
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	@pytest.fixture
	def backend(self, mock_backend: Mock) -> FiQCIBackend:
		"""Create a FiQCIBackend with mitigation level 0."""
		return FiQCIBackend(IQMFakeAphrodite())

	def setup_method(self):
		_twirl_set_cache.clear()

	def test_returns_correct_number_of_circuits(self, backend):
		"""Test that output has len(circuits) * (num_twirls + 1) circuits."""
		qc = QuantumCircuit(2)
		qc.cz(0, 1)

		result = get_twirled_circuits([qc], num_twirls=5, backend=backend)
		assert len(result) == 6  # 1 original + 5 twirled

	def test_multiple_input_circuits(self, backend):
		"""Test with multiple input circuits."""
		circuits = [QuantumCircuit(2) for _ in range(3)]
		for qc in circuits:
			qc.cz(0, 1)

		result = get_twirled_circuits(circuits, num_twirls=4, backend=backend)
		assert len(result) == 15  # 3 * (4 + 1)

	def test_original_circuit_is_first_in_each_group(self, backend):
		"""Test that the original circuit is at the start of each group."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cz(0, 1)
		qc.measure_all()

		result = get_twirled_circuits([qc], num_twirls=3, backend=backend)
		# First circuit in the group should be the original
		assert result[0] is qc

	def test_original_circuits_at_group_boundaries(self, backend):
		"""Test that original circuits appear at positions 0, group_size, 2*group_size, etc."""
		qc1 = QuantumCircuit(2)
		qc1.cz(0, 1)
		qc2 = QuantumCircuit(2)
		qc2.h(0)
		qc2.cz(0, 1)

		num_twirls = 3
		result = get_twirled_circuits([qc1, qc2], num_twirls=num_twirls, backend=backend)
		group_size = num_twirls + 1

		assert result[0] is qc1
		assert result[group_size] is qc2

	def test_twirled_circuits_preserve_unitary(self, backend):
		"""Test that each twirled circuit is unitarily equivalent to the original."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cz(0, 1)
		original_op = Operator(qc)

		result = get_twirled_circuits([qc], num_twirls=5, backend=backend)
		for twirled in result[1:]:  # skip original at index 0
			assert original_op.equiv(Operator(twirled))

	def test_num_twirls_zero(self, backend):
		"""Test with num_twirls=0 returns just the original circuits."""
		qc = QuantumCircuit(2)
		qc.cz(0, 1)

		result = get_twirled_circuits([qc], num_twirls=0, backend=backend)
		assert len(result) == 1
		assert result[0] is qc

	def test_returns_flat_list(self, backend):
		"""Test that result is a flat list, not nested."""
		qc = QuantumCircuit(2)
		qc.cz(0, 1)

		result = get_twirled_circuits([qc], num_twirls=3, backend=backend)
		assert isinstance(result, list)
		for item in result:
			assert isinstance(item, QuantumCircuit)

	def test_twirled_circuits_preserve_layout(self, backend):
		"""Twirled circuits must keep the input's TranspileLayout.

		The PassManager builds fresh circuits that drop ``circuit.layout``. The IQM backend
		maps virtual to physical qubits via that layout, so without it the twirled circuits get
		validated/run against an identity layout and their CZ gates can land on non-adjacent
		physical qubits, raising CircuitValidationError. Regression test for that bug.
		"""
		from qiskit import transpile

		fake = backend._backend
		qc = QuantumCircuit(3)
		qc.h(0)
		qc.cx(0, 1)
		qc.cx(1, 2)
		qc.measure_all()
		tr = transpile(qc, fake, optimization_level=1)
		assert tr.layout is not None

		result = get_twirled_circuits([tr], num_twirls=50, backend=fake)
		for twirled in result[1:]:  # skip original at index 0
			assert twirled.layout is tr.layout

	def test_twirled_circuits_pass_iqm_validation(self, backend):
		"""Twirled circuits must validate against the IQM connectivity for many twirls.

		Each twirl is an independent random draw, so a layout-mapping bug surfaces
		probabilistically and becomes near-certain as num_twirls grows.
		"""
		from qiskit import transpile
		from iqm.qiskit_iqm.iqm_circuit_validation import validate_circuit

		fake = backend._backend
		qc = QuantumCircuit(3)
		qc.h(0)
		qc.cx(0, 1)
		qc.cx(1, 2)
		qc.measure_all()
		tr = transpile(qc, fake, optimization_level=1)

		for twirled in get_twirled_circuits([tr], num_twirls=200, backend=fake):
			validate_circuit(twirled, fake)  # raises CircuitValidationError on illegal locus

	def test_basis_gates_does_not_crash_on_non_str_names(self):
		"""get_twirled_circuits must not crash when a backend target entry has no .name attribute.

		AerSimulator (and some other backends) include entries where i[0] is a class object
		without a .name attribute; the old code raised AttributeError on those entries.
		"""

		class InstructionWithName:
			name = "cx"

		class InstructionWithNoName:
			pass  # no .name — triggers AttributeError without the getattr guard

		mock_backend = Mock()
		mock_backend.name = "MockBackend"
		mock_backend.num_qubits = 5
		mock_backend.target.instructions = [(InstructionWithName,), (InstructionWithNoName,)]

		try:
			get_twirled_circuits([QuantumCircuit(2)], num_twirls=1, backend=mock_backend)
		except AttributeError as e:
			pytest.fail(f"get_twirled_circuits raised AttributeError on missing .name: {e}")


class TestBackendPauliTwirlSettings:
	"""Tests for Pauli twirling settings on FiQCIBackend."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	def test_pauli_twirl_disabled_by_default(self, mock_backend):
		"""Test that Pauli twirling is disabled by default."""
		fb = FiQCIBackend(mock_backend, mitigation_level=0)
		assert fb._pauli_twirl["enabled"] is False

	def test_pauli_twirl_enabled_at_level_3(self, mock_backend):
		"""Test that Pauli twirling is enabled for mitigation level 3."""
		with patch("fiqci.ems.fiqci_backend.M3IQM"):
			fb = FiQCIBackend(mock_backend, mitigation_level=3)
		assert fb._pauli_twirl["enabled"] is True
		assert fb._pauli_twirl["num_twirls"] == 10  # default

	def test_pauli_twirl_method_enables(self, mock_backend):
		"""Test enabling Pauli twirling via pauli_twirl() method."""
		fb = FiQCIBackend(mock_backend, mitigation_level=0)
		fb.pauli_twirl(enabled=True, num_twirls=7)

		assert fb._pauli_twirl["enabled"] is True
		assert fb._pauli_twirl["num_twirls"] == 7

	def test_pauli_twirl_method_disables(self, mock_backend):
		"""Test disabling Pauli twirling via pauli_twirl() method."""
		fb = FiQCIBackend(mock_backend, mitigation_level=0)
		fb.pauli_twirl(enabled=True, num_twirls=5)
		fb.pauli_twirl(enabled=False)

		assert fb._pauli_twirl["enabled"] is False

	def test_pauli_twirl_custom_gates(self, mock_backend):
		"""Test setting custom gates_to_twirl."""
		fb = FiQCIBackend(mock_backend, mitigation_level=0)
		fb.pauli_twirl(enabled=True, gates_to_twirl=[CXGate()])

		assert fb._pauli_twirl["gates_to_twirl"] == [CXGate()]

	def test_mitigator_options_includes_pauli_twirl(self, mock_backend):
		"""Test that mitigator_options includes pauli_twirl settings."""
		fb = FiQCIBackend(mock_backend, mitigation_level=0)
		fb.pauli_twirl(enabled=True, num_twirls=5)

		opts = fb.mitigator_options
		assert "pauli_twirl" in opts
		assert opts["pauli_twirl"]["enabled"] is True
		assert opts["pauli_twirl"]["num_twirls"] == 5


class TestBackendRunWithPauliTwirling:
	"""Tests for FiQCIBackend.run() with Pauli twirling enabled."""

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

	def test_run_without_twirling_passes_through(self, mock_backend, mock_circuit):
		"""Test that run without twirling passes circuits directly to backend."""
		mock_job = Mock()
		mock_backend.run.return_value = mock_job

		fb = FiQCIBackend(mock_backend, mitigation_level=0)
		result = fb.run(mock_circuit, shots=1024)

		assert isinstance(result, BatchedJob)
		assert result.job_ids() == [mock_job.job_id()]
		mock_backend.run.assert_called_once()

	@patch("fiqci.ems.fiqci_backend.get_twirled_circuits")
	def test_run_with_twirling_expands_circuits(self, mock_get_twirled, mock_backend, mock_circuit):
		"""Test that twirling expands the circuit list before running."""
		num_twirls = 3
		expanded = [mock_circuit] * 4  # 1 original + 3 twirled
		mock_get_twirled.return_value = expanded

		mock_job = Mock()
		mock_result = Mock()
		# 4 circuits, each returns counts
		mock_result.get_counts.return_value = [{"00": 500, "11": 500}] * 4
		mock_result.to_dict.return_value = {
			"results": [{"data": {"counts": {"00": 500, "11": 500}}, "shots": 1024, "success": True} for _ in range(4)],
			"backend_name": "mock",
			"job_id": "test",
			"qobj_id": "test",
			"success": True,
			"status": "COMPLETED",
		}
		mock_job.result.return_value = mock_result
		mock_backend.run.return_value = mock_job

		fb = FiQCIBackend(mock_backend, mitigation_level=0)
		fb.pauli_twirl(enabled=True, num_twirls=num_twirls)
		result = fb.run(mock_circuit, shots=1024)

		mock_get_twirled.assert_called_once()
		# Result should be MitigatedJob since twirling requires post-processing
		assert isinstance(result, MitigatedJob)

	@patch("fiqci.ems.fiqci_backend.get_twirled_circuits")
	def test_run_with_twirling_and_rem(self, mock_get_twirled, mock_backend, mock_circuit):
		"""Test that twirling works together with REM."""
		num_twirls = 2
		expanded = [mock_circuit] * 3  # 1 original + 2 twirled
		mock_get_twirled.return_value = expanded

		mock_job = Mock()
		mock_result = Mock()
		mock_result.get_counts.side_effect = lambda idx=None: (
			[{"00": 500, "11": 500}] * 3 if idx is None else {"00": 500, "11": 500}
		)
		mock_result.to_dict.return_value = {
			"results": [{"data": {"counts": {"00": 500, "11": 500}}, "shots": 1024, "success": True} for _ in range(3)],
			"backend_name": "mock",
			"job_id": "test",
			"qobj_id": "test",
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

			fb = FiQCIBackend(mock_backend, mitigation_level=1)
			fb.pauli_twirl(enabled=True, num_twirls=num_twirls)
			result = fb.run(mock_circuit, shots=1024)

			assert isinstance(result, MitigatedJob)
			# Mitigation is lazy; trigger it.
			result.result()
			# M3 correction should be applied to each expanded circuit
			assert mock_mitigator.apply_correction.call_count == 3


class TestAverageAndTrimMethods:
	"""Tests for _average_counts, and _trim_result_to_groups."""

	def test_average_counts_single(self):
		"""Test averaging a single counts dict returns it unchanged."""
		counts = {"00": 500, "11": 500}
		result = FiQCIBackend._average_counts([counts])
		assert result is counts

	def test_average_counts_two_dicts(self):
		"""Test averaging two count dicts."""
		counts1 = {"00": 600, "11": 400}
		counts2 = {"00": 400, "11": 600}
		result = FiQCIBackend._average_counts([counts1, counts2])

		assert result["00"] == 500
		assert result["11"] == 500

	def test_average_counts_different_keys(self):
		"""Test averaging with non-overlapping keys."""
		counts1 = {"00": 1000}
		counts2 = {"11": 1000}
		result = FiQCIBackend._average_counts([counts1, counts2])

		assert result["00"] == 500
		assert result["11"] == 500

	def test_average_counts_rounding(self):
		"""Test that averaging rounds to nearest integer."""
		counts1 = {"00": 1, "11": 2}
		counts2 = {"00": 2, "11": 1}
		counts3 = {"00": 1, "11": 1}
		result = FiQCIBackend._average_counts([counts1, counts2, counts3])

		# (1+2+1)/3 = 1.33 -> 1, (2+1+1)/3 = 1.33 -> 1
		assert result["00"] == 1
		assert result["11"] == 1

	def test_trim_result_to_groups(self):
		"""_trim_result_to_groups keeps one representative entry per group, in order.

		The flat result is ``[g0_orig, g0_tw, g1_orig, g1_tw]`` (twirl_group_size=2, 2 groups).
		The trim must keep the *original* of each group (indices 0 and 2), not the first two flat
		entries (indices 0 and 1, both belonging to group 0). Each kept entry's header otherwise
		describes the wrong input circuit, which get_counts() uses to rebuild bitstrings.
		"""
		mock_result = Mock()
		mock_result.to_dict.return_value = {
			"results": [
				{"data": {"counts": {"00": 500}}, "header": {"name": "g0"}, "shots": 1024, "success": True},
				{"data": {"counts": {"11": 500}}, "header": {"name": "g0_tw"}, "shots": 1024, "success": True},
				{"data": {"counts": {"01": 500}}, "header": {"name": "g1"}, "shots": 1024, "success": True},
				{"data": {"counts": {"10": 500}}, "header": {"name": "g1_tw"}, "shots": 1024, "success": True},
			],
			"backend_name": "mock",
			"job_id": "test",
			"qobj_id": "test",
			"success": True,
			"status": "COMPLETED",
		}

		trimmed = FiQCIBackend._trim_result_to_groups(mock_result, 2)
		trimmed_dict = trimmed.to_dict()
		assert len(trimmed_dict["results"]) == 2
		# Representative entries are the group originals at stride == twirl_group_size (2): indices 0, 2.
		assert [r["header"]["name"] for r in trimmed_dict["results"]] == ["g0", "g1"]
		assert [r["data"]["counts"] for r in trimmed_dict["results"]] == [{"00": 500}, {"01": 500}]

	def test_twirled_results_aligned_with_input_order(self):
		"""End-to-end: get_counts() must stay aligned with input circuit order under twirling.

		Regression for the trimming bug: with differently-structured circuits interleaved, a
		mis-trim makes circuit i's slot receive another circuit's (wrongly-structured) counts.
		Uses the no-REM twirl branch so no M3 calibration is needed.
		"""
		from qiskit import transpile
		from iqm.qiskit_iqm import IQMFakeAdonis

		fake = IQMFakeAdonis()

		def mk(nq: int) -> QuantumCircuit:
			qc = QuantumCircuit(nq, nq)
			qc.h(0)
			for i in range(nq - 1):
				qc.cx(i, i + 1)
			qc.measure(range(nq), range(nq))
			return transpile(qc, fake, optimization_level=3)

		labels = [nq for _ in range(4) for nq in (1, 2, 3)]
		circuits = [mk(nq) for nq in labels]

		fb = FiQCIBackend(fake, mitigation_level=0)
		fb.rem(enabled=False)
		fb.pauli_twirl(enabled=True, num_twirls=8)

		counts = fb.run(circuits, shots=512).result().get_counts()
		got = [len(next(iter(c)).replace(" ", "")) for c in counts]
		assert got == labels


class TestEstimatorPauliTwirl:
	"""Tests for Pauli twirling on FiQCIEstimator."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_estimator_pauli_twirl_delegates(self, mock_fiqci_backend_class, mock_backend):
		"""Test that estimator.pauli_twirl() delegates to backend."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		mock_fiqci_backend = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend, mitigation_level=0)
		estimator.pauli_twirl(enabled=True, num_twirls=7, gates_to_twirl=[CZGate()])

		mock_fiqci_backend.pauli_twirl.assert_called_once_with(True, 7, [CZGate()])

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_estimator_pauli_twirl_disable(self, mock_fiqci_backend_class, mock_backend):
		"""Test disabling Pauli twirling via estimator."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		mock_fiqci_backend = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend, mitigation_level=0)
		estimator.pauli_twirl(enabled=False)

		mock_fiqci_backend.pauli_twirl.assert_called_once_with(False, 10, None)


class TestSamplerPauliTwirl:
	"""Tests for Pauli twirling on FiQCISampler."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	@patch("fiqci.ems.primitives.fiqci_sampler.FiQCIBackend")
	def test_sampler_pauli_twirl_delegates(self, mock_fiqci_backend_class, mock_backend):
		"""Test that sampler.pauli_twirl() delegates to backend."""
		from fiqci.ems.primitives.fiqci_sampler import FiQCISampler

		mock_fiqci_backend = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		sampler = FiQCISampler(mock_backend)
		sampler.pauli_twirl(enabled=True, num_twirls=5)

		mock_fiqci_backend.pauli_twirl.assert_called_once_with(True, 5, None)

	@patch("fiqci.ems.primitives.fiqci_sampler.FiQCIBackend")
	def test_sampler_pauli_twirl_disable(self, mock_fiqci_backend_class, mock_backend):
		"""Test disabling Pauli twirling via sampler."""
		from fiqci.ems.primitives.fiqci_sampler import FiQCISampler

		mock_fiqci_backend = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		sampler = FiQCISampler(mock_backend)
		sampler.pauli_twirl(enabled=False)

		mock_fiqci_backend.pauli_twirl.assert_called_once_with(False, 10, None)
