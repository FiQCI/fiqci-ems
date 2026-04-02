"""Unit tests for dynamical decoupling functionality."""

from unittest.mock import Mock, patch

import pytest
from iqm.iqm_client import CircuitCompilationOptions, DDMode, STANDARD_DD_STRATEGY
from qiskit import QuantumCircuit

from fiqci.ems.mitigators.dd import build_dd_options, DDGateSequenceEntry
from fiqci.ems.fiqci_backend import FiQCIBackend


class TestBuildDDOptions:
	"""Tests for the build_dd_options function."""

	def test_returns_circuit_compilation_options(self) -> None:
		"""Test that build_dd_options returns a CircuitCompilationOptions instance."""
		gate_sequences: list[DDGateSequenceEntry] = [(2, "XY", "asap")]
		result = build_dd_options(gate_sequences)
		assert isinstance(result, CircuitCompilationOptions)

	def test_dd_mode_is_enabled(self) -> None:
		"""Test that DD mode is set to ENABLED."""
		result = build_dd_options([(2, "XY", "asap")])
		assert result.dd_mode == DDMode.ENABLED

	def test_dd_strategy_contains_gate_sequences(self) -> None:
		"""Test that the DD strategy contains the provided gate sequences."""
		gate_sequences: list[DDGateSequenceEntry] = [(5, "YXYX", "asap"), (2, "XX", "center")]
		result = build_dd_options(gate_sequences)
		assert result.dd_strategy is not None
		assert result.dd_strategy.gate_sequences == gate_sequences

	def test_single_gate_sequence(self) -> None:
		"""Test with a single gate sequence entry."""
		result = build_dd_options([(9, "XYXYYXYX", "alap")])
		assert result.dd_strategy is not None
		assert len(result.dd_strategy.gate_sequences) == 1
		assert result.dd_strategy.gate_sequences[0] == (9, "XYXYYXYX", "alap")

	def test_prx_sequence(self) -> None:
		"""Test with PRX rotation angle tuples as the sequence."""
		prx_seq = [(0.5, 0.25), (0.75, 0.5)]
		result = build_dd_options([(4, prx_seq, "center")])
		assert result.dd_strategy is not None
		assert result.dd_strategy.gate_sequences[0] == (4, prx_seq, "center")

	def test_standard_dd_strategy_gate_sequences(self) -> None:
		"""Test with the standard DD strategy gate sequences from iqm_client."""
		result = build_dd_options(STANDARD_DD_STRATEGY.gate_sequences)
		assert result.dd_mode == DDMode.ENABLED
		assert result.dd_strategy.gate_sequences == STANDARD_DD_STRATEGY.gate_sequences


class TestFiQCIBackendDD:
	"""Tests for DD settings on FiQCIBackend."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		"""Create a mock IQM backend."""
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	@pytest.fixture
	def backend_level0(self, mock_backend: Mock) -> FiQCIBackend:
		"""Create a FiQCIBackend with mitigation level 0."""
		return FiQCIBackend(mock_backend, mitigation_level=0)

	@pytest.fixture
	def backend_level2(self, mock_backend: Mock) -> FiQCIBackend:
		"""Create a FiQCIBackend with mitigation level 2 (REM + DD)."""
		with patch("fiqci.ems.fiqci_backend.M3IQM"):
			return FiQCIBackend(mock_backend, mitigation_level=2)

	def test_dd_disabled_by_default_level0(self, backend_level0: FiQCIBackend) -> None:
		"""Test that DD is disabled by default at level 0."""
		assert backend_level0._dd["enabled"] is False
		assert backend_level0._dd["gate_sequences"] == []

	def test_dd_enabled_at_level2(self, backend_level2: FiQCIBackend) -> None:
		"""Test that DD is enabled at mitigation level 2."""
		assert backend_level2._dd["enabled"] is True
		assert backend_level2._dd["gate_sequences"] == STANDARD_DD_STRATEGY.gate_sequences

	def test_dd_enable_with_defaults(self, backend_level0: FiQCIBackend) -> None:
		"""Test enabling DD with default gate sequences uses STANDARD_DD_STRATEGY."""
		backend_level0.dd(enabled=True)
		assert backend_level0._dd["enabled"] is True
		assert backend_level0._dd["gate_sequences"] == STANDARD_DD_STRATEGY.gate_sequences

	def test_dd_enable_with_custom_sequences(self, backend_level0: FiQCIBackend) -> None:
		"""Test enabling DD with custom gate sequences."""
		custom_sequences: list[DDGateSequenceEntry] = [(3, "XYX", "alap")]
		backend_level0.dd(enabled=True, gate_sequences=custom_sequences)
		assert backend_level0._dd["enabled"] is True
		assert backend_level0._dd["gate_sequences"] == [(3, "XYX", "alap")]

	def test_dd_disable(self, backend_level2: FiQCIBackend) -> None:
		"""Test disabling DD."""
		backend_level2.dd(enabled=False)
		assert backend_level2._dd["enabled"] is False

	def test_dd_enable_with_none_sequences_uses_defaults(self, backend_level0: FiQCIBackend) -> None:
		"""Test that passing None gate_sequences uses the standard strategy."""
		backend_level0.dd(enabled=True, gate_sequences=None)
		assert backend_level0._dd["gate_sequences"] == STANDARD_DD_STRATEGY.gate_sequences

	def test_dd_enable_with_empty_sequences_uses_defaults(self, backend_level0: FiQCIBackend) -> None:
		"""Test that passing empty gate_sequences uses the standard strategy."""
		backend_level0.dd(enabled=True, gate_sequences=[])
		assert backend_level0._dd["gate_sequences"] == STANDARD_DD_STRATEGY.gate_sequences

	def test_dd_reported_in_mitigator_options(self, backend_level0: FiQCIBackend) -> None:
		"""Test that DD settings appear in mitigator_options."""
		backend_level0.dd(enabled=True)
		options = backend_level0.mitigator_options
		assert "dd" in options
		assert options["dd"]["enabled"] is True


class TestDDValidation:
	"""Tests for DD gate_sequences validation in _init_dd."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	@pytest.fixture
	def backend(self, mock_backend: Mock) -> FiQCIBackend:
		return FiQCIBackend(mock_backend, mitigation_level=0)

	def test_invalid_entry_not_tuple(self, backend: FiQCIBackend) -> None:
		"""Test that non-tuple entries raise ValueError."""
		with pytest.raises(ValueError, match="must be a tuple"):
			backend.dd(enabled=True, gate_sequences=["bad"])

	def test_invalid_entry_wrong_length(self, backend: FiQCIBackend) -> None:
		"""Test that entries with wrong length raise ValueError."""
		with pytest.raises(ValueError, match="must be a tuple"):
			backend.dd(enabled=True, gate_sequences=[(1, "X")])

	def test_invalid_strategy(self, backend: FiQCIBackend) -> None:
		"""Test that invalid strategy raises ValueError."""
		with pytest.raises(ValueError, match="Invalid strategy"):
			backend.dd(enabled=True, gate_sequences=[(2, "XY", "invalid_strategy")])

	def test_invalid_threshold_type(self, backend: FiQCIBackend) -> None:
		"""Test that non-integer threshold raises ValueError."""
		with pytest.raises(ValueError, match="treshold_length must be an integer"):
			backend.dd(enabled=True, gate_sequences=[(2.5, "XY", "asap")])

	def test_invalid_sequence_type(self, backend: FiQCIBackend) -> None:
		"""Test that invalid sequence type raises ValueError."""
		with pytest.raises(ValueError, match="sequence must be a string"):
			backend.dd(enabled=True, gate_sequences=[(2, 123, "asap")])

	def test_none_threshold_defaults_to_sequence_length(self, backend: FiQCIBackend) -> None:
		"""Test that None threshold defaults to sequence length."""
		backend.dd(enabled=True, gate_sequences=[(None, "XYXY", "asap")])
		assert backend._dd["gate_sequences"][0][0] == 4

	def test_none_threshold_none_sequence_defaults_to_2(self, backend: FiQCIBackend) -> None:
		"""Test that None threshold with None sequence defaults to 2."""
		backend.dd(enabled=True, gate_sequences=[(None, None, "asap")])
		assert backend._dd["gate_sequences"][0][0] == 2

	def test_none_strategy_defaults_to_asap(self, backend: FiQCIBackend) -> None:
		"""Test that None strategy defaults to 'asap'."""
		backend.dd(enabled=True, gate_sequences=[(2, "XY", None)])
		assert backend._dd["gate_sequences"][0][2] == "asap"

	def test_none_sequence_defaults_to_xy(self, backend: FiQCIBackend) -> None:
		"""Test that None sequence defaults to 'XY'."""
		backend.dd(enabled=True, gate_sequences=[(2, None, "asap")])
		assert backend._dd["gate_sequences"][0][1] == "XY"

	def test_all_none_fields_get_defaults(self, backend: FiQCIBackend) -> None:
		"""Test that all None fields get their defaults."""
		backend.dd(enabled=True, gate_sequences=[(None, None, None)])
		entry = backend._dd["gate_sequences"][0]
		assert entry == (2, "XY", "asap")

	def test_valid_strategies(self, backend: FiQCIBackend) -> None:
		"""Test all valid strategy values."""
		for strategy in ["asap", "alap", "center"]:
			backend.dd(enabled=True, gate_sequences=[(2, "XY", strategy)])
			assert backend._dd["gate_sequences"][0][2] == strategy


class TestDDRunIntegration:
	"""Tests for DD options being passed to the backend during run."""

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

	def test_dd_options_passed_to_backend_run_level0(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that DD compilation options are passed to backend.run when DD is enabled at level 0."""
		mock_job = Mock()
		mock_backend.run.return_value = mock_job

		backend = FiQCIBackend(mock_backend, mitigation_level=0)
		backend.dd(enabled=True)
		backend.run(mock_circuit, shots=1024)

		mock_backend.run.assert_called_once()
		call_kwargs = mock_backend.run.call_args[1]
		assert "circuit_compilation_options" in call_kwargs
		opts = call_kwargs["circuit_compilation_options"]
		assert opts.dd_mode == DDMode.ENABLED

	def test_no_dd_options_when_disabled(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that no DD compilation options are passed when DD is disabled."""
		mock_job = Mock()
		mock_backend.run.return_value = mock_job

		backend = FiQCIBackend(mock_backend, mitigation_level=0)
		backend.run(mock_circuit, shots=1024)

		call_kwargs = mock_backend.run.call_args[1]
		assert "circuit_compilation_options" not in call_kwargs

	def test_dd_options_passed_with_rem(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that DD options are passed when both DD and REM are enabled."""
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

			backend = FiQCIBackend(mock_backend, mitigation_level=2)
			backend.run(mock_circuit, shots=1024)

			call_kwargs = mock_backend.run.call_args[1]
			assert "circuit_compilation_options" in call_kwargs
			opts = call_kwargs["circuit_compilation_options"]
			assert opts.dd_mode == DDMode.ENABLED


class TestSamplerEstimatorDD:
	"""Tests for DD settings on FiQCISampler and FiQCIEstimator."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	@patch("fiqci.ems.primitives.fiqci_sampler.FiQCIBackend")
	def test_sampler_dd_delegates_to_backend(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that FiQCISampler.dd() delegates to FiQCIBackend.dd()."""
		from fiqci.ems.primitives.fiqci_sampler import FiQCISampler

		mock_fiqci_backend = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		sampler = FiQCISampler(mock_backend)
		sequences: list[DDGateSequenceEntry] = [(2, "XY", "asap")]
		sampler.dd(enabled=True, gate_sequences=sequences)

		mock_fiqci_backend.dd.assert_called_once_with(True, sequences)

	@patch("fiqci.ems.primitives.fiqci_sampler.FiQCIBackend")
	def test_sampler_dd_disable(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test disabling DD via FiQCISampler."""
		from fiqci.ems.primitives.fiqci_sampler import FiQCISampler

		mock_fiqci_backend = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		sampler = FiQCISampler(mock_backend)
		sampler.dd(enabled=False)

		mock_fiqci_backend.dd.assert_called_once_with(False, None)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_estimator_dd_delegates_to_backend(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that FiQCIEstimator.dd() delegates to FiQCIBackend.dd()."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		mock_fiqci_backend = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(mock_backend, mitigation_level=0)
		sequences: list[DDGateSequenceEntry] = [(3, "XYX", "center")]
		estimator.dd(enabled=True, gate_sequences=sequences)

		mock_fiqci_backend.dd.assert_called_once_with(True, sequences)
