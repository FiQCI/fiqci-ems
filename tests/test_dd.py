"""Unit tests for dynamical decoupling functionality."""

import warnings
from unittest.mock import Mock, patch

import pytest
from mthree.classes import QuasiDistribution
from iqm.iqm_client import (
	CircuitCompilationOptions,
	DDMode,
	DDStrategy,
	HeraldingMode,
	MoveGateFrameTrackingMode,
	MoveGateValidationMode,
	STANDARD_DD_STRATEGY,
)
from qiskit import QuantumCircuit

from fiqci.ems.mitigators.dd import build_dd_options, DDGateSequenceEntry
from fiqci.ems.backend import FiQCIBackend


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
		backend.has_resonators.return_value = False
		return backend

	@pytest.fixture
	def backend_level0(self, mock_backend: Mock) -> FiQCIBackend:
		"""Create a FiQCIBackend with mitigation level 0."""
		return FiQCIBackend(mock_backend, mitigation_level=0)

	@pytest.fixture
	def backend_level2(self, mock_backend: Mock) -> FiQCIBackend:
		"""Create a FiQCIBackend with mitigation level 2 (REM + DD)."""
		with patch("fiqci.ems.backend.core.M3IQM"):
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
		backend.has_resonators.return_value = False
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
		with pytest.raises(ValueError, match="threshold_length must be an integer"):
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
		backend.has_resonators.return_value = False
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
			patch("fiqci.ems.backend.core.M3IQM") as mock_m3iqm_class,
			patch("fiqci.ems.backend.core.final_measurement_mapping", return_value={0: 0, 1: 1}),
			patch("fiqci.ems.backend.core.probabilities_to_counts", return_value=[{"00": 480, "11": 520}]),
		):
			mock_mitigator = Mock()
			mock_mitigator.apply_correction.return_value = QuasiDistribution(
				{"00": 0.48, "11": 0.52}, shots=1024, mitigation_overhead=1.0
			)
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
		backend.has_resonators.return_value = False
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


class TestUserSuppliedCompilationOptions:
	"""DD must be layered onto caller-supplied compilation options, not replace them.

	``CircuitCompilationOptions`` carries settings that have nothing to do with DD — heralding, MOVE
	gate validation and frame tracking — and are the only way to reach them, since ems does not wrap
	them. Overwriting the whole object when DD is on would silently drop them.
	"""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		backend.has_resonators.return_value = False
		return backend

	@pytest.fixture
	def mock_circuit(self) -> QuantumCircuit:
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)
		qc.measure_all()
		return qc

	def test_build_dd_options_keeps_non_dd_fields_of_the_base(self) -> None:
		base = CircuitCompilationOptions(
			move_gate_validation=MoveGateValidationMode.ALLOW_PRX,
			move_gate_frame_tracking=MoveGateFrameTrackingMode.NONE,
			heralding_mode=HeraldingMode.ZEROS,
		)

		options = build_dd_options([(2, "XX", "center")], base=base)

		assert options.dd_mode == DDMode.ENABLED
		assert options.dd_strategy.gate_sequences == [(2, "XX", "center")]
		assert options.move_gate_validation == MoveGateValidationMode.ALLOW_PRX
		assert options.move_gate_frame_tracking == MoveGateFrameTrackingMode.NONE
		assert options.heralding_mode == HeraldingMode.ZEROS
		# The caller's object is not mutated.
		assert base.dd_mode == DDMode.DISABLED

	def test_run_merges_dd_into_user_options(self, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		mock_backend.run.return_value = Mock()

		backend = FiQCIBackend(mock_backend, mitigation_level=0)
		backend.dd(enabled=True, gate_sequences=[(2, "XX", "center")])
		backend.run(
			mock_circuit,
			shots=1024,
			circuit_compilation_options=CircuitCompilationOptions(
				move_gate_validation=MoveGateValidationMode.ALLOW_PRX
			),
		)

		options = mock_backend.run.call_args[1]["circuit_compilation_options"]
		assert options.dd_mode == DDMode.ENABLED
		assert options.move_gate_validation == MoveGateValidationMode.ALLOW_PRX

	def test_run_passes_user_options_through_unchanged_when_dd_is_off(
		self, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		mock_backend.run.return_value = Mock()
		options = CircuitCompilationOptions(move_gate_frame_tracking=MoveGateFrameTrackingMode.NONE)

		FiQCIBackend(mock_backend, mitigation_level=0).run(
			mock_circuit, shots=1024, circuit_compilation_options=options
		)

		assert mock_backend.run.call_args[1]["circuit_compilation_options"] is options

	def test_run_warns_when_user_options_already_configure_dd(
		self, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		mock_backend.run.return_value = Mock()

		backend = FiQCIBackend(mock_backend, mitigation_level=0)
		backend.dd(enabled=True, gate_sequences=[(2, "XX", "center")])

		with pytest.warns(UserWarning, match="already configures dynamical decoupling"):
			backend.run(
				mock_circuit,
				shots=1024,
				circuit_compilation_options=CircuitCompilationOptions(
					dd_mode=DDMode.ENABLED, dd_strategy=DDStrategy(gate_sequences=[(9, "XYXYYXYX", "asap")])
				),
			)

		options = mock_backend.run.call_args[1]["circuit_compilation_options"]
		assert options.dd_strategy.gate_sequences == [(2, "XX", "center")]

	def test_run_rejects_a_non_compilation_options_value(
		self, mock_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		mock_backend.run.return_value = Mock()

		backend = FiQCIBackend(mock_backend, mitigation_level=0)
		backend.dd(enabled=True)

		with pytest.raises(TypeError, match="CircuitCompilationOptions"):
			backend.run(mock_circuit, shots=1024, circuit_compilation_options={"dd_mode": "enabled"})


class TestStarArchitectureWarning:
	"""DD is not validated on Star devices, so enabling it there must say so before submitting."""

	@pytest.fixture
	def star_backend(self) -> Mock:
		backend = Mock()
		backend.name = "MockStarBackend"
		backend.num_qubits = 5
		backend.has_resonators.return_value = True
		backend.run.return_value = Mock()
		return backend

	@pytest.fixture
	def mock_circuit(self) -> QuantumCircuit:
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)
		qc.measure_all()
		return qc

	def test_warns_when_dd_is_submitted_to_a_resonator_device(
		self, star_backend: Mock, mock_circuit: QuantumCircuit
	) -> None:
		backend = FiQCIBackend(star_backend, mitigation_level=0)
		backend.dd(enabled=True)

		with pytest.warns(UserWarning, match="corrupts MOVE-routed circuits"):
			backend.run(mock_circuit, shots=1024)

	def test_warns_before_anything_is_submitted(self, star_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Shots are spent at submission, so the warning is only actionable if it comes first."""
		backend = FiQCIBackend(star_backend, mitigation_level=0)
		backend.dd(enabled=True)

		already_submitted: list[bool] = []
		with warnings.catch_warnings():
			warnings.simplefilter("always")
			warnings.showwarning = lambda *args, **kwargs: already_submitted.append(star_backend.run.called)
			backend.run(mock_circuit, shots=1024)

		assert already_submitted == [False]

	def test_no_warning_when_dd_is_disabled(self, star_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		with warnings.catch_warnings():
			warnings.simplefilter("error")
			FiQCIBackend(star_backend, mitigation_level=0).run(mock_circuit, shots=1024)

	def test_no_warning_on_a_device_without_resonators(self, mock_circuit: QuantumCircuit) -> None:
		crystal_backend = Mock()
		crystal_backend.name = "MockBackend"
		crystal_backend.num_qubits = 5
		crystal_backend.has_resonators.return_value = False
		crystal_backend.run.return_value = Mock()

		backend = FiQCIBackend(crystal_backend, mitigation_level=0)
		backend.dd(enabled=True)

		with warnings.catch_warnings():
			warnings.simplefilter("error")
			backend.run(mock_circuit, shots=1024)

	def test_no_warning_when_the_backend_cannot_report_resonators(self, mock_circuit: QuantumCircuit) -> None:
		"""A backend without ``has_resonators`` is not assumed to be a Star device."""
		plain_backend = Mock(spec=["name", "num_qubits", "run"])
		plain_backend.name = "MockBackend"
		plain_backend.num_qubits = 5
		plain_backend.run.return_value = Mock()

		backend = FiQCIBackend(plain_backend, mitigation_level=0)
		backend.dd(enabled=True)

		with warnings.catch_warnings():
			warnings.simplefilter("error")
			backend.run(mock_circuit, shots=1024)
