"""Unit tests for Zero-Noise Extrapolation (ZNE) functionality."""

import warnings
from unittest.mock import Mock, patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from fiqci.ems.mitigators.zne import exponential_extrapolation, polynomial_extrapolation, richardson_extrapolation
from fiqci.ems.transpiler_passes.zne_circuits import ZNECircuits, _get_zne_circuits


class TestExponentialExtrapolation:
	"""Tests for exponential_extrapolation function."""

	def test_known_exponential_decay(self) -> None:
		"""Test extrapolation recovers zero-noise value for exact exponential data."""
		# y = 0.8 * exp(-0.1 * x)  => at x=0, y = 0.8
		scale_factors = [1, 3, 5]
		expectation_values = [[0.8 * np.exp(-0.1 * s)] for s in scale_factors]

		result = exponential_extrapolation(expectation_values, scale_factors)

		assert len(result) == 1
		assert result[0] == pytest.approx(0.8, abs=0.01)

	def test_two_point_extrapolation(self) -> None:
		"""Test with minimum two data points."""
		scale_factors = [1, 3]
		expectation_values = [[0.9], [0.7]]

		result = exponential_extrapolation(expectation_values, scale_factors)

		assert len(result) == 1
		# Extrapolated value should be larger than the value at scale=1
		assert result[0] > 0.9

	def test_negative_expectation_values(self) -> None:
		"""Test that negative expectation values are handled correctly."""
		scale_factors = [1, 3, 5]
		expectation_values = [[-0.8 * np.exp(-0.1 * s)] for s in scale_factors]

		result = exponential_extrapolation(expectation_values, scale_factors)

		assert len(result) == 1
		assert result[0] == pytest.approx(-0.8, abs=0.01)

	def test_multiple_observables(self) -> None:
		"""Test extrapolation with multiple observables per scale factor."""
		scale_factors = [1, 3, 5]
		expectation_values = [[0.8 * np.exp(-0.1 * s), 0.5 * np.exp(-0.2 * s)] for s in scale_factors]

		result = exponential_extrapolation(expectation_values, scale_factors)

		assert len(result) == 2
		assert result[0] == pytest.approx(0.8, abs=0.01)
		assert result[1] == pytest.approx(0.5, abs=0.01)

	def test_zero_value_in_data_is_finite(self) -> None:
		"""A literal 0.0 must not produce log(0) -> -inf -> nan."""
		scale_factors = [1, 3, 5]
		expectation_values = [[0.5], [0.3], [0.0]]

		result = exponential_extrapolation(expectation_values, scale_factors)

		assert len(result) == 1
		assert np.isfinite(result[0])

	def test_sign_flip_near_zero_is_finite(self) -> None:
		"""Values that flip sign around zero due to noise must stay finite."""
		scale_factors = [1, 3, 5]
		expectation_values = [[0.02], [-0.01], [0.015]]

		result = exponential_extrapolation(expectation_values, scale_factors)

		assert len(result) == 1
		assert np.isfinite(result[0])

	def test_all_zero_column_returns_zero(self) -> None:
		"""A column with no signal should extrapolate to zero, not nan."""
		scale_factors = [1, 3, 5]
		expectation_values = [[0.0], [0.0], [0.0]]

		result = exponential_extrapolation(expectation_values, scale_factors)

		assert result == [0.0]

	def test_real_world_near_zero_observables(self) -> None:
		"""Regression: noisy near-zero data that previously yielded all-nan output."""
		scale_factors = [1, 3, 5]
		expectation_values = [
			[
				-0.01568627450980392,
				-0.0196078431372549,
				-0.006862745098039216,
				-0.025490196078431372,
				-0.03137254901960784,
				0.7254901960784313,
				-0.026470588235294117,
			],
			[
				0.03740157480314961,
				0.02952755905511811,
				0.05905511811023622,
				0.06496062992125984,
				0.06299212598425197,
				0.46062992125984253,
				0.07874015748031496,
			],
			[
				0.003937007874015748,
				0.006889763779527559,
				0.01968503937007874,
				0.008858267716535433,
				0.0,
				0.26968503937007876,
				-0.011811023622047244,
			],
		]

		result = exponential_extrapolation(expectation_values, scale_factors)

		assert len(result) == 7
		assert all(np.isfinite(v) for v in result)

	def test_too_few_points_raises_error(self) -> None:
		"""Test that fewer than 2 expectation values raises ValueError."""
		with pytest.raises(ValueError, match="At least two expectation values"):
			exponential_extrapolation([[0.5]], [1])

	def test_empty_list_raises_error(self) -> None:
		"""Test that empty list raises ValueError."""
		with pytest.raises(ValueError, match="At least two expectation values"):
			exponential_extrapolation([], [])


class TestRichardsonExtrapolation:
	"""Tests for richardson_extrapolation function."""

	def test_linear_data_two_points(self) -> None:
		"""Test Richardson extrapolation with linear data and two points."""
		# y = 1.0 - 0.1 * x => at x=0, y = 1.0
		scale_factors = [1, 3]
		expectation_values = [[1.0 - 0.1 * s] for s in scale_factors]

		result = richardson_extrapolation(expectation_values, scale_factors)

		assert len(result) == 1
		assert result[0] == pytest.approx(1.0, abs=1e-10)

	def test_quadratic_data_three_points(self) -> None:
		"""Test Richardson extrapolation with quadratic data and three points."""
		# y = 1.0 - 0.05*x^2 => at x=0, y = 1.0
		scale_factors = [1, 3, 5]
		expectation_values = [[1.0 - 0.05 * s**2] for s in scale_factors]

		result = richardson_extrapolation(expectation_values, scale_factors)

		assert len(result) == 1
		assert result[0] == pytest.approx(1.0, abs=1e-10)

	def test_flat_input(self) -> None:
		"""Test with 1D input (no nested lists)."""
		scale_factors = [1, 3]
		expectation_values = [0.9, 0.7]

		result = richardson_extrapolation(expectation_values, scale_factors)

		assert len(result) == 1
		assert result[0] == pytest.approx(1.0, abs=1e-10)

	def test_multiple_observables(self) -> None:
		"""Test with multiple observables."""
		scale_factors = [1, 3]
		expectation_values = [[0.9, 0.8], [0.7, 0.6]]

		result = richardson_extrapolation(expectation_values, scale_factors)

		assert len(result) == 2

	def test_length_mismatch_raises_error(self) -> None:
		"""Test that mismatched scales and values raises ValueError."""
		with pytest.raises(ValueError, match="Length mismatch"):
			richardson_extrapolation([[0.9], [0.7], [0.5]], [1, 3])


class TestPolynomialExtrapolation:
	"""Tests for polynomial_extrapolation function."""

	def test_linear_data_degree_1(self) -> None:
		"""Test polynomial extrapolation with linear data and degree 1."""
		scale_factors = [1, 3, 5]
		expectation_values = [[1.0 - 0.1 * s] for s in scale_factors]

		result = polynomial_extrapolation(expectation_values, scale_factors, degree=1)

		assert len(result) == 1
		assert result[0] == pytest.approx(1.0, abs=1e-10)

	def test_quadratic_data_degree_2(self) -> None:
		"""Test polynomial extrapolation with quadratic data and degree 2."""
		scale_factors = [1, 3, 5]
		expectation_values = [[1.0 - 0.02 * s**2] for s in scale_factors]

		result = polynomial_extrapolation(expectation_values, scale_factors, degree=2)

		assert len(result) == 1
		assert result[0] == pytest.approx(1.0, abs=1e-10)

	def test_default_degree(self) -> None:
		"""Test that default degree is min(n_scales - 1, 2)."""
		scale_factors = [1, 3, 5, 7]
		expectation_values = [[1.0 - 0.02 * s**2] for s in scale_factors]

		# Default degree should be min(4-1, 2) = 2
		result = polynomial_extrapolation(expectation_values, scale_factors)

		assert len(result) == 1
		assert result[0] == pytest.approx(1.0, abs=0.01)

	def test_flat_input(self) -> None:
		"""Test with 1D flat input."""
		scale_factors = [1, 3, 5]
		expectation_values = [0.9, 0.7, 0.5]

		result = polynomial_extrapolation(expectation_values, scale_factors, degree=1)

		assert len(result) == 1
		assert result[0] == pytest.approx(1.0, abs=1e-10)

	def test_multiple_observables(self) -> None:
		"""Test with multiple observables."""
		scale_factors = [1, 3, 5]
		expectation_values = [[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]]

		result = polynomial_extrapolation(expectation_values, scale_factors, degree=1)

		assert len(result) == 2

	def test_length_mismatch_raises_error(self) -> None:
		"""Test that mismatched scales and values raises ValueError."""
		with pytest.raises(ValueError, match="Length mismatch"):
			polynomial_extrapolation([[0.9], [0.7]], [1, 3, 5])


class TestZNECircuitsPass:
	"""Tests for ZNECircuits transpiler pass."""

	def test_scale_factor_1_returns_unchanged_circuit(self) -> None:
		"""Test that scale_factor=1 does not modify the circuit."""
		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		pm = PassManager(ZNECircuits(scale_factor=1))
		result = pm.run(qc)

		assert result.size() == qc.size()

	def test_scale_factor_3_triples_two_qubit_gates(self) -> None:
		"""Test that scale_factor=3 triples two-qubit gates."""
		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		pm = PassManager(ZNECircuits(scale_factor=3))
		result = pm.run(qc)

		# Original 1 CX should become 3
		cx_count = sum(1 for inst in result.data if inst.operation.name == "cx")
		assert cx_count == 3

	def test_single_qubit_gates_not_folded(self) -> None:
		"""Test that single-qubit gates are not folded."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)

		pm = PassManager(ZNECircuits(scale_factor=3))
		result = pm.run(qc)

		h_count = sum(1 for inst in result.data if inst.operation.name == "h")
		cx_count = sum(1 for inst in result.data if inst.operation.name == "cx")

		assert h_count == 1  # H gate unchanged
		assert cx_count == 3  # CX gate tripled

	def test_barriers_not_folded(self) -> None:
		"""Test that barrier operations are not folded."""
		qc = QuantumCircuit(2)
		qc.cx(0, 1)
		qc.barrier()
		qc.cx(0, 1)

		pm = PassManager(ZNECircuits(scale_factor=3))
		result = pm.run(qc)

		cx_count = sum(1 for inst in result.data if inst.operation.name == "cx")
		assert cx_count == 6  # 2 CX gates each tripled

	def test_fold_specific_gates_only(self) -> None:
		"""Test that only specified gates are folded."""
		qc = QuantumCircuit(3)
		qc.cx(0, 1)
		qc.cz(1, 2)

		pm = PassManager(ZNECircuits(fold_gates=["cx"], scale_factor=3))
		result = pm.run(qc)

		cx_count = sum(1 for inst in result.data if inst.operation.name == "cx")
		cz_count = sum(1 for inst in result.data if inst.operation.name == "cz")

		assert cx_count == 3  # CX folded
		assert cz_count == 1  # CZ not folded

	def test_scale_factor_5(self) -> None:
		"""Test scale_factor=5 produces 5x two-qubit gates."""
		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		pm = PassManager(ZNECircuits(scale_factor=5))
		result = pm.run(qc)

		cx_count = sum(1 for inst in result.data if inst.operation.name == "cx")
		assert cx_count == 5

	def test_multiple_two_qubit_gates(self) -> None:
		"""Test folding with multiple two-qubit gates in the circuit."""
		qc = QuantumCircuit(2)
		qc.cx(0, 1)
		qc.cx(0, 1)
		qc.cx(0, 1)

		pm = PassManager(ZNECircuits(scale_factor=3))
		result = pm.run(qc)

		cx_count = sum(1 for inst in result.data if inst.operation.name == "cx")
		assert cx_count == 9  # 3 original * 3 scale

	def test_local_folding_preserves_semantics_for_non_self_inverse_gate(self) -> None:
		"""Local folding G G† G must equal G for non-self-inverse gates (e.g. CRX)."""
		from qiskit.quantum_info import Operator

		qc = QuantumCircuit(2)
		qc.crx(0.5, 0, 1)

		pm = PassManager(ZNECircuits(scale_factor=3, folding_method="local"))
		folded = pm.run(qc)

		assert np.allclose(Operator(qc).data, Operator(folded).data, atol=1e-10)

	def test_global_folding_preserves_semantics_for_non_self_inverse_gate(self) -> None:
		"""Global folding C C† C must equal C for non-self-inverse gates (e.g. CRX)."""
		from qiskit.quantum_info import Operator

		qc = QuantumCircuit(2, 2)
		qc.crx(0.5, 0, 1)
		qc.measure([0, 1], [0, 1])

		pm = PassManager(ZNECircuits(scale_factor=3, folding_method="global"))
		folded = pm.run(qc)

		qc_nm = QuantumCircuit(2)
		qc_nm.crx(0.5, 0, 1)
		folded_nm = QuantumCircuit(2)
		for inst in folded.data:
			if inst.operation.name not in ("measure", "barrier"):
				folded_nm.append(inst.operation, inst.qubits)

		assert np.allclose(Operator(qc_nm).data, Operator(folded_nm).data, atol=1e-10)

	def test_even_integer_scale_factor_local(self) -> None:
		"""Even-integer scale factor 2 folds half the gates so the average matches (4 -> 8 gates)."""
		qc = QuantumCircuit(2)
		for _ in range(4):
			qc.cx(0, 1)

		pm = PassManager(ZNECircuits(scale_factor=2, seed=0))
		result = pm.run(qc)

		# num_folds = round((2-1)*4/2) = 2 gates folded once each: 2*1 + 2*3 = 8 instances.
		cx_count = sum(1 for inst in result.data if inst.operation.name == "cx")
		assert cx_count == 8

	def test_fractional_scale_factor_local(self) -> None:
		"""Fractional scale factor 1.5 folds a subset to approximate the average (4 -> 6 gates)."""
		qc = QuantumCircuit(2)
		for _ in range(4):
			qc.cx(0, 1)

		pm = PassManager(ZNECircuits(scale_factor=1.5, seed=0))
		result = pm.run(qc)

		# num_folds = round((1.5-1)*4/2) = 1 gate folded once: 3*1 + 1*3 = 6 instances.
		cx_count = sum(1 for inst in result.data if inst.operation.name == "cx")
		assert cx_count == 6

	def test_fractional_local_folding_preserves_semantics(self) -> None:
		"""Partial random folding must still leave the overall unitary unchanged."""
		from qiskit.quantum_info import Operator

		qc = QuantumCircuit(2)
		for theta in (0.5, 0.3, 0.7, 0.2):
			qc.crx(theta, 0, 1)

		pm = PassManager(ZNECircuits(scale_factor=1.5, folding_method="local", seed=0))
		folded = pm.run(qc)

		assert np.allclose(Operator(qc).data, Operator(folded).data, atol=1e-10)

	def _instruction_layout(self, circuit) -> list[tuple[str, tuple[int, ...]]]:
		return [(inst.operation.name, tuple(circuit.find_bit(q).index for q in inst.qubits)) for inst in circuit.data]

	def test_seed_reproducible(self) -> None:
		"""Same seed yields identical folded circuits for a non-odd-integer scale factor."""
		qc = QuantumCircuit(2)
		for _ in range(6):
			qc.cx(0, 1)

		a = PassManager(ZNECircuits(scale_factor=1.5, seed=42)).run(qc)
		b = PassManager(ZNECircuits(scale_factor=1.5, seed=42)).run(qc)

		assert self._instruction_layout(a) == self._instruction_layout(b)

	def test_different_seeds_can_differ(self) -> None:
		"""Different seeds generally sample different gates to fold (not all layouts identical)."""
		# Alternate the qubit pair so the *position* of each folded gate is distinguishable.
		qc = QuantumCircuit(3)
		for i in range(6):
			qc.cx(0, 1) if i % 2 == 0 else qc.cx(1, 2)

		layouts = {
			tuple(self._instruction_layout(PassManager(ZNECircuits(scale_factor=1.5, seed=s)).run(qc)))
			for s in range(6)
		}

		assert len(layouts) > 1

	def test_fractional_global_folding_preserves_semantics(self) -> None:
		"""Fractional global folding appends a suffix fold but preserves the unitary."""
		from qiskit.quantum_info import Operator

		qc = QuantumCircuit(2, 2)
		qc.crx(0.5, 0, 1)
		qc.crx(0.3, 0, 1)
		qc.measure([0, 1], [0, 1])

		pm = PassManager(ZNECircuits(scale_factor=2, folding_method="global"))
		folded = pm.run(qc)

		qc_nm = QuantumCircuit(2)
		qc_nm.crx(0.5, 0, 1)
		qc_nm.crx(0.3, 0, 1)
		folded_nm = QuantumCircuit(2)
		for inst in folded.data:
			if inst.operation.name not in ("measure", "barrier"):
				folded_nm.append(inst.operation, inst.qubits)

		# Suffix fold of 1 gate grew the circuit beyond the original 2 gates.
		assert folded_nm.size() > qc_nm.size()
		assert np.allclose(Operator(qc_nm).data, Operator(folded_nm).data, atol=1e-10)


class TestGetZNECircuits:
	"""Tests for _get_zne_circuits helper function."""

	def test_default_scale_factors(self) -> None:
		"""Test that default scale factors [1, 3, 5] produce correct number of circuits."""
		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		result = _get_zne_circuits([qc])

		# 1 input circuit * 3 scale factors = 3 output circuits
		assert len(result) == 3

	def test_custom_scale_factors(self) -> None:
		"""Test with custom scale factors."""
		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		result = _get_zne_circuits([qc], scale_factors=[1, 3])

		assert len(result) == 2

	def test_multiple_input_circuits(self) -> None:
		"""Test with multiple input circuits."""
		qc1 = QuantumCircuit(2)
		qc1.cx(0, 1)

		qc2 = QuantumCircuit(2)
		qc2.cx(0, 1)
		qc2.cx(0, 1)

		result = _get_zne_circuits([qc1, qc2], scale_factors=[1, 3, 5])

		# 2 circuits * 3 scale factors = 6
		assert len(result) == 6

	def test_none_scale_factors_uses_default(self) -> None:
		"""Test that None scale_factors falls back to [1, 3, 5]."""
		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		result = _get_zne_circuits([qc], scale_factors=None)

		assert len(result) == 3

	def test_fold_gates_parameter_forwarded(self) -> None:
		"""Test that fold_gates is passed through to ZNECircuits."""
		qc = QuantumCircuit(3)
		qc.cx(0, 1)
		qc.cz(1, 2)

		result = _get_zne_circuits([qc], fold_gates=["cx"], scale_factors=[1, 3])

		# At scale=3, only CX should be tripled, CZ unchanged
		scale3_circuit = result[1]
		cx_count = sum(1 for inst in scale3_circuit.data if inst.operation.name == "cx")
		cz_count = sum(1 for inst in scale3_circuit.data if inst.operation.name == "cz")

		assert cx_count == 3
		assert cz_count == 1

	def test_invalid_scale_factors_raises_error(self) -> None:
		"""Test that scale factors below 1 (or negative) raise ValueError."""
		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		with pytest.raises(ValueError, match="Scale factors must be real numbers >= 1"):
			_get_zne_circuits([qc], scale_factors=[-1, 3, 5])  # -1 is negative

		with pytest.raises(ValueError, match="Scale factors must be real numbers >= 1"):
			_get_zne_circuits([qc], scale_factors=[1, 3, 0.2])  # 0.2 is below 1

	def test_arbitrary_scale_factors_accepted(self) -> None:
		"""Even integers and fractions (>= 1) are now valid scale factors."""
		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")  # small circuit can't reach all scales exactly
			result = _get_zne_circuits([qc], scale_factors=[1, 1.5, 2, 2.5])

		assert len(result) == 4


class TestEstimatorZNESettings:
	"""Tests for ZNE settings on FiQCIEstimator."""

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_default_zne_disabled(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that ZNE is disabled by default (mitigation_level=1)."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock(), mitigation_level=1)

		assert estimator._zne["enabled"] is False

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_mitigation_level_3_enables_zne(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that mitigation_level=3 enables ZNE."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock(), mitigation_level=3)

		assert estimator._zne["enabled"] is True

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_default_settings(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test default ZNE configuration values."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock(), mitigation_level=3)

		assert estimator._zne["scale_factors"] == [1, 3, 5]
		assert estimator._zne["extrapolation_method"] == "exponential"
		assert estimator._zne["fold_gates"] is None
		assert estimator._zne["extrapolation_degree"] is None

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_configure_custom_settings(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test configuring ZNE with custom settings."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, fold_gates=["cx"], scale_factors=[1, 3], extrapolation_method="richardson")

		assert estimator._zne["enabled"] is True
		assert estimator._zne["fold_gates"] == ["cx"]
		assert estimator._zne["scale_factors"] == [1, 3]
		assert estimator._zne["extrapolation_method"] == "richardson"

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_arbitrary_scale_factors_and_seed(self, mock_fiqci_backend_class: Mock) -> None:
		"""Arbitrary scale factors, folding_method, and seed are stored on the estimator."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, scale_factors=[1, 1.5, 2, 3], folding_method="global", seed=7)

		assert estimator._zne["scale_factors"] == [1, 1.5, 2, 3]
		assert estimator._zne["folding_method"] == "global"
		assert estimator._zne["seed"] == 7
		assert estimator.mitigator_options["zne"]["seed"] == 7

	def test_zne_job_exposes_achieved_scale_factors(self) -> None:
		"""Unreachable scales warn at run and are exposed on the job; estimator config is untouched."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(enabled=True, scale_factors=[1, 2], folding_method="local", seed=0)

		qc = QuantumCircuit(2)
		for _ in range(3):  # 3 foldable CX gates -> scale 2 rounds to (3 + 2*2)/3 = 7/3
			qc.cx(0, 1)

		with pytest.warns(UserWarning, match="Access them via the job's achieved_scale_factors"):
			job = estimator.run(qc, SparsePauliOp(["ZZ"]), shots=1024)

		# The estimator's user-defined config is never mutated.
		assert estimator._zne["scale_factors"] == [1, 2]
		# Requested and achieved are per-pair (always a list of lists) and live on the job.
		assert np.allclose(job.requested_scale_factors(), [[1.0, 2.0]])
		assert np.allclose(job.achieved_scale_factors(), [[1.0, 7 / 3]])
		assert np.allclose(job.achieved_scale_factors(0), [1.0, 7 / 3])

	def test_zne_job_mitigator_options_snapshot(self) -> None:
		"""The estimator job reports a frozen snapshot merging ZNE config with the backend stack."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(
			enabled=True, scale_factors=[1, 3], folding_method="global", extrapolation_method="richardson", seed=7
		)

		qc = QuantumCircuit(2)
		qc.cx(0, 1)
		job = estimator.run(qc, SparsePauliOp(["ZZ"]), shots=1024)

		options = job.mitigator_options
		# ZNE settings the run used.
		assert options["zne"]["enabled"] is True
		assert options["zne"]["folding_method"] == "global"
		assert options["zne"]["extrapolation_method"] == "richardson"
		assert options["zne"]["seed"] == 7
		# Merged with the underlying backend job's snapshot (level 0 here -> all off).
		assert options["mitigation_level"] == 0
		assert options["rem"]["enabled"] is False

		# Mutating the estimator afterwards does not change the job's snapshot.
		estimator.zne(enabled=True, folding_method="local", seed=99)
		assert job.mitigator_options["zne"]["folding_method"] == "global"
		assert job.mitigator_options["zne"]["seed"] == 7

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_scale_factor_below_one_raises(self, mock_fiqci_backend_class: Mock) -> None:
		"""Scale factors below 1 are rejected."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())

		with pytest.raises(ValueError, match="Scale factors must be real numbers >= 1"):
			estimator.zne(enabled=True, scale_factors=[0.5, 1.5])

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_nested_scale_factors_stored(self, mock_fiqci_backend_class: Mock) -> None:
		"""A list of per-circuit scale-factor lists is accepted and stored."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, scale_factors=[[1, 3, 5], [1, 2, 4]])

		assert estimator._zne["scale_factors"] == [[1, 3, 5], [1, 2, 4]]

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_nested_scale_factors_invalid_sublist_raises(self, mock_fiqci_backend_class: Mock) -> None:
		"""Each per-circuit list must itself have at least two real numbers >= 1."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())

		with pytest.raises(ValueError, match="Each per-circuit scale factor list"):
			estimator.zne(enabled=True, scale_factors=[[1, 3], [0.5, 2]])  # 0.5 < 1

		with pytest.raises(ValueError, match="Each per-circuit scale factor list"):
			estimator.zne(enabled=True, scale_factors=[[1, 3], [3]])  # sublist too short

	def test_zne_nested_scale_factors_length_mismatch_raises(self) -> None:
		"""Number of per-circuit lists must match the number of submitted circuits."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(enabled=True, scale_factors=[[1, 3], [1, 3]], seed=0)

		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		with pytest.raises(ValueError, match="Per-circuit scale_factors has 2 entr"):
			estimator.run([qc], SparsePauliOp(["ZZ"]), shots=256)  # only 1 circuit

	def test_zne_nested_scale_factors_per_circuit_run(self) -> None:
		"""Each circuit folds with and extrapolates against its own scale factors."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(enabled=True, scale_factors=[[1, 3, 5], [1, 3]], folding_method="local", seed=0)

		qc0 = QuantumCircuit(2)
		qc0.h(0)
		qc0.cx(0, 1)
		qc1 = QuantumCircuit(2)
		qc1.h(0)
		qc1.cx(0, 1)

		job = estimator.run([qc0, qc1], SparsePauliOp(["ZZ"]), shots=1024)
		vals = job.expectation_values()

		# Two circuit/observable pairs, each with a single ZZ expectation value.
		assert len(vals) == 2
		# Config is never mutated; each pair reports its own requested/achieved scales on the job
		# (per-pair lists differ in length, so compare pair by pair).
		assert estimator._zne["scale_factors"] == [[1, 3, 5], [1, 3]]
		assert np.allclose(job.requested_scale_factors(0), [1, 3, 5])
		assert np.allclose(job.requested_scale_factors(1), [1, 3])
		# Odd integers are reachable exactly, so achieved equals requested per pair.
		assert np.allclose(job.achieved_scale_factors(0), [1, 3, 5])
		assert np.allclose(job.achieved_scale_factors(1), [1, 3])

	def test_zne_flat_request_reports_per_circuit_achieved(self) -> None:
		"""A flat request yields per-circuit achieved scales on the job when circuits differ in size."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(enabled=True, scale_factors=[1, 2], folding_method="local", seed=0)

		# Different foldable-gate counts -> different achieved scale for the requested 2.
		qc_small = QuantumCircuit(2)
		qc_small.cx(0, 1)  # N=1: scale 2 -> achieved 1.0 (round(0.5)=0 folds)
		qc_big = QuantumCircuit(2)
		for _ in range(3):
			qc_big.cx(0, 1)  # N=3: scale 2 -> achieved 7/3

		with pytest.warns(UserWarning, match="not all exactly reachable"):
			job = estimator.run([qc_small, qc_big], SparsePauliOp(["ZZ"]), shots=512)

		# Config unchanged; both pairs share the flat request but report their own achieved scales.
		assert estimator._zne["scale_factors"] == [1, 2]
		assert np.allclose(job.requested_scale_factors(), [[1.0, 2.0], [1.0, 2.0]])
		assert np.allclose(job.achieved_scale_factors(), [[1.0, 1.0], [1.0, 7 / 3]])

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_invalid_extrapolation_method_raises_error(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that invalid extrapolation method raises ValueError."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())

		with pytest.raises(ValueError, match="Unsupported extrapolation method"):
			estimator.zne(enabled=True, extrapolation_method="invalid")

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_invalid_fold_gates_raises_error(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that non-list fold_gates raises ValueError."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())

		with pytest.raises(ValueError, match="fold_gates must be a list"):
			estimator.zne(enabled=True, fold_gates="cx")

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_polynomial_with_degree(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test polynomial extrapolation with explicit degree."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, extrapolation_method="polynomial", extrapolation_degree=2)

		assert estimator._zne["extrapolation_method"] == "polynomial"
		assert estimator._zne["extrapolation_degree"] == 2

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_degree_ignored_for_non_polynomial(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that extrapolation_degree is ignored for non-polynomial methods."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			estimator.zne(enabled=True, extrapolation_method="exponential", extrapolation_degree=2)

		assert estimator._zne["extrapolation_degree"] is None

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_degree_warning_for_non_polynomial(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that a warning is raised when degree is set for non-polynomial methods."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())

		with pytest.warns(UserWarning, match="only applicable for polynomial"):
			estimator.zne(enabled=True, extrapolation_method="exponential", extrapolation_degree=2)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_polynomial_degree_1_warning(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test warning when polynomial degree=1 (equivalent to linear)."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())

		with pytest.warns(UserWarning, match="equivalent to linear"):
			estimator.zne(enabled=True, extrapolation_method="polynomial", extrapolation_degree=1)

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_disable(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test disabling ZNE after enabling it."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock(), mitigation_level=3)
		assert estimator._zne["enabled"] is True

		estimator.zne(enabled=False)
		assert estimator._zne["enabled"] is False

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_mitigator_options_includes_zne(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that mitigator_options() includes ZNE settings."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		mock_fiqci_backend = Mock()
		mock_fiqci_backend.mitigator_options = {"rem": {"enabled": False}}
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		estimator = FiQCIEstimator(Mock())
		options = estimator.mitigator_options

		print(options)  # For debugging

		assert "zne" in options
		assert options["zne"]["enabled"] is False

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	@pytest.mark.parametrize("method", ["exponential", "richardson", "polynomial", "linear"])
	def test_zne_accepts_all_valid_methods(self, mock_fiqci_backend_class: Mock, method: str) -> None:
		"""Test that all valid extrapolation methods are accepted."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, extrapolation_method=method)

		assert estimator._zne["extrapolation_method"] == method

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_extraplation_degree_only_for_polynomial(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that extrapolation_degree is only set for polynomial method."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, extrapolation_method="linear", extrapolation_degree=3)

		assert estimator._zne["extrapolation_method"] == "linear"
		assert estimator._zne["extrapolation_degree"] is None
