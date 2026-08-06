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


class TestExtrapolationErrors:
	"""Tests for the optional ``sigmas`` error-propagation path of the extrapolation functions."""

	def test_omitting_sigmas_returns_plain_value_list(self) -> None:
		"""Without sigmas, every extrapolator returns a plain list (backward compatible)."""
		scales = [1, 3, 5]
		vals = [[0.9], [0.7], [0.5]]
		assert isinstance(richardson_extrapolation(vals, scales), list)
		assert isinstance(polynomial_extrapolation(vals, scales, degree=1), list)
		assert isinstance(exponential_extrapolation(vals, scales), list)

	def test_passing_sigmas_returns_values_and_errors(self) -> None:
		"""With sigmas, each extrapolator returns a (values, errors) tuple of matching length."""
		scales = [1, 3, 5]
		vals = [[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]]
		sig = [[0.01, 0.02], [0.01, 0.02], [0.01, 0.02]]
		for fn in (
			lambda: richardson_extrapolation(vals, scales, sigmas=sig),
			lambda: polynomial_extrapolation(vals, scales, degree=1, sigmas=sig),
			lambda: exponential_extrapolation(vals, scales, sigmas=sig),
		):
			values, errors = fn()
			assert len(values) == 2
			assert len(errors) == 2
			assert all(e >= 0 for e in errors)

	def test_richardson_error_matches_hand_computed_propagation(self) -> None:
		"""Richardson error equals sqrt(sum_i c_i^2 sigma_i^2) with the Lagrange coefficients."""
		scales = [1, 3]
		# c_0 = 3/(3-1) = 1.5, c_1 = 1/(1-3) = -0.5
		sig = [[0.1], [0.2]]
		_, errors = richardson_extrapolation([[0.9], [0.7]], scales, sigmas=sig)
		expected = np.sqrt((1.5**2) * 0.1**2 + (0.5**2) * 0.2**2)
		assert errors[0] == pytest.approx(expected)

	@pytest.mark.parametrize(
		"fn",
		[
			lambda vals, scales, sig: richardson_extrapolation(vals, scales, sigmas=sig),
			lambda vals, scales, sig: polynomial_extrapolation(vals, scales, degree=1, sigmas=sig),
			lambda vals, scales, sig: exponential_extrapolation(vals, scales, sigmas=sig),
		],
	)
	def test_zero_sigma_gives_zero_error(self, fn) -> None:
		"""No shot noise in -> no extrapolation error out."""
		scales = [1, 3, 5]
		vals = [[0.8], [0.6], [0.45]]
		_, errors = fn(vals, scales, [[0.0], [0.0], [0.0]])
		assert errors[0] == pytest.approx(0.0)

	@pytest.mark.parametrize(
		"fn",
		[
			lambda vals, scales, sig: richardson_extrapolation(vals, scales, sigmas=sig),
			lambda vals, scales, sig: polynomial_extrapolation(vals, scales, degree=1, sigmas=sig),
			lambda vals, scales, sig: exponential_extrapolation(vals, scales, sigmas=sig),
		],
	)
	def test_error_increases_with_sigma(self, fn) -> None:
		"""Larger per-scale shot errors propagate to a larger extrapolation error."""
		scales = [1, 3, 5]
		vals = [[0.8], [0.6], [0.45]]
		_, small = fn(vals, scales, [[0.01], [0.01], [0.01]])
		_, large = fn(vals, scales, [[0.05], [0.05], [0.05]])
		assert large[0] > small[0]


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
	def test_zne_accepts_user_defined_extrapolation_callable(self, mock_fiqci_backend_class: Mock) -> None:
		"""A user-defined callable is accepted and stored verbatim as the extrapolation method."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		def my_extrapolation(expectation_values, scales):
			return [0.0 for _ in expectation_values[0]]

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, extrapolation_method=my_extrapolation)

		assert estimator._zne["extrapolation_method"] is my_extrapolation

	def test_zne_user_defined_extrapolation_used_in_run(self) -> None:
		"""The user-defined callable is invoked with (expectation_values, scales) and its result used."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		calls: list[tuple] = []

		def my_extrapolation(expectation_values, scales):
			calls.append((expectation_values, scales))
			# Return a sentinel value per observable so we can assert it flowed through.
			return [42.0 for _ in expectation_values[0]]

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(
			enabled=True, scale_factors=[1, 3], folding_method="local", seed=0, extrapolation_method=my_extrapolation
		)

		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)

		job = estimator.run(qc, SparsePauliOp(["ZZ"]), shots=512)
		vals = job.expectation_values()

		assert vals == [[42.0]]
		# Callable was invoked once for the single pair with the achieved scale factors.
		assert len(calls) == 1
		_, scales = calls[0]
		assert np.allclose(scales, job.achieved_scale_factors(0))

	def test_zne_user_defined_extrapolation_without_sigmas_reports_no_extrapolation_error(self) -> None:
		"""A two-argument callable still works and leaves the extrapolation error unreported."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		def my_extrapolation(expectation_values, scales):
			return [42.0 for _ in expectation_values[0]]

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(
			enabled=True, scale_factors=[1, 3], folding_method="global", extrapolation_method=my_extrapolation
		)

		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)

		obs = SparsePauliOp(["ZZ", "ZI"])
		job = estimator.run(qc, obs, shots=512)
		errors = job.standard_errors(0)

		assert job.expectation_values(0) == [42.0, 42.0]
		# The shot error at the unfolded point is still measured and reported.
		assert len(errors["shot_error"]) == len(obs.paulis)
		assert errors["zne_extrapolation_error"] is None
		assert errors["total"] is None

	def test_zne_user_defined_extrapolation_receives_sigmas_and_reports_errors(self) -> None:
		"""A callable accepting sigmas is given the per-scale shot errors and may return SEs."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		calls: list[tuple] = []

		def my_extrapolation(expectation_values, scales, sigmas=None):
			calls.append((expectation_values, scales, sigmas))
			values = [42.0 for _ in expectation_values[0]]
			errors = [0.5 for _ in expectation_values[0]]
			return values, errors

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(
			enabled=True, scale_factors=[1, 3], folding_method="global", extrapolation_method=my_extrapolation
		)

		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)

		obs = SparsePauliOp(["ZZ", "ZI"])
		job = estimator.run(qc, obs, shots=512)
		errors = job.standard_errors(0)

		assert job.expectation_values(0) == [42.0, 42.0]
		assert errors["zne_extrapolation_error"] == [0.5, 0.5]
		assert errors["total"] == [0.5, 0.5]

		# sigmas mirrors the (n_scales, n_obs) shape of the expectation values handed to the callable.
		assert len(calls) == 1
		expvs, scales, sigmas = calls[0]
		assert sigmas is not None
		assert len(sigmas) == len(scales) == len(expvs)
		assert all(len(s) == len(obs.paulis) for s in sigmas)
		assert all(e >= 0 for s in sigmas for e in s)

	def test_zne_user_defined_extrapolation_kwargs_callable_receives_sigmas(self) -> None:
		"""A callable that only declares **kwargs is also handed the sigmas."""
		from fiqci.ems.primitives.fiqci_estimator import _apply_custom_extrapolation

		seen: dict = {}

		def my_extrapolation(expectation_values, scales, **kwargs):
			seen.update(kwargs)
			return [0.0 for _ in expectation_values[0]]

		values, errors = _apply_custom_extrapolation(my_extrapolation, [[0.9], [0.7]], [1.0, 3.0], [[0.01], [0.02]])

		assert seen["sigmas"] == [[0.01], [0.02]]
		assert values == [0.0]
		assert errors is None

	def test_zne_user_defined_extrapolation_builtin_as_callable_reports_errors(self) -> None:
		"""Passing a built-in extrapolator as the callable propagates errors exactly as the string does."""
		from fiqci.ems.primitives.fiqci_estimator import _apply_custom_extrapolation

		expvs = [[0.9, 0.5], [0.7, 0.3]]
		scales = [1.0, 3.0]
		sigmas = [[0.01, 0.02], [0.03, 0.04]]

		values, errors = _apply_custom_extrapolation(richardson_extrapolation, expvs, scales, sigmas)
		want_values, want_errors = richardson_extrapolation(expvs, scales, sigmas=sigmas)

		assert values == pytest.approx(want_values)
		assert errors == pytest.approx(want_errors)

	def test_zne_user_defined_extrapolation_error_length_mismatch_raises(self) -> None:
		"""Returning a different number of standard errors than values is rejected."""
		from fiqci.ems.primitives.fiqci_estimator import _apply_custom_extrapolation

		def my_extrapolation(expectation_values, scales, sigmas=None):
			return [0.0, 0.0], [0.1]

		with pytest.raises(ValueError, match="standard error"):
			_apply_custom_extrapolation(my_extrapolation, [[0.9, 0.5], [0.7, 0.3]], [1.0, 3.0], [[0.01, 0.01]] * 2)

	def test_zne_user_defined_extrapolation_two_observables_not_read_as_pair(self) -> None:
		"""A plain two-element value list is not mistaken for a (values, errors) pair."""
		from fiqci.ems.primitives.fiqci_estimator import _apply_custom_extrapolation

		def my_extrapolation(expectation_values, scales):
			return [0.25, -0.5]

		values, errors = _apply_custom_extrapolation(
			my_extrapolation, [[0.9, 0.5], [0.7, 0.3]], [1.0, 3.0], [[0.01, 0.01], [0.02, 0.02]]
		)

		assert values == [0.25, -0.5]
		assert errors is None

	@patch("fiqci.ems.primitives.fiqci_estimator.FiQCIBackend")
	def test_zne_extraplation_degree_only_for_polynomial(self, mock_fiqci_backend_class: Mock) -> None:
		"""Test that extrapolation_degree is only set for polynomial method."""
		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(Mock())
		estimator.zne(enabled=True, extrapolation_method="linear", extrapolation_degree=3)

		assert estimator._zne["extrapolation_method"] == "linear"
		assert estimator._zne["extrapolation_degree"] is None


class TestDegenerateAchievedScaleFactors:
	"""Folding is discrete, so distinct requested scales can collapse onto the same achieved value.

	Extrapolation then has too few distinct x-values and yields nan/inf or a meaningless fit, so
	``run()`` warns at submission time (while the job is still cancellable).
	"""

	def _estimator(self, **zne_kwargs):
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(enabled=True, **zne_kwargs)
		return estimator

	def test_no_two_qubit_gates_collapses_every_scale_and_warns(self) -> None:
		"""A circuit with no foldable 2q gates cannot be locally folded: every scale becomes 1.0."""
		from qiskit.quantum_info import SparsePauliOp

		estimator = self._estimator(scale_factors=[1, 3, 5], folding_method="local")

		qc = QuantumCircuit(2)  # single-qubit gates only -> nothing to fold
		qc.h(0)
		qc.h(1)

		with pytest.warns(UserWarning, match="collapsed every requested scale factor"):
			job = estimator.run(qc, SparsePauliOp(["ZZ"]), shots=512)

		assert np.allclose(job.achieved_scale_factors(0), [1.0, 1.0, 1.0])

	def test_collapse_warning_is_emitted_before_results_are_fetched(self) -> None:
		"""The warning must fire at run() so the user can still cancel; not at result time."""
		from qiskit.quantum_info import SparsePauliOp

		estimator = self._estimator(scale_factors=[1, 3], folding_method="local")

		qc = QuantumCircuit(2)
		qc.h(0)

		with warnings.catch_warnings(record=True) as caught:
			warnings.simplefilter("always")
			job = estimator.run(qc, SparsePauliOp(["ZZ"]), shots=512)

		messages = [str(w.message) for w in caught if w.category is UserWarning]
		assert any("no extrapolation is possible" in m for m in messages)
		# The handle is usable for cancellation: ids are available without fetching results.
		assert job.job_ids()

	def test_partial_collapse_warns_about_reduced_distinct_scales(self) -> None:
		"""Some (not all) scales collapsing warns that the fit uses duplicated points."""
		from qiskit.quantum_info import SparsePauliOp

		# One foldable CX: scale 1.05 rounds to 0 folds -> 1.0, colliding with the requested 1.0.
		estimator = self._estimator(scale_factors=[1, 1.05, 3], folding_method="local")

		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		with pytest.warns(UserWarning, match="only 2 distinct achieved scale factor"):
			job = estimator.run(qc, SparsePauliOp(["ZZ"]), shots=512)

		assert np.allclose(job.achieved_scale_factors(0), [1.0, 1.0, 3.0])

	def test_resolvable_scale_factors_do_not_warn(self) -> None:
		"""A circuit with enough foldable gates to resolve every scale stays silent."""
		from qiskit.quantum_info import SparsePauliOp

		estimator = self._estimator(scale_factors=[1, 3, 5], folding_method="local")

		qc = QuantumCircuit(2)
		qc.cx(0, 1)

		with warnings.catch_warnings(record=True) as caught:
			warnings.simplefilter("always")
			job = estimator.run(qc, SparsePauliOp(["ZZ"]), shots=512)

		user_warnings = [str(w.message) for w in caught if w.category is UserWarning]
		assert user_warnings == []
		assert np.allclose(job.achieved_scale_factors(0), [1.0, 3.0, 5.0])


class TestCustomExtrapolationReturnShapes:
	"""``_apply_custom_extrapolation`` normalises the shapes a user callable may return."""

	def test_values_with_explicit_none_errors(self) -> None:
		"""``(values, None)`` is the natural way to say "no standard errors" and must be accepted."""
		from fiqci.ems.primitives.fiqci_estimator import _apply_custom_extrapolation

		def my_extrapolation(expectation_values, scales, sigmas=None):
			return [0.25, -0.5], None

		values, errors = _apply_custom_extrapolation(
			my_extrapolation, [[0.9, 0.5], [0.7, 0.3]], [1.0, 3.0], [[0.01, 0.01], [0.02, 0.02]]
		)

		assert values == [0.25, -0.5]
		assert errors is None

	def test_numpy_values_with_none_errors(self) -> None:
		"""The ``(values, None)`` form also works when values are a numpy array."""
		from fiqci.ems.primitives.fiqci_estimator import _apply_custom_extrapolation

		def my_extrapolation(expectation_values, scales, sigmas=None):
			return np.array([0.25, -0.5]), None

		values, errors = _apply_custom_extrapolation(
			my_extrapolation, [[0.9, 0.5], [0.7, 0.3]], [1.0, 3.0], [[0.01, 0.01], [0.02, 0.02]]
		)

		assert values == [0.25, -0.5]
		assert errors is None

	def test_none_errors_reported_as_no_extrapolation_error_end_to_end(self) -> None:
		"""A ``(values, None)`` callable leaves the job's extrapolation error unset."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		def my_extrapolation(expectation_values, scales, sigmas=None):
			return [float(v) for v in expectation_values[0]], None

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(enabled=True, scale_factors=[1, 3], extrapolation_method=my_extrapolation)

		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)

		job = estimator.run(qc, SparsePauliOp(["ZZ"]), shots=1024)

		assert job.standard_errors(0)["zne_extrapolation_error"] is None
		assert job.standard_errors(0)["total"] is None
		assert job.standard_errors(0)["shot_error"] is not None

	@pytest.mark.parametrize(
		"returned",
		[
			pytest.param({"a": 1}, id="dict"),
			pytest.param(["not", "a", "float"], id="strings"),
			pytest.param(([0.1], "oops"), id="non-numeric-errors"),
		],
	)
	def test_uninterpretable_return_raises_a_clear_type_error(self, returned) -> None:
		"""A return value we cannot read as floats names the offending value instead of leaking float()."""
		from fiqci.ems.primitives.fiqci_estimator import _apply_custom_extrapolation

		def my_extrapolation(expectation_values, scales, sigmas=None):
			return returned

		with pytest.raises(TypeError, match="must return a sequence of floats"):
			_apply_custom_extrapolation(my_extrapolation, [[0.9], [0.7]], [1.0, 3.0], [[0.01], [0.02]])


def _ghz(num_qubits: int) -> QuantumCircuit:
	qc = QuantumCircuit(num_qubits)
	qc.h(0)
	for qubit in range(num_qubits - 1):
		qc.cx(qubit, qubit + 1)
	return qc


class TestZNEFoldsBeforeBasisRotations:
	"""Folding runs on the bare circuit, so a pair's achieved scale is basis-independent.

	Global folding counts every non-measurement gate, so folding the finished measurement-basis
	subcircuits made an X/Y group fold by a different amount than the Z group of the same pair,
	while a single achieved-scale list (the first group's) was reported and extrapolated against.
	"""

	def _submitted_gate_counts(self, circuit, obs, scales, folding_method) -> list[int]:
		"""Non-measurement gate count of every submitted circuit, in submission order.

		The estimator flattens scale-major, so index ``s * num_groups + g`` is scale ``s`` of
		measurement group ``g``.
		"""
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		counts: list[int] = []
		unpatched = AerSimulator.run

		def spy(backend, circuits, **kwargs):
			for candidate in circuits if isinstance(circuits, list) else [circuits]:
				counts.append(sum(1 for ins in candidate.data if ins.operation.name not in ("measure", "barrier")))
			return unpatched(backend, circuits, **kwargs)

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(
			enabled=True, scale_factors=scales, folding_method=folding_method, extrapolation_method="linear", seed=3
		)
		with patch.object(AerSimulator, "run", spy), warnings.catch_warnings():
			warnings.simplefilter("ignore")
			estimator.run(circuit, obs, shots=64).expectation_values(0)
		return counts

	@pytest.mark.parametrize("folding_method", ["local", "global"])
	@pytest.mark.parametrize("scales", [[1, 3, 5], [1, 2, 3.5]])
	def test_every_group_of_a_pair_shares_one_folded_core(self, folding_method: str, scales: list) -> None:
		"""Each group's gate count grows by the same amount per scale: one shared folded core."""
		from qiskit.quantum_info import SparsePauliOp

		obs = SparsePauliOp(["ZZZ", "XXX", "YYY"])  # 3 groups, differing rotation overhead
		num_groups = 3

		counts = self._submitted_gate_counts(_ghz(3), obs, scales, folding_method)
		assert len(counts) == num_groups * len(scales)

		# Growth over the unfolded circuit comes purely from the folded core, so it must not
		# depend on which basis a group measures in.
		for scale_index in range(len(scales)):
			growth = {
				counts[scale_index * num_groups + group] - counts[group]  # vs the same group at scale 1
				for group in range(num_groups)
			}
			assert len(growth) == 1, f"scale index {scale_index}: per-group growth diverged: {growth}"

	def test_achieved_scales_do_not_depend_on_the_observable_bases(self) -> None:
		"""The same circuit reports the same achieved scales whatever bases the observable needs.

		Both observables have a single measurement group, so the group whose gate count used to be
		measured differs: all-Z adds no rotations, all-X adds one per qubit.
		"""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		achieved = []
		for obs in (SparsePauliOp(["ZZZ"]), SparsePauliOp(["XXX"])):
			estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
			estimator.zne(enabled=True, scale_factors=[1, 2, 3.5], folding_method="global", seed=3)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				achieved.append(estimator.run(_ghz(3), obs, shots=64).achieved_scale_factors(0))

		# 3 foldable gates in the bare circuit either way.
		assert np.allclose(achieved[0], achieved[1]), achieved
		assert np.allclose(achieved[0], [1.0, 1 + 2 * 2 / 3, 3 + 2 * 1 / 3])

	def test_local_folding_achieved_scales_track_two_qubit_gates_only(self) -> None:
		"""Local folding is unaffected: 3 CX gates, so scale 2 lands on (3 + 2*3)/3 rounding."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(enabled=True, scale_factors=[1, 2, 4], folding_method="local", seed=3)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			job = estimator.run(_ghz(4), SparsePauliOp(["ZZZZ", "XXXX"]), shots=64)

		# 3 CX gates, and round() is banker's: scale 2 -> round(1.5)=2 folds -> 7/3;
		# scale 4 -> round(4.5)=4 folds -> 11/3. Basis rotations are single-qubit, so they never count.
		assert np.allclose(job.achieved_scale_factors(0), [1.0, 7 / 3, 11 / 3])

	@pytest.mark.parametrize("folding_method", ["local", "global"])
	@pytest.mark.parametrize("scales", [[1, 3, 5], [1, 2, 3.5]])
	def test_noiseless_values_stay_exact_and_group_aligned(self, folding_method: str, scales: list) -> None:
		"""Verify the circuit ordering is consistent between `_run` and `_compute`."""
		from qiskit.quantum_info import SparsePauliOp
		from qiskit_aer import AerSimulator

		from fiqci.ems.primitives.fiqci_estimator import FiQCIEstimator

		estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
		estimator.zne(
			enabled=True, scale_factors=scales, folding_method=folding_method, extrapolation_method="linear", seed=3
		)
		# Distinct values per basis, so a group mix-up cannot pass unnoticed.
		obs = SparsePauliOp(["ZZ", "XX", "YY"])
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			values = estimator.run(_ghz(2), obs, shots=16384).expectation_values(0)

		assert values == pytest.approx([1.0, 1.0, -1.0], abs=0.05)
