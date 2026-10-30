"""
Test that FiQCIEstimator results match Statevector reference within shot noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from fiqci.ems import FiQCIEstimator


SHOTS = 16384
TOL = 5 / np.sqrt(SHOTS)


def _bell() -> QuantumCircuit:
	qc = QuantumCircuit(2)
	qc.h(0)
	qc.cx(0, 1)
	return qc


def _ghz(n: int) -> QuantumCircuit:
	qc = QuantumCircuit(n)
	qc.h(0)
	for i in range(n - 1):
		qc.cx(i, i + 1)
	return qc


def _superposition(n: int) -> QuantumCircuit:
	qc = QuantumCircuit(n)
	for q in range(n):
		qc.h(q)
	return qc


def _exact(circuit: QuantumCircuit, obs: SparsePauliOp) -> list[float]:
	sv = Statevector(circuit)
	return [sv.expectation_value(p).real for p in obs.paulis]


@pytest.fixture(scope="module")
def estimator() -> FiQCIEstimator:
	return FiQCIEstimator(AerSimulator(), mitigation_level=0)


@pytest.mark.parametrize(
	"name, circuit, obs",
	[
		("bell-uniform-support", _bell(), SparsePauliOp(["ZZ", "XX", "YY"])),
		("bell-mixed-support", _bell(), SparsePauliOp(["IZ", "ZI", "ZZ", "IX", "XI", "XX"])),
		("ghz3-mixed-support", _ghz(3), SparsePauliOp(["IIZ", "IZI", "ZII", "ZZI", "IZZ", "ZZZ", "XXX"])),
		("superposition-x-basis", _superposition(3), SparsePauliOp(["IIX", "IXI", "XII", "XXX", "IIZ", "ZZZ"])),
	],
)
def test_estimator_matches_statevector(
	estimator: FiQCIEstimator, name: str, circuit: QuantumCircuit, obs: SparsePauliOp
) -> None:
	"""Each <P_i> from FiQCIEstimator agrees with Statevector reference within shot noise."""
	got = estimator.run(circuit, list(obs), shots=SHOTS).expectation_values(0)
	want = _exact(circuit, obs)
	assert got == pytest.approx(want, abs=TOL), f"{name}: got={got}, want={want}"


def test_estimator_distinguishes_single_qubit_supports(estimator: FiQCIEstimator) -> None:

	qc = QuantumCircuit(3)
	qc.x(1)  # <Z_1> = -1
	qc.h(2)  # <Z_2> = 0

	obs = SparsePauliOp(["IIZ", "IZI", "ZII"])
	got = estimator.run(qc, list(obs), shots=SHOTS).expectation_values(0)

	want = [1.0, -1.0, 0.0]
	assert got == pytest.approx(want, abs=TOL)


def test_standard_errors_match_shot_noise_and_bound_deviation(estimator: FiQCIEstimator) -> None:
	"""shot_error ~ sqrt((1 - <P>^2)/shots) per term, and |got - exact| stays within ~5 SE."""
	circuit = _ghz(3)
	obs = SparsePauliOp(["IIZ", "IZI", "ZII", "ZZI", "IZZ", "ZZZ", "XXX"])
	job = estimator.run(circuit, list(obs), shots=SHOTS)

	got = job.expectation_values(0)
	errors = job.standard_errors(0)
	want = _exact(circuit, obs)

	# ZNE is off at level 0: only shot error is reported, and it equals the total.
	assert errors["zne_extrapolation_error"] is None
	assert errors["total"] == errors["shot_error"]

	for got_v, se, want_v in zip(got, errors["shot_error"], want):
		expected_se = np.sqrt(max(0.0, 1.0 - got_v**2) / SHOTS)
		assert se == pytest.approx(expected_se, rel=1e-6)
		# Deviation from the exact value should sit within a few standard errors (5 SE + a small
		# floor so terms with se ~ 0, e.g. <ZZZ> = 0 on GHZ, don't make the bound vanish).
		assert abs(got_v - want_v) <= 5 * se + TOL


def test_shot_error_counts_every_twirled_variants_shots() -> None:
	"""Twirled counts are averaged back to `shots`, so N must be scaled by the group size."""
	num_twirls = 3
	circuit = _ghz(2)
	obs = SparsePauliOp(["ZZ", "XX"])

	estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
	estimator.pauli_twirl(enabled=True, num_twirls=num_twirls, seed=0)
	job = estimator.run(circuit, list(obs), shots=SHOTS)

	got = job.expectation_values(0)
	errors = job.standard_errors(0)

	assert got == pytest.approx(_exact(circuit, obs), abs=TOL)
	for got_v, se in zip(got, errors["shot_error"]):
		expected_se = np.sqrt(max(0.0, 1.0 - got_v**2) / (SHOTS * (num_twirls + 1)))
		assert se == pytest.approx(expected_se, rel=1e-6)


def test_standard_errors_with_zne_enabled() -> None:
	"""With ZNE on, shot_error (at scale 1) and a propagated zne_extrapolation_error are reported."""
	estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
	# Global folding so the single-qubit-gate circuit reaches distinct achieved scales [1, 3, 5];
	# local folding of a circuit with no foldable multi-qubit gates would collapse them to [1, 1, 1].
	estimator.zne(enabled=True, scale_factors=[1, 3, 5], extrapolation_method="linear", folding_method="global")

	# Superposition gives <ZZ> = <ZI> = 0, i.e. maximal (non-zero) shot noise to propagate.
	obs = SparsePauliOp(["ZZ", "ZI"])
	errors = estimator.run(_superposition(2), list(obs), shots=SHOTS).standard_errors(0)

	assert errors["zne_extrapolation_error"] is not None
	assert errors["total"] == errors["zne_extrapolation_error"]
	assert len(errors["shot_error"]) == len(obs.paulis)
	assert all(e >= 0 for e in errors["shot_error"])
	assert all(e > 0 for e in errors["zne_extrapolation_error"])
	# Extrapolation amplifies variance, so the zero-noise SE exceeds the raw scale-1 shot SE.
	for raw_se, ext_se in zip(errors["shot_error"], errors["zne_extrapolation_error"]):
		assert ext_se >= raw_se


def _bell_with_clbits(num_clbits: int) -> QuantumCircuit:
	qc = QuantumCircuit(2, num_clbits)
	qc.h(0)
	qc.cx(0, 1)
	return qc


def _bell_with_named_meas_register() -> QuantumCircuit:
	qc = _bell()
	qc.add_register(ClassicalRegister(2, "meas"))
	return qc


@pytest.mark.parametrize(
	"name, circuit",
	[
		("no-creg", _bell()),
		("idle-creg-equal-width", _bell_with_clbits(2)),
		("idle-creg-wider-than-qubits", _bell_with_clbits(5)),
		("idle-creg-narrower-than-qubits", _bell_with_clbits(1)),
		("creg-already-named-meas", _bell_with_named_meas_register()),
	],
)
def test_extra_classical_registers_do_not_corrupt_expectation_values(
	estimator: FiQCIEstimator,
	name: str,
	circuit: QuantumCircuit,
	obs: SparsePauliOp = SparsePauliOp(["ZZ", "IZ", "ZI"]),
) -> None:
	"""Classical bits the input circuit already carries must not shift the measured bit positions.

	The basis measurements used to be written into the input circuit's register, leaving the
	pass's own register all zeros, so every parity read ``+1`` and each observable came back as
	``1.0`` regardless of the state.
	"""
	got = estimator.run(circuit, list(obs), shots=SHOTS).expectation_values(0)
	want = _exact(_bell(), obs)

	assert got == pytest.approx(want, abs=TOL), f"{name}: got={got}, want={want}"


def test_mid_circuit_measurement_keeps_its_own_result(estimator: FiQCIEstimator) -> None:
	"""A non-final measurement must survive and still agree with the basis measurement."""
	qc = QuantumCircuit(2)
	aux = ClassicalRegister(1, "aux")
	qc.add_register(aux)
	qc.h(0)
	qc.measure(0, aux[0])
	qc.cx(0, 1)

	# Measuring q0 then copying it onto q1 leaves a 50/50 mix of |00> and |11>.
	obs = SparsePauliOp(["ZZ", "IZ", "ZI"])
	job = estimator.run(qc, list(obs), shots=SHOTS)

	assert job.expectation_values(0) == pytest.approx([1.0, 0.0, 0.0], abs=TOL)
	# Keys are "<meas bits> <aux bit>"; aux must track the basis measurement, not be overwritten.
	for key in job.result().get_counts(0):
		meas_bits, aux_bit = key.split(" ")
		assert set(meas_bits) == {aux_bit}


# Observables behave as in Qiskit's Estimator: one SparsePauliOp is one observable, sum(c * <P>).


@pytest.mark.parametrize(
	"circuit, obs",
	[
		(_bell(), SparsePauliOp(["ZZ", "XX", "YY"], [0.5, 0.25, -1.0])),
		(_ghz(3), SparsePauliOp(["III", "ZZI", "XXX", "IIZ"], [2.0, -0.5, 1.5, 3.0])),
		(_superposition(3), SparsePauliOp(["IIX", "XII", "ZZZ"], [1.0, -2.0, 0.3])),
	],
)
def test_weighted_observable_matches_statevector_estimator(
	estimator: FiQCIEstimator, circuit: QuantumCircuit, obs: SparsePauliOp
) -> None:
	got = estimator.run(circuit, obs, shots=SHOTS).expectation_values(0)
	want = StatevectorEstimator().run([(circuit, obs)]).result()[0].data.evs

	assert len(got) == 1
	assert got[0] == pytest.approx(float(want), abs=sum(abs(c) for c in obs.coeffs) * TOL)


def test_standard_errors_match_backend_estimator_v2(estimator: FiQCIEstimator) -> None:
	# <ZZ> = <ZI> = 0 and <XX> = 1 on |++>, so the stds are (0.5 + 2) / sqrt(N) and 0.
	circuit = _superposition(2)
	observables = [SparsePauliOp(["ZZ", "ZI", "XX"], [0.5, -2.0, 1.0]), SparsePauliOp("XX")]

	got = estimator.run(circuit, observables, shots=SHOTS).standard_errors(0)
	reference = BackendEstimatorV2(backend=AerSimulator())
	reference.options.default_precision = 1 / np.sqrt(SHOTS)
	want = reference.run([(circuit, observables)]).result()[0].data.stds

	assert got["total"] == got["shot_error"]
	assert got["total"] == pytest.approx(list(want), rel=0.05, abs=1e-9)
	assert got["total"][0] == pytest.approx(2.5 / np.sqrt(SHOTS), rel=0.05)


def test_observable_input_types_match(estimator: FiQCIEstimator) -> None:
	from qiskit.quantum_info import Pauli, SparseObservable

	circuit = _bell()
	observables = ["XX", {"ZZ": 0.5, "XX": 0.5}, Pauli("YY"), SparseObservable.from_label("ZZ")]

	got = estimator.run(circuit, observables, shots=SHOTS).expectation_values(0)

	assert got == pytest.approx([1.0, 1.0, -1.0, 1.0], abs=TOL)


def test_list_of_observables_per_circuit(estimator: FiQCIEstimator) -> None:
	job = estimator.run([_bell(), _superposition(2)], [["ZZ", "YY"], "XI"], shots=SHOTS)

	assert job.expectation_values(0) == pytest.approx([1.0, -1.0], abs=TOL)
	assert job.expectation_values(1) == pytest.approx([1.0], abs=TOL)
	assert [len(obs) for obs in job.observables()] == [2, 1]
	assert len(job.standard_errors(0)["total"]) == 2


def test_observables_sharing_a_basis_are_measured_once(estimator: FiQCIEstimator) -> None:
	observables = ["ZZ", "ZI", SparsePauliOp(["IZ", "ZZ"], [1.0, 2.0])]

	job = estimator.run(_bell(), observables, shots=SHOTS)

	assert len(job.result().results) == 1
	assert estimator.total_circuits_generated(1, observables) == 1
	assert job.expectation_values(0) == pytest.approx([1.0, 0.0, 2.0], abs=3 * TOL)


def test_duplicate_terms_are_merged(estimator: FiQCIEstimator) -> None:
	obs = SparsePauliOp(["ZZ", "ZZ"], [1.0, 2.0])

	assert estimator.run(_bell(), obs, shots=SHOTS).expectation_values(0) == pytest.approx([3.0], abs=TOL)


def test_identity_terms_add_their_coefficient(estimator: FiQCIEstimator) -> None:
	observables = [SparsePauliOp(["II", "ZZ"], [3.0, 0.5]), SparsePauliOp("II", -2.0)]

	job = estimator.run(_bell(), observables, shots=SHOTS)

	assert job.expectation_values(0) == pytest.approx([3.5, -2.0], abs=TOL)
	assert job.standard_errors(0)["total"][1] == 0.0


def test_identity_only_pair_next_to_a_measured_pair(estimator: FiQCIEstimator) -> None:
	job = estimator.run([_bell(), _bell()], ["II", "ZZ"], shots=SHOTS)

	assert np.array(job.expectation_values()) == pytest.approx(np.array([[1.0], [1.0]]), abs=TOL)


def test_identity_only_run_raises(estimator: FiQCIEstimator) -> None:
	with pytest.raises(ValueError, match="nothing to measure"):
		estimator.run(_bell(), SparsePauliOp("II", 3.0))


@pytest.mark.parametrize("obs", ["Z", "ZZZ"])
def test_observable_qubit_count_must_match_circuit(estimator: FiQCIEstimator, obs: str) -> None:
	with pytest.raises(ValueError, match="does not match the number of qubits"):
		estimator.run(_bell(), obs)


def test_complex_coefficients_raise(estimator: FiQCIEstimator) -> None:
	with pytest.raises(ValueError, match="Non-Hermitian"):
		estimator.run(_bell(), SparsePauliOp("ZZ", 1j))


@pytest.mark.parametrize("method", ["linear", "exponential", "richardson"])
def test_zne_adds_identity_offset_after_extrapolation(method: str) -> None:
	estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
	estimator.zne(enabled=True, scale_factors=[1, 3, 5], folding_method="global", extrapolation_method=method)
	obs = SparsePauliOp(["II", "ZZ", "XX"], [3.0, 0.5, 0.5])

	job = estimator.run(_bell(), [obs, "II"], shots=SHOTS)

	assert job.expectation_values(0) == pytest.approx([4.0, 1.0], abs=0.05)
	assert np.array(job.raw_expectation_values(0)) == pytest.approx(np.array([[4.0, 1.0]] * 3), abs=TOL)


def test_zne_custom_extrapolator_gets_one_column_per_observable_without_identity() -> None:
	calls: list = []

	def my_extrapolation(expectation_values, scales):
		calls.append(expectation_values)
		return expectation_values[0]

	estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
	estimator.zne(enabled=True, scale_factors=[1, 3], folding_method="global", extrapolation_method=my_extrapolation)
	obs = SparsePauliOp(["II", "ZZ", "XX"], [3.0, 0.5, 0.5])

	values = estimator.run(_bell(), [obs, "YY"], shots=SHOTS).expectation_values(0)

	assert len(calls) == 1
	assert np.array(calls[0]) == pytest.approx(np.array([[1.0, -1.0], [1.0, -1.0]]), abs=TOL)
	assert values == pytest.approx([4.0, -1.0], abs=TOL)
