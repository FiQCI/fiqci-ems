"""
Test that FiQCIEstimator results match Statevector reference within shot noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
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
	got = estimator.run(circuit, obs, shots=SHOTS).expectation_values(0)
	want = _exact(circuit, obs)
	assert got == pytest.approx(want, abs=TOL), f"{name}: got={got}, want={want}"


def test_estimator_distinguishes_single_qubit_supports(estimator: FiQCIEstimator) -> None:

	qc = QuantumCircuit(3)
	qc.x(1)  # <Z_1> = -1
	qc.h(2)  # <Z_2> = 0

	obs = SparsePauliOp(["IIZ", "IZI", "ZII"])
	got = estimator.run(qc, obs, shots=SHOTS).expectation_values(0)

	want = [1.0, -1.0, 0.0]
	assert got == pytest.approx(want, abs=TOL)


def test_standard_errors_match_shot_noise_and_bound_deviation(estimator: FiQCIEstimator) -> None:
	"""shot_error ~ sqrt((1 - <P>^2)/shots) per term, and |got - exact| stays within ~5 SE."""
	circuit = _ghz(3)
	obs = SparsePauliOp(["IIZ", "IZI", "ZII", "ZZI", "IZZ", "ZZZ", "XXX"])
	job = estimator.run(circuit, obs, shots=SHOTS)

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


def test_standard_errors_with_zne_enabled() -> None:
	"""With ZNE on, shot_error (at scale 1) and a propagated zne_extrapolation_error are reported."""
	estimator = FiQCIEstimator(AerSimulator(), mitigation_level=0)
	estimator.zne(enabled=True, scale_factors=[1, 3, 5], extrapolation_method="linear")

	# Superposition gives <ZZ> = <ZI> = 0, i.e. maximal (non-zero) shot noise to propagate.
	obs = SparsePauliOp(["ZZ", "ZI"])
	errors = estimator.run(_superposition(2), obs, shots=SHOTS).standard_errors(0)

	assert errors["zne_extrapolation_error"] is not None
	assert errors["total"] == errors["zne_extrapolation_error"]
	assert len(errors["shot_error"]) == len(obs.paulis)
	assert all(e >= 0 for e in errors["shot_error"])
	assert all(e > 0 for e in errors["zne_extrapolation_error"])
	# Extrapolation amplifies variance, so the zero-noise SE exceeds the raw scale-1 shot SE.
	for raw_se, ext_se in zip(errors["shot_error"], errors["zne_extrapolation_error"]):
		assert ext_se >= raw_se
