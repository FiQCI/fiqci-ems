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
