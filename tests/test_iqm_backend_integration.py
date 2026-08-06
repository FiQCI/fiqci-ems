"""Integration tests against an IQM fake backend, which uses the real device validation path.

The rest of the suite runs on ``AerSimulator``, which happily accepts circuits IQM hardware would
reject: it ignores the native gate set, the coupling map and the qubit layout. ``IQMFakeAdonis``
runs the same ``validate_circuit`` as the real backend and carries a noise model, so these tests
cover the gap between "works on Aer" and "works on hardware".

Note that EMS does not raise on submission failure: ``run()`` logs a warning and returns a handle
whose ``result()`` raises :class:`BatchFailedError`. Every test therefore fetches results, which is
what surfaces a rejected circuit.
"""

from __future__ import annotations

import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from iqm.qiskit_iqm import IQMFakeAdonis, transpile_to_IQM

from fiqci.ems import FiQCIEstimator, FiQCISampler

SHOTS = 2048
LEVELS = [0, 1, 2, 3]


def _bell() -> QuantumCircuit:
	qc = QuantumCircuit(2)
	qc.h(0)
	qc.cx(0, 1)
	return qc


@pytest.fixture(scope="module")
def backend() -> IQMFakeAdonis:
	return IQMFakeAdonis()


@pytest.fixture(scope="module")
def transpiled(backend: IQMFakeAdonis) -> QuantumCircuit:
	"""A Bell circuit compiled for the device.

	``remove_final_rzs=False`` is required for the estimator: it measures in the X and Y bases, and
	dropping the final RZ gates changes those expectation values.
	"""
	return transpile_to_IQM(_bell(), backend, remove_final_rzs=False, optimization_level=3)


@pytest.fixture(scope="module")
def measured(transpiled: QuantumCircuit) -> QuantumCircuit:
	"""The transpiled circuit with measurements on the qubits its virtual qubits were mapped onto."""
	positions = transpiled.layout.final_index_layout()
	qc = transpiled.copy()
	creg = ClassicalRegister(len(positions), "c")
	qc.add_register(creg)
	for virtual, position in enumerate(positions):
		qc.measure(qc.qubits[position], creg[virtual])
	return qc


@pytest.fixture(scope="module")
def device_observables(transpiled: QuantumCircuit) -> SparsePauliOp:
	"""Observables mapped onto the transpiled circuit's layout, as the docs instruct."""
	return SparsePauliOp(["ZZ", "XX"]).apply_layout(transpiled.layout)


class TestSamplerOnRealDeviceValidation:
	"""The sampler's submitted circuits must satisfy IQM's circuit validation."""

	@pytest.mark.parametrize("level", LEVELS)
	def test_runs_and_returns_counts(self, backend: IQMFakeAdonis, measured: QuantumCircuit, level: int) -> None:
		counts = FiQCISampler(backend, mitigation_level=level).run(measured, shots=SHOTS).result().get_counts()

		assert counts
		# Mitigation renormalises, so allow a small deviation rather than requiring exactly SHOTS.
		assert sum(counts.values()) == pytest.approx(SHOTS, rel=0.02)
		assert all(set(key.replace(" ", "")) <= {"0", "1"} for key in counts)
		total = sum(counts.values())
		probabilities = {key.replace(" ", ""): count / total for key, count in counts.items()}
		assert probabilities.get("00", 0.0) > 0.3
		assert probabilities.get("11", 0.0) > 0.3

	def test_untranspiled_circuit_is_rejected(self, backend: IQMFakeAdonis) -> None:
		"""Guards the tests above: the backend really does validate, so passing them means something."""
		qc = _bell()  # h and cx are not IQM native gates
		qc.measure_all()

		with pytest.raises(Exception) as excinfo:
			FiQCISampler(backend, mitigation_level=0).run(qc, shots=64).result()

		assert "natively supported" in str(excinfo.value) or "not allowed as locus" in str(excinfo.value)


class TestEstimatorOnRealDeviceValidation:
	"""The estimator builds its own measurement circuits, so they need validating too."""

	@pytest.mark.parametrize("level", LEVELS)
	def test_runs_and_returns_expectation_values(
		self, backend: IQMFakeAdonis, transpiled: QuantumCircuit, device_observables: SparsePauliOp, level: int
	) -> None:
		values = (
			FiQCIEstimator(backend, mitigation_level=level)
			.run(transpiled, device_observables, shots=SHOTS)
			.expectation_values(0)
		)

		assert len(values) == len(device_observables.paulis)
		# A noisy device pulls these below the ideal 1.0, but they must stay physical.
		assert all(-1.05 <= value <= 1.05 for value in values)

	def test_standard_errors_are_reported(
		self, backend: IQMFakeAdonis, transpiled: QuantumCircuit, device_observables: SparsePauliOp
	) -> None:
		errors = (
			FiQCIEstimator(backend, mitigation_level=0)
			.run(transpiled, device_observables, shots=SHOTS)
			.standard_errors(0)
		)

		assert len(errors["shot_error"]) == len(device_observables.paulis)
		assert all(error > 0 for error in errors["shot_error"])


class TestMitigationImprovesAccuracy:
	"""Beyond running, mitigation should measurably help on a noisy device.

	Nothing else in the suite checks this: Aer's default noiseless simulation leaves no error to
	mitigate, so a no-op mitigator would pass every other test.
	"""

	def test_readout_mitigation_moves_zz_towards_the_ideal_value(
		self, backend: IQMFakeAdonis, transpiled: QuantumCircuit, device_observables: SparsePauliOp
	) -> None:
		shots = 8192  # keep shot noise well below the readout-error gap being measured
		raw = (
			FiQCIEstimator(backend, mitigation_level=0)
			.run(transpiled, device_observables, shots=shots)
			.expectation_values(0)[0]
		)
		mitigated = (
			FiQCIEstimator(backend, mitigation_level=1)
			.run(transpiled, device_observables, shots=shots)
			.expectation_values(0)[0]
		)

		# Ideal <ZZ> on a Bell pair is 1.0. Adonis' readout error costs roughly 0.2 of that, so the
		# improvement from M3 is far larger than the ~0.01 shot noise at this shot count.
		assert raw < 0.95, f"expected visible readout error to mitigate, got <ZZ>={raw}"
		assert mitigated > raw, f"REM did not improve <ZZ>: raw={raw}, mitigated={mitigated}"
		assert abs(1.0 - mitigated) < abs(1.0 - raw)
