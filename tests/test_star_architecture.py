"""Submission tests against a Star-architecture fake backend (IQMFakeDeneb).

``IQMFakeAdonis`` (used by ``test_iqm_backend_integration.py``) is a Crystal device: every qubit
couples directly, there is no computational resonator and no MOVE gate. A Star device routes
two-qubit interactions through a resonator, so a transpiled circuit contains ``move`` gates and its
gates land on device components (``QB1``, ``CR1``) rather than on plain qubit indices.

That makes a whole class of bug invisible to the rest of the suite: anything that drops or rewrites a
circuit's ``TranspileLayout`` shifts which physical component each gate acts on, and on a Star device
that can put a single-qubit gate on the resonator. ``serialize_instructions`` resolves a circuit to
exactly the device components the submission would carry, so this is checkable offline.
"""

from __future__ import annotations

import warnings

import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from iqm.qiskit_iqm import IQMFakeDeneb, transpile_to_IQM
from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions

from fiqci.ems import FiQCIEstimator, FiQCISampler

SHOTS = 64
LEVELS = [0, 1, 2, 3]


def _bell() -> QuantumCircuit:
	qc = QuantumCircuit(2)
	qc.h(0)
	qc.cx(0, 1)
	return qc


@pytest.fixture(scope="module")
def backend() -> IQMFakeDeneb:
	return IQMFakeDeneb()


@pytest.fixture(scope="module")
def resonators(backend: IQMFakeDeneb) -> list[str]:
	return list(backend.architecture.computational_resonators)


@pytest.fixture(scope="module")
def transpiled(backend: IQMFakeDeneb) -> QuantumCircuit:
	"""A Bell circuit compiled for the device, MOVE-routed through the resonator.

	``remove_final_rzs=False`` is required for the estimator, which measures in the X and Y bases.
	``initial_layout`` is deliberately non-trivial: with the default layout the virtual-to-physical map
	happens to be the identity over the first qubits, so a circuit that lost its layout would serialize
	to the same components anyway and these tests would pass vacuously.
	"""
	circuit = transpile_to_IQM(
		_bell(), backend, remove_final_rzs=False, optimization_level=3, seed_transpiler=0, initial_layout=[2, 4]
	)
	assert circuit.count_ops().get("move", 0), "test needs a MOVE-routed circuit to be meaningful"
	return circuit


@pytest.fixture(scope="module")
def measured(transpiled: QuantumCircuit) -> QuantumCircuit:
	"""The transpiled circuit with measurements on the qubits its virtual qubits were mapped onto."""
	positions = transpiled.layout.final_index_layout()
	circuit = transpiled.copy()
	creg = ClassicalRegister(len(positions), "c")
	circuit.add_register(creg)
	for virtual, position in enumerate(positions):
		circuit.measure(circuit.qubits[position], creg[virtual])
	return circuit


@pytest.fixture(scope="module")
def device_observables(transpiled: QuantumCircuit) -> SparsePauliOp:
	"""Observables mapped onto the transpiled circuit's layout, as the docs instruct."""
	return SparsePauliOp(["ZZ", "XX", "YY"]).apply_layout(transpiled.layout)


def _loci(backend: IQMFakeDeneb, circuit: QuantumCircuit) -> dict[str, list[str]]:
	"""The device components each native operation of ``circuit`` acts on, as submitted."""
	loci: dict[str, set[str]] = {}
	for instruction in serialize_instructions(circuit, qubit_index_to_name=backend._idx_to_qb):
		loci.setdefault(instruction.name, set()).update(instruction.locus)
	return {name: sorted(components) for name, components in loci.items()}


def _capture_submissions(backend: IQMFakeDeneb, run) -> list[QuantumCircuit]:
	"""Run ``run()`` with ``backend.run`` spying on the circuits it is handed."""
	captured: list[QuantumCircuit] = []
	original = backend.run

	def spy(circuits, **kwargs):
		captured.extend(circuits if isinstance(circuits, list) else [circuits])
		return original(circuits, **kwargs)

	backend.run = spy
	try:
		with warnings.catch_warnings():
			# A noisy fake device can make M3 over-subtract; the warning is not what is under test.
			warnings.simplefilter("ignore")
			run()
	finally:
		backend.run = original

	assert captured, "nothing was submitted to the backend"
	return captured


def _assert_loci_are_sound(
	backend: IQMFakeDeneb, submitted: list[QuantumCircuit], reference: QuantumCircuit, resonators: list[str]
) -> None:
	"""Every submitted circuit must place its gates on the same components as the transpiled input.

	The mitigators only append measurement-basis rotations (single-qubit), duplicate existing gates
	(ZNE folding) or conjugate them with Paulis (twirling), so the two-qubit and MOVE loci are
	invariant. A single-qubit gate on a resonator, or a MOVE between the wrong pair, means the layout
	was lost somewhere between the input circuit and submission: without one, ``serialize_instructions``
	falls back to reading qubit indices as component indices, which shifts every gate.

	Only the MOVE-routed circuits are checked. Level 1+ also submits M3's own calibration circuits
	through the same ``backend.run``; those are not routed and carry no layout.
	"""
	reference_loci = _loci(backend, reference)
	routed = [circuit for circuit in submitted if circuit.count_ops().get("move", 0)]
	assert routed, "no MOVE-routed circuit reached the backend"

	for circuit in routed:
		assert circuit.layout is not None, "submitted circuit lost its TranspileLayout"
		loci = _loci(backend, circuit)

		for resonator in resonators:
			assert resonator not in loci.get("prx", []), f"single-qubit gate on resonator {resonator}: {loci}"
			assert resonator not in loci.get("measure", []), f"measurement on resonator {resonator}: {loci}"

		for name in ("move", "cz"):
			assert loci.get(name, []) == reference_loci.get(name, []), (
				f"{name} loci changed on submission: {loci.get(name)} != {reference_loci.get(name)}"
			)


class TestSamplerOnStarArchitecture:
	@pytest.mark.parametrize("level", LEVELS)
	def test_submitted_circuits_keep_their_device_loci(
		self, backend: IQMFakeDeneb, measured: QuantumCircuit, resonators: list[str], level: int
	) -> None:
		sampler = FiQCISampler(backend, mitigation_level=level)
		submitted = _capture_submissions(backend, lambda: sampler.run(measured, shots=SHOTS).result())

		_assert_loci_are_sound(backend, submitted, measured, resonators)


class TestEstimatorOnStarArchitecture:
	@pytest.mark.parametrize("level", LEVELS)
	def test_submitted_circuits_keep_their_device_loci(
		self,
		backend: IQMFakeDeneb,
		transpiled: QuantumCircuit,
		device_observables: SparsePauliOp,
		resonators: list[str],
		level: int,
	) -> None:
		estimator = FiQCIEstimator(backend, mitigation_level=level)
		submitted = _capture_submissions(
			backend, lambda: estimator.run(transpiled, device_observables, shots=SHOTS).expectation_values(0)
		)

		_assert_loci_are_sound(backend, submitted, transpiled, resonators)

	def test_submitted_circuits_keep_their_device_loci_with_zne_and_twirling(
		self,
		backend: IQMFakeDeneb,
		transpiled: QuantumCircuit,
		device_observables: SparsePauliOp,
		resonators: list[str],
	) -> None:
		"""The full stack at once: folding duplicates gates and twirling wraps them in Paulis."""
		estimator = FiQCIEstimator(backend, mitigation_level=1)
		estimator.zne(enabled=True, scale_factors=[1, 3], folding_method="local")
		estimator.pauli_twirl(enabled=True, num_twirls=2, seed=0)

		submitted = _capture_submissions(
			backend, lambda: estimator.run(transpiled, device_observables, shots=SHOTS).expectation_values(0)
		)

		_assert_loci_are_sound(backend, submitted, transpiled, resonators)

	def test_estimator_values_are_physical_on_a_star_device(
		self, backend: IQMFakeDeneb, transpiled: QuantumCircuit, device_observables: SparsePauliOp
	) -> None:
		"""Guards the loci tests: a circuit whose gates moved would not measure a Bell state at all."""
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			values = (
				FiQCIEstimator(backend, mitigation_level=0)
				.run(transpiled, device_observables, shots=4096)
				.expectation_values(0)
			)

		zz, xx, yy = values
		assert zz > 0.5, f"<ZZ> should be near +1 on a Bell state, got {zz}"
		assert xx > 0.5, f"<XX> should be near +1 on a Bell state, got {xx}"
		assert yy < -0.5, f"<YY> should be near -1 on a Bell state, got {yy}"


class TestDDWarningOnStarArchitecture:
	"""DD is unvalidated on this architecture, so every interface must say so before submitting."""

	@pytest.mark.parametrize("level", [2, 3])
	def test_sampler_warns_at_dd_levels(self, backend: IQMFakeDeneb, measured: QuantumCircuit, level: int) -> None:
		with pytest.warns(UserWarning, match="corrupts MOVE-routed circuits"):
			FiQCISampler(backend, mitigation_level=level).run(measured, shots=SHOTS)

	@pytest.mark.parametrize("level", [2, 3])
	def test_estimator_warns_at_dd_levels(
		self, backend: IQMFakeDeneb, transpiled: QuantumCircuit, device_observables: SparsePauliOp, level: int
	) -> None:
		with pytest.warns(UserWarning, match="corrupts MOVE-routed circuits"):
			FiQCIEstimator(backend, mitigation_level=level).run(transpiled, device_observables, shots=SHOTS)

	@pytest.mark.parametrize("level", [0, 1])
	def test_no_warning_below_the_dd_levels(
		self, backend: IQMFakeDeneb, transpiled: QuantumCircuit, device_observables: SparsePauliOp, level: int
	) -> None:
		with warnings.catch_warnings():
			warnings.simplefilter("error", UserWarning)
			FiQCIEstimator(backend, mitigation_level=level).run(transpiled, device_observables, shots=SHOTS)
