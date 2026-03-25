"""
A class that runs quantum circuits and calculates expectation values of observables with error mitigation techniques.
"""
from __future__ import annotations
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from fiqci.ems import FiQCIBackend
from fiqci.ems.basis_measurement import get_obs_subcircuits, _get_observable_circuit_index, _combine_pauli_ops
from .utils import _remove_idle_wires


class FiQCIEstimator:
	"""
	FiQCIEstimator wraps a backend with built-in error mitigation (readout error mitigation via M3,
	zero-noise extrapolation) and computes expectation values of observables directly from circuits,
	eliminating the need for manual post-processing of measurement counts.

	Mitigation levels:
		- 0: No error mitigation (raw results)
		- 1: Readout error mitigation using M3 (default)
	
	Args:
		backend: An IQMBackendBase instance to wrap.
		mitigation_level: Level of error mitigation to apply (default: 1).
		calibration_shots: Number of shots to use for calibration circuits (default: 1000).
		calibration_files: Optional list of calibration files to use for readout error mitigation.

	"""
	def __init__(self, backend, mitigation_level=1, calibration_shots=1000, calibration_files=None):
		super().__init__()
		self.mitigation_level = mitigation_level

		self.backend = FiQCIBackend(backend, mitigation_level, calibration_shots, calibration_files)

	@property
	def mitigator_options(self) -> dict[str, Any]:
		"""Get current mitigator settings."""
		return {**self.backend.mitigator_options()}

	def _make_meas_instruction(self, circuit: QuantumCircuit, label: str):
		"""Transpile a measurement circuit to basis gates and wrap as an instruction."""
		circuit = transpile(circuit, target=self.backend.target, optimization_level=3)
		circuit = _remove_idle_wires(circuit)
		return circuit.to_instruction(label=label)

	def _run(
		self,
		circuits: QuantumCircuit | list[QuantumCircuit],
		observables: SparsePauliOp | list[SparsePauliOp],
		shots: int = 2048,
		**options,
	) -> FiQCIEstimatorJobCollection:
		x_meas = QuantumCircuit(1)
		x_meas.h(0)

		y_meas = QuantumCircuit(1)
		y_meas.sdg(0)
		y_meas.h(0)

		ops = {
			"X-meas": self._make_meas_instruction(x_meas, "X-meas"),
			"Y-meas": self._make_meas_instruction(y_meas, "Y-meas"),
		}

		# if observables and circuits are both lists, they must be of the same length and we pair them elementwise
		if isinstance(observables, list) and isinstance(circuits, list):
			if len(observables) != len(circuits):
				# raise error if lengths don't match
				raise ValueError("Length of observables and circuits lists must match.")

			# if lengths match, we pair them elementwise
			else:
				obs_circuits = [
					get_obs_subcircuits([circ], _combine_pauli_ops(obs), ops)
					for circ, obs in zip(circuits, observables)
				]

		# if observables is a single SparsePauliOp and circuits is a list, we use the same observables for all circuits
		elif isinstance(observables, SparsePauliOp) and isinstance(circuits, list):
			obs_circuits = [get_obs_subcircuits([circ], _combine_pauli_ops(observables), ops) for circ in circuits]

		# if observables is a single SparsePauliOp and circuits is a single QuantumCircuit, we just pair them
		elif isinstance(observables, SparsePauliOp) and isinstance(circuits, QuantumCircuit):
			obs_circuits = [get_obs_subcircuits([circuits], _combine_pauli_ops(observables), ops)]
		else:
			raise TypeError(f"Unsupported types: circuits={type(circuits)}, observables={type(observables)}")

		expectation_values = []

		jobs = []

		for i, obs_circ_groups in enumerate(obs_circuits):
			obs_circs_list = [group[0] for group in obs_circ_groups]

			measurement_settings = _combine_pauli_ops(
				observables if isinstance(observables, SparsePauliOp) else observables[i]
			)

			job = self.backend.run(obs_circs_list, shots=shots, **options)

			jobs.append(job)

			results = job.result()

			counts = results.get_counts()

			expvs = self._calculate_expectation_values(
				counts, observables if isinstance(observables, SparsePauliOp) else observables[i], measurement_settings
			)

			expectation_values.append(expvs)

		return FiQCIEstimatorJobCollection(jobs, expectation_values, observables)

	def run(self, circuits: QuantumCircuit | list[QuantumCircuit], observables: SparsePauliOp | list[SparsePauliOp], shots: int = 2048, **options) -> FiQCIEstimatorJobCollection:
		"""
		Execute the given circuits on the backend and calculate expectation values for the provided observables.

		Args:
			circuits: A QuantumCircuit or list of QuantumCircuits to execute.
			observables: A SparsePauliOp or list of SparsePauliOps representing the observables for which to calculate expectation values.
			shots: Number of shots to execute each circuit (default: 2048).
			**options: Additional options to pass to the backend's run method.
		
		Returns:
			A FiQCIEstimatorJobCollection containing the jobs and calculated expectation values.
		"""
		return self._run(circuits, observables, shots=shots, **options)

	def _calculate_expectation_values(
		self,
		counts: dict[str, int] | list[dict[str, int]],
		obs: SparsePauliOp,
		measurement_settings: list[dict[int, str]],
	) -> list[float]:
		if not isinstance(counts, list):
			counts = [counts]
		expectation_values = []
		for pauli in obs.paulis:
			obs_info = _get_observable_circuit_index(pauli, measurement_settings)
			if obs_info["circuit_index"] is not None:
				circuit_counts = counts[obs_info["circuit_index"]]
				# Calculate expectation value from counts
				exp_val = 0
				for bitstring, count in circuit_counts.items():
					parity = 1
					for idx in obs_info["obs_indices"]:
						if bitstring[idx] == "1":
							parity *= -1
					exp_val += parity * count
				exp_val /= sum(circuit_counts.values())
				expectation_values.append(exp_val)
			else:
				expectation_values.append(0)  # No measurement setting covers this observable
		return expectation_values

	def rem(self, enabled: bool, calibration_shots: int = 1000, calibration_file: str | None = None) -> None:
		"""
		Set readout error mitigation settings for the estimator. This will configure the underlying backend's
		readout error mitigation accordingly.

		Args:
			enabled: Whether to enable readout error mitigation.
			calibration_shots: Number of shots to use for calibration circuits (default: 1000).
			calibration_file: Optional calibration file to use for readout error mitigation.
		"""
		self.backend.rem(enabled, calibration_shots, calibration_file)

class FiQCIEstimatorJobCollection:
	"""Wrapper for job results with mitigated data.

	This class wraps the original job and provides access to mitigated results.
	"""

	def __init__(self, mitigated_jobs, expectation_values, observables) -> None:
		"""Initialize mitigated job wrapper.

		Args:
		    original_job: Original job from backend.
		    mitigated_result: Result object with mitigated counts.
		"""
		self.mitigated_jobs = mitigated_jobs
		self._expectation_values = expectation_values
		self._observables = observables

	def results(self):
		"""Get all results for this estimator."""
		return [job.result() for job in self.mitigated_jobs]

	def jobs(self):
		"""Get all jobs ran for this estimator."""
		return self.mitigated_jobs

	def expectation_values(self, index: int | None = None) -> list[float]:
		"""Get the calculated expectation values."""
		if index is not None:
			return self._expectation_values[index]
		return self._expectation_values

	def observables(self, index: int | None = None) -> SparsePauliOp:
		"""Get the observables for which expectation values were calculated."""
		if index is not None:
			return self._observables[index]
		return self._observables
