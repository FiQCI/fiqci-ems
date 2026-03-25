"""
A class that runs quantum circuits and calculates expectation values of observables with error mitigation techniques.

FiQCIEstimator wraps a backend with built-in error mitigation (readout error mitigation via M3,
zero-noise extrapolation) and computes expectation values of observables directly from circuits,
eliminating the need for manual post-processing of measurement counts.
"""

import warnings
from typing import TypedDict, cast

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Pauli
from fiqci.ems import FiQCIBackend
from fiqci.ems.transpiler_passes.basis_measurement import (
	_get_obs_subcircuits,
	_get_observable_circuit_index,
	_combine_pauli_ops,
)
from fiqci.ems.utils import _remove_idle_wires
from fiqci.ems.transpiler_passes.zne_circuits import _get_zne_circuits
from fiqci.ems.mitigators.zne import exponential_extrapolation, richardson_extrapolation, polynomial_extrapolation


class FiQCIEstimator:
	def __init__(self, backend, mitigation_level=1, calibration_shots=1000, calibration_files=None):
		super().__init__()
		self._mitigation_level = mitigation_level

		class ZNESettings(TypedDict):
			enabled: bool
			fold_gates: list | None
			scale_factors: list[int]
			folding_method: str
			extrapolation_method: str
			extrapolation_degree: int | None

		self._zne: ZNESettings = {
			"enabled": mitigation_level == 3,
			"fold_gates": None,  # if None, fold all gates. Otherwise, should be a list of gate names to fold (e.g. ["cx", "u3"])
			"scale_factors": [1, 3, 5],  # odd integers
			"folding_method": "local",  # global or local folding
			"extrapolation_method": "exponential",  # exponential, richardson, linear, polynomial
			"extrapolation_degree": None,  # only for polynomial
		}

		if self._mitigation_level in [0, 1, 2]:
			self.backend = FiQCIBackend(backend, mitigation_level, calibration_shots, calibration_files)
		elif self._mitigation_level == 3:
			self.backend = FiQCIBackend(backend, 2, calibration_shots, calibration_files)
			self.zne(enabled=True)
		else:
			raise NotImplementedError(f"Unknown mitigation level {mitigation_level}")

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
	):
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
					_get_obs_subcircuits([circ], _combine_pauli_ops(obs), ops)
					for circ, obs in zip(circuits, observables)
				]
		# TODO: Better batching for estimator. No need to run multiple separate jobs if total number of circuits
		# is not too large. We can just combine circuits into a single job and split the results afterwards.

		# if observables is a single SparsePauliOp and circuits is a list, we use the same observables for all circuits
		elif isinstance(observables, SparsePauliOp) and isinstance(circuits, list):
			obs_circuits = [_get_obs_subcircuits([circ], _combine_pauli_ops(observables), ops) for circ in circuits]

		# if observables is a single SparsePauliOp and circuits is a single QuantumCircuit, we just pair them
		elif isinstance(observables, SparsePauliOp) and isinstance(circuits, QuantumCircuit):
			obs_circuits = [_get_obs_subcircuits([circuits], _combine_pauli_ops(observables), ops)]
		else:
			raise TypeError(f"Unsupported types: circuits={type(circuits)}, observables={type(observables)}")

		expectation_values = []

		jobs = []

		for i, obs_circ_groups in enumerate(obs_circuits):
			obs_circs_list = [group[0] for group in obs_circ_groups]

			measurement_settings = _combine_pauli_ops(
				observables if isinstance(observables, SparsePauliOp) else observables[i]
			)

			if self._zne["enabled"]:
				obs_circs_list = _get_zne_circuits(
					obs_circs_list, self._zne["fold_gates"], self._zne["scale_factors"], self._zne["folding_method"]
				)

			job = self.backend.run(obs_circs_list, shots=shots, **options)

			jobs.append(job)

			results = job.result()

			counts = results.get_counts()

			if self._zne["enabled"]:
				split_counts = []
				num_circs_per_zne = len(measurement_settings)
				for j in range(0, len(counts), num_circs_per_zne):
					split_counts.append(counts[j : j + num_circs_per_zne])

				zne_expvs = []
				for c in split_counts:
					expvs = self.calculate_expectation_values(
						c,
						observables if isinstance(observables, SparsePauliOp) else observables[i],
						measurement_settings,
					)
					zne_expvs.append(expvs)

				if self._zne["extrapolation_method"] == "exponential":
					expvs = exponential_extrapolation(zne_expvs, self._zne["scale_factors"])
				elif self._zne["extrapolation_method"] == "richardson":
					expvs = richardson_extrapolation(zne_expvs, self._zne["scale_factors"])
				elif self._zne["extrapolation_method"] == "polynomial":
					expvs = polynomial_extrapolation(
						zne_expvs, self._zne["scale_factors"], degree=self._zne["extrapolation_degree"]
					)
				elif self._zne["extrapolation_method"] == "linear":
					expvs = polynomial_extrapolation(zne_expvs, self._zne["scale_factors"], degree=1)
			else:
				expvs = self.calculate_expectation_values(
					counts,
					observables if isinstance(observables, SparsePauliOp) else observables[i],
					measurement_settings,
				)

			expectation_values.append(expvs)

		return FiQCIEstimatorJobCollection(jobs, expectation_values, observables)

	def run(self, circuits, observables, shots=2048, **options):
		return self._run(circuits, observables, shots=shots, **options)

	def calculate_expectation_values(
		self,
		counts: dict[str, int] | list[dict[str, int]],
		obs: SparsePauliOp,
		measurement_settings: list[dict[int, str]],
	) -> list[float]:
		if not isinstance(counts, list):
			counts = [counts]
		expectation_values = []
		for pauli in obs.paulis:
			pauli = cast(Pauli, pauli)
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

	def rem(self, enabled, calibration_shots=1000, calibration_file=None):
		"""Enable or disable readout error mitigation."""
		self.backend.rem(enabled, calibration_shots, calibration_file)

	def mitigator_options(self):
		"""Get current mitigator settings."""
		return {"zne": self._zne, **self.backend.mitigator_options()}

	def zne(
		self,
		enabled: bool,
		fold_gates: list | None = None,
		scale_factors: list[int] = [1, 3, 5],
		folding_method: str = "local",
		extrapolation_method: str = "exponential",
		extrapolation_degree: int | None = None,
	):
		# TODO: Support any real >= 1 scale factor
		# TODO: More extrapolation methods, allow user-defined extrapolation functions
		"""Configure zero-noise extrapolation settings."""
		if extrapolation_method not in ["exponential", "richardson", "polynomial", "linear"]:
			raise ValueError(f"Unsupported extrapolation method: {extrapolation_method}")
		if folding_method not in ["local", "global"]:
			raise ValueError(f"Unsupported folding method: {folding_method}")
		if folding_method == "global" and fold_gates is not None:
			warnings.warn("fold_gates is not applicable for global folding and will be ignored.")
			fold_gates = None
		if len(scale_factors) < 2:
			raise ValueError("At least two scale factors are required for extrapolation.")
		if (
			isinstance(scale_factors, list)
			and any(s <= 0 for s in scale_factors)
			and any(s % 2 == 0 for s in scale_factors)
		):
			raise ValueError("Scale factors must be positive odd integers.")
		if fold_gates is not None and not isinstance(fold_gates, list):
			raise ValueError("fold_gates must be a list of gate names or None.")
		if extrapolation_degree is not None and extrapolation_degree < 1 and extrapolation_method == "polynomial":
			raise ValueError("Extrapolation degree must be at least 1 for polynomial extrapolation.")
		if extrapolation_method not in ("polynomial") and extrapolation_degree is not None:
			warnings.warn(
				"Extrapolation degree is only applicable for polynomial extrapolation and will be ignored for other methods."
			)
		if extrapolation_method == "polynomial" and extrapolation_degree == 1:
			warnings.warn(
				"Extrapolation degree of 1 for polynomial extrapolation is equivalent to linear extrapolation. Consider using 'linear' as the extrapolation method instead."
			)

		self._zne["enabled"] = enabled
		self._zne["fold_gates"] = fold_gates
		self._zne["scale_factors"] = scale_factors
		self._zne["extrapolation_method"] = extrapolation_method
		if extrapolation_method in ("polynomial", "linear"):
			self._zne["extrapolation_degree"] = extrapolation_degree
		else:
			self._zne["extrapolation_degree"] = None


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
