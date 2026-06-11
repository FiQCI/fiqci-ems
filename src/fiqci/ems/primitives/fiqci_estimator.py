"""
A class that runs quantum circuits and calculates expectation values of observables with error mitigation techniques.
"""

from __future__ import annotations
import logging
import threading
import warnings
from collections.abc import Callable
from typing import Any, TypedDict, cast

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Pauli
from fiqci.ems import FiQCIBackend
from fiqci.ems.transpiler_passes.basis_measurement import (
	get_obs_subcircuits,
	_get_observable_circuit_index,
	_combine_pauli_ops,
)
from fiqci.ems.utils import _remove_idle_wires
from fiqci.ems.transpiler_passes.zne_circuits import _get_zne_circuits
from fiqci.ems.mitigators.zne import exponential_extrapolation, richardson_extrapolation, polynomial_extrapolation
from fiqci.ems.mitigators.dd import DDGateSequenceEntry

logger: logging.Logger = logging.getLogger(__name__)


class FiQCIEstimator:
	"""
	FiQCIEstimator wraps a backend with built-in error mitigation (readout error mitigation via M3,
	zero-noise extrapolation) and computes expectation values of observables directly from circuits,
	eliminating the need for manual post-processing of measurement counts.

	Mitigation levels:
		- 0: No error mitigation (raw results)
		- 1: Readout error mitigation using M3 (default)
		- 2: Level 1 + dynamical decoupling (DD)
		- 3: Level 2 + zero-noise extrapolation (ZNE) with local folding and exponential extrapolation

	Args:
		backend: An IQMBackendBase instance to wrap.
		mitigation_level: Level of error mitigation to apply (default: 1).
		calibration_shots: Number of shots to use for calibration circuits (default: 1000).
		calibration_file: Optional calibration file to use for readout error mitigation.

	"""

	def __init__(self, backend, mitigation_level=1, calibration_shots=1000, calibration_file=None):
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
			"fold_gates": None,  # if None, fold all gates. Otherwise, should be a list of gate names to fold (e.g. ["cx", "cz"])
			"scale_factors": [1, 3, 5],  # odd integers
			"folding_method": "local",  # global or local folding
			"extrapolation_method": "exponential",  # exponential, richardson, linear, polynomial
			"extrapolation_degree": None,  # int only for polynomial, None defaults to min(n_scales - 1, 2), where n_scales is the number of scale factors
		}

		if self._mitigation_level in [0, 1, 2]:
			self.backend = FiQCIBackend(backend, mitigation_level, calibration_shots, calibration_file)
		elif self._mitigation_level == 3:
			self.backend = FiQCIBackend(backend, 2, calibration_shots, calibration_file)
			self.zne(enabled=True)
		else:
			raise NotImplementedError(f"Unknown mitigation level {mitigation_level}")

	@property
	def mitigator_options(self) -> dict[str, Any]:
		"""Get current mitigator settings."""
		return {"zne": self._zne, **self.backend.mitigator_options}

	def total_circuits_generated(
		self, num_base_circuits: int, observables: SparsePauliOp | list[SparsePauliOp], detailed: bool = False
	) -> int | dict[str, int]:
		"""Calculate total circuits generated for a given number of base circuits and observables."""
		measurement_settings = _combine_pauli_ops(
			observables if isinstance(observables, SparsePauliOp) else observables[0]
		)
		num_measurement_circuits = len(measurement_settings)
		zne_circuits_multiplier = 1
		pauli_twirl_circuits_multiplier = 1

		if self._zne["enabled"]:
			zne_circuits_multiplier = len(self._zne["scale_factors"])
		if self.backend._pauli_twirl["enabled"]:
			pauli_twirl_circuits_multiplier = (
				self.backend._pauli_twirl["num_twirls"] + 1
			)  # +1 for the original circuit without twirling

		total_circuits = (
			num_base_circuits * num_measurement_circuits * zne_circuits_multiplier * pauli_twirl_circuits_multiplier
		)

		if detailed:
			print(
				f"The total number of circuits is {total_circuits}, calculated as follows: base circuits ({num_base_circuits}) * circuits for conflicting basis measurements ({num_measurement_circuits}) * ZNE multiplier ({zne_circuits_multiplier}) * Pauli twirl multiplier ({pauli_twirl_circuits_multiplier}). This does not include circuits ran to calibrate readout error mitigation (REM)."
			)
			return {
				"base_circuits": num_base_circuits,
				"measurement_circuits_per_basis": num_measurement_circuits,
				"zne_multiplier": zne_circuits_multiplier,
				"pauli_twirl_multiplier": pauli_twirl_circuits_multiplier,
				"total_circuits": total_circuits,
			}
		else:
			return total_circuits

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
		max_batch_size: int = 100,
		**options,
	) -> FiQCIEstimatorJob:
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

		num_pairs = len(obs_circuits)
		if self._zne["enabled"]:
			logger.info(
				"FiQCIEstimator.run: %d circuit/observable pair(s); mitigation_level=%d, "
				"ZNE=on (scales=%s, folding=%s, extrapolation=%s)",
				num_pairs,
				self._mitigation_level,
				self._zne["scale_factors"],
				self._zne["folding_method"],
				self._zne["extrapolation_method"],
			)
		else:
			logger.info(
				"FiQCIEstimator.run: %d circuit/observable pair(s); mitigation_level=%d, ZNE=off",
				num_pairs,
				self._mitigation_level,
			)

		flat_circuits: list[QuantumCircuit] = []
		pair_lengths: list[int] = []
		pair_measurement_settings: list[list[dict[int, str]]] = []
		num_base_circuits = 0  # measurement-basis circuits before ZNE scale-factor expansion

		for i, obs_circ_groups in enumerate(obs_circuits):
			obs_circs_list = [group[0] for group in obs_circ_groups]

			measurement_settings = _combine_pauli_ops(
				observables if isinstance(observables, SparsePauliOp) else observables[i]
			)
			pair_measurement_settings.append(measurement_settings)
			num_base_circuits += len(obs_circs_list)

			if self._zne["enabled"]:
				obs_circs_list = _get_zne_circuits(
					obs_circs_list, self._zne["fold_gates"], self._zne["scale_factors"], self._zne["folding_method"]
				)

			pair_lengths.append(len(obs_circs_list))
			flat_circuits.extend(obs_circs_list)

		if self._zne["enabled"]:
			logger.info(
				"Flattened %d pair(s) into %d measurement-basis circuit(s), expanded to %d after %dx ZNE; "
				"forwarding to backend with max_batch_size=%d",
				num_pairs,
				num_base_circuits,
				len(flat_circuits),
				len(self._zne["scale_factors"]),
				max_batch_size,
			)
		else:
			logger.info(
				"Flattened %d pair(s) into %d measurement-basis circuit(s); forwarding to backend with max_batch_size=%d",
				num_pairs,
				len(flat_circuits),
				max_batch_size,
			)

		# Backend returns a lazy handle immediately (it does not wait for results). The expectation
		# values are computed on first access via the deferred closure below.
		job = self.backend.run(flat_circuits, shots=shots, max_batch_size=max_batch_size, **options)

		# Snapshot ZNE settings at submission time so the deferred computation stays consistent with
		# the circuits that were actually submitted, even if the user mutates settings via zne()
		# before accessing the results.
		zne_enabled = self._zne["enabled"]
		zne_scale_factors = self._zne["scale_factors"]
		zne_extrapolation_method = self._zne["extrapolation_method"]
		zne_extrapolation_degree = self._zne["extrapolation_degree"]

		def _compute() -> tuple[list, list]:
			"""Fetch results and compute per-pair (and ZNE-extrapolated) expectation values.

			Returns ``(expectation_values, raw_expectation_values)`` where the second element holds
			the pre-extrapolation ZNE values when ZNE is enabled, otherwise the same values.
			"""
			all_counts = job.result().get_counts()
			if not isinstance(all_counts, list):
				all_counts = [all_counts]

			expectation_values: list = []
			all_zne_expvs: list = []
			offset = 0
			for i, length in enumerate(pair_lengths):
				counts = all_counts[offset : offset + length]
				offset += length

				measurement_settings = pair_measurement_settings[i]
				zne_expvs = []

				if zne_enabled:
					split_counts = []
					num_circs_per_zne = len(measurement_settings)
					for j in range(0, len(counts), num_circs_per_zne):
						split_counts.append(counts[j : j + num_circs_per_zne])

					for c in split_counts:
						expvs = self._calculate_expectation_values(
							c,
							observables if isinstance(observables, SparsePauliOp) else observables[i],
							measurement_settings,
						)
						zne_expvs.append(expvs)

					if zne_extrapolation_method == "exponential":
						expvs = exponential_extrapolation(zne_expvs, zne_scale_factors)
					elif zne_extrapolation_method == "richardson":
						expvs = richardson_extrapolation(zne_expvs, zne_scale_factors)
					elif zne_extrapolation_method == "polynomial":
						expvs = polynomial_extrapolation(zne_expvs, zne_scale_factors, degree=zne_extrapolation_degree)
					elif zne_extrapolation_method == "linear":
						expvs = polynomial_extrapolation(zne_expvs, zne_scale_factors, degree=1)
				else:
					expvs = self._calculate_expectation_values(
						counts,
						observables if isinstance(observables, SparsePauliOp) else observables[i],
						measurement_settings,
					)

				expectation_values.append(expvs)
				if zne_enabled and len(zne_expvs) > 0:
					all_zne_expvs.append(zne_expvs)

			if zne_enabled and len(all_zne_expvs) > 0:
				return expectation_values, all_zne_expvs
			return expectation_values, expectation_values

		return FiQCIEstimatorJob(job, _compute, observables)

	def run(
		self,
		circuits: QuantumCircuit | list[QuantumCircuit],
		observables: SparsePauliOp | list[SparsePauliOp],
		shots: int = 2048,
		max_batch_size: int = 100,
		**options,
	) -> FiQCIEstimatorJob:
		"""
		Execute the given circuits on the backend and calculate expectation values for the provided observables.

		Args:
			circuits: A QuantumCircuit or list of QuantumCircuits to execute.
			observables: A SparsePauliOp or list of SparsePauliOps representing the observables for which to calculate expectation values.
			shots: Number of shots to execute each circuit (default: 2048).
			max_batch_size: Maximum number of circuits to send in a single backend job. All measurement-basis subcircuits
				(across all circuit/observable pairs and ZNE scale factors) are flattened and split into batches of this
				size (default: 100).
			**options: Additional options to pass to the backend's run method.

		Returns:
			A FiQCIEstimatorJob containing the jobs and calculated expectation values.
		"""
		return self._run(circuits, observables, shots=shots, max_batch_size=max_batch_size, **options)

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

	def dd(self, enabled: bool, gate_sequences: list[DDGateSequenceEntry] | None = None) -> None:
		"""
		Set dynamical decoupling settings for the estimator. This will configure the underlying backend's
		dynamical decoupling accordingly.

		Args:
			enabled: Whether to enable dynamical decoupling.
			gate_sequences: List of (threshold_length, sequence, strategy) tuples defining DD behavior.
				See build_dd_options for details on each field.
		"""
		self.backend.dd(enabled, gate_sequences)

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
		if extrapolation_method not in ["polynomial"] and extrapolation_degree is not None:
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
		if extrapolation_method in ["polynomial"] and extrapolation_degree is not None:
			self._zne["extrapolation_degree"] = extrapolation_degree
		else:
			self._zne["extrapolation_degree"] = None

	def pauli_twirl(self, enabled: bool, num_twirls: int = 10, gates_to_twirl: list | None = None) -> None:
		"""Configure Pauli twirling settings for the estimator."""
		self.backend.pauli_twirl(enabled, num_twirls, gates_to_twirl)


class FiQCIEstimatorJob:
	"""Lazy wrapper around the backend job that produced an estimator's results.

	The estimator flattens all per-pair measurement-basis circuits into one backend call, so there
	is exactly one underlying job (which may itself batch internally — see ``BatchedJob``). This
	class is returned immediately from :meth:`FiQCIEstimator.run`; the expectation-value
	computation is deferred until :meth:`expectation_values` / :meth:`raw_expectation_values` is
	first called (it fetches the underlying results and computes once, then caches). Polling the
	underlying job (``status``/``done``/``job_ids``) works before the values are computed.
	"""

	def __init__(self, job, compute_fn: Callable[[], tuple[list, list]], observables) -> None:
		"""Initialize the estimator job.

		Args:
		    job: The underlying job that produced the results (BatchedJob or MitigatedJob).
		    compute_fn: Deferred callable returning ``(expectation_values, raw_expectation_values)``.
		    observables: Observable(s) for which expectation values were calculated.
		"""
		self.mitigated_job = job
		self._compute_fn = compute_fn
		self._observables = observables
		self._expectation_values: list | None = None
		self._raw_expectation_values: list | None = None
		self._computed = False
		self._lock = threading.Lock()

	def _ensure_computed(self) -> None:
		"""Run the deferred computation once (blocking on results), caching the outcome."""
		if self._computed:
			return
		with self._lock:
			if self._computed:
				return
			self._expectation_values, self._raw_expectation_values = self._compute_fn()
			self._computed = True

	def result(self):
		"""Get the underlying combined result for this estimator run (blocks until ready)."""
		return self.mitigated_job.result()

	def job(self):
		"""Get the underlying job for this estimator run."""
		return self.mitigated_job

	def raw_expectation_values(self, index: int | None = None) -> list[float]:
		"""Get the raw (unmitigated) expectation values before extrapolation (computes lazily)."""
		self._ensure_computed()
		assert self._raw_expectation_values is not None
		if index is not None:
			return self._raw_expectation_values[index]
		return self._raw_expectation_values

	def expectation_values(self, index: int | None = None) -> list[float]:
		"""Get the calculated expectation values (computes lazily on first access)."""
		self._ensure_computed()
		assert self._expectation_values is not None
		if index is not None:
			return self._expectation_values[index]
		return self._expectation_values

	def observables(self, index: int | None = None) -> SparsePauliOp:
		"""Get the observables for which expectation values were calculated."""
		if index is not None:
			return self._observables[index]
		return self._observables

	def __getattr__(self, name: str) -> Any:
		"""Delegate polling/attribute access (status, done, job_ids, …) to the underlying job."""
		return getattr(self.mitigated_job, name)
