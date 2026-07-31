"""
A class that runs quantum circuits and calculates expectation values of observables with error mitigation techniques.
"""

from __future__ import annotations
import logging
import math
import threading
import warnings
from collections.abc import Callable
from typing import Any, TypedDict, cast

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Pauli
from fiqci.ems import FiQCIBackend
from fiqci.ems.transpiler_passes.basis_measurement import (
	get_obs_subcircuits,
	get_measurement_settings,
	_get_observable_circuit_index,
)
from fiqci.ems.utils import _remove_idle_wires
from fiqci.ems.transpiler_passes.zne_circuits import _achieved_scale_factors, _get_zne_circuits
from fiqci.ems.mitigators.zne import exponential_extrapolation, richardson_extrapolation, polynomial_extrapolation
from fiqci.ems.mitigators.dd import DDGateSequenceEntry

logger: logging.Logger = logging.getLogger(__name__)


def _is_nested_scale_factors(scale_factors) -> bool:
	"""True when ``scale_factors`` is a list of per-circuit lists rather than a single flat list."""
	return len(scale_factors) > 0 and all(isinstance(s, (list, tuple)) for s in scale_factors)


def _valid_flat_scale_factors(scale_factors) -> bool:
	"""True when ``scale_factors`` is a flat list of at least two real numbers >= 1."""
	return len(scale_factors) >= 2 and all(
		isinstance(s, (int, float)) and not isinstance(s, bool) and s >= 1 for s in scale_factors
	)


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
			scale_factors: list[float] | list[list[float]]
			folding_method: str
			extrapolation_method: str
			extrapolation_degree: int | None
			seed: int | None

		self._zne: ZNESettings = {
			"enabled": mitigation_level == 3,
			"fold_gates": None,  # if None, fold all gates. Otherwise, should be a list of gate names to fold (e.g. ["cx", "cz"])
			"scale_factors": [1, 3, 5],  # any real numbers >= 1
			"folding_method": "local",  # global or local folding
			"extrapolation_method": "exponential",  # exponential, richardson, linear, polynomial
			"extrapolation_degree": None,  # int only for polynomial, None defaults to min(n_scales - 1, 2), where n_scales is the number of scale factors
			"seed": None,  # seed for random gate sampling when approximating non-odd-integer scale factors
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
	) -> int | dict[str, Any]:
		"""Calculate total circuits generated for a given number of base circuits and observables.

		When ``scale_factors`` is a per-circuit list of lists, each circuit expands by its own count, so
		the ZNE contribution is the sum of the per-circuit counts (this assumes ``num_base_circuits``
		matches the number of per-circuit scale-factor lists).
		"""
		measurement_settings = get_measurement_settings(
			observables if isinstance(observables, SparsePauliOp) else observables[0]
		)
		num_measurement_circuits = len(measurement_settings)
		zne_circuits_multiplier: int | list[int] = 1
		pauli_twirl_circuits_multiplier = 1

		if self._zne["enabled"]:
			if _is_nested_scale_factors(self._zne["scale_factors"]):
				zne_circuits_multiplier = [len(s) for s in cast(list[list[float]], self._zne["scale_factors"])]
			else:
				zne_circuits_multiplier = len(self._zne["scale_factors"])
		if self.backend._pauli_twirl["enabled"]:
			pauli_twirl_circuits_multiplier = (
				self.backend._pauli_twirl["num_twirls"] + 1
			)  # +1 for the original circuit without twirling

		if isinstance(zne_circuits_multiplier, list):
			# Per-circuit scale factors: each circuit contributes its own number of ZNE circuits.
			total_circuits = num_measurement_circuits * sum(zne_circuits_multiplier) * pauli_twirl_circuits_multiplier
		else:
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
				obs_circuits = [get_obs_subcircuits([circ], obs, ops) for circ, obs in zip(circuits, observables)]

		# if observables is a single SparsePauliOp and circuits is a list, we use the same observables for all circuits
		elif isinstance(observables, SparsePauliOp) and isinstance(circuits, list):
			obs_circuits = [get_obs_subcircuits([circ], observables, ops) for circ in circuits]

		# if observables is a single SparsePauliOp and circuits is a single QuantumCircuit, we just pair them
		elif isinstance(observables, SparsePauliOp) and isinstance(circuits, QuantumCircuit):
			obs_circuits = [get_obs_subcircuits([circuits], observables, ops)]
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
		# Scale factors actually realised per pair. Folding can only approximate the requested values, so
		# extrapolation uses these achieved x-values. They depend only on each circuit's foldable-gate
		# count (not the random seed), and equal the requested values exactly for odd integers. Both the
		# requested and achieved per-pair lists are attached to the returned job (the estimator's
		# scale_factors config is never mutated).
		pair_requested_scale_factors: list[list[float]] = []
		pair_scale_factors: list[list[float]] = []
		num_base_circuits = 0  # measurement-basis circuits before ZNE scale-factor expansion

		# scale_factors may be a single flat list (same for every circuit) or one list per circuit.
		zne_nested = self._zne["enabled"] and _is_nested_scale_factors(self._zne["scale_factors"])
		if zne_nested and len(self._zne["scale_factors"]) != num_pairs:
			raise ValueError(
				f"Per-circuit scale_factors has {len(self._zne['scale_factors'])} entr(y/ies) but {num_pairs} "
				"circuit/observable pair(s) were submitted; provide one scale-factor list per circuit."
			)

		for i, obs_circ_groups in enumerate(obs_circuits):
			obs_circs_list = [group[0] for group in obs_circ_groups]

			measurement_settings = get_measurement_settings(
				observables if isinstance(observables, SparsePauliOp) else observables[i]
			)
			pair_measurement_settings.append(measurement_settings)
			num_base_circuits += len(obs_circs_list)

			if self._zne["enabled"]:
				requested_scales = cast(
					list[float], self._zne["scale_factors"][i] if zne_nested else self._zne["scale_factors"]
				)
				pair_requested_scale_factors.append([float(s) for s in requested_scales])
				# All measurement-basis subcircuits in a pair share the same foldable-gate count (basis
				# rotations are single-qubit), so one achieved-scale list applies to the whole pair.
				pair_scale_factors.append(
					_achieved_scale_factors(
						obs_circs_list[0], requested_scales, self._zne["folding_method"], self._zne["fold_gates"]
					)
				)
				obs_circs_list = _get_zne_circuits(
					obs_circs_list,
					self._zne["fold_gates"],
					requested_scales,
					self._zne["folding_method"],
					self._zne["seed"],
				)

			pair_lengths.append(len(obs_circs_list))
			flat_circuits.extend(obs_circs_list)

		if self._zne["enabled"] and pair_scale_factors:
			# The estimator's scale_factors config is left untouched; the requested and achieved per-pair
			# values are attached to the job below. Warn whenever folding could not reach the request
			# exactly (per pair, since sublists may differ in length).
			if not all(np.allclose(pair_scale_factors[i], pair_requested_scale_factors[i]) for i in range(num_pairs)):
				warnings.warn(
					"Requested ZNE scale factors are not all exactly reachable by folding; extrapolation uses "
					f"the achieved values {pair_scale_factors}. Access them via the job's achieved_scale_factors()."
				)

		if self._zne["enabled"]:
			logger.info(
				"Flattened %d pair(s) into %d measurement-basis circuit(s), expanded to %d ZNE circuit(s); "
				"forwarding to backend with max_batch_size=%d",
				num_pairs,
				num_base_circuits,
				len(flat_circuits),
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
		zne_scale_factors_per_pair = pair_scale_factors
		zne_extrapolation_method = self._zne["extrapolation_method"]
		zne_extrapolation_degree = self._zne["extrapolation_degree"]

		# Freeze the full ZNE configuration for reporting on the returned job. Copied so later zne()
		# mutations do not alter what the job reports it ran with. The realised per-pair scale factors
		# are exposed separately via the job's requested/achieved_scale_factors() accessors.
		zne_options_snapshot: dict[str, Any] = {
			"enabled": self._zne["enabled"],
			"fold_gates": list(self._zne["fold_gates"]) if self._zne["fold_gates"] is not None else None,
			"folding_method": self._zne["folding_method"],
			"extrapolation_method": self._zne["extrapolation_method"],
			"extrapolation_degree": self._zne["extrapolation_degree"],
			"seed": self._zne["seed"],
		}

		def _compute() -> tuple[list, list, list]:
			"""Fetch results and compute per-pair (and ZNE-extrapolated) expectation values.

			Returns ``(expectation_values, raw_expectation_values, standard_errors)`` where the
			second element holds the pre-extrapolation ZNE values when ZNE is enabled (otherwise the
			same values), and the third holds the per-pair standard-error dicts (see
			:meth:`FiQCIEstimatorJob.standard_errors`).
			"""
			all_counts = job.result().get_counts()
			if not isinstance(all_counts, list):
				all_counts = [all_counts]

			expectation_values: list = []
			all_zne_expvs: list = []
			standard_errors: list = []
			offset = 0
			for i, length in enumerate(pair_lengths):
				counts = all_counts[offset : offset + length]
				offset += length

				obs = observables if isinstance(observables, SparsePauliOp) else observables[i]
				measurement_settings = pair_measurement_settings[i]
				zne_expvs = []

				if zne_enabled:
					split_counts = []
					num_circs_per_zne = len(measurement_settings)
					for j in range(0, len(counts), num_circs_per_zne):
						split_counts.append(counts[j : j + num_circs_per_zne])

					per_scale_sigmas = []
					for c in split_counts:
						zne_expvs.append(self._calculate_expectation_values(c, obs, measurement_settings))
						per_scale_sigmas.append(self._calculate_shot_errors(c, obs, measurement_settings))

					scales = zne_scale_factors_per_pair[i]
					if zne_extrapolation_method == "exponential":
						expvs, ext_err = exponential_extrapolation(zne_expvs, scales, sigmas=per_scale_sigmas)
					elif zne_extrapolation_method == "richardson":
						expvs, ext_err = richardson_extrapolation(zne_expvs, scales, sigmas=per_scale_sigmas)
					elif zne_extrapolation_method == "polynomial":
						expvs, ext_err = polynomial_extrapolation(
							zne_expvs, scales, degree=zne_extrapolation_degree, sigmas=per_scale_sigmas
						)
					elif zne_extrapolation_method == "linear":
						expvs, ext_err = polynomial_extrapolation(zne_expvs, scales, degree=1, sigmas=per_scale_sigmas)
					else:
						raise ValueError(f"Unsupported extrapolation method: {zne_extrapolation_method}")

					# Report the raw shot error at the unfolded (scale 1) point when available.
					scale_1_idx = scales.index(1) if 1 in scales else scales.index(min(scales))
					shot_err = per_scale_sigmas[scale_1_idx]
					standard_errors.append(
						{"shot_error": shot_err, "zne_extrapolation_error": ext_err, "total": ext_err}
					)
				else:
					expvs = self._calculate_expectation_values(counts, obs, measurement_settings)
					shot_err = self._calculate_shot_errors(counts, obs, measurement_settings)
					standard_errors.append({"shot_error": shot_err, "zne_extrapolation_error": None, "total": shot_err})

				expectation_values.append(expvs)
				if zne_enabled and len(zne_expvs) > 0:
					all_zne_expvs.append(zne_expvs)

			if zne_enabled and len(all_zne_expvs) > 0:
				return expectation_values, all_zne_expvs, standard_errors
			return expectation_values, expectation_values, standard_errors

		return FiQCIEstimatorJob(
			job, _compute, observables, pair_requested_scale_factors, pair_scale_factors, zne_options_snapshot
		)

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

	def _calculate_shot_errors(
		self,
		counts: dict[str, int] | list[dict[str, int]],
		obs: SparsePauliOp,
		measurement_settings: list[dict[int, str]],
	) -> list[float]:
		"""Per-Pauli-term shot-noise standard error, mirroring ``_calculate_expectation_values``.

		Each ⟨P⟩ is the sample mean of a ±1 random variable over ``N`` shots, so its standard error
		is ``sqrt((1 - ⟨P⟩²) / N)``. Uncovered terms (no measurement setting) report 0.0.
		"""
		if not isinstance(counts, list):
			counts = [counts]
		shot_errors = []
		for pauli in obs.paulis:
			pauli = cast(Pauli, pauli)
			obs_info = _get_observable_circuit_index(pauli, measurement_settings)
			if obs_info["circuit_index"] is not None:
				circuit_counts = counts[obs_info["circuit_index"]]
				total = sum(circuit_counts.values())
				if total == 0:
					shot_errors.append(0.0)
					continue
				exp_val = 0
				for bitstring, count in circuit_counts.items():
					parity = 1
					for idx in obs_info["obs_indices"]:
						if bitstring[idx] == "1":
							parity *= -1
					exp_val += parity * count
				exp_val /= total
				shot_errors.append(math.sqrt(max(0.0, 1.0 - exp_val**2) / total))
			else:
				shot_errors.append(0.0)  # No measurement setting covers this observable
		return shot_errors

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
		scale_factors: list[float] | list[list[float]] = [1, 3, 5],
		folding_method: str = "local",
		extrapolation_method: str = "exponential",
		extrapolation_degree: int | None = None,
		seed: int | None = None,
	):
		# TODO: More extrapolation methods, allow user-defined extrapolation functions
		"""Configure zero-noise extrapolation settings.

		Scale factors may be any real numbers >= 1. Non-odd-integer values (even integers, fractions)
		are approximated by partially folding a randomly-sampled subset of gates; ``seed`` makes that
		sampling reproducible. Extrapolation uses the achieved scale factors as the x-axis.

		``scale_factors`` may be either a single flat list applied to every submitted circuit, or a list
		of lists (one per submitted circuit) so each circuit uses its own scale factors — the number of
		lists must then match the number of circuit/observable pairs passed to :meth:`run`.
		"""
		if extrapolation_method not in ["exponential", "richardson", "polynomial", "linear"]:
			raise ValueError(f"Unsupported extrapolation method: {extrapolation_method}")
		if folding_method not in ["local", "global"]:
			raise ValueError(f"Unsupported folding method: {folding_method}")
		if folding_method == "global" and fold_gates is not None:
			warnings.warn("fold_gates is not applicable for global folding and will be ignored.")
			fold_gates = None
		if _is_nested_scale_factors(scale_factors):
			if not all(_valid_flat_scale_factors(sub) for sub in scale_factors):
				raise ValueError("Each per-circuit scale factor list must contain at least two real numbers >= 1.")
		else:
			if len(scale_factors) < 2:
				raise ValueError("At least two scale factors are required for extrapolation.")
			if not all(isinstance(s, (int, float)) and not isinstance(s, bool) and s >= 1 for s in scale_factors):
				raise ValueError("Scale factors must be real numbers >= 1.")
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
		self._zne["folding_method"] = folding_method
		self._zne["seed"] = seed
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

	def __init__(
		self,
		job,
		compute_fn: Callable[[], tuple[list, list, list]],
		observables,
		requested_scale_factors: list[list[float]] | None = None,
		achieved_scale_factors: list[list[float]] | None = None,
		zne_options: dict[str, Any] | None = None,
	) -> None:
		"""Initialize the estimator job.

		Args:
		    job: The underlying job that produced the results (BatchedJob or MitigatedJob).
		    compute_fn: Deferred callable returning
		        ``(expectation_values, raw_expectation_values, standard_errors)``.
		    observables: Observable(s) for which expectation values were calculated.
		    requested_scale_factors: ZNE scale factors requested for each circuit/observable pair (empty
		        when ZNE is disabled).
		    achieved_scale_factors: ZNE scale factors actually realised by folding for each pair — the
		        x-axis used for extrapolation (empty when ZNE is disabled).
		    zne_options: Frozen snapshot of the ZNE configuration used at submission (folding/extrapolation
		        settings), surfaced via :attr:`mitigator_options`. ``None`` when unknown.
		"""
		self.mitigated_job = job
		self._compute_fn = compute_fn
		self._observables = observables
		self._requested_scale_factors = requested_scale_factors if requested_scale_factors is not None else []
		self._achieved_scale_factors = achieved_scale_factors if achieved_scale_factors is not None else []
		self._zne_options = zne_options if zne_options is not None else {}
		self._expectation_values: list | None = None
		self._raw_expectation_values: list | None = None
		self._standard_errors: list | None = None
		self._computed = False
		self._lock = threading.Lock()

	def _ensure_computed(self) -> None:
		"""Run the deferred computation once (blocking on results), caching the outcome."""
		if self._computed:
			return
		with self._lock:
			if self._computed:
				return
			self._expectation_values, self._raw_expectation_values, self._standard_errors = self._compute_fn()
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

	def standard_errors(self, index: int | None = None) -> list[dict] | dict:
		"""Standard errors of the expectation values (computes lazily on first access).

		Mirrors the shape of :meth:`expectation_values`: one entry per circuit/observable pair, each
		a dict of per-Pauli-term standard errors with keys:

		- ``"shot_error"``: statistical SE of the raw measurement, ``sqrt((1 - ⟨P⟩²) / N)`` per term.
		  When ZNE is enabled this is taken at the unfolded (scale 1) point.
		- ``"zne_extrapolation_error"``: SE of the extrapolated value, the per-scale shot errors
		  propagated through the (linear) extrapolator; ``None`` when ZNE is disabled.
		- ``"total"``: SE of the value :meth:`expectation_values` actually returns — ``"shot_error"``
		  when ZNE is off, ``"zne_extrapolation_error"`` when ZNE is on. Not a quadrature sum, since
		  the extrapolation error already incorporates the shot noise.

		Args:
		    index: If given, return the dict for that pair; otherwise the list of all pairs.
		"""
		self._ensure_computed()
		assert self._standard_errors is not None
		if index is not None:
			return self._standard_errors[index]
		return self._standard_errors

	def requested_scale_factors(self, index: int | None = None) -> list[list[float]] | list[float]:
		"""ZNE scale factors requested for this run, as one list per circuit/observable pair.

		Returns all pairs' lists (a list of lists) when ``index`` is None, or a single pair's list when
		``index`` is given. Empty when ZNE was disabled for the run. See :meth:`achieved_scale_factors`
		for the values folding could actually realise.
		"""
		if index is not None:
			return self._requested_scale_factors[index]
		return self._requested_scale_factors

	def achieved_scale_factors(self, index: int | None = None) -> list[list[float]] | list[float]:
		"""ZNE scale factors actually realised by folding, as one list per circuit/observable pair.

		Folding can only approximate the requested scale factors, so these (the x-axis used for
		extrapolation) may differ from :meth:`requested_scale_factors`. Returns all pairs' lists (a list
		of lists) when ``index`` is None, or a single pair's list when ``index`` is given. Empty when ZNE
		was disabled for the run.
		"""
		if index is not None:
			return self._achieved_scale_factors[index]
		return self._achieved_scale_factors

	@property
	def mitigator_options(self) -> dict[str, Any]:
		"""Mitigation settings frozen at submission time for this estimator run.

		Merges the ZNE configuration with the underlying backend job's snapshot (``mitigation_level``,
		``rem``, ``dd``, ``pauli_twirl``), so the returned dict describes the full mitigation stack the
		run actually used. Unlike :attr:`FiQCIEstimator.mitigator_options` (which is live and mutable),
		this never changes after submission. Per-pair scale factors are available via
		:meth:`requested_scale_factors` / :meth:`achieved_scale_factors`.
		"""
		backend_options = getattr(self.mitigated_job, "mitigator_options", None) or {}
		return {"zne": self._zne_options, **backend_options}

	def __getattr__(self, name: str) -> Any:
		"""Delegate polling/attribute access (status, done, job_ids, …) to the underlying job."""
		return getattr(self.mitigated_job, name)
