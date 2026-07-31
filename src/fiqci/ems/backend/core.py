"""
FiQCI backend wrapper for seamless error mitigation.

FiQCIBackend wraps an IQM backend and applies error mitigation (e.g. M3 readout
error correction) to every circuit execution. It handles calibration, caching, and result
post-processing automatically, so users get mitigated results through the standard Qiskit
backend interface without additional code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict
from collections.abc import Iterable

from iqm.iqm_client import STANDARD_DD_STRATEGY
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from mthree.utils import final_measurement_mapping

from fiqci.ems.mitigators.rem import M3IQM
from fiqci.ems.mitigators.dd import DDGateSequenceEntry, build_dd_options
from fiqci.ems.transpiler_passes.pauli_twirl import get_twirled_circuits
from fiqci.ems.utils import probabilities_to_counts
from fiqci.ems.backend.jobs import MitigatedJob, BatchedJob, _UnsubmittedBatch
from fiqci.ems.backend.counts import (
	_key_layout,
	_reduce_counts,
	_expand_counts,
	_average_counts,
	_trim_result_to_groups,
)

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers import JobStatus, JobV1
from qiskit.result import Result

logger: logging.Logger = logging.getLogger(__name__)


class FiQCIBackend:
	"""FiQCI backend wrapper that applies error mitigation automatically.

	Mitigation levels:
		- 0: No error mitigation (raw results)
		- 1: Readout error mitigation using M3 (default)
		- 2: Level 1 + dynamical decoupling (DD)
		- 3: Level 2 + Pauli twirling

	Args:
		backend: An IQMBackendBase instance to wrap.
		mitigation_level: Error mitigation level (0-3). Default is 1.
		calibration_shots: Number of shots for calibration circuits. Default is 1000.
		calibration_file: Optional path to save/load M3 calibration data.
	"""

	def __init__(
		self,
		backend: IQMBackendBase,
		mitigation_level: int = 1,
		calibration_shots: int = 1000,
		calibration_file: str | None = None,
	) -> None:
		"""Initialize the FiQCI backend wrapper.

		Args:
			backend: An IQMBackendBase instance to wrap.
			mitigation_level: Error mitigation level (0-3). Default is 1.
			calibration_shots: Number of shots for calibration circuits. Default is 1000.
			calibration_file: Optional path to save/load M3 calibration data.

		Raises:
			ValueError: If mitigation_level is not in range 0-3.
		"""
		if mitigation_level not in range(4):
			raise ValueError(f"mitigation_level must be 0-3, got {mitigation_level}")

		self._backend = backend
		self._mitigation_level = mitigation_level
		self._raw_counts_cache: list[dict[str, int]] | None = None

		class REMSettings(TypedDict):
			enabled: bool
			calibration_shots: int
			calibration_file: str | None
			mitigator: M3IQM | None

		self._rem: REMSettings = {
			"enabled": False,
			"calibration_shots": calibration_shots,
			"calibration_file": calibration_file,
			"mitigator": None,
		}

		class DDSettings(TypedDict):
			enabled: bool
			gate_sequences: list[DDGateSequenceEntry]

		self._dd: DDSettings = {"enabled": False, "gate_sequences": []}

		class PauliTwirlSettings(TypedDict):
			enabled: bool
			num_twirls: int
			gates_to_twirl: Iterable[Gate] | None

		self._pauli_twirl: PauliTwirlSettings = {"enabled": False, "num_twirls": 0, "gates_to_twirl": None}

		# Initialize mitigator for level 1 (readout error mitigation using M3)
		if self._mitigation_level == 0:
			pass  # No mitigation, just pass through to backend
		elif self._mitigation_level == 1:
			self._init_rem(calibration_shots, calibration_file)
		elif self._mitigation_level == 2:
			self._init_rem(calibration_shots, calibration_file)
			self._init_dd()  # Use default DD settings
		elif self._mitigation_level == 3:
			self._init_rem(calibration_shots, calibration_file)
			self._init_dd()  # Use default DD settings
			self.init_pauli_twirl(enabled=True)
		else:
			raise ValueError(f"mitigation_level must be 0-3, got {mitigation_level}")

	@property
	def backend(self) -> IQMBackendBase:
		"""Get the underlying backend."""
		return self._backend

	@property
	def mitigation_level(self) -> int:
		"""Get the current mitigation level."""
		return self._mitigation_level

	@property
	def raw_counts(self) -> list[dict[str, int]] | None:
		"""Get the raw (unmitigated) counts from the most recent run.

		The list is flat with one entry per circuit submitted to the backend, in submission order.
		With Pauli twirling enabled this is the per-twirl counts before any averaging, so the length
		is ``num_input_circuits * (num_twirls + 1)`` and entries for the same input circuit are
		contiguous.

		Note:
			Because post-processing is now lazy, this is populated only after the run's mitigated
			``result()`` has been retrieved at least once. It returns ``None`` until then.

		Returns:
			List of raw count dictionaries, or None if no run's result has been computed yet.
		"""
		return self._raw_counts_cache

	@property
	def mitigator_options(self) -> dict[str, Any]:
		"""
		Get current mitigator settings.

		Returns:
			A dictionary of current mitigator settings and their values.
		"""
		return {"rem": self._rem, "dd": self._dd, "pauli_twirl": self._pauli_twirl}

	def _snapshot_mitigator_options(self) -> dict[str, Any]:
		"""Freeze the current mitigator settings for attachment to a submitted job.

		Unlike :attr:`mitigator_options` (which reflects the backend's live, mutable settings),
		the returned dict is a copy taken at submission time, so a job can faithfully report the
		configuration it actually ran with even if the backend's settings are later mutated. The
		live ``M3IQM`` mitigator is intentionally omitted.
		"""
		return {
			"mitigation_level": self._mitigation_level,
			"rem": {
				"enabled": self._rem["enabled"],
				"calibration_shots": self._rem["calibration_shots"],
				"calibration_file": self._rem["calibration_file"],
			},
			"dd": {"enabled": self._dd["enabled"], "gate_sequences": list(self._dd["gate_sequences"])},
			"pauli_twirl": dict(self._pauli_twirl),
		}

	def _snapshot_mitigator_options(self) -> dict[str, Any]:
		"""Freeze the current mitigator settings for attachment to a submitted job.

		Unlike :attr:`mitigator_options` (which reflects the backend's live, mutable settings),
		the returned dict is a copy taken at submission time, so a job can faithfully report the
		configuration it actually ran with even if the backend's settings are later mutated. The
		live ``M3IQM`` mitigator is intentionally omitted.
		"""
		return {
			"mitigation_level": self._mitigation_level,
			"rem": {
				"enabled": self._rem["enabled"],
				"calibration_shots": self._rem["calibration_shots"],
				"calibration_file": self._rem["calibration_file"],
			},
			"dd": {"enabled": self._dd["enabled"], "gate_sequences": list(self._dd["gate_sequences"])},
			"pauli_twirl": dict(self._pauli_twirl),
		}

	def total_circuits_generated(self, num_base_circuits: int, detailed: bool = False) -> int | dict[str, int]:
		"""Calculate total circuits generated for a given number of base circuits and observables."""
		pauli_twirl_circuits_multiplier = 1

		if self._pauli_twirl["enabled"]:
			pauli_twirl_circuits_multiplier = (
				self._pauli_twirl["num_twirls"] + 1
			)  # +1 for the original circuit without twirling

		total_circuits = num_base_circuits * pauli_twirl_circuits_multiplier

		if detailed:
			print(
				f"The total number of circuits is {total_circuits}, calculated as follows: base circuits ({num_base_circuits}) * Pauli twirl multiplier ({pauli_twirl_circuits_multiplier}). This does not include circuits ran to calibrate readout error mitigation (REM)."
			)
			return {
				"base_circuits": num_base_circuits,
				"pauli_twirl_multiplier": pauli_twirl_circuits_multiplier,
				"total_circuits": total_circuits,
			}
		else:
			return total_circuits

	def init_pauli_twirl(
		self, enabled: bool, num_twirls: int = 10, gates_to_twirl: Iterable[Gate] | None = None
	) -> None:
		"""
		Initialize Pauli twirling settings.

		Args:
			enabled: Whether Pauli twirling is enabled.
			num_twirls: Number of twirled circuits to generate per input circuit.
			gates_to_twirl: Optional list of gates to twirl, if None, all two-qubit basis gates will be twirled.
		"""

		self._pauli_twirl["enabled"] = enabled
		self._pauli_twirl["num_twirls"] = num_twirls
		self._pauli_twirl["gates_to_twirl"] = gates_to_twirl

	def _init_dd(self, gate_sequences: list[DDGateSequenceEntry] | None = None) -> None:
		"""Initialize dynamical decoupling settings.

		Args:
			gate_sequences: List of (threshold_length, sequence, strategy) tuples defining DD behavior.
				See build_dd_options for details on each field.
		"""
		if gate_sequences is None or len(gate_sequences) == 0:
			gate_sequences = STANDARD_DD_STRATEGY.gate_sequences
		else:
			# Validate gate_sequences format
			valid_gate_sequences = []
			for entry in gate_sequences:
				if not isinstance(entry, (list, tuple)) or len(entry) != 3:
					raise ValueError(
						f"Each gate sequence entry must be a tuple of (threshold_length, sequence, strategy), got {entry}"
					)
				threshold_length, sequence, strategy = entry
				if strategy is not None and strategy not in ["asap", "alap", "center"]:
					raise ValueError(f"Invalid strategy: {strategy} in entry {entry}")
				if threshold_length is not None and not isinstance(threshold_length, int):
					raise ValueError(
						f"threshold_length must be an integer or None, got {threshold_length} in entry {entry}"
					)
				if sequence is not None and not isinstance(sequence, (str, list)):
					raise ValueError(
						f"sequence must be a string, list of tuples, or None, got {sequence} in entry {entry}"
					)

				if threshold_length is None and sequence is not None:
					threshold_length = len(sequence)
				elif threshold_length is None:
					threshold_length = 2

				if strategy is None:
					strategy = "asap"

				if sequence is None:
					sequence = "XY"

				valid_gate_sequences.append((threshold_length, sequence, strategy))
			gate_sequences = valid_gate_sequences

		self._dd["enabled"] = True
		self._dd["gate_sequences"] = gate_sequences

	def _init_rem(self, calibration_shots: int = 1000, calibration_file: str | None = None) -> None:
		"""Initialize readout error mitigation (M3).

		Args:
			calibration_shots: Number of shots for calibration circuits. Default is 1000.
			calibration_file: Path to save/load calibration data. Default is None.
		"""
		self._rem["enabled"] = True
		self._rem["calibration_file"] = calibration_file
		self._rem["mitigator"] = M3IQM(self._backend)

		# Try to load calibration from file if specified
		# Do not load if calibration_shots has changed since last calibration, as the calibration data would be invalid
		if calibration_file and (calibration_shots == self._rem["calibration_shots"]):
			cal_path = Path(calibration_file)
			if cal_path.exists():
				try:
					self._rem["mitigator"].cals_from_file(calibration_file, validate_calibration_set=True)
					logger.info("Loaded existing M3 calibration from %s", calibration_file)
				except Exception as e:
					error_msg = str(e)
					if "Calibration set ID mismatch" in error_msg:
						logger.error(
							"Calibration set ID mismatch: %s. Backend configuration has changed. "
							"Will recalibrate on first run.",
							error_msg,
						)
					else:
						logger.warning(
							"Could not load calibration from %s: %s. Will calibrate on first run.", calibration_file, e
						)
			else:
				self._rem["calibration_shots"] = calibration_shots
				logger.info(
					"Calibration file %s does not exist yet. Will calibrate and save on first run.", calibration_file
				)
		else:
			self._rem["calibration_shots"] = calibration_shots
			logger.info("Calibration shots set to %d. Will calibrate on first run.", calibration_shots)

	def dd(self, enabled: bool = True, gate_sequences: list[DDGateSequenceEntry] | None = None) -> None:
		"""
		Set dynamical decoupling settings for the backend.

		Args:
			enabled: Whether to enable dynamical decoupling.
			gate_sequences: List of (threshold_length, sequence, strategy) tuples defining DD behavior.
				See build_dd_options for details on each field.
		"""
		if enabled:
			self._init_dd(gate_sequences)
		else:
			self._dd["enabled"] = False

	def rem(self, enabled: bool = True, calibration_shots: int = 1000, calibration_file: str | None = None) -> None:
		"""
		Set readout error mitigation settings for the backend.

		Args:
			enabled: Whether to enable readout error mitigation.
			calibration_shots: Number of shots to use for calibration circuits (default: 1000).
			calibration_file: Optional calibration file to use for readout error mitigation.
		"""
		if not enabled:
			self._rem["enabled"] = False
			self._rem["mitigator"] = None
			return

		settings_changed = (
			calibration_shots != self._rem["calibration_shots"] or calibration_file != self._rem["calibration_file"]
		)
		if not self._rem["enabled"] or settings_changed:
			self._init_rem(calibration_shots, calibration_file)

	def pauli_twirl(self, enabled: bool, num_twirls: int = 10, gates_to_twirl: list | None = None) -> None:
		"""
		Set Pauli twirling settings for the backend.

		Args:
			enabled: Whether to enable Pauli twirling.
			num_twirls: Number of twirled circuits to generate per input circuit (default: 10).
			gates_to_twirl: Optional list of gates to twirl, if None, all two-qubit basis gates will be twirled.
		"""
		self.init_pauli_twirl(enabled, num_twirls, gates_to_twirl)

	def run(
		self,
		circuits: QuantumCircuit | list[QuantumCircuit],
		shots: int = 1024,
		max_batch_size: int = 100,
		**kwargs: Any,
	) -> MitigatedJob | BatchedJob:
		"""Submit quantum circuits and return a lazy job handle immediately.

		Circuits are submitted to the backend (in batches), and a handle is returned right away
		without waiting for results: the per-batch ``job_id()``s and ``status()`` are available
		immediately, and any configured error mitigation is deferred until the handle's
		``result()`` is first called.

		Args:
			circuits: Single quantum circuit or list of circuits to execute.
			shots: Number of shots. Default is 1024.
			max_batch_size: Maximum number of circuits per backend job. The (post-twirl) circuit list is
				flattened and split into batches of this size; the resulting jobs are wrapped so that the
				returned handle's ``result()`` exposes a single combined Result indexed in submission order
				(default: 100).
			**kwargs: Additional keyword arguments passed to backend.run().

		Returns:
			A :class:`BatchedJob` handle (level 0) or a :class:`MitigatedJob` view (level 1+). In
			both cases mitigation/combination is computed lazily on the first ``result()`` call.

		Raises:
			ValueError: If circuits is empty or invalid.
		"""

		# Normalize to list
		circuits_list = circuits if isinstance(circuits, list) else [circuits]

		if not circuits_list:
			raise ValueError("No circuits provided")

		input_circuit_count = len(circuits_list)
		logger.info(
			"FiQCIBackend.run: %d input circuit(s); mitigation_level=%d (REM=%s, DD=%s, pauli_twirl=%s)",
			input_circuit_count,
			self._mitigation_level,
			"on" if self._rem["enabled"] else "off",
			"on" if self._dd["enabled"] else "off",
			"on" if self._pauli_twirl["enabled"] else "off",
		)

		# Freeze the mitigation config now so every returned handle reports the exact settings it
		# ran with, unaffected by any later mutation of the backend's settings.
		options_snapshot = self._snapshot_mitigator_options()

		# If Pauli Twirling is enabled, replace circuits with twirled versions
		twirl_group_size = 0
		if self._pauli_twirl["enabled"]:
			circuits_list = get_twirled_circuits(
				circuits_list,
				num_twirls=self._pauli_twirl["num_twirls"],
				gates_to_twirl=self._pauli_twirl["gates_to_twirl"],
				backend=self._backend,
			)
			twirl_group_size = self._pauli_twirl["num_twirls"] + 1
			logger.info(
				"Pauli twirling expanded %d -> %d circuits (group size %d)",
				input_circuit_count,
				len(circuits_list),
				twirl_group_size,
			)

		# Build run kwargs (DD options if enabled)
		run_kwargs = dict(kwargs)
		if self._dd["enabled"]:
			run_kwargs["circuit_compilation_options"] = build_dd_options(self._dd["gate_sequences"])

		# Submit circuits in batches of at most max_batch_size, preserving submission order so the
		# combined result indices match circuits_list.
		num_batches = (len(circuits_list) + max_batch_size - 1) // max_batch_size
		logger.info(
			"Submitting %d circuit(s) to backend in %d batch(es) of up to %d",
			len(circuits_list),
			num_batches,
			max_batch_size,
		)
		# Submission is not atomic: a later batch may be rejected after earlier ones were accepted.
		# Rather than throwing (which would lose the already-submitted jobs), stop submitting on the
		# first rejection and still return a handle covering every intended batch. Submitted batches
		# hold real jobs; the rejected batch and the ones skipped afterwards hold placeholders that
		# report a terminal ERROR/CANCELLED status. A warning is logged so the failure is visible.
		batch_jobs: list[JobV1] = []
		batch_ranges: list[tuple[int, int]] = []
		submission_failed = False
		for batch_start in range(0, len(circuits_list), max_batch_size):
			batch = circuits_list[batch_start : batch_start + max_batch_size]
			batch_range = (batch_start, batch_start + len(batch))

			if submission_failed:
				# An earlier batch was rejected; do not submit any further batches.
				# _UnsubmittedBatch is a duck-typed JobV1 stand-in (no real job_id, result() raises).
				batch_jobs.append(_UnsubmittedBatch(batch_range, JobStatus.CANCELLED))  # type: ignore[bad-argument-type]
				batch_ranges.append(batch_range)
				continue

			try:
				batch_job = self._backend.run(batch, shots=shots, **run_kwargs)
				assert batch_job is not None, "Backend returned None job"
			except Exception as exc:
				submission_failed = True
				logger.warning(
					"Backend rejected batch at circuit indices %d-%d: %s. Stopping submission; "
					"returning a job handle for the %d batch(es) submitted so far (remaining batches "
					"are marked cancelled). Inspect status()/statuses() and partial_results().",
					batch_start,
					batch_start + len(batch) - 1,
					exc,
					len(batch_jobs),
				)
				batch_jobs.append(_UnsubmittedBatch(batch_range, JobStatus.ERROR, error=str(exc)))  # type: ignore[bad-argument-type]
				batch_ranges.append(batch_range)
				continue

			logger.info(
				"Submitted batch of %d circuit(s) (indices %d-%d) to backend, got job ID %s",
				len(batch),
				batch_start,
				batch_start + len(batch) - 1,
				batch_job.job_id(),
			)

			batch_jobs.append(batch_job)
			batch_ranges.append(batch_range)

		# No REM: return a lazy handle immediately. With twirling, the averaging is deferred to the
		# handle's result() via a post-process callback (no blocking here).
		if not self._rem["enabled"]:
			if twirl_group_size == 0:
				return BatchedJob(batch_jobs, batch_ranges, mitigator_options=options_snapshot)

			# Snapshot mitigation metadata at submission time so deferred post-processing is not
			# affected by later changes to the backend's settings.
			mitigation_level = self._mitigation_level
			calibration_shots = self._rem["calibration_shots"]

			def _twirl_post(result: Result) -> Result:
				all_counts = result.get_counts()
				if not isinstance(all_counts, list):
					all_counts = [all_counts]
				raw_counts_list: list[dict[str, int]] = list(all_counts)
				self._raw_counts_cache = raw_counts_list

				averaged_counts_list = [
					_average_counts(raw_counts_list[i : i + twirl_group_size])
					for i in range(0, len(raw_counts_list), twirl_group_size)
				]

				num_groups = len(circuits_list) // twirl_group_size
				result_to_use = _trim_result_to_groups(result, num_groups)
				return self._create_mitigated_result(
					result_to_use,
					averaged_counts_list,
					raw_counts_list,
					twirl_group_size=twirl_group_size,
					mitigation_level=mitigation_level,
					calibration_shots=calibration_shots,
				)

			return MitigatedJob(
				BatchedJob(batch_jobs, batch_ranges, post_process=_twirl_post, mitigator_options=options_snapshot)
			)

		# REM enabled: run with M3 mitigation (deferred to the handle's result()).
		return self._run_with_m3_mitigation(
			batch_jobs,
			batch_ranges,
			circuits_list,
			shots,
			twirl_group_size=twirl_group_size,
			max_batch_size=max_batch_size,
			mitigator_options=options_snapshot,
		)

	def _run_with_m3_mitigation(
		self,
		batch_jobs: list[JobV1],
		batch_ranges: list[tuple[int, int]],
		circuits: list[QuantumCircuit],
		shots: int,
		twirl_group_size: int = 0,
		max_batch_size: int = 100,
		mitigator_options: dict[str, Any] | None = None,
	) -> MitigatedJob:
		"""Build a lazy M3-mitigated handle for already-submitted batch jobs.

		Calibration is kicked off eagerly here (``cals_from_system`` is non-blocking and mthree runs
		it in a background thread) so it proceeds in parallel with the circuit jobs. The blocking
		result fetch and the per-circuit M3 correction are deferred to a ``post_process`` callback
		that runs the first time the returned handle's ``result()`` is called.

		Args:
			batch_jobs: Per-batch jobs already submitted to the backend, in submission order.
			batch_ranges: ``(start, end_exclusive)`` circuit index range for each batch.
			circuits: List of quantum circuits that were executed.
			shots: Number of measurement shots.
			twirl_group_size: Size of each twirl group (num_twirls + 1), or 0 if no twirling.
			max_batch_size: Maximum circuits per calibration job (default: 100).
			mitigator_options: Frozen snapshot of the mitigation settings used at submission, attached
				to the returned handle so it can report the configuration it ran with.

		Returns:
			A MitigatedJob view whose ``result()`` lazily applies M3 mitigation.
		"""
		# Get qubit mappings for each circuit
		qubits_list = [final_measurement_mapping(circuit) for circuit in circuits]

		# Calibrate M3 mitigator if not already done. This is non-blocking (async_cal), so it can
		# start now and run alongside the circuit jobs; apply_correction (in the callback below)
		# waits for it to finish if necessary.
		if self._rem["mitigator"] is not None and self._rem["mitigator"].single_qubit_cals is None:
			all_qubits: set[int] = set()
			for qubit_mapping in qubits_list:
				all_qubits.update(qubit_mapping.values())  # type: ignore[arg-type]
			calibration_qubits = sorted(all_qubits)

			if self._rem["calibration_file"]:
				logger.info(
					"Calibrating M3 mitigator for qubits %s with %d shots and saving to %s",
					calibration_qubits,
					self._rem["calibration_shots"],
					self._rem["calibration_file"],
				)
			else:
				logger.info(
					"Calibrating M3 mitigator for qubits %s with %d shots",
					calibration_qubits,
					self._rem["calibration_shots"],
				)

			assert self._rem["mitigator"] is not None, "Mitigator should be initialized for level 1"
			self._rem["mitigator"].cals_from_system(
				calibration_qubits,
				shots=self._rem["calibration_shots"],
				cals_file=self._rem["calibration_file"],
				max_batch_size=max_batch_size,
			)

		# Snapshot the mitigator and metadata at submission time. The closure runs later (on the
		# first result() call), so binding these now keeps the deferred correction consistent with
		# the configuration used at submission, even if the user mutates settings (e.g. disables
		# REM, which would otherwise clear self._rem["mitigator"]) in the meantime.
		mitigator = self._rem["mitigator"]
		assert mitigator is not None, "Mitigator should be initialized for level 1"
		mitigation_level = self._mitigation_level
		calibration_shots = self._rem["calibration_shots"]

		def _m3_post(result: Result) -> Result:
			# Apply M3 correction to each circuit's results
			raw_counts_list: list[dict[str, int]] = []
			mitigated_counts_list: list[dict[str, int]] = []

			for idx in range(len(circuits)):
				raw_counts = result.get_counts(idx)
				assert isinstance(raw_counts, dict), f"Expected dict from get_counts({idx}), got {type(raw_counts)}"
				raw_counts_list.append(raw_counts)
				qubits = qubits_list[idx]
				assert isinstance(qubits, dict), f"Expected dict mapping for circuit {idx}, got {type(qubits)}"

				# M3 expects a spaceless bitstring with exactly one bit per measured qubit. Result
				# keys instead carry a bit for every classical bit (with spaces between registers and
				# zero-filled bits for unmeasured registers), so reduce them to the measured bits
				# before correction and restore the original structure afterwards.
				layout = _key_layout(raw_counts, qubits)
				counts_for_correction = _reduce_counts(raw_counts, layout)

				quasi_dist = mitigator.apply_correction(counts_for_correction, qubits)
				mitigated_probs = quasi_dist.nearest_probability_distribution()  # type: ignore[union-attr]
				mitigated_counts = probabilities_to_counts(mitigated_probs, shots)
				mitigated_counts_list.append(_expand_counts(mitigated_counts[0], layout))

			result_to_use = result
			# If Pauli twirling, average mitigated counts across groups (raw counts stay flat)
			if twirl_group_size:
				averaged_mitigated: list[dict[str, int]] = []
				for i in range(0, len(mitigated_counts_list), twirl_group_size):
					averaged_mitigated.append(_average_counts(mitigated_counts_list[i : i + twirl_group_size]))
				mitigated_counts_list = averaged_mitigated
				num_groups = len(circuits) // twirl_group_size
				result_to_use = _trim_result_to_groups(result, num_groups)

			self._raw_counts_cache = raw_counts_list
			return self._create_mitigated_result(
				result_to_use,
				mitigated_counts_list,
				raw_counts_list,
				twirl_group_size=twirl_group_size or 1,
				mitigation_level=mitigation_level,
				calibration_shots=calibration_shots,
			)

		return MitigatedJob(
			BatchedJob(batch_jobs, batch_ranges, post_process=_m3_post, mitigator_options=mitigator_options)
		)

	def _create_mitigated_result(
		self,
		original_result: Result,
		mitigated_counts: list[dict[str, int]],
		raw_counts: list[dict[str, int]],
		twirl_group_size: int = 1,
		mitigation_level: int | None = None,
		calibration_shots: int | None = None,
	) -> Result:
		"""Create a new Result object with mitigated counts and metadata.

		Args:
			original_result: Original result from backend, already trimmed to one entry per group.
			mitigated_counts: List of mitigated count dictionaries, one per group.
			raw_counts: Flat list of raw (unmitigated) count dictionaries, one per submitted circuit.
				When ``twirl_group_size > 1`` this contains all per-twirl counts in submission order.
			twirl_group_size: Number of submitted circuits per group (``num_twirls + 1``), or 1 when
				twirling is disabled. Used to slice ``raw_counts`` into the per-group entries that get
				attached to each result header.
			mitigation_level: Mitigation level to record in the result metadata. Defaults to the
				backend's current level; callers running deferred post-processing pass the value
				snapshotted at submission time so the metadata stays consistent.
			calibration_shots: Calibration shots to record in the result metadata. Defaults to the
				backend's current setting; snapshotted by deferred callers for the same reason.

		Returns:
			New Result object with mitigated data and FiQCI EMS metadata. Each header's
			``fiqci_ems["raw_counts"]`` is a single dict when ``twirl_group_size == 1`` and a list of
			per-twirl dicts when ``twirl_group_size > 1``.
		"""
		if mitigation_level is None:
			mitigation_level = self._mitigation_level
		if calibration_shots is None:
			calibration_shots = self._rem["calibration_shots"]

		# Get original result data
		results_data = original_result.to_dict()

		# Update counts and add metadata in each experiment result
		results_list = results_data.get("results")
		if results_list is not None:
			for idx, counts in enumerate(mitigated_counts):
				if idx < len(results_list):
					# Update counts with mitigated values
					results_list[idx]["data"]["counts"] = counts  # type: ignore[index]

					# Add FiQCI EMS metadata to header
					if "header" not in results_list[idx]:
						results_list[idx]["header"] = {}  # type: ignore[index]

					if twirl_group_size > 1:
						raw_for_group: dict[str, int] | list[dict[str, int]] = raw_counts[
							idx * twirl_group_size : (idx + 1) * twirl_group_size
						]
					else:
						raw_for_group = raw_counts[idx]

					results_list[idx]["header"]["fiqci_ems"] = {  # type: ignore[index]
						"mitigation_level": mitigation_level,
						"mitigation_method": "M3" if mitigation_level == 1 else None,
						"calibration_shots": calibration_shots if mitigation_level == 1 else None,
						"raw_counts": raw_for_group,
					}

		# Create new result from modified data
		from qiskit.result import Result as QiskitResult

		return QiskitResult.from_dict(results_data)

	def __getattr__(self, name: str) -> Any:
		"""Delegate attribute access to underlying backend object."""
		return getattr(self._backend, name)
