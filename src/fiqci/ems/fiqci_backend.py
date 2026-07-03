"""
FiQCI backend wrapper for seamless error mitigation.

FiQCIBackend wraps an IQM backend and applies error mitigation (e.g. M3 readout
error correction) to every circuit execution. It handles calibration, caching, and result
post-processing automatically, so users get mitigated results through the standard Qiskit
backend interface without additional code.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple, TypedDict
from collections.abc import Callable, Iterable

from iqm.iqm_client import STANDARD_DD_STRATEGY
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from mthree.utils import final_measurement_mapping

from fiqci.ems.mitigators.rem import M3IQM
from fiqci.ems.mitigators.dd import DDGateSequenceEntry, build_dd_options
from fiqci.ems.transpiler_passes.pauli_twirl import get_twirled_circuits
from fiqci.ems.utils import probabilities_to_counts

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers import JobStatus, JobV1
from qiskit.result import Result

logger: logging.Logger = logging.getLogger(__name__)

# Job statuses that indicate a batch will not change further.
_TERMINAL_STATUSES = frozenset({JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED})


class BatchFailedError(RuntimeError):
	"""Raised when one or more batches of a submitted job fail.

	The message identifies the failing batch by its index and the range of original circuit
	indices it covered, so callers can map the failure back to their input.
	"""


class _UnsubmittedBatch:
	"""Stand-in for a batch that was not successfully submitted to the backend.

	Submission is not atomic: when a batch is rejected mid-stream, ``run()`` stops submitting and
	still returns a handle covering every intended batch. The rejected batch and any batches skipped
	after it are represented by this placeholder so the handle exposes a uniform, index-aligned
	per-batch view. It reports a terminal status (``ERROR`` for the batch that failed to submit,
	``CANCELLED`` for batches skipped afterwards), has no backend job id, and raises when its result
	is requested.
	"""

	def __init__(self, circuit_range: tuple[int, int], status: JobStatus, error: str | None = None) -> None:
		self._range = circuit_range
		self._status = status
		self._error = error

	def job_id(self) -> None:
		"""Unsubmitted batches have no backend job id."""
		return None

	def status(self) -> JobStatus:
		return self._status

	def result(self, timeout: float | None = None) -> Result:
		start, end = self._range
		detail = f": {self._error}" if self._error else ""
		raise BatchFailedError(f"Batch covering circuits {start}-{end - 1} was not submitted{detail}")


class PartialBatch(NamedTuple):
	"""Snapshot of a single batch within a multi-batch job.

	Attributes:
		index: Position of the batch in submission order.
		circuit_range: ``(start, end_exclusive)`` original circuit indices covered by the batch.
		status: Current :class:`~qiskit.providers.JobStatus` of the batch job.
		job_id: Backend job id of the batch.
		result: The batch's :class:`~qiskit.result.Result` if it has completed, else ``None``.
	"""

	index: int  # type: ignore[bad-override]  # NamedTuple field shadows the read-write tuple.index method
	circuit_range: tuple[int, int]
	status: JobStatus
	job_id: str
	result: Result | None


class _KeyLayout(NamedTuple):
	"""Structure of result count keys, used to reduce them for M3 and restore them afterwards.

	Attributes:
		num_clbits: Total number of classical bits (length of a key with spaces removed).
		measured: Classical bit indices that are actually measured; all others are always zero.
		space_positions: Indices (in the original spaced key) at which spaces occur.
	"""

	num_clbits: int
	measured: frozenset[int]
	space_positions: tuple[int, ...]


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
					self._average_counts(raw_counts_list[i : i + twirl_group_size])
					for i in range(0, len(raw_counts_list), twirl_group_size)
				]

				num_groups = len(circuits_list) // twirl_group_size
				result_to_use = self._trim_result_to_groups(result, num_groups)
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

	@staticmethod
	def _key_layout(counts: dict[str, int], mapping: dict[int, int]) -> _KeyLayout:
		"""Describe the structure of count dictionary keys for M3 correction.

		M3 expects a bitstring with exactly one bit per measured qubit and no spaces. Result
		keys, however, contain a bit for every classical bit in the circuit, including spaces
		between classical registers (e.g. ``"001 00"``) and zero-filled bits for classical
		registers that are never measured. This computes the information needed to strip those
		extra bits before correction and restore them afterwards.

		Bitstrings are little-endian on the classical bit index: the character at spaceless
		index ``i`` corresponds to classical bit ``num_clbits - 1 - i`` (Qiskit MSB-left). A
		classical bit is "measured" iff it is a key of ``mapping`` (from
		``final_measurement_mapping``); every other bit is always zero in the results.

		Args:
			counts: Count dictionary whose keys may contain spaces and unmeasured bits.
			mapping: ``{classical_bit: qubit}`` mapping for the circuit's final measurements.

		Returns:
			A :class:`_KeyLayout` describing the total bit count, the measured bits, and the
			space positions of the original keys.
		"""
		sample_key = next(iter(counts))
		space_positions = tuple(i for i, char in enumerate(sample_key) if char == " ")
		num_clbits = len(sample_key) - len(space_positions)
		return _KeyLayout(num_clbits=num_clbits, measured=frozenset(mapping), space_positions=space_positions)

	@staticmethod
	def _reduce_counts(counts: dict[str, int], layout: _KeyLayout) -> dict[str, int]:
		"""Strip spaces and unmeasured (always-zero) bits from count keys for M3 correction.

		Keys differing only in unmeasured bits collapse to the same reduced key (those bits are
		always zero), so their values are summed; in practice no collision occurs.

		Args:
			counts: Count dictionary with full, spaced keys.
			layout: Layout describing the key structure, from :meth:`_key_layout`.

		Returns:
			Count dictionary keyed by measured bits only, with no spaces.
		"""
		reduced: dict[str, int] = {}
		for key, value in counts.items():
			spaceless = key.replace(" ", "")
			measured_bits = "".join(
				char for i, char in enumerate(spaceless) if (layout.num_clbits - 1 - i) in layout.measured
			)
			reduced[measured_bits] = reduced.get(measured_bits, 0) + value
		return reduced

	@staticmethod
	def _expand_counts(counts: dict[str, int], layout: _KeyLayout) -> dict[str, int]:
		"""Restore unmeasured zero bits and register spaces to reduced count keys.

		Inverse of :meth:`_reduce_counts`: each measured bit is placed back at its original
		position, unmeasured bits are filled with ``"0"``, and spaces are reinserted.

		Args:
			counts: Count dictionary keyed by measured bits only (e.g. M3 output).
			layout: Layout describing the key structure, from :meth:`_key_layout`.

		Returns:
			Count dictionary with full, spaced keys matching the original structure.
		"""
		expanded: dict[str, int] = {}
		for key, value in counts.items():
			measured_iter = iter(key)
			chars = [
				next(measured_iter) if (layout.num_clbits - 1 - i) in layout.measured else "0"
				for i in range(layout.num_clbits)
			]
			full = "".join(chars)
			for pos in layout.space_positions:
				full = full[:pos] + " " + full[pos:]
			expanded[full] = expanded.get(full, 0) + value
		return expanded

	@staticmethod
	def _average_counts(counts_list: list[dict[str, int]]) -> dict[str, int]:
		"""Average multiple count dictionaries.

		Args:
			counts_list: List of count dictionaries to average.

		Returns:
			Averaged count dictionary with integer values.
		"""
		if len(counts_list) == 1:
			return counts_list[0]

		totals: dict[str, float] = {}
		for counts in counts_list:
			for key, value in counts.items():
				totals[key] = totals.get(key, 0.0) + value
		n = len(counts_list)
		return {key: round(value / n) for key, value in totals.items()}

	@staticmethod
	def _trim_result_to_groups(result: Result, num_groups: int) -> Result:
		"""Trim a flat (per-twirl) Result down to one entry per twirl group.

		The flat result list is ``[g0_orig, g0_tw1..g0_twN, g1_orig, g1_tw1.., ...]`` of length
		``num_groups * twirl_group_size``. We must keep the *representative* entry of each group
		(its original circuit, at stride ``twirl_group_size``) rather than the first ``num_groups``
		flat entries: a plain ``[:num_groups]`` slice keeps mostly group 0's twirl copies, so every
		kept entry carries group 0's header. Since ``Result.get_counts()`` reconstructs each
		bitstring's structure (creg sizes, memory slots) from the per-entry header, wrong headers
		yield misaligned, wrongly-structured counts even though ``_create_mitigated_result`` later
		overwrites ``data["counts"]``. Keeping ``results_list[i * stride]`` makes each kept entry's
		header match input circuit ``i``.

		Args:
			result: Original Result object (flat, one entry per submitted twirl circuit).
			num_groups: Number of twirl groups (== number of input circuits).

		Returns:
			New Result object with one representative entry per group, in input-circuit order.
		"""
		from qiskit.result import Result as QiskitResult

		result_data = result.to_dict()
		results_list = result_data.get("results")
		if results_list is not None and num_groups:
			stride = len(results_list) // num_groups  # == twirl_group_size
			result_data["results"] = [results_list[i * stride] for i in range(num_groups)]
		return QiskitResult.from_dict(result_data)

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
				layout = self._key_layout(raw_counts, qubits)
				counts_for_correction = self._reduce_counts(raw_counts, layout)

				quasi_dist = mitigator.apply_correction(counts_for_correction, qubits)
				mitigated_probs = quasi_dist.nearest_probability_distribution()  # type: ignore[union-attr]
				mitigated_counts = probabilities_to_counts(mitigated_probs, shots)
				mitigated_counts_list.append(self._expand_counts(mitigated_counts[0], layout))

			result_to_use = result
			# If Pauli twirling, average mitigated counts across groups (raw counts stay flat)
			if twirl_group_size:
				averaged_mitigated: list[dict[str, int]] = []
				for i in range(0, len(mitigated_counts_list), twirl_group_size):
					averaged_mitigated.append(self._average_counts(mitigated_counts_list[i : i + twirl_group_size]))
				mitigated_counts_list = averaged_mitigated
				num_groups = len(circuits) // twirl_group_size
				result_to_use = self._trim_result_to_groups(result, num_groups)

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


class BatchedJob:
	"""Lazy handle over one or more backend jobs submitted as ordered batches.

	A larger circuit list is split into batches that are each submitted to the backend; this
	wrapper holds the resulting per-batch jobs. It is returned immediately from
	:meth:`FiQCIBackend.run` (the submission loop does not wait for results), so callers can
	inspect ``job_ids()`` and poll ``status()``/``done()`` right away.

	Calling :meth:`result` blocks until every batch reaches a terminal state, then concatenates
	the batches' ``results`` lists in submission order (so ``get_counts(idx)`` on the combined
	Result corresponds to the original circuit index) and runs the optional ``post_process``
	callback that applies error mitigation. The combined/post-processed result is computed once
	and cached. If any batch failed, :meth:`result` raises :class:`BatchFailedError` identifying
	the batch and the original circuit indices it covered.
	"""

	def __init__(
		self,
		jobs: list[JobV1],
		batch_ranges: list[tuple[int, int]] | None = None,
		post_process: Callable[[Result], Result] | None = None,
		mitigator_options: dict[str, Any] | None = None,
	) -> None:
		"""Initialize the handle.

		Args:
			jobs: Per-batch jobs in submission order. Must be non-empty.
			batch_ranges: ``(start, end_exclusive)`` original circuit-index range for each batch,
				used for partial-result reporting and failure messages. If omitted, ranges are
				reported as best-effort placeholders.
			post_process: Optional callback mapping the combined raw Result to the final
				(mitigated) Result. Runs once on the first :meth:`result` call.
			mitigator_options: Frozen snapshot of the mitigation settings used at submission time,
				exposed via :attr:`mitigator_options`. ``None`` when unknown.
		"""
		assert jobs, "BatchedJob must wrap at least one job"
		self._jobs = jobs
		self._batch_ranges = batch_ranges
		self._post_process = post_process
		self._mitigator_options = mitigator_options
		self._combined_result: Result | None = None
		self._final_result: Result | None = None
		self._lock = threading.Lock()

	# -- identity / polling (available immediately, before any results) --

	@property
	def mitigator_options(self) -> dict[str, Any] | None:
		"""Mitigation settings frozen at submission time (``mitigation_level``, ``rem``, ``dd``,
		``pauli_twirl``).

		Unlike :attr:`FiQCIBackend.mitigator_options` (which is live and mutable), this reports the
		configuration this job actually ran with and never changes. ``None`` if no snapshot was
		attached.
		"""
		return self._mitigator_options

	def job_id(self) -> str:
		"""Backend job id of the first batch (for single-job back-compat)."""
		return self._jobs[0].job_id()

	def job_ids(self) -> list[str]:
		"""Backend job ids of every batch, in submission order."""
		return [job.job_id() for job in self._jobs]

	def statuses(self) -> list[JobStatus]:
		"""Current :class:`JobStatus` of each batch, in submission order."""
		return [job.status() for job in self._jobs]

	def status(self) -> JobStatus:
		"""Single aggregated status across all batches (see :meth:`_aggregate_status`)."""
		return self._aggregate_status(self.statuses())

	def done(self) -> bool:
		"""True once every batch has reached a terminal state (DONE/ERROR/CANCELLED)."""
		return all(status in _TERMINAL_STATUSES for status in self.statuses())

	def all_succeeded(self) -> bool:
		"""True once every batch has completed successfully (DONE)."""
		return all(status == JobStatus.DONE for status in self.statuses())

	@staticmethod
	def _aggregate_status(statuses: list[JobStatus]) -> JobStatus:
		"""Collapse per-batch statuses into one by priority.

		A failure anywhere dominates (ERROR, then CANCELLED). Otherwise the job is DONE only when
		all batches are done; if any batch is still progressing the least-advanced active state is
		reported (RUNNING > VALIDATING > QUEUED > INITIALIZING).
		"""
		if not statuses:
			return JobStatus.DONE
		if any(status == JobStatus.ERROR for status in statuses):
			return JobStatus.ERROR
		if any(status == JobStatus.CANCELLED for status in statuses):
			return JobStatus.CANCELLED
		if all(status == JobStatus.DONE for status in statuses):
			return JobStatus.DONE
		for active in (JobStatus.RUNNING, JobStatus.VALIDATING, JobStatus.QUEUED):
			if any(status == active for status in statuses):
				return active
		return JobStatus.INITIALIZING

	def _range(self, index: int) -> tuple[int, int]:
		"""Original circuit-index range for batch ``index`` (best-effort if unknown)."""
		if self._batch_ranges is not None and index < len(self._batch_ranges):
			return self._batch_ranges[index]
		return (index, index + 1)

	# -- partial results (batch-granular) --

	def partial_results(self) -> list[PartialBatch]:
		"""Per-batch snapshot, exposing results for batches that have already completed.

		Each entry carries the batch's status and, for completed (DONE) batches, its Result.
		Batches that are still running or have failed report ``result=None``. Results are exposed
		at batch granularity only; the globally-indexed combined Result is available from
		:meth:`result` once all batches are terminal.
		"""
		snapshots: list[PartialBatch] = []
		for index, job in enumerate(self._jobs):
			status = job.status()
			result: Result | None = None
			if status == JobStatus.DONE:
				try:
					result = job.result()
				except Exception:  # pragma: no cover - defensive
					result = None
			snapshots.append(
				PartialBatch(
					index=index, circuit_range=self._range(index), status=status, job_id=job.job_id(), result=result
				)
			)
		return snapshots

	# -- combined / post-processed result --

	def result(self, timeout: float | None = None) -> Result:
		"""Return the combined, post-processed result, computed once and cached.

		Blocks until every batch is terminal. Raises :class:`BatchFailedError` if any batch
		failed, otherwise concatenates batch results in submission order and applies the
		``post_process`` callback (if any).

		Args:
			timeout: Best-effort total budget (seconds) shared across all batches.
		"""
		if self._final_result is not None:
			return self._final_result
		with self._lock:
			if self._final_result is not None:
				return self._final_result
			combined = self._combine(timeout)
			self._final_result = self._post_process(combined) if self._post_process is not None else combined
			return self._final_result

	def _combine(self, timeout: float | None) -> Result:
		"""Wait for all batches and concatenate their results in submission order."""
		if self._combined_result is not None:
			return self._combined_result

		from qiskit.result import Result as QiskitResult

		deadline = None if timeout is None else time.monotonic() + timeout
		results: list[Result] = []
		failures: list[tuple[int, str, str]] = []
		for index, job in enumerate(self._jobs):
			remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
			try:
				# Concrete IQM jobs accept a timeout; the abstract JobV1.result stub does not declare one.
				result = job.result() if remaining is None else job.result(remaining)  # type: ignore[bad-argument-count]
			except Exception as exc:  # batch failed at the backend
				failures.append((index, job.job_id(), str(exc)))
				continue
			if job.status() in (JobStatus.ERROR, JobStatus.CANCELLED):
				failures.append((index, job.job_id(), str(job.status())))
				continue
			results.append(result)

		if failures:
			raise BatchFailedError(self._failure_message(failures))

		combined_data: dict[str, Any] = results[0].to_dict()
		merged_results: list[Any] = list(combined_data.get("results") or [])
		for result in results[1:]:
			merged_results.extend(result.to_dict().get("results") or [])
		combined_data["results"] = merged_results
		self._combined_result = QiskitResult.from_dict(combined_data)
		return self._combined_result

	def _failure_message(self, failures: list[tuple[int, str, str]]) -> str:
		"""Build a human-readable message naming each failed batch and its circuit range."""
		parts = []
		for index, job_id, detail in failures:
			start, end = self._range(index)
			parts.append(f"batch {index} (circuits {start}-{end - 1}, job_id={job_id}): {detail}")
		return f"{len(failures)} of {len(self._jobs)} batch(es) failed: " + "; ".join(parts)

	def __getattr__(self, name: str) -> Any:
		"""Delegate attribute access to the first underlying job."""
		return getattr(self._jobs[0], name)


class MitigatedJob:
	"""Lazy view over a :class:`BatchedJob` whose result has error mitigation applied.

	The mitigation work is deferred to the wrapped handle's ``post_process`` callback, so this
	wrapper simply exposes the handle's polling API and a :meth:`result` that returns the
	mitigated, combined Result. It exists as a distinct type so callers and tests can detect that
	mitigation was configured for the run.
	"""

	def __init__(self, handle: BatchedJob) -> None:
		"""Initialize the wrapper.

		Args:
			handle: The submission handle carrying the batch jobs and the mitigation
				``post_process`` callback.
		"""
		self._handle = handle

	def result(self, timeout: float | None = None) -> Result:
		"""Return the mitigated, combined result (computed once by the underlying handle)."""
		return self._handle.result(timeout)

	def __getattr__(self, name: str) -> Any:
		"""Delegate attribute access (status, done, job_ids, partial_results, …) to the handle."""
		return getattr(self._handle, name)
