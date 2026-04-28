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
from typing import Any, Iterable, Optional, TypedDict

from iqm.iqm_client import STANDARD_DD_STRATEGY
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from mthree.utils import final_measurement_mapping

from fiqci.ems.mitigators.rem import M3IQM
from fiqci.ems.mitigators.dd import DDGateSequenceEntry, build_dd_options
from fiqci.ems.transpiler_passes.pauli_twirl import get_twirled_circuits
from fiqci.ems.utils import probabilities_to_counts

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.providers import JobV1
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
			gates_to_twirl: Optional[Iterable[Gate]]
		
		self._pauli_twirl: PauliTwirlSettings = {
			"enabled": False,
			"num_twirls": 0,
			"gates_to_twirl": None,
		}

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

		Returns:
			List of raw count dictionaries, or None if no run has been performed yet.
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

	def init_pauli_twirl(self, enabled: bool, num_twirls: int = 10, gates_to_twirl: Optional[Iterable[Gate]] = None) -> None:
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
		self, circuits: QuantumCircuit | list[QuantumCircuit], shots: int = 1024, **kwargs: Any
	) -> JobV1 | MitigatedJob:
		"""Run quantum circuits with error mitigation.

		This method runs the specified quantum circuit(s) on the backend and applies
		error mitigation based on the configured mitigation level.

		Args:
			circuits: Single quantum circuit or list of circuits to execute.
			shots: Number of shots. Default is 1024.
			**kwargs: Additional keyword arguments passed to backend.run().

		Returns:
			A JobV1 instance (level 0) or MitigatedJob instance (level 1+) with mitigated results.

		Raises:
			ValueError: If circuits is empty or invalid.
		"""

		# Normalize to list
		circuits_list = circuits if isinstance(circuits, list) else [circuits]

		if not circuits_list:
			raise ValueError("No circuits provided")

		# If Pauli Twirling is enabled, replace circuits with twirled versions
		twirl_group_size = 0
		if self._pauli_twirl["enabled"]:
			circuits_list = get_twirled_circuits(
				circuits_list, num_twirls=self._pauli_twirl["num_twirls"], gates_to_twirl=self._pauli_twirl["gates_to_twirl"]
			)
			circuits_list = transpile(circuits_list, backend=self._backend, optimization_level=0)
			twirl_group_size = self._pauli_twirl["num_twirls"] + 1

		# Run circuits on backend (with DD options if enabled)
		if self._dd["enabled"]:
			dd_options = build_dd_options(self._dd["gate_sequences"])
			job = self._backend.run(circuits_list, shots=shots, circuit_compilation_options=dd_options, **kwargs)
		else:
			job = self._backend.run(circuits_list, shots=shots, **kwargs)
		assert job is not None, "Backend returned None job"

		# No REM: return raw job (or averaged twirl results)
		if not self._rem["enabled"]:
			if twirl_group_size == 0:
				return job

			result = job.result()
			raw_counts_list = self._average_group_counts(result, twirl_group_size)
			self._raw_counts_cache = raw_counts_list
			num_groups = len(circuits_list) // twirl_group_size
			result_to_use = self._trim_result_to_groups(result, num_groups)
			mitigated_result = self._create_mitigated_result(result_to_use, raw_counts_list, raw_counts_list)
			return MitigatedJob(job, mitigated_result)

		# REM enabled: run with M3 mitigation
		return self._run_with_m3_mitigation(job, circuits_list, shots, twirl_group_size=twirl_group_size)

	def _average_group_counts(self, result: Result, group_size: int) -> list[dict[str, int]]:
		"""Average raw counts across twirled circuit groups.

		Args:
			result: Result object from backend.
			group_size: Number of circuits per group (num_twirls + 1).

		Returns:
			List of averaged count dictionaries, one per group.
		"""
		all_counts = result.get_counts()
		if not isinstance(all_counts, list):
			all_counts = [all_counts]
		averaged = []
		for i in range(0, len(all_counts), group_size):
			averaged.append(self._average_counts(all_counts[i:i + group_size]))
		return averaged

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
		"""Trim a Result to only include the first num_groups experiment results.

		Args:
			result: Original Result object.
			num_groups: Number of experiment results to keep.

		Returns:
			New Result object with only the first num_groups results.
		"""
		from qiskit.result import Result as QiskitResult

		result_data = result.to_dict()
		results_list = result_data.get("results")
		if results_list is not None:
			result_data["results"] = results_list[:num_groups]
		return QiskitResult.from_dict(result_data)

	def _run_with_m3_mitigation(
		self, job: JobV1, circuits: list[QuantumCircuit], shots: int, twirl_group_size: int = 0
	) -> MitigatedJob:
		"""Run circuits with M3 readout error mitigation.

		Args:
			job: Already-submitted job from backend.
			circuits: List of quantum circuits that were executed.
			shots: Number of measurement shots.
			twirl_group_size: Size of each twirl group (num_twirls + 1), or 0 if no twirling.

		Returns:
			A MitigatedJob instance with mitigated results.
		"""
		# Get qubit mappings for each circuit
		qubits_list = [final_measurement_mapping(circuit) for circuit in circuits]

		# Calibrate M3 mitigator if not already done
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
				calibration_qubits, shots=self._rem["calibration_shots"], cals_file=self._rem["calibration_file"]
			)

		result = job.result()

		# Apply M3 correction to each circuit's results
		raw_counts_list: list[dict[str, int]] = []
		mitigated_counts_list: list[dict[str, int]] = []

		for idx in range(len(circuits)):
			raw_counts = result.get_counts(idx)
			assert isinstance(raw_counts, dict), f"Expected dict from get_counts({idx}), got {type(raw_counts)}"
			raw_counts_list.append(raw_counts)
			qubits = qubits_list[idx]

			assert self._rem["mitigator"] is not None, "Mitigator should be initialized for level 1"
			quasi_dist = self._rem["mitigator"].apply_correction(raw_counts, qubits)
			mitigated_probs = quasi_dist.nearest_probability_distribution()  # type: ignore[union-attr]
			mitigated_counts = probabilities_to_counts(mitigated_probs, shots)
			mitigated_counts_list.append(mitigated_counts[0])

		# If Pauli twirling, average across groups
		if twirl_group_size:
			raw_counts_list = self._average_group_counts(result, twirl_group_size)
			# Average mitigated counts by group
			averaged_mitigated: list[dict[str, int]] = []
			for i in range(0, len(mitigated_counts_list), twirl_group_size):
				averaged_mitigated.append(self._average_counts(mitigated_counts_list[i:i + twirl_group_size]))
			mitigated_counts_list = averaged_mitigated
			num_groups = len(circuits) // twirl_group_size
			result = self._trim_result_to_groups(result, num_groups)

		self._raw_counts_cache = raw_counts_list
		mitigated_result = self._create_mitigated_result(result, mitigated_counts_list, raw_counts_list)
		return MitigatedJob(job, mitigated_result)
	
	def _create_mitigated_result(
		self, original_result: Result, mitigated_counts: list[dict[str, int]], raw_counts: list[dict[str, int]]
	) -> Result:
		"""Create a new Result object with mitigated counts and metadata.

		Args:
			original_result: Original result from backend.
			mitigated_counts: List of mitigated count dictionaries.
			raw_counts: List of raw (unmitigated) count dictionaries.

		Returns:
			New Result object with mitigated data and FiQCI EMS metadata.
		"""
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

					results_list[idx]["header"]["fiqci_ems"] = {  # type: ignore[index]
						"mitigation_level": self._mitigation_level,
						"mitigation_method": "M3" if self._mitigation_level == 1 else None,
						"calibration_shots": self._rem["calibration_shots"] if self._mitigation_level == 1 else None,
						"raw_counts": raw_counts[idx],
					}

		# Create new result from modified data
		from qiskit.result import Result as QiskitResult

		return QiskitResult.from_dict(results_data)

	def __getattr__(self, name: str) -> Any:
		"""Delegate attribute access to underlying backend object."""
		return getattr(self._backend, name)


class MitigatedJob:
	"""Wrapper for job results with mitigated data.

	This class wraps the original job and provides access to mitigated results.
	"""

	def __init__(self, original_job: JobV1, mitigated_result: Result) -> None:
		"""Initialize mitigated job wrapper.

		Args:
			original_job: Original job from backend.
			mitigated_result: Result object with mitigated counts.
		"""
		self._original_job = original_job
		self._mitigated_result = mitigated_result

	def result(self, timeout: float | None = None) -> Result:
		"""Get the mitigated result.

		Args:
			timeout: Maximum time to wait for result (unused, job already complete).

		Returns:
			Result object with mitigated counts.
		"""
		return self._mitigated_result

	def __getattr__(self, name: str) -> Any:
		"""Delegate attribute access to original job object."""
		return getattr(self._original_job, name)
