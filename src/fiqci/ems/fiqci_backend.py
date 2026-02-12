"""FiQCI backend wrapper for seamless error mitigation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.iqm_client.models import CircuitCompilationOptions, DDMode
from mthree.utils import final_measurement_mapping

from fiqci.ems.rem import M3IQM
from fiqci.ems.utils import probabilities_to_counts
from fiqci.ems.paulitwirl import PauliTwirl

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.providers import JobV1
from qiskit.result import Result

logger: logging.Logger = logging.getLogger(__name__)


class FiQCIBackend:
	"""FiQCI backend wrapper that applies error mitigation automatically.

	Mitigation levels:
		0: No error mitigation (raw results)
		1: Readout error mitigation using M3 (default)

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
		self._calibration_shots = calibration_shots
		self._calibration_file = calibration_file
		self._mitigator: M3IQM | None = None
		self._raw_counts_cache: list[dict[str, int]] | None = None
		self.num_pauli_twirls = 50 # Number of twirled circuits to generate

		# Initialize mitigator for level 1 (readout error mitigation using M3)
		if self._mitigation_level > 0 and self._mitigation_level < 4:
			self._mitigator = M3IQM(self._backend)

			# Load calibration from file if it exists
			if self._calibration_file:
				cal_path = Path(self._calibration_file)
				if cal_path.exists():
					try:
						# M3IQM.cals_from_file will validate calibration_set_id matches
						self._mitigator.cals_from_file(self._calibration_file, validate_calibration_set=True)
						logger.info("Loaded existing M3 calibration from %s", self._calibration_file)
					except Exception as e:
						# Log the specific error and fall back to calibration
						error_msg = str(e)
						if "Calibration set ID mismatch" in error_msg:
							logger.error(
								"Calibration set ID mismatch: %s. Backend configuration has changed. "
								"Will recalibrate on first run.",
								error_msg,
							)
						else:
							logger.warning(
								"Could not load calibration from %s: %s. Will calibrate on first run.",
								self._calibration_file,
								e,
							)
				else:
					logger.info(
						"Calibration file %s does not exist yet. Will calibrate and save on first run.",
						self._calibration_file,
					)

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

		# Track original circuit indices for level 3
		circuit_groups: list[list[int]] = []
		flattened_circuits: list[QuantumCircuit] = []

		# Apply Pauli twirling for mitigation level 3
		if self._mitigation_level == 3:
			pm = PassManager([PauliTwirl()])

			for circ in circuits_list:
				start_idx = len(flattened_circuits)
				twirled_qcs = [transpile(pm.run(circ), self._backend) for _ in range(self.num_pauli_twirls)]
				twirled_qcs.append(circ)  # Include original circuit
				flattened_circuits.extend(twirled_qcs)
				# Track indices for this original circuit
				circuit_groups.append(list(range(start_idx, len(flattened_circuits))))

			circuits_list = flattened_circuits
		else:
			circuit_groups = [[i] for i in range(len(circuits_list))]

		if not circuits_list:
			raise ValueError("No circuits provided")

		# Level 0: No mitigation, pass through to backend
		if self._mitigation_level == 0:
			job = self._backend.run(circuits_list, shots=shots, **kwargs)
			assert job is not None, "Backend returned None job"
			return job

		# Level 1: Readout error mitigation with M3
		if self._mitigation_level == 1:
			return self._run_with_m3_mitigation(circuits_list, shots, dd=False, circuit_groups=None, **kwargs)
		
		# Level 2: Readout error mitigation with M3 and dynamical decoupling (DD)
		if self._mitigation_level == 2:
			return self._run_with_m3_mitigation(circuits_list, shots, dd=True, circuit_groups=None, **kwargs)

		# Level 3: Pauli twirling + M3 mitigation + dynmical decoupling
		if self._mitigation_level == 3:
			return self._run_with_m3_mitigation(circuits_list, shots, dd=True, circuit_groups=circuit_groups, **kwargs)

		raise NotImplementedError(f"Mitigation level {self._mitigation_level} not yet implemented")

	def _run_with_m3_mitigation(
		self, 
		circuits: list[QuantumCircuit], 
		shots: int, 
		dd: bool = False, 
		circuit_groups: list[list[int]] | None = None,
		**kwargs: Any
	) -> MitigatedJob:
		"""Run circuits with M3 readout error mitigation.

		Args:
			circuits: List of quantum circuits to execute.
			shots: Number of measurement shots.
			dd: Whether to enable dynamical decoupling.
			circuit_groups: For level 3, groups of circuit indices to average. None for levels 1-2.
			**kwargs: Additional keyword arguments passed to backend.run().

		Returns:
			A MitigatedJob instance with mitigated results.
		"""
		# Get qubit mappings for each circuit
		qubits_list = [final_measurement_mapping(circuit) for circuit in circuits]

		# Calibrate M3 mitigator if not already done
		if self._mitigator is not None and self._mitigator.single_qubit_cals is None:
			# Extract unique qubits from all circuits for calibration
			all_qubits: set[int] = set()
			for qubit_mapping in qubits_list:
				all_qubits.update(qubit_mapping.values())  # type: ignore[arg-type]
			calibration_qubits = sorted(all_qubits)

			if self._calibration_file:
				logger.info(
					"Calibrating M3 mitigator for qubits %s with %d shots and saving to %s",
					calibration_qubits,
					self._calibration_shots,
					self._calibration_file,
				)
			else:
				logger.info(
					"Calibrating M3 mitigator for qubits %s with %d shots", calibration_qubits, self._calibration_shots
				)

			assert self._mitigator is not None, "Mitigator should be initialized for level 1"
			self._mitigator.cals_from_system(
				calibration_qubits, shots=self._calibration_shots, cals_file=self._calibration_file
			)

		# Run circuits on backend
		if dd:
			circuit_compilation_options_val = CircuitCompilationOptions(dd_mode=DDMode.ENABLED)
		else:
			circuit_compilation_options_val = CircuitCompilationOptions(dd_mode=DDMode.DISABLED)

		# Avoid passing circuit_compilation_options twice
		if "circuit_compilation_options" in kwargs:
			logger.debug(
				"Dropping user-provided circuit_compilation_options; using dd_mode=%s",
				"ENABLED" if dd else "DISABLED",
			)
			kwargs.pop("circuit_compilation_options")
		
		job = self._backend.run(circuits, shots=shots, circuit_compilation_options=circuit_compilation_options_val, **kwargs)
		assert job is not None, "Backend returned None job"
		result = job.result()

		# Store raw counts and apply mitigation to each circuit's results
		raw_counts_list: list[dict[str, int]] = []
		mitigated_counts_list: list[dict[str, int]] = []

		# For level 3, we need to average the counts of twirled variants
		if circuit_groups:
        	# Process groups of circuits and average their results
			for group_indices in circuit_groups:
				group_raw_counts_list: list[dict[str, int]] = []
				group_mitigated_counts_list: list[dict[str, int]] = []

				for idx in group_indices:
					raw_counts = result.get_counts(idx)
					group_raw_counts_list.append(raw_counts)
					qubits = qubits_list[idx]

					# Apply M3 correction
					assert self._mitigator is not None, "Mitigator should be initialized"
					quasi_dist = self._mitigator.apply_correction(raw_counts, qubits)
					mitigated_probs = quasi_dist.nearest_probability_distribution()  # type: ignore[union-attr]
					mitigated_counts = probabilities_to_counts(mitigated_probs, shots)
					group_mitigated_counts_list.append(mitigated_counts[0])

				# Average the raw counts
				averaged_raw = self._average_counts(group_raw_counts_list)
				raw_counts_list.append(averaged_raw)

				# Average the mitigated counts
				averaged_mitigated = self._average_counts(group_mitigated_counts_list)
				mitigated_counts_list.append(averaged_mitigated)
		else:
			# Level 1-2: Process each circuit individually
			for idx, circuit in enumerate(circuits):
				raw_counts = result.get_counts(idx)
				raw_counts_list.append(raw_counts)
				qubits = qubits_list[idx]

				# Apply M3 correction
				assert self._mitigator is not None, "Mitigator should be initialized for level 1"
				quasi_dist = self._mitigator.apply_correction(raw_counts, qubits)
				mitigated_probs = quasi_dist.nearest_probability_distribution()  # type: ignore[union-attr]
				mitigated_counts = probabilities_to_counts(mitigated_probs, shots)
				mitigated_counts_list.append(mitigated_counts[0])

		# Cache raw counts for access via property
		self._raw_counts_cache = raw_counts_list

		# Create new result with mitigated counts and metadata
		# For level 3, trim result to only include original circuits (not twirled variants)
		result_to_use = result
		if circuit_groups:
			# Create a trimmed result with only the averaged results
			result_to_use = self._trim_result_to_groups(result, len(circuit_groups))
    

		# Create new result with mitigated counts and metadata
		mitigated_result = self._create_mitigated_result(result_to_use, mitigated_counts_list, raw_counts_list)

		# Wrap in job-like object
		return MitigatedJob(job, mitigated_result)
	
	def _average_counts(self, counts_list: list[dict[str, int]]) -> dict[str, int]:
		"""Average a list of count dictionaries.

		Args:
			counts_list: List of count dictionaries to average.

		Returns:
			Averaged count dictionary rounded to nearest integer.
		"""
		if not counts_list:
			return {}

		# Get all possible states
		all_states = set()
		for counts in counts_list:
			all_states.update(counts.keys())

		# Average the counts for each state
		averaged = {}
		for state in all_states:
			total = sum(counts.get(state, 0) for counts in counts_list)
			averaged[state] = round(total / len(counts_list))

		return averaged

	def _trim_result_to_groups(self, result: Result, num_groups: int) -> Result:
		"""Trim result to keep only the first N experiments (for level 3 averaging).

		Args:
			result: Original result with all circuits.
			num_groups: Number of groups (original circuits) to keep.

		Returns:
			Trimmed Result object.
		"""
		results_data = result.to_dict()
		results_list = results_data.get("results", [])
		
		# Keep only the first num_groups results
		results_data["results"] = results_list[:num_groups]
		
		from qiskit.result import Result as QiskitResult
		return QiskitResult.from_dict(results_data)

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
						"calibration_shots": self._calibration_shots if self._mitigation_level == 1 else None,
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
