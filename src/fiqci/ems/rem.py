"""Functions related to Readout Error Mitigation (REM)."""

import logging
from dataclasses import dataclass

import numpy as np
import orjson
from mthree import M3Mitigation
from mthree.exceptions import M3Error
from mthree.mitigation import _faulty_qubit_checker

logger: logging.Logger = logging.getLogger(__name__)


def balanced_cal_strings(num_qubits: int) -> list[str]:
	"""Generate balanced calibration strings for the given number of qubits.

	Balanced calibration strings ensure equal representation of 0 and 1 states
	across all qubits during calibration.

	Args:
		num_qubits: Number of qubits to generate calibration strings for.

	Returns:
		List of balanced calibration bit strings.

	Raises:
		ValueError: If num_qubits is less than 1.
	"""
	if num_qubits < 1:
		raise ValueError("Number of qubits must be at least 1")

	# Generate all possible bit strings for num_qubits
	num_strings = 2**num_qubits
	return [format(i, f"0{num_qubits}b") for i in range(num_strings)]


@dataclass
class Config:
	"""Configuration for the backend"""

	num_qubits: int
	max_shots: int
	simulator: bool
	max_experiments: int
	max_circuits: int


class M3IQM(M3Mitigation):
	"""M3 readout mitigation class modified to work with IQM devices.
	Adapted from IQM Benchmarks which is adapted from M3 both of which are licensed under Apache 2.0
	"""

	def __init__(self, backend):
		self.backend = backend
		if not hasattr(self.backend, "configuration"):
			self.backend.configuration = lambda: Config(
				num_qubits=backend.num_qubits, max_shots=10000, simulator=False, max_experiments=2, max_circuits=100
			)

		super().__init__(self.backend)

		# Track which qubits were calibrated for validation
		self._calibrated_qubits: list[int] | None = None

	def cals_from_system(  # type: ignore[override]
		self,
		qubits=None,
		shots=None,
		method=None,
		initial_reset=False,
		rep_delay=None,
		cals_file=None,
		async_cal=True,
		runtime_mode=None,
		cal_id=None,
	):
		"""Grab calibration data from system.

		Overrides M3's method to:
		1. Default to 'balanced' calibration method for IQM
		2. Support IQM's calibration_set_id parameter
		3. Use IQM-specific job thread for bit-string handling

		Parameters:
			qubits (array_like): Qubits over which to correct calibration data. Default is all.
			shots (int): Number of shots per circuit. min(1e4, max_shots).
			method (str): Type of calibration, 'balanced' (default for IQM),
						 'independent', or 'marginal'.
			initial_reset (bool): Use resets at beginning of calibration circuits, default=False.
			rep_delay (float): Delay between circuits on IBM Quantum backends.
			cals_file (str): Output path to write JSON calibration data to.
			async_cal (bool): Do calibration async in a separate thread, default is True.
			runtime_mode: Mode to run jobs in if using IBM system, default=None
			cal_id (str): Optional calibration set ID for IQM backends.

		Returns:
			list: List of jobs submitted.

		Raises:
			M3Error: Called while a calibration currently in progress.
		"""
		# Store cal_id for use in _grab_additional_cals
		self._cal_id = cal_id

		# Force balanced method for IQM if not specified
		if method is None:
			method = "balanced"

		# Call parent's method
		return super().cals_from_system(
			qubits=qubits,
			shots=shots,
			method=method,
			initial_reset=initial_reset,
			rep_delay=rep_delay,
			cals_file=cals_file,
			async_cal=async_cal,
			runtime_mode=runtime_mode,
		)

	def _grab_additional_cals(  # type: ignore[override]
		self, qubits, shots=None, method="balanced", rep_delay=None, initial_reset=False, async_cal=False
	):
		"""Grab missing calibration data from backend.

		Minimal override to track calibrated qubits for FiQCI validation.
		All calibration logic handled by M3's parent implementation.

		Parameters:
			qubits (array_like): List of measured qubits.
			shots (int): Number of shots to take, min(1e4, max_shots).
			method (str): Type of calibration, 'balanced' (default for IQM), 'independent', or 'marginal'.
			rep_delay (float): Delay between circuits on IBM Quantum backends.
			initial_reset (bool): Use resets at beginning of calibration circuits, default=False.
			async_cal (bool): Do calibration async in a separate thread, default is False.

		Raises:
			M3Error: Backend not set.
			M3Error: Faulty qubits found.
		"""
		# Extract qubits for tracking (handle dict/list of dicts formats)
		if isinstance(qubits, dict):
			qubits_to_track = list(set(qubits.values()))
		elif isinstance(qubits, list) and qubits and isinstance(qubits[0], dict):
			_qubits = []
			for item in qubits:
				_qubits.extend(list(set(item.values())))
			qubits_to_track = list(set(_qubits))
		else:
			qubits_to_track = qubits

		# Track which qubits will be calibrated
		if hasattr(self, "_calibrated_qubits") and self._calibrated_qubits is not None:
			all_qubits = set(self._calibrated_qubits) | set(qubits_to_track)
			self._calibrated_qubits = sorted(all_qubits)
		else:
			self._calibrated_qubits = sorted(qubits_to_track)

		# Call parent's implementation - M3's logic works correctly for IQM backends
		return super()._grab_additional_cals(
			qubits=qubits,
			shots=shots,
			method=method,
			rep_delay=rep_delay,
			initial_reset=initial_reset,
			async_cal=async_cal,
		)

	def cals_to_file(self, cals_file: str | None = None) -> None:
		"""Save calibration data to JSON file with FiQCI-specific metadata.

		Extends M3's cals_to_file to include calibration_set_id and qubits.

		Parameters:
			cals_file: File in which to store calibrations.

		Raises:
			M3Error: Calibration filename missing.
			M3Error: Mitigator is not calibrated.
		"""
		if not cals_file:
			raise M3Error("cals_file must be explicitly set.")
		if not self.single_qubit_cals:
			raise M3Error("Mitigator is not calibrated.")

		# Get calibration set ID from backend if available
		calibration_set_id = None
		if hasattr(self.backend, "_calibration_set_id"):
			cal_id = self.backend._calibration_set_id
			# Convert UUID to string for JSON serialization
			calibration_set_id = str(cal_id) if cal_id else None

		save_dict = {
			"timestamp": self.cal_timestamp,
			"backend": self.system_info.get("name", None),
			"shots": self.cal_shots,
			"cals": self.single_qubit_cals,
			"calibration_set_id": calibration_set_id,
			"qubits": self._calibrated_qubits,
		}

		with open(cals_file, "wb") as fd:
			fd.write(orjson.dumps(save_dict, option=orjson.OPT_SERIALIZE_NUMPY))

		logger.info(
			"Saved calibration to %s (calibration_set_id=%s, qubits=%s)",
			cals_file,
			calibration_set_id,
			self._calibrated_qubits,
		)

	def cals_from_file(self, cals_file: str, validate_calibration_set: bool = True) -> None:
		"""Load calibration data from JSON file with FiQCI-specific validation.

		Extends M3's cals_from_file to validate calibration_set_id matches backend.

		Parameters:
			cals_file: Path to the saved calibration file.
			validate_calibration_set: Whether to validate calibration_set_id matches backend.

		Raises:
			M3Error: Calibration in progress.
			M3Error: Calibration set ID mismatch.
			M3Error: Invalid calibration file format.
			FileNotFoundError: Calibration file not found.
		"""
		if self._thread:
			raise M3Error("Calibration currently in progress.")

		with open(cals_file, encoding="utf-8") as fd:
			loaded_data = orjson.loads(fd.read())

		# Only support dict format with required fields
		if not isinstance(loaded_data, dict):
			raise M3Error("Invalid calibration file format. ")

		# Load calibration data
		self.single_qubit_cals = [  # type: ignore[assignment]
			np.asarray(cal, dtype=np.float32) if cal else None for cal in loaded_data["cals"]
		]
		self.cal_timestamp = loaded_data.get("timestamp")
		self.cal_shots = loaded_data.get("shots", None)
		self._calibrated_qubits = loaded_data.get("qubits", None)

		# Validate calibration set ID if present and validation enabled
		if validate_calibration_set and "calibration_set_id" in loaded_data:
			saved_cal_id = loaded_data["calibration_set_id"]

			if saved_cal_id is not None and hasattr(self.backend, "_calibration_set_id"):
				current_cal_id = str(self.backend._calibration_set_id) if self.backend._calibration_set_id else None

				if current_cal_id != saved_cal_id:
					raise M3Error(
						f"Calibration set ID mismatch! "
						f"Saved calibration is for calibration_set_id={saved_cal_id}, "
						f"but current backend has calibration_set_id={current_cal_id}. "
						f"The backend configuration has changed. Please recalibrate."
					)

		self.faulty_qubits = _faulty_qubit_checker(self.single_qubit_cals)
