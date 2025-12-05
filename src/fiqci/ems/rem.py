"""Functions related to Readout Error Mitigation (REM)."""

import logging
import threading
from math import ceil

import numpy as np
import orjson
from mthree import M3Mitigation
from mthree.circuits import _marg_meas_states, _tensor_meas_states, balanced_cal_circuits
from mthree.exceptions import M3Error
from mthree.generators import HadamardGenerator
from mthree.mitigation import _faulty_qubit_checker
from dataclasses import dataclass

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

		This method is identical to M3's version except it calls _iqm_job_thread
		instead of _job_thread to handle IQM's bit-string reversal.

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
		if self.system is None:
			raise M3Error("System is not set.  Use 'cals_from_file'.")
		if self.single_qubit_cals is None:
			self.single_qubit_cals = [None] * self.num_qubits  # type: ignore[operator]
		if shots is None:
			shots = min(self.system_info["max_shots"], 10000)
		self.cal_shots = shots  # type: ignore[assignment]
		if self.rep_delay is None:
			self.rep_delay = rep_delay

		logger.info("Grabbing calibration data for qubits=%s, method=%s, async_cal=%s", qubits, method, async_cal)

		if method not in ["independent", "balanced", "marginal"]:
			raise M3Error(f"Invalid calibration method {method}.")

		if isinstance(qubits, dict):
			# Assuming passed a mapping
			qubits = list(set(qubits.values()))
		elif isinstance(qubits, list):
			# Check if passed a list of mappings
			if isinstance(qubits[0], dict):
				# Assuming list of mappings, need to get unique elements
				_qubits = []
				for item in qubits:
					_qubits.extend(list(set(item.values())))
				qubits = list(set(_qubits))

		# Do check for inoperable qubits here
		inoperable_overlap = list(set(qubits) & set(self.system_info["inoperable_qubits"]))
		if any(inoperable_overlap):
			raise M3Error(f"Attempting to calibrate inoperable qubits: {inoperable_overlap}")

		num_cal_qubits = len(qubits)
		generator = None
		# shots is needed here because balanced cals will use a value
		# different from cal_shots
		shots = self.cal_shots
		logger.info("Generating calibration circuits.")
		if method == "marginal":
			trans_qcs = _marg_meas_states(qubits, self.num_qubits, initial_reset=initial_reset)
		elif method == "balanced":
			generator = HadamardGenerator(num_cal_qubits)
			trans_qcs = balanced_cal_circuits(generator, qubits, self.num_qubits, initial_reset=initial_reset)
			shots = 2 * self.cal_shots // generator.length  # type: ignore[operator]
			if 2 * self.cal_shots / generator.length != shots:  # type: ignore[operator]
				shots += 1
			self._balanced_shots = shots * generator.length  # type: ignore[assignment]
		# Independent
		else:
			trans_qcs = []
			for kk in qubits:
				trans_qcs.extend(_tensor_meas_states(kk, self.num_qubits, initial_reset=initial_reset))

		num_circs = len(trans_qcs)
		max_circuits = self.system_info["max_circuits"]
		# Determine the number of jobs required
		num_jobs = ceil(num_circs / max_circuits)
		logger.info("Generated %s circuits, which will run in %s jobs using %s shots", num_circs, num_jobs, shots)
		# Get the slice length
		circ_slice = ceil(num_circs / num_jobs)
		circs_list = [trans_qcs[kk * circ_slice : (kk + 1) * circ_slice] for kk in range(num_jobs - 1)] + [
			trans_qcs[(num_jobs - 1) * circ_slice :]
		]

		# Do job submission here
		jobs = []
		for circs in circs_list:
			# Check if we have a calibration_set_id to pass
			if hasattr(self, "_cal_id") and self._cal_id is not None:
				_job = self.system.run(
					circs,
					shots=shots,
					rep_delay=self.rep_delay,
					job_tags=["M3 calibration"],
					calibration_set_id=self._cal_id,
				)
			else:
				_job = self.system.run(circs, shots=shots, rep_delay=self.rep_delay, job_tags=["M3 calibration"])
			jobs.append(_job)

		# Track which qubits were calibrated BEFORE calling _job_thread
		if hasattr(self, "_calibrated_qubits") and self._calibrated_qubits is not None:
			all_qubits = set(self._calibrated_qubits) | set(qubits)
			self._calibrated_qubits = sorted(all_qubits)
		else:
			self._calibrated_qubits = sorted(qubits)

		# Execute job and cal building in new thread.
		# KEY CHANGE: Use _iqm_job_thread instead of M3's _job_thread
		self._job_error = None
		if async_cal:
			thread = threading.Thread(target=_iqm_job_thread, args=(jobs, self, qubits, num_cal_qubits, generator))
			self._thread = thread  # type: ignore[assignment]
			self._thread.start()  # type: ignore[union-attr]
		else:
			_iqm_job_thread(jobs, self, qubits, num_cal_qubits, generator)

		return jobs

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


def _iqm_job_thread(jobs, mit, qubits, num_cal_qubits, generator):
	"""Custom job thread for IQM backends that accounts for bit-string reversal.

	IQM's qiskit integration reverses bit strings when formatting results (see iqm_job.py:145).
	This conflicts with M3's balanced calibration which also does reversal.
	This function removes one reversal layer to make it work correctly.

	Parameters:
		jobs (list): A list of job instances
		mit (M3Mitigator): The mitigator instance
		qubits (list): List of qubits used
		num_cal_qubits (int): Number of calibration qubits
		generator (None or list): Generator for bit-arrays for balanced cals
	"""
	import datetime

	counts = []
	timestamp = None
	for job in jobs:
		result = job.result()
		if timestamp is None:
			timestamp = result.date
		for exp in result.results:
			counts.append(exp.data.counts)

	logger.info("All jobs are done.")

	# Handle timestamp
	if timestamp is None:
		timestamp = datetime.datetime.now()
	if isinstance(timestamp, datetime.datetime):
		dt = timestamp
	else:
		dt = datetime.datetime.fromisoformat(timestamp)

	# Convert to UTC
	try:
		dt_utc = dt.astimezone(datetime.UTC)
	except ValueError:
		dt_utc = dt.replace(tzinfo=datetime.UTC)

	mit.cal_timestamp = dt_utc.isoformat()

	# A list of qubits with bad meas cals
	bad_list = []

	if mit.cal_method == "independent":
		# Independent calibration - works as-is, no bit reversal issues
		for idx, qubit in enumerate(qubits):
			mit.single_qubit_cals[qubit] = np.zeros((2, 2), dtype=np.float32)
			# Counts 0 has all P00, P10 data
			prep0_counts = counts[2 * idx]
			P10 = prep0_counts.get("1", 0) / mit.cal_shots
			P00 = 1 - P10
			mit.single_qubit_cals[qubit][:, 0] = [P00, P10]
			# Counts 1 has all P01, P11 data
			prep1_counts = counts[2 * idx + 1]
			P01 = prep1_counts.get("0", 0) / mit.cal_shots
			P11 = 1 - P01
			mit.single_qubit_cals[qubit][:, 1] = [P01, P11]
			if P01 >= P00:
				bad_list.append(qubit)

	elif mit.cal_method == "marginal":
		# Marginal calibration
		prep0_counts = counts[0]
		prep1_counts = counts[1]
		for idx, qubit in enumerate(qubits):
			bit0 = format(qubit, f"0{num_cal_qubits}b")
			P10 = prep0_counts.get(bit0, 0) / mit.cal_shots
			P00 = 1 - P10
			P01 = prep1_counts.get(bit0, 0) / mit.cal_shots
			P11 = 1 - P01
			mit.single_qubit_cals[qubit] = np.asarray([[P00, P01], [P10, P11]], dtype=np.float32)
			if P01 >= P00:
				bad_list.append(qubit)

	# Balanced calibration - Use M3's logic exactly
	else:
		cals = [np.zeros((2, 2), dtype=np.float32) for kk in range(num_cal_qubits)]

		for idx, target in enumerate(generator):
			count = counts[idx]
			good_prep = np.zeros(num_cal_qubits, dtype=np.float32)
			# divide by 2 since total shots is double
			denom = mit._balanced_shots / 2
			# Reverse target to match classical bit ordering
			target = target[::-1]
			for key, val in count.items():
				# Reverse key to match classical bit ordering
				key = key[::-1]
				for kk in range(num_cal_qubits):
					if int(key[kk]) == target[kk]:
						good_prep[kk] += val

			for kk, cal in enumerate(cals):
				if target[kk] == 0:
					cal[0, 0] += good_prep[kk] / denom
				else:
					cal[1, 1] += good_prep[kk] / denom

		for cal in cals:
			cal[1, 0] = 1.0 - cal[0, 0]
			cal[0, 1] = 1.0 - cal[1, 1]

		for idx, cal in enumerate(cals):
			mit.single_qubit_cals[qubits[idx]] = cal

	# save cals to file, if requested
	if mit.cals_file:
		mit.cals_to_file(mit.cals_file)

	# faulty qubits, if any
	mit.faulty_qubits = _faulty_qubit_checker(mit.single_qubit_cals)
