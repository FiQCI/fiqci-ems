"""Functions related to Readout Error Mitigation (REM)."""

import logging
import threading
import warnings
from collections.abc import Iterable, Mapping
from math import ceil
from typing import Any

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from mthree import M3Mitigation
from mthree.circuits import _marg_meas_states, _tensor_meas_states, balanced_cal_circuits
from mthree.classes import QuasiCollection
from mthree.exceptions import M3Error
from mthree.generators import HadamardGenerator
from mthree.mitigation import _job_thread
from mthree.utils import final_measurement_mapping
from qiskit import QuantumCircuit, transpile
from qiskit.providers import BackendV2
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

	def cals_from_system(  # type: ignore[override]
		self,
		qubits=None,
		shots=None,
		method=None,
		initial_reset=False,
		rep_delay=None,
		cals_file=None,
		async_cal=False,
		cal_id=None,
	):
		"""Grab calibration data from system.

		Parameters:
			qubits (array_like): Qubits over which to correct calibration data. Default is all.
			shots (int): Number of shots per circuit. min(1e4, max_shots).
			method (str): Type of calibration, 'balanced' (default for hardware),
						 'independent' (default for simulators), or 'marginal'.
			initial_reset (bool): Use resets at beginning of calibration circuits, default=False.
			rep_delay (float): Delay between circuits on IBM Quantum backends.
			cals_file (str): Output path to write JSON calibration data to.
			async_cal (bool): Do calibration async in a separate thread, default is False.

		Raises:
			M3Error: Called while a calibration currently in progress.
		"""
		if self._thread:
			raise M3Error("Calibration currently in progress.")
		if qubits is None:
			qubits = range(self.num_qubits)  # type: ignore[arg-type]
			# Remove faulty qubits if any
			if any(self.system_info["inoperable_qubits"]):
				inoperable = self.system_info["inoperable_qubits"]
				qubits = list(
					filter(lambda item: item not in inoperable, list(range(self.num_qubits)))  # type: ignore[arg-type]
				)
				warnings.warn(
					f"Backend reporting inoperable qubits. Skipping calibrations for: {self.system_info['inoperable_qubits']}",
					UserWarning,
					stacklevel=2,
				)

		if method is None:
			method = "balanced"
			# if self.system_info["simulator"]:
			#     method = "independent"
		self.cal_method = method
		self.rep_delay = rep_delay
		self.cals_file = cals_file
		self.cal_timestamp = None
		self._grab_additional_cals(
			qubits,
			shots=shots,
			method=method,
			rep_delay=rep_delay,
			initial_reset=initial_reset,
			async_cal=async_cal,
			cal_id=cal_id,
		)

	def _grab_additional_cals(  # type: ignore[override]
		self, qubits, shots=None, method="balanced", rep_delay=None, initial_reset=False, async_cal=False, cal_id=None
	):
		"""Grab missing calibration data from backend.

		Parameters:
			qubits (array_like): List of measured qubits.
			shots (int): Number of shots to take, min(1e4, max_shots).
			method (str): Type of calibration, 'balanced' (default), 'independent', or 'marginal'.
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
		if self.cal_shots is None:
			if shots is None:
				shots = min(self.system_info["max_shots"], 10000)
			self.cal_shots = shots  # type: ignore[assignment]
		if self.rep_delay is None:
			self.rep_delay = rep_delay

		if method not in ["independent", "balanced", "marginal"]:
			raise M3Error(
				f"Invalid calibration method: {method}. Valid methods are 'independent', 'balanced', or 'marginal'."
			)

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
		cal_strings = []
		# shots is needed here because balanced cals will use a value
		# different from cal_shots
		shots = self.cal_shots
		if method == "marginal":
			trans_qcs = _marg_meas_states(qubits, self.num_qubits, initial_reset=initial_reset)
		elif method == "balanced":
			cal_strings = balanced_cal_strings(num_cal_qubits)
			generator = HadamardGenerator(num_cal_qubits)
			trans_qcs = balanced_cal_circuits(generator, qubits, self.num_qubits, initial_reset=initial_reset)
			shots = self.cal_shots // num_cal_qubits  # type: ignore[operator]
			if self.cal_shots / num_cal_qubits != shots:  # type: ignore[operator]
				shots += 1
			self._balanced_shots = shots * num_cal_qubits  # type: ignore[assignment]
		# Independent
		else:
			trans_qcs = []
			for qubit in qubits:
				trans_qcs.extend(_tensor_meas_states(qubit, self.num_qubits, initial_reset=initial_reset))

		num_circs = len(trans_qcs)

		# Check for max number of circuits per job
		if isinstance(self.system, BackendV2):
			max_circuits = self.system.max_circuits
			# Needed for https://github.com/Qiskit/qiskit-terra/issues/9947
			if max_circuits is None:
				max_circuits = 300
		else:
			# For BackendV1 or other backend types
			max_circuits = getattr(self.system, "max_circuits", 300)

		# Determine the number of jobs required
		num_jobs = ceil(num_circs / max_circuits)
		# Get the slice length
		circ_slice = ceil(num_circs / num_jobs)
		# Split circuits into chunks for each job
		circs_list = []
		for kk in range(num_jobs):
			start_idx = kk * circ_slice
			end_idx = (kk + 1) * circ_slice if kk < num_jobs - 1 else None
			circs_list.append(trans_qcs[start_idx:end_idx])

		# Do job submission
		jobs = []
		for circs in circs_list:
			transpiled_circuit = transpile(circs, self.system, optimization_level=0)
			if cal_id is None:
				_job = self.system.run(transpiled_circuit, shots=shots, rep_delay=self.rep_delay)
			else:
				_job = self.system.run(
					transpiled_circuit, shots=shots, rep_delay=self.rep_delay, calibration_set_id=cal_id
				)
			jobs.append(_job)
			logger.info("REM: %d calibration circuits to be executed!", len(circs))

		# Execute job and cal building in new thread
		self._job_error = None
		if async_cal:
			thread = threading.Thread(target=_job_thread, args=(jobs, self, qubits, num_cal_qubits, cal_strings))
			self._thread = thread  # type: ignore[assignment]
			self._thread.start()  # type: ignore[union-attr]
		else:
			_job_thread(jobs, self, qubits, num_cal_qubits, cal_strings)


def readout_error_m3(
	counts: Mapping[str, int | float], mit: M3IQM, qubits: Iterable[int] | dict[Any, int]
) -> QuasiCollection:
	"""Apply M3IQM readout error mitigation to measurement counts.

	Args:
		counts: Dictionary of measurement outcomes and their counts/probabilities.
		mit: Initialized M3IQM mitigation object with calibration data.
		qubits: Either a dictionary from `mthree.utils.final_measurement_mapping`
			or an iterable of qubit indices.

	Returns:
		QuasiCollection object containing mitigated quasiprobabilities.
	"""
	return mit.apply_correction(counts, qubits)  # type: ignore[return-value]


def apply_readout_error_mitigation(
	backend: IQMBackendBase,
	transpiled_circuits: list[QuantumCircuit],
	counts: list[dict[str, int]],
	mit_shots: int = 1000,
) -> list[QuasiCollection]:
	"""Apply readout error mitigation to measurement results.

	This function initializes an M3IQM mitigator, calibrates it against the specified
	backend, and applies correction to the provided measurement counts.

	Args:
		backend: An IQMBackendBase instance to calibrate against.
		transpiled_circuits: List of transpiled quantum circuits.
		counts: List of measurement count dictionaries corresponding to the circuits.
		mit_shots: Number of shots per calibration circuit. Default is 1000.

	Returns:
		List of QuasiCollection objects containing mitigated quasiprobabilities
		for each circuit's measurement outcomes.
	"""
	# M3IQM uses mthree.mitigation, which displays many INFO messages

	# Initialize with the given system and get calibration data
	qubits_rem = [final_measurement_mapping(c) for c in transpiled_circuits]

	mit = M3IQM(backend)
	mit.cals_from_system(qubits_rem, shots=mit_shots)

	# Apply the REM correction to the given measured counts
	rem_quasidistro = [mit.apply_correction(c, q) for c, q in zip(counts, qubits_rem, strict=True)]

	return rem_quasidistro  # type: ignore[return-value]
