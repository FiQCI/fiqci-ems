"""
A lightweight wrapper around FiQCIBackend for sampling quantum circuits with error mitigation.
"""

import logging
from typing import Any

from fiqci.ems import FiQCIBackend
from qiskit import QuantumCircuit
from fiqci.ems.backend import BatchedJob, MitigatedJob
from fiqci.ems.mitigators.dd import DDGateSequenceEntry

logger: logging.Logger = logging.getLogger(__name__)


class FiQCISampler:
	"""
	FiQCISampler provides a simple interface for running circuits and obtaining mitigated measurement
	counts without needing to configure the backend directly. It applies readout error mitigation
	based on the chosen mitigation level and chosen settings, so users get improved sampling results
	with minimal setup.

	Mitigation levels:
		- 0: No error mitigation (raw results)
		- 1: Readout error mitigation using M3 (default)
		- 2: Level 1 + dynamical decoupling (DD)
		- 3: Level 2 + Pauli twirling

	Args:
		backend: An IQMBackendBase instance to wrap.
		mitigation_level: Level of error mitigation to apply (default: 1).
		calibration_shots: Number of shots to use for calibration circuits (default: 1000).
		calibration_file: Optional calibration file to use for readout error mitigation.
	"""

	def __init__(self, backend, mitigation_level=1, calibration_shots=1000, calibration_file=None):
		super().__init__()
		self.backend = FiQCIBackend(backend, mitigation_level, calibration_shots, calibration_file)

	@property
	def mitigator_options(self) -> dict[str, Any]:
		"""Get current mitigator settings.

		The returned dict is a copy, so mutating it does not change the sampler's configuration;
		use :meth:`rem` / :meth:`dd` / :meth:`pauli_twirl`, which validate their input.
		"""
		return {**self.backend.mitigator_options}

	def total_circuits_generated(self, num_base_circuits: int, detailed: bool = False) -> int | dict[str, int]:
		"""Calculate total circuits generated for a given number of base circuits and observables."""
		return self.backend.total_circuits_generated(num_base_circuits, detailed)

	def _run(
		self, circuits: QuantumCircuit | list[QuantumCircuit], shots: int = 2048, max_batch_size: int = 100, **options
	) -> MitigatedJob | BatchedJob:
		num_circuits = len(circuits) if isinstance(circuits, list) else 1
		logger.info(
			"FiQCISampler.run: %d input circuit(s), shots=%d, max_batch_size=%d", num_circuits, shots, max_batch_size
		)
		return self.backend.run(circuits, shots=shots, max_batch_size=max_batch_size, **options)

	def run(
		self, circuits: QuantumCircuit | list[QuantumCircuit], shots: int = 2048, max_batch_size: int = 100, **options
	) -> MitigatedJob | BatchedJob:
		"""
		Execute the given circuits on the backend and return mitigated measurement counts.

		Args:
			circuits: A QuantumCircuit or list of QuantumCircuits to execute.
			shots: Number of shots to execute each circuit (default: 2048).
			max_batch_size: Maximum number of circuits per backend job. Inputs longer than this are split
				into multiple jobs whose results are combined into a single Result indexed in submission
				order (default: 100).
			**options: Additional options to pass to the backend's run method.

		Returns:
			A lazy job handle (``BatchedJob`` at level 0, or a ``MitigatedJob`` view at level 1+).
			It is returned immediately; ``job_ids()``/``status()`` are available right away and the
			combined/mitigated counts are computed on the first ``result()`` call.
		"""
		return self._run(circuits, shots, max_batch_size=max_batch_size, **options)

	def rem(self, enabled: bool, calibration_shots: int = 1000, calibration_file: str | None = None) -> None:
		"""
		Set readout error mitigation settings for the sampler. This will configure the underlying backend's
		readout error mitigation accordingly.

		Args:
			enabled: Whether to enable readout error mitigation.
			calibration_shots: Number of shots to use for calibration circuits (default: 1000).
			calibration_file: Optional calibration file to use for readout error mitigation.
		"""
		self.backend.rem(enabled, calibration_shots, calibration_file)

	def dd(self, enabled: bool, gate_sequences: list[DDGateSequenceEntry] | None = None) -> None:
		"""
		Set dynamical decoupling settings for the sampler. This will configure the underlying backend's
		dynamical decoupling accordingly.

		Args:
			enabled: Whether to enable dynamical decoupling.
			gate_sequences: List of (threshold_length, sequence, strategy) tuples defining DD behavior.
				See build_dd_options for details on each field.
		"""
		self.backend.dd(enabled, gate_sequences)

	def pauli_twirl(self, enabled: bool, num_twirls: int = 10, gates_to_twirl: list | None = None) -> None:
		"""Configure Pauli twirling settings for the estimator."""
		self.backend.pauli_twirl(enabled, num_twirls, gates_to_twirl)
