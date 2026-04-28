"""
A lightweight wrapper around FiQCIBackend for sampling quantum circuits with error mitigation.
"""

from typing import Any

from fiqci.ems import FiQCIBackend
from qiskit import QuantumCircuit
from qiskit.providers import JobV1
from fiqci.ems.fiqci_backend import MitigatedJob
from fiqci.ems.mitigators.dd import DDGateSequenceEntry


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
		"""Get current mitigator settings."""
		return {**self.backend.mitigator_options}

	def _run(
		self, circuits: QuantumCircuit | list[QuantumCircuit], shots: int = 2048, **options
	) -> JobV1 | MitigatedJob:
		return self.backend.run(circuits, shots=shots, **options)

	def run(
		self, circuits: QuantumCircuit | list[QuantumCircuit], shots: int = 2048, **options
	) -> JobV1 | MitigatedJob:
		"""
		Execute the given circuits on the backend and return mitigated measurement counts.

		Args:
			circuits: A QuantumCircuit or list of QuantumCircuits to execute.
			shots: Number of shots to execute each circuit (default: 2048).
			**options: Additional options to pass to the backend's run method.

		Returns:
			A JobV1 or MitigatedJob instance containing the results of the execution.
		"""
		return self._run(circuits, shots, **options)

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
