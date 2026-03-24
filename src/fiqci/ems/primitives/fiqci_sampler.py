"""
A lightweight wrapper around FiQCIBackend for sampling quantum circuits with error mitigation.

FiQCISampler provides a simple interface for running circuits and obtaining mitigated measurement
counts without needing to configure the backend directly. It applies readout error mitigation
based on the chosen mitigation level and chosen settings, so users get improved sampling results
with minimal setup.
"""

from fiqci.ems import FiQCIBackend


class FiQCISampler:
	def __init__(self, backend, mitigation_level=1, calibration_shots=1000, calibration_files=None):
		super().__init__()
		self.backend = FiQCIBackend(backend, mitigation_level, calibration_shots, calibration_files)

	def _run(self, circuits, shots=2048, **options):
		return self.backend.run(circuits, shots=shots, **options)

	def run(self, circuits, **options):
		return self._run(circuits, **options)

	def rem(self, enable, calibration_shots=1000, calibration_file=None):
		self.backend.rem(enable, calibration_shots, calibration_file)

	def mitigator_options(self):
		"""Get current mitigator settings."""
		return {**self.backend.mitigator_options()}
