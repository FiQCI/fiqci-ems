""" 
For now just to test how BaseSampler works.
Only wraps FiQCIBackend and exposes a run method that calls the backend's run method.
"""

from qiskit.primitives import BaseSamplerV2
from fiqci.ems import FiQCIBackend

class FiQCISampler(BaseSamplerV2):
    def __init__(self, backend, mitigation_level=1, calibration_shots=1000, calibration_files=None):
        super().__init__()
        self.backend = FiQCIBackend(backend, mitigation_level, calibration_shots, calibration_files)

    def _run(self, circuits, shots=2048, **options):
        return self.backend.run(circuits, shots=shots, **options)
    
    def run(self, circuits, **options):
        return self._run(circuits, **options)