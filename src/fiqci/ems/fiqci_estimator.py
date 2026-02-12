""" 
For now just to test how BaseEstimator works.
Only wraps FiQCIBackend and exposes a run method that calls the backend's run method.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.providers import JobV1
from qiskit.result import Result
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimatorV2
from fiqci.ems import FiQCIBackend
from fiqci.ems.basis_measurement import _get_obs_subcircuits

from typing import Any


class FiQCIEstimator(BaseEstimatorV2):
    def __init__(self, backend, mitigation_level=1, calibration_shots=1000, calibration_files=None):
        super().__init__()
        self.backend = FiQCIBackend(backend, mitigation_level, calibration_shots, calibration_files)

    def _run(self, circuits, observables, shots=2048, **options):

        x_meas = QuantumCircuit(1)
        x_meas.h(0)
        x_meas = transpile(x_meas, basis_gates=list(self.backend.target.operation_names))
        x_meas = x_meas.to_instruction(label="X-meas")

        y_meas = QuantumCircuit(1)
        y_meas.sdg(0)
        y_meas.h(0)
        y_meas = transpile(y_meas, basis_gates=list(self.backend.target.operation_names))
        y_meas = y_meas.to_instruction(label="Y-meas")

        ops = {
            "X-meas": x_meas,
            "Y-meas": y_meas,
        }

        measurement_settings = self._combine_pauli_ops(observables)

        obs_circuits = _get_obs_subcircuits([circuits], measurement_settings, ops)

        obs_circs_list = [obs_circuits[i][0] for i in range(len(obs_circuits))] #TODO handle multiple input circuits, with multiple input circuits return value becomes even more complicated :/
        
        job = self.backend.run(obs_circs_list, shots=shots, **options)

        results = job.result()

        counts = results.get_counts()

        expectation_values = self.calculate_expectation_values(counts, observables, measurement_settings)

        return FiQCIEstimatorJob(job, expectation_values, observables)
    
    def run(self, circuits, observables, shots=2048, **options):
        return self._run(circuits, observables, shots=shots, **options)
    
    def _get_observable_circuit_index(self, pauli, combined: list[dict[int, str]]):
        """Find which measurement setting covers the non-identity letters of `pauli`,
        and return the indices of the qubits involved."""
        label = pauli
        non_identity = {i: p for i, p in enumerate(label) if p.to_label() != "I"}

        for idx, setting in enumerate(combined):
            # All non-identity qubits must be measured in the matching basis
            if all(setting.get(q) == p.to_label() for q, p in non_identity.items()):
                return {"circuit_index": idx, "obs_indices": list(range(len(non_identity))), "num_meas": len(non_identity)}

        return {"circuit_index": None, "obs_indices": [], "num_meas": 0}
    
    def calculate_expectation_values(self, counts, obs, measurement_settings):
        if not isinstance(counts, list):
            counts = [counts]
        expectation_values = []
        for pauli in obs.paulis:
            obs_info = self._get_observable_circuit_index(pauli, measurement_settings)
            if obs_info["circuit_index"] is not None:
                circuit_counts = counts[obs_info["circuit_index"]]
                # Calculate expectation value from counts
                exp_val = 0
                for bitstring, count in circuit_counts.items():
                    parity = 1
                    for idx in obs_info["obs_indices"]:
                        if bitstring[idx] == '1':
                            parity *= -1
                    exp_val += parity * count
                exp_val /= sum(circuit_counts.values())
                expectation_values.append(exp_val)
            else:
                expectation_values.append(0)  # No measurement setting covers this observable
        return expectation_values
    
    def _combine_pauli_ops(self, op: SparsePauliOp) -> list[dict[int, str]]:  # noqa: C901
        """Combine Pauli operators that have no conflicting non-identity components.
        
        Args:
            op (SparsePauliOp): The SparsePauliOp to analyze.
        
        Returns:
            list[dict[int, str]]: A list of combined measurement settings, where each dict
                                maps qubit indices to Pauli basis measurements.
        """

        pauli_strings = [pauli.to_label()[::-1] for pauli in op.paulis]
        
        combined_settings = []
        used = [False] * len(pauli_strings)
        
        for i, pauli_string in enumerate(pauli_strings):
            if used[i]:
                continue
            
            # Start a new combined setting with the current Pauli string
            combined = {}
            for qubit_index, pauli in enumerate(pauli_string):
                if pauli != "I":
                    combined[qubit_index] = pauli
            
            used[i] = True
            
            # Try to combine with remaining Pauli strings
            for j in range(i + 1, len(pauli_strings)):
                if used[j]:
                    continue
                
                # Check if pauli_strings[j] can be combined with current combined setting
                can_combine = True
                for qubit_index, pauli in enumerate(pauli_strings[j]):
                    if pauli != "I":
                        if qubit_index in combined and combined[qubit_index] != pauli:
                            can_combine = False
                            break
                
                # If compatible, add to combined setting
                if can_combine:
                    for qubit_index, pauli in enumerate(pauli_strings[j]):
                        if pauli != "I":
                            combined[qubit_index] = pauli
                    used[j] = True
            
            combined_settings.append(combined)
        
        return combined_settings

class FiQCIEstimatorJob:
    """Wrapper for job results with mitigated data.

	This class wraps the original job and provides access to mitigated results.
	"""
    def __init__(self, mitigated_job, expectation_values, observables) -> None:
        """Initialize mitigated job wrapper.

        Args:
            original_job: Original job from backend.
            mitigated_result: Result object with mitigated counts.
        """
        self._original_job = mitigated_job._original_job
        self._mitigated_result = mitigated_job._mitigated_result
        self._expectation_values = expectation_values
        self._observables = observables

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
    
    def expectation_values(self, index: int | None = None) -> list[float]:
        """Get the calculated expectation values."""
        if index is not None:
            return self._expectation_values[index]
        return self._expectation_values
    
    def observables(self, index: int | None = None) -> SparsePauliOp:
        """Get the observables for which expectation values were calculated."""
        if index is not None:
            return self._observables[index]
        return self._observables