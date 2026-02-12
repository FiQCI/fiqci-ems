from __future__ import annotations

from copy import deepcopy

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import (
    Instruction,
)
from qiskit.circuit.library import HGate, SdgGate, Measure
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass

from qiskit.transpiler.passes import RemoveFinalMeasurements



class _ModifyMeasurementBasis(TransformationPass):
 
    def __init__(
        self,
        measurement_settings: list[dict[int, str]],
        ops: dict[str, Instruction] | None = None,

    ):
        
        self.measurement_settings = measurement_settings
        self.ops = ops
        super().__init__()
 
    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        
        cloned_dag = deepcopy(dag)

        cloned_dag.add_creg(ClassicalRegister(len(self.measurement_settings[0]), name="meas"))

        clbit_index = 0

        for setting in self.measurement_settings:
            for qubit, basis in setting.items():
                if basis == "X":
                    if self.ops and "X-meas" in self.ops:
                        cloned_dag.apply_operation_back(
                            self.ops["X-meas"], [cloned_dag.qubits[qubit]]
                        )
                    else:
                        cloned_dag.apply_operation_back(
                            HGate(), [cloned_dag.qubits[qubit]]
                        )
                    cloned_dag.apply_operation_back(Measure(), [cloned_dag.qubits[qubit]], [cloned_dag.clbits[clbit_index]])
                    clbit_index += 1
                elif basis == "Y":
                    if self.ops and "Y-meas" in self.ops:
                        cloned_dag.apply_operation_back(
                            self.ops["Y-meas"], [cloned_dag.qubits[qubit]]
                        )
                    else:
                        cloned_dag.apply_operation_back(
                            SdgGate(), [cloned_dag.qubits[qubit]]
                        )
                        cloned_dag.apply_operation_back(
                            HGate(), [cloned_dag.qubits[qubit]]
                        )
                    cloned_dag.apply_operation_back(Measure(), [cloned_dag.qubits[qubit]], [cloned_dag.clbits[clbit_index]])
                    clbit_index += 1
                elif basis == "Z":
                    cloned_dag.apply_operation_back(Measure(), [cloned_dag.qubits[qubit]], [cloned_dag.clbits[clbit_index]])
                    clbit_index += 1
                    # No change needed for Z-basis measurement
                    pass
                else:
                    raise ValueError(f"Unsupported measurement basis: {basis}")
                
        return cloned_dag
    

def _get_obs_subcircuits(subcircuits: list[QuantumCircuit], 
                        measurement_settings: list[dict[int, str]],
                        ops: dict[str, Instruction] | None = None
                        ) -> list[dict[int, QuantumCircuit]]:
    pms = [PassManager([_ModifyMeasurementBasis([setting], ops)]) 
           for setting in measurement_settings]
    
    remove_meas_pm = PassManager([RemoveFinalMeasurements()])

    obs_subcircuits = []
    for pm in pms:
        pm_circs = {}
        for ind, subcircuit in enumerate(subcircuits):
            modified_circuit = pm.run(remove_meas_pm.run(subcircuit)).decompose(gates_to_decompose=["X-meas", "Y-meas"])  # Decompose custom measurement
            if modified_circuit.num_qubits == 0:
                continue
            pm_circs[ind] = modified_circuit
        obs_subcircuits.append(pm_circs)
    return obs_subcircuits