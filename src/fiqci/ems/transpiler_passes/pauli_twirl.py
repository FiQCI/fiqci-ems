from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister, Gate
from qiskit.circuit.library import CZGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import Operator, pauli_basis
 
import numpy as np
 
from typing import Iterable, Optional

class PauliTwirl(TransformationPass):
    """Add Pauli twirls to two-qubit gates."""
 
    def __init__(
        self,
        gates_to_twirl: Optional[Iterable[Gate]] = None,
    ):
        """
        Args:
            gates_to_twirl: Names of gates to twirl. The default behavior is to twirl all
                two-qubit basis gates.
        """
        if gates_to_twirl is None:
            gates_to_twirl = [CZGate()]
        self.gates_to_twirl = gates_to_twirl
        self.build_twirl_set()
        super().__init__()
 
    def build_twirl_set(self):
        """
        Build a set of Paulis to twirl for each gate and store internally as .twirl_set.
        """
        self.twirl_set = {}
 
        # iterate through gates to be twirled
        for twirl_gate in self.gates_to_twirl:
            twirl_list = []
 
            # iterate through Paulis on left of gate to twirl
            for pauli_left in pauli_basis(2):
                # iterate through Paulis on right of gate to twirl
                for pauli_right in pauli_basis(2):
                    # save pairs that produce identical operation as gate to twirl
                    if (Operator(pauli_left) @ Operator(twirl_gate)).equiv(
                        Operator(twirl_gate) @ pauli_right
                    ):
                        twirl_list.append((pauli_left, pauli_right))
 
            self.twirl_set[twirl_gate.name] = twirl_list
 
    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        # collect all nodes in DAG and proceed if it is to be twirled
        twirling_gate_classes = tuple(
            gate.base_class for gate in self.gates_to_twirl
        )
        for node in dag.op_nodes():
            if not isinstance(node.op, twirling_gate_classes):
                continue
 
            # random integer to select Pauli twirl pair
            pauli_index = np.random.randint(
                0, len(self.twirl_set[node.op.name])
            )
            twirl_pair = self.twirl_set[node.op.name][pauli_index]
 
            # instantiate mini_dag and attach quantum register
            mini_dag = DAGCircuit()
            register = QuantumRegister(2)
            mini_dag.add_qreg(register)
 
            # apply left Pauli, gate to twirl, and right Pauli to empty mini-DAG
            mini_dag.apply_operation_back(
                twirl_pair[0].to_instruction(), [register[0], register[1]]
            )
            mini_dag.apply_operation_back(node.op, [register[0], register[1]])
            mini_dag.apply_operation_back(
                twirl_pair[1].to_instruction(), [register[0], register[1]]
            )
 
            # substitute gate to twirl node with twirling mini-DAG
            dag.substitute_node_with_dag(node, mini_dag)
 
        return dag
    
def get_twirled_circuits(
    circuits: list[QuantumCircuit],  # list of QuantumCircuits to generate twirled circuits from
    num_twirls: int,  # number of twirled circuits to generate per input circuit
    gates_to_twirl: Optional[Iterable[Gate]] = None,  # optional list of gate names to twirl, if None, all two-qubit basis gates will be twirled
) -> tuple[list[QuantumCircuit], list[list[int]]]:
    """
    Generate twirled circuits from input circuits.

    Args:
        circuits: List of QuantumCircuits to generate twirled circuits from.
        num_twirls: Number of twirled circuits to generate per input circuit.
        gates_to_twirl: Optional list of gate names to twirl, if None, all two-qubit basis gates will be twirled.
    Returns:
        Tuple of (list of twirled QuantumCircuits, list of circuit groups).
    """
    twirled_circuits = []
    
    circuit_groups = []

    pm = PassManager(PauliTwirl(gates_to_twirl=gates_to_twirl))

    for circuit in circuits:

        start_idx = len(twirled_circuits)
        
        twirled_circuit = [pm.run(circuit) for _ in range(num_twirls)]
        twirled_circuits.append(circuit) # include original circuit in list of twirled circuits
        twirled_circuits.extend(twirled_circuit)

        circuit_groups.append(list(range(start_idx, len(twirled_circuits))))
    return twirled_circuits, circuit_groups