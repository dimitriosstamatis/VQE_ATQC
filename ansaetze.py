import qiskit.circuit as qkc
import qiskit_nature.second_q.circuit.library as sqc

def create_hea_ansatz(num_qubits, num_layers):
    '''Creates a hardware efficient ansatz'''
    num_rgates = 2
    num_params = num_qubits * num_rgates * num_layers

    he_ansatz = qkc.QuantumCircuit(num_qubits)

    # Initialize circuit in Hartree-Fock state
    he_ansatz.x(0)
    he_ansatz.x(1)

    params = qkc.ParameterVector('theta', num_params)
    param_idx = 0

    # Build alternating layers of rotation and entangling gates
    for _ in range(num_layers):
        # rotation gates
        for i in range (num_qubits):
            he_ansatz.rx(params[param_idx], i)
            param_idx += 1
        for i in range(num_qubits):
            he_ansatz.ry(params[param_idx], i)
            param_idx += 1
        # entangling gates
        for i in range(num_qubits - 1):
            he_ansatz.cz(i, i + 1)

    return he_ansatz

def create_ucc_ansatz(driver_run, mapper):
    '''Creates a Unitary Coupled Cluster (UCC) ansatz'''
    num_spatial_orbitals = driver_run.num_spatial_orbitals
    num_particles = driver_run.num_particles

    ucc_ansatz = sqc.UCC(num_spatial_orbitals, num_particles, mapper)

    return ucc_ansatz
