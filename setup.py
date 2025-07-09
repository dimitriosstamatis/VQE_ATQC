import qiskit_nature.second_q.drivers as sqd
import qiskit_nature.second_q.mappers as sqm
import qiskit_algorithms as qka

def check_commutativity(pauli_1, pauli_2):
    '''Checks if two Pauli strings are qubit wise commuting'''
    for p_1, p_2 in zip(str(pauli_1), str(pauli_2)):
        if p_1 != 'I' and p_2 != 'I' and p_1 != p_2:
            return False
        
    return True

def group_pauli_strings(hamiltonian):
    '''Groups Hamiltonian terms into qubit wise commuting sets'''
    pauli_strings = list(zip(hamiltonian.paulis, hamiltonian.coeffs))
    groups = []

    for pauli, coeff in pauli_strings:
        in_group = False
        for group in groups:
            if all(check_commutativity(pauli, pauli_in_group) for pauli_in_group, _ in group):
                group.append((pauli, coeff))
                in_group = True
                break
        if not in_group:
            groups.append([(pauli, coeff)])
    
    print(f"Hamiltonian with {len(pauli_strings)} terms was reduced to {len(groups)} groups")
    return groups

def setup(seperation):
    '''Initializes Hamiltonian for HeH+ molecule'''
    # Define molecule using PySCF driver
    driver = sqd.PySCFDriver(atom=f'He 0 0 0; H 0 0 {seperation}', charge=1)
    driver_run = driver.run()
    second_q_hamiltonian = driver_run.hamiltonian.second_q_op()

    # Map fermionic Hamiltonian to qubit Hamiltonian
    mapper = sqm.JordanWignerMapper()
    qubit_hamiltonian = mapper.map(second_q_hamiltonian)

    # Group Pauli strings
    pauli_groups = group_pauli_strings(qubit_hamiltonian)
    
    # Calculate exact energy
    numpy_solver = qka.minimum_eigensolvers.NumPyMinimumEigensolver()
    numpy_solver_result = numpy_solver.compute_minimum_eigenvalue(qubit_hamiltonian)

    print(f'Exact energy: {numpy_solver_result.eigenvalue.real:.10f}')
    return qubit_hamiltonian, pauli_groups, numpy_solver_result.eigenvalue.real