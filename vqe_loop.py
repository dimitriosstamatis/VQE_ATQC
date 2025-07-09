import qiskit as qk
import qiskit.circuit as qkc
import qiskit_aer as aer
import numpy as np
import optimizer as op

simulator = aer.AerSimulator()
np.random.seed(21)

def vqe_step(params, pauli_groups, ansatz):
    '''Calculates expectation value using Pauli grouping'''
    weighted_sum = 0.0
    
    # Iterate over each group of Pauli strings
    for group in pauli_groups:
        # Create a rotation circuit based on the first operator in the group
        rotation_circuit = qkc.QuantumCircuit(ansatz.num_qubits)
        reference_pauli = group[0][0]
        
        for i, pauli in enumerate(str(reference_pauli)):
            if pauli == 'X':
                rotation_circuit.h(i)
            elif pauli == 'Y':
                rotation_circuit.sdg(i)
                rotation_circuit.h(i)
        
        # Combine ansatz and rotation
        composed_circuit = ansatz.compose(rotation_circuit)
        composed_circuit.measure_all()

        # Run simulation
        parameterized_circuit = composed_circuit.assign_parameters(params)
        result = simulator.run(parameterized_circuit, shots=8192).result()
        shot_counts = result.get_counts()
        total_shots = sum(shot_counts.values())

        # Calculate expectation value for each Pauli string in the group
        for pauli, coeff in group:
            expectation_value = 0.0
            
            for bitstring, num_shots in shot_counts.items():
                parity = sum(int(b) 
                             for i, b in enumerate(reversed(bitstring)) 
                             if str(pauli)[i] != 'I')
                sign = (-1)**parity
                expectation_value += sign * (num_shots / total_shots)
            
            weighted_sum += coeff.real * expectation_value
            
    return weighted_sum

def run_vqe(num_params, pauli_groups, ansatz, optimizer, n_iterations):
    '''Runs VQE optimization'''
    current_params = np.random.rand(num_params) * 2 * np.pi

    # Initialize optimizer state variables
    if optimizer == 1:
        E_g2 = np.zeros_like(current_params)
    elif optimizer == 2:
        m = np.zeros_like(current_params)
        v = np.zeros_like(current_params)

    energy_list = []

    for i in range(n_iterations):
        # Calculate gradient using the parameter shift rule
        gradient = np.zeros_like(current_params)

        for k in range(len(current_params)):
            params_plus = current_params.copy()
            params_plus[k] += np.pi / 2
            energy_plus = vqe_step(params_plus, pauli_groups, ansatz)

            params_minus = current_params.copy()
            params_minus[k] -= np.pi / 2
            energy_minus = vqe_step(params_minus, pauli_groups, ansatz)
            gradient[k] = 0.5 * (energy_plus - energy_minus)

        # Update parameters using the optimizer
        if optimizer == 1:
            current_params, E_g2 = op.rmsprop_optimizer(current_params, gradient, E_g2)
        elif optimizer == 2:
            current_params, m, v = op.adam_optimizer(current_params, gradient, m, v, i)
        else:
            current_params = op.simple_descent(current_params, gradient)
        
        # Calculate and store the energy for the new parameters
        current_energy = vqe_step(current_params, pauli_groups, ansatz)
        energy_list.append(current_energy)
        print(f'Iteration {i+1}/{n_iterations}, Energy: {current_energy:.10f}')

    return energy_list