from setup import *
from ansaetze import *
from vqe_loop import run_vqe
import matplotlib.pyplot as plt

def main():
    '''Main function to run the VQE prototype.'''
    # Setup Hamiltonian and calculate exact energy
    qubit_hamiltonian, pauli_groups, numpy_solver_result = setup(0.9)
    
    # Create a hardware efficient ansatz
    num_layers = 2
    he_ansatz = create_hea_ansatz(qubit_hamiltonian.num_qubits, num_layers)

    print('\nHEA - Circuit:')
    print(he_ansatz.draw('text'))
    
    # Run VQE optimization with different optimizers
    print('\nHEA - Simple descent:')
    hea_simple = run_vqe(he_ansatz.num_parameters, pauli_groups, he_ansatz, 0, 200)
    print('\nHEA - RMSPROP optimizer:')
    hea_rmsprop = run_vqe(he_ansatz.num_parameters, pauli_groups, he_ansatz, 1, 200)
    print('\nHEA - Adam optimizer:')
    hea_adam = run_vqe(he_ansatz.num_parameters, pauli_groups, he_ansatz, 2, 200)

    # Plot results for comparison
    plt.figure(figsize=(15,10))
    plt.axhline(y=numpy_solver_result, color='r', label=f'Exact Energy ({numpy_solver_result:.3f})')
    plt.plot(hea_simple, label='HEA - Simple descent')
    plt.plot(hea_rmsprop, label='HEA - RMSPROP optimizer')
    plt.plot(hea_adam, label='HEA - Adam optimizer')
    plt.xlabel('Iterations')
    plt.ylabel('Energy')
    plt.title('Finding the ground state of He-H+ for a specific molecular separation R = 90 pm')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
