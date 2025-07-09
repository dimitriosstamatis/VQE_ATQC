import numpy as np

def simple_descent(current_params, gradient):
    '''Performs one optimization step using simple gradient descent'''
    learning_rate = 0.1

    current_params -= learning_rate * gradient
    
    return current_params

def rmsprop_optimizer(current_params, gradient, E_g2):
    '''Performs one optimization step using RMSProp optimizer'''
    learning_rate = 0.01
    gamma = 0.9
    epsilon = 1e-8

    E_g2 = gamma * E_g2 + (1 - gamma) * np.square(gradient)
    current_params -= learning_rate / (np.sqrt(E_g2) + epsilon) * gradient
    
    return current_params, E_g2

def adam_optimizer(current_params, gradient, m, v, i):
    '''Performs one optimization step using the Adam optimizer'''
    learning_rate = 0.01
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    
    t = i + 1
    m = beta_1 * m + (1 - beta_1) * gradient
    v = beta_2 * v + (1 - beta_2) * np.square(gradient)
    
    m_corr = m / (1 - beta_1**t)
    v_corr = v / (1 - beta_2**t)
    current_params -= learning_rate / (np.sqrt(v_corr) + epsilon) * m_corr
    
    return current_params, m, v