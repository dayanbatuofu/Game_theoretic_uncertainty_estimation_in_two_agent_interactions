'''
Run this script to train the NN. Loads the problem and configuration
according to examples/choose_problem.py.
'''

import numpy as np
import torch
import scipy.io
from HJI_Vehicle_costate.examples.choose_problem import system, problem, config, time_dependent
# import time

from HJI_Vehicle_costate.utilities.other import int_input, load_NN
from HJI_Vehicle_costate.examples.choose_problem import system, problem, config, time_dependent

if time_dependent:
    from HJI_Vehicle_costate.utilities.neural_networks_costate import HJI_network
    system += '/tspan'
else:
    from HJI_Vehicle_costate.utilities.neural_networks_costate import HJI_network_t0 as HJI_network
    system += '/t0'

# Loads neural network
# start_time = time.time()

def get_Hamilton(X, t, U, theta):
    theta1, theta2 = theta
    if theta1 == 5 and theta2 == 5:
        model_path = 'HJI_Vehicle_costate/examples/' + system + '/V_model_na_na.mat'
        parameters, scaling = load_NN(model_path)
    if theta1 == 1 and theta2 == 1:
        model_path = 'HJI_Vehicle_costate/examples/' + system + '/V_model_a_a.mat'
        parameters, scaling = load_NN(model_path)
    if theta1 == 1 and theta2 == 5:
        model_path = 'HJI_Vehicle_costate/examples/' + system + '/V_model_a_na.mat'
        parameters, scaling = load_NN(model_path)
    if theta1 == 5 and theta2 == 1:
        model_path = 'HJI_Vehicle_costate/examples/' + system + '/V_model_na_a.mat'
        parameters, scaling = load_NN(model_path)
    model = HJI_network(problem, scaling, config, parameters)
    H1, H2 = model.Hamilton_value(X, t, U, theta1, theta2)
    return H1, H2

def get_Costate(X, t, theta):
    theta1, theta2 = theta
    if theta1 == 5 and theta2 == 5:
        model_path = 'HJI_Vehicle_costate/examples/' + system + '/V_model_na_na.mat'
        parameters, scaling = load_NN(model_path)
    if theta1 == 1 and theta2 == 1:
        model_path = 'HJI_Vehicle_costate/examples/' + system + '/V_model_a_a.mat'
        parameters, scaling = load_NN(model_path)
    if theta1 == 1 and theta2 == 5:
        model_path = 'HJI_Vehicle_costate/examples/' + system + '/V_model_a_na.mat'
        parameters, scaling = load_NN(model_path)
    if theta1 == 5 and theta2 == 1:
        model_path = 'HJI_Vehicle_costate/examples/' + system + '/V_model_na_a.mat'
        parameters, scaling = load_NN(model_path)
    model = HJI_network(problem, scaling, config, parameters)
    U1, U2 = model.get_costate(X, t, theta1, theta2)
    return U1, U2

if __name__ == '__main__':
    X = np.array([[15.], [18.], [15.], [18.]])
    t = np.array([[0]])
    U = np.array([[-5.], [10.]])
    theta = (5, 5)
    Q1, Q2 = get_Hamilton(X, t, U, theta)
    print(Q1)
    print(Q2)
