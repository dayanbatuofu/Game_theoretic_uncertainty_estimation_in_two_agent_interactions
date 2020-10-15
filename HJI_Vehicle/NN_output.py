'''
Run this script to train the NN. Loads the problem and configuration
according to examples/choose_problem.py.
'''

import numpy as np
import torch
import scipy.io
import time

from utilities.other import int_input, load_NN
from examples.choose_problem import system, problem, config, time_dependent

if time_dependent:
    from utilities.neural_networks import HJB_network
    system += '/tspan'
else:
    from utilities.neural_networks import HJB_network_t0 as HJB_network
    system += '/t0'

# Loads neural network
# start_time = time.time()

def get_Q_value(X, U, theta):
    theta1, theta2 = theta
    if theta1 == 1 and theta2 == 1:
        model_path = 'HJI_Vehicle/examples/' + system + '/V_model.mat'
        parameters, scaling = load_NN(model_path)
    model = HJB_network(problem, scaling, config, parameters)
    Q1, Q2 = model.Q_value(X, U, theta1, theta2)
    return Q1, Q2

# model_running = time.time() - start_time
# print('Computation time: %.0f' % (model_running), 'sec')

if __name__ == '__main__':
    X = np.array([[15.], [18.], [15.], [18.]])
    U = np.array([[5.], [5.]])
    theta = (1, 1)
    Q1, Q2 = get_Q_value(X, U, theta)
    print(Q1)
    print(Q2)


