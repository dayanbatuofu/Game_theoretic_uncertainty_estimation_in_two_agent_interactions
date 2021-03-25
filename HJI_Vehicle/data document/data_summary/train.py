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
else:
    from utilities.neural_networks import HJB_network_t0 as HJB_network

c = np.random.seed(config.random_seeds['train'])
# torch.manual_seed(config.random_seeds['train'])
# tf.set_random_seed(config.random_seeds['train'])

# ---------------------------------------------------------------------------- #

##### Loads data sets #####

train_data = scipy.io.loadmat('examples/' + system + '/data_train.mat')
val_data = scipy.io.loadmat('examples/' + system + '/data_val.mat')

if time_dependent:
    system += '/tspan'
else:
    system += '/t0'
    for data in [train_data, val_data]:
        idx0 = np.nonzero(np.equal(data.pop('t'), 0.))[1]
        data.update({'X': data['X'][:, idx0],
                     'A': data['A'][:, idx0],
                     'V': data['V'][:, idx0]})

N_train = train_data['X'].shape[1]
N_val = val_data['X'].shape[1]

print('')
print('Number of training data:', N_train)
print('Number of validation data:', N_val)
print('')

# ---------------------------------------------------------------------------- #

##### Builds and trains the neural net #####
model_path = 'examples/' + system + '/V_model.mat'
data_path = 'examples/' + system + '/model_data.mat'

if int_input('Load pre-trained model? Enter 0 for no, 1 for yes:'):
    # Loads pre-trained model
    parameters, scaling = load_NN(model_path)
else:
    # Initializes the model from scratch
    parameters = None
    lb_1 = train_data.pop('lb_1')
    lb_2 = train_data.pop('lb_2')

    ub_1 = train_data.pop('ub_1')
    ub_2 = train_data.pop('ub_2')

    A_lb_11 = train_data.pop('A_lb_11')
    A_lb_12 = train_data.pop('A_lb_12')
    A_lb_21 = train_data.pop('A_lb_21')
    A_lb_22 = train_data.pop('A_lb_22')

    A_ub_11 = train_data.pop('A_ub_11')
    A_ub_12 = train_data.pop('A_ub_12')
    A_ub_21 = train_data.pop('A_ub_21')
    A_ub_22 = train_data.pop('A_ub_22')

    U_lb_1 = train_data.pop('U_lb_1')
    U_lb_2 = train_data.pop('U_lb_2')

    U_ub_1 = train_data.pop('U_ub_1')
    U_ub_2 = train_data.pop('U_ub_2')

    V_min_1 = train_data.pop('V_min_1')
    V_min_2 = train_data.pop('V_min_2')

    V_max_1 = train_data.pop('V_max_1')
    V_max_2 = train_data.pop('V_max_2')

    scaling = {
        'lb': np.vstack((lb_1, lb_2)), 'ub': np.vstack((ub_1, ub_2)),
        'A_lb': np.vstack((A_lb_11, A_lb_12, A_lb_21, A_lb_22)),
        'A_ub': np.vstack((A_ub_11, A_ub_12, A_ub_21, A_ub_22)),
        'U_lb': np.vstack((U_lb_1, U_lb_2)), 'U_ub': np.vstack((U_ub_1, U_ub_2)),
        'V_min': np.vstack((V_min_1, V_min_2)), 'V_max': np.vstack((V_max_1, V_max_2))
        }

start_time = time.time()

model = HJB_network(problem, scaling, config, parameters)

# use validation data to train the model and use train_data to verify
errors, model_data = model.train(train_data, val_data, EPISODE=5000, LR=0.01)

train_time = time.time() - start_time
print('Computation time: %.0f' % (train_time), 'sec')

# ---------------------------------------------------------------------------- #
save_dict = {'train_time': train_time,
             'train_err': errors[0], 'train_grad_err': errors[1],
             'val_err': errors[2], 'val_grad_err': errors[3]}

scipy.io.savemat('examples/' + system + '/results/train_results.mat', save_dict)

# Saves model parameters
save_me = int_input('Save model parameters? Enter 0 for no, 1 for yes:')

save_data = int_input('Save data? Enter 0 for no, 1 for yes:')

if save_me:
    weights1, biases1, weights2, biases2 = model.export_model()
    save_dict = scaling
    save_dict.update({'weights1': weights1,
                      'biases1': biases1,
                      'weights2': weights2,
                      'biases2': biases2,
                      'train_time': train_time})
    scipy.io.savemat(model_path, save_dict)

if save_data:
    scipy.io.savemat(data_path, model_data)
