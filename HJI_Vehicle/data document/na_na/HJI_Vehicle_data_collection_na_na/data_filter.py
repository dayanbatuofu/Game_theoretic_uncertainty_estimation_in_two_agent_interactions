'''
This file is used to remove some weird trajectory manually
'''

import numpy as np
import scipy.io
from examples.choose_problem import system, problem
from utilities.other import int_input

N_states = problem.N_states

data_type = int_input('What kind of data do you want to generate? Enter 0 for validation, 1 for training:')
if data_type:
    data_type = 'train_update'
    train_data = scipy.io.loadmat('examples/' + system + '/data_train.mat')
    data_number = 120
else:
    data_type = 'val_update'
    train_data = scipy.io.loadmat('examples/' + system + '/data_val.mat')
    data_number = 35

train_data.update({'t0': train_data['t']})

idx0 = np.nonzero(np.equal(train_data.pop('t0'), 0.))[1]

length = len(idx0)

t_OUT = np.empty((1, 0))
X_OUT = np.empty((2 * N_states, 0))
A_OUT = np.empty((4 * N_states, 0))
V_OUT = np.empty((2, 0))

for i in range(length):
    x1 = train_data['X'][0, idx0[i]].astype(np.float32)
    x2 = train_data['X'][2, idx0[i]].astype(np.float32)
    if x1 == np.array(17.1).astype(np.float32) and x2 == np.array(15.0).astype(np.float32):
        continue
    if x1 == np.array(16.6).astype(np.float32) and x2 == np.array(15.5).astype(np.float32):
        continue
    if x1 == np.array(18.1).astype(np.float32) and x2 == np.array(15.5).astype(np.float32):
        continue
    if x1 == np.array(15.1).astype(np.float32) and x2 == np.array(16.0).astype(np.float32):
        continue
    if x1 == np.array(16.1).astype(np.float32) and x2 == np.array(16.0).astype(np.float32):
        continue
    if x1 == np.array(17.1).astype(np.float32) and x2 == np.array(16.0).astype(np.float32):
        continue
    if x1 == np.array(18.1).astype(np.float32) and x2 == np.array(16.0).astype(np.float32):
        continue
    if x1 == np.array(16.1).astype(np.float32) and x2 == np.array(16.5).astype(np.float32):
        continue
    if x1 == np.array(16.6).astype(np.float32) and x2 == np.array(16.5).astype(np.float32):
        continue
    if x1 == np.array(17.1).astype(np.float32) and x2 == np.array(16.5).astype(np.float32):
        continue
    if x1 == np.array(18.1).astype(np.float32) and x2 == np.array(16.5).astype(np.float32):
        continue
    if x1 == np.array(16.6).astype(np.float32) and x2 == np.array(17.0).astype(np.float32):
        continue
    if x1 == np.array(17.6).astype(np.float32) and x2 == np.array(17.0).astype(np.float32):
        continue
    if x1 == np.array(18.1).astype(np.float32) and x2 == np.array(17.0).astype(np.float32):
        continue
    if x1 == np.array(16.6).astype(np.float32) and x2 == np.array(17.5).astype(np.float32):
        continue
    if x1 == np.array(17.1).astype(np.float32) and x2 == np.array(17.5).astype(np.float32):
        continue
    if x1 == np.array(17.6).astype(np.float32) and x2 == np.array(17.5).astype(np.float32):
        continue
    if x1 == np.array(18.1).astype(np.float32) and x2 == np.array(17.5).astype(np.float32):
        continue
    if x1 == np.array(18.6).astype(np.float32) and x2 == np.array(17.5).astype(np.float32):
        continue
    if x1 == np.array(19.1).astype(np.float32) and x2 == np.array(17.5).astype(np.float32):
        continue
    if x1 == np.array(17.1).astype(np.float32) and x2 == np.array(18.0).astype(np.float32):
        continue
    if x1 == np.array(17.6).astype(np.float32) and x2 == np.array(18.0).astype(np.float32):
        continue
    if x1 == np.array(18.1).astype(np.float32) and x2 == np.array(18.0).astype(np.float32):
        continue
    if x1 == np.array(18.6).astype(np.float32) and x2 == np.array(18.0).astype(np.float32):
        continue
    if x1 == np.array(19.1).astype(np.float32) and x2 == np.array(18.0).astype(np.float32):
        continue
    if x1 == np.array(19.6).astype(np.float32) and x2 == np.array(18.0).astype(np.float32):
        continue
    if x1 == np.array(17.1).astype(np.float32) and x2 == np.array(18.5).astype(np.float32):
        continue
    if x1 == np.array(17.6).astype(np.float32) and x2 == np.array(18.5).astype(np.float32):
        continue
    if x1 == np.array(17.6).astype(np.float32) and x2 == np.array(19.0).astype(np.float32):
        continue
    if x1 == np.array(17.6).astype(np.float32) and x2 == np.array(19.5).astype(np.float32):
        continue
    if x1 == np.array(18.6).astype(np.float32) and x2 == np.array(20.0).astype(np.float32):
        continue
    else:
        if i == data_number:
            t_OUT = np.hstack((t_OUT, train_data['t'][:, idx0[i]:]))
            X_OUT = np.hstack((X_OUT, train_data['X'][:, idx0[i]:]))
            A_OUT = np.hstack((A_OUT, train_data['A'][:, idx0[i]:]))
            V_OUT = np.hstack((V_OUT, train_data['V'][:, idx0[i]:]))
        else:
            t_OUT = np.hstack((t_OUT, train_data['t'][:, idx0[i]:idx0[i + 1]]))
            X_OUT = np.hstack((X_OUT, train_data['X'][:, idx0[i]:idx0[i + 1]]))
            A_OUT = np.hstack((A_OUT, train_data['A'][:, idx0[i]:idx0[i + 1]]))
            V_OUT = np.hstack((V_OUT, train_data['V'][:, idx0[i]:idx0[i + 1]]))

save_data = int_input('Save data? Enter 0 for no, 1 for yes:')
if save_data:
    save_path = 'examples/' + system + '/data_' + data_type + '.mat'
    try:
        save_dict = scipy.io.loadmat(save_path)

        overwrite_data = int_input('Overwrite existing data? Enter 0 for no, 1 for yes:')

        if overwrite_data:
            raise

        save_dict.update({'t': np.hstack((save_dict['t'], t_OUT)),
                          'X': np.hstack((save_dict['X'], X_OUT)),
                          'A': np.hstack((save_dict['A'], A_OUT)),
                          'V': np.hstack((save_dict['V'], V_OUT))})
    except:
        U1, U2 = problem.U_star(np.vstack((X_OUT, A_OUT)))
        U = np.vstack((U1, U2))

        save_dict = {'lb_1': np.min(X_OUT[:N_states], axis=1, keepdims=True),
                     'ub_1': np.max(X_OUT[:N_states], axis=1, keepdims=True),
                     'lb_2': np.min(X_OUT[N_states:2*N_states], axis=1, keepdims=True),
                     'ub_2': np.max(X_OUT[N_states:2*N_states], axis=1, keepdims=True),
                     'A_lb_11': np.min(A_OUT[:N_states], axis=1, keepdims=True),
                     'A_ub_11': np.max(A_OUT[:N_states], axis=1, keepdims=True),
                     'A_lb_12': np.min(A_OUT[N_states:2*N_states], axis=1, keepdims=True),
                     'A_ub_12': np.max(A_OUT[N_states:2*N_states], axis=1, keepdims=True),
                     'A_lb_21': np.min(A_OUT[2*N_states:3*N_states], axis=1, keepdims=True),
                     'A_ub_21': np.max(A_OUT[2*N_states:3*N_states], axis=1, keepdims=True),
                     'A_lb_22': np.min(A_OUT[3*N_states:4*N_states], axis=1, keepdims=True),
                     'A_ub_22': np.max(A_OUT[3*N_states:4*N_states], axis=1, keepdims=True),
                     'U_lb_1': np.min(U1, axis=1, keepdims=True),
                     'U_ub_1': np.max(U1, axis=1, keepdims=True),
                     'U_lb_2': np.min(U2, axis=1, keepdims=True),
                     'U_ub_2': np.max(U2, axis=1, keepdims=True),
                     'V_min_1': np.min(V_OUT[-2:-1,:]), 'V_max_1': np.max(V_OUT[-2:-1,:]),
                     'V_min_2': np.min(V_OUT[-1,:]), 'V_max_2': np.max(V_OUT[-1,:]),
                     't': t_OUT, 'X': X_OUT, 'A': A_OUT, 'V': V_OUT}
        scipy.io.savemat(save_path, save_dict)


