'''
This file is used to select better data from 4 different samplings: 100, 200, 400, 800
'''

import numpy as np
import scipy.io
from examples.choose_problem import system, problem
from utilities.other import int_input

N_states = problem.N_states

data_type = int_input('What kind of data do you want to generate? Enter 0 for validation, 1 for training:')
if data_type:
    data_type = 'train'
    train_data1 = scipy.io.loadmat('examples/' + system + '/data_train_100.mat')
    train_data2 = scipy.io.loadmat('examples/' + system + '/data_train_200.mat')
    train_data3 = scipy.io.loadmat('examples/' + system + '/data_train_400.mat')
    train_data4 = scipy.io.loadmat('examples/' + system + '/data_train_800.mat')
    data_number = 120
else:
    data_type = 'val'
    train_data1 = scipy.io.loadmat('examples/' + system + '/data_val_100.mat')
    train_data2 = scipy.io.loadmat('examples/' + system + '/data_val_200.mat')
    train_data3 = scipy.io.loadmat('examples/' + system + '/data_val_400.mat')
    train_data4 = scipy.io.loadmat('examples/' + system + '/data_val_800.mat')
    data_number = 35


train_data1.update({'t0': train_data1['t']})
train_data2.update({'t0': train_data2['t']})
train_data3.update({'t0': train_data3['t']})
train_data4.update({'t0': train_data4['t']})

idx0_1 = np.nonzero(np.equal(train_data1.pop('t0'), 0.))[1]
idx0_2 = np.nonzero(np.equal(train_data2.pop('t0'), 0.))[1]
idx0_3 = np.nonzero(np.equal(train_data3.pop('t0'), 0.))[1]
idx0_4 = np.nonzero(np.equal(train_data4.pop('t0'), 0.))[1]

length = len(idx0_1)

t_OUT = np.empty((1, 0))
X_OUT = np.empty((2 * N_states, 0))
A_OUT = np.empty((4 * N_states, 0))
V_OUT = np.empty((2, 0))

for i in range(length):
    V1_train_data1 = train_data1['V'][0, idx0_1[i]]
    V2_train_data1 = train_data1['V'][1, idx0_1[i]]
    V1_train_data2 = train_data2['V'][0, idx0_2[i]]
    V2_train_data2 = train_data2['V'][1, idx0_2[i]]
    V1_train_data3 = train_data3['V'][0, idx0_3[i]]
    V2_train_data3 = train_data3['V'][1, idx0_3[i]]
    V1_train_data4 = train_data4['V'][0, idx0_4[i]]
    V2_train_data4 = train_data4['V'][1, idx0_4[i]]
    if V1_train_data1 > 0 or V2_train_data1 > 0:
        V1_train_data1 = -10000
    if V1_train_data2 > 0 or V2_train_data2 > 0:
        V1_train_data2 = -10000
    if V1_train_data3 > 0 or V2_train_data3 > 0:
        V1_train_data3 = -10000
    if V1_train_data4 > 0 or V2_train_data4 > 0:
        V1_train_data4 = -10000
    V1_max = np.max(np.array([V1_train_data1, V1_train_data2, V1_train_data3, V1_train_data4]))
    if V1_max == V1_train_data1:
        if i == data_number:
            t_OUT = np.hstack((t_OUT, train_data1['t'][:, idx0_1[i]:]))
            X_OUT = np.hstack((X_OUT, train_data1['X'][:, idx0_1[i]:]))
            A_OUT = np.hstack((A_OUT, train_data1['A'][:, idx0_1[i]:]))
            V_OUT = np.hstack((V_OUT, train_data1['V'][:, idx0_1[i]:]))
        else:
            t_OUT = np.hstack((t_OUT, train_data1['t'][:, idx0_1[i]:idx0_1[i + 1]]))
            X_OUT = np.hstack((X_OUT, train_data1['X'][:, idx0_1[i]:idx0_1[i + 1]]))
            A_OUT = np.hstack((A_OUT, train_data1['A'][:, idx0_1[i]:idx0_1[i + 1]]))
            V_OUT = np.hstack((V_OUT, train_data1['V'][:, idx0_1[i]:idx0_1[i + 1]]))
        continue
    if V1_max == V1_train_data2:
        if i == data_number:
            t_OUT = np.hstack((t_OUT, train_data2['t'][:, idx0_2[i]:]))
            X_OUT = np.hstack((X_OUT, train_data2['X'][:, idx0_2[i]:]))
            A_OUT = np.hstack((A_OUT, train_data2['A'][:, idx0_2[i]:]))
            V_OUT = np.hstack((V_OUT, train_data2['V'][:, idx0_2[i]:]))
        else:
            t_OUT = np.hstack((t_OUT, train_data2['t'][:, idx0_2[i]:idx0_2[i + 1]]))
            X_OUT = np.hstack((X_OUT, train_data2['X'][:, idx0_2[i]:idx0_2[i + 1]]))
            A_OUT = np.hstack((A_OUT, train_data2['A'][:, idx0_2[i]:idx0_2[i + 1]]))
            V_OUT = np.hstack((V_OUT, train_data2['V'][:, idx0_2[i]:idx0_2[i + 1]]))
        continue
    if V1_max == V1_train_data3:
        if i == data_number:
            t_OUT = np.hstack((t_OUT, train_data3['t'][:, idx0_3[i]:]))
            X_OUT = np.hstack((X_OUT, train_data3['X'][:, idx0_3[i]:]))
            A_OUT = np.hstack((A_OUT, train_data3['A'][:, idx0_3[i]:]))
            V_OUT = np.hstack((V_OUT, train_data3['V'][:, idx0_3[i]:]))
        else:
            t_OUT = np.hstack((t_OUT, train_data3['t'][:, idx0_3[i]:idx0_3[i + 1]]))
            X_OUT = np.hstack((X_OUT, train_data3['X'][:, idx0_3[i]:idx0_3[i + 1]]))
            A_OUT = np.hstack((A_OUT, train_data3['A'][:, idx0_3[i]:idx0_3[i + 1]]))
            V_OUT = np.hstack((V_OUT, train_data3['V'][:, idx0_3[i]:idx0_3[i + 1]]))
        continue
    if V1_max == V1_train_data4:
        if i == data_number:
            t_OUT = np.hstack((t_OUT, train_data4['t'][:, idx0_4[i]:]))
            X_OUT = np.hstack((X_OUT, train_data4['X'][:, idx0_4[i]:]))
            A_OUT = np.hstack((A_OUT, train_data4['A'][:, idx0_4[i]:]))
            V_OUT = np.hstack((V_OUT, train_data4['V'][:, idx0_4[i]:]))
        else:
            t_OUT = np.hstack((t_OUT, train_data4['t'][:, idx0_4[i]:idx0_4[i + 1]]))
            X_OUT = np.hstack((X_OUT, train_data4['X'][:, idx0_4[i]:idx0_4[i + 1]]))
            A_OUT = np.hstack((A_OUT, train_data4['A'][:, idx0_4[i]:idx0_4[i + 1]]))
            V_OUT = np.hstack((V_OUT, train_data4['V'][:, idx0_4[i]:idx0_4[i + 1]]))
        continue

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


