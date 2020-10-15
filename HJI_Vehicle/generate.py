'''
This script generates data from scratch using time-marching.
'''

import numpy as np
from scipy.integrate import solve_bvp
import scipy.io
import time
import warnings
import copy

from utilities.other import int_input

from examples.choose_problem import system, problem, config

np.seterr(over='warn', divide='warn', invalid='warn')
warnings.filterwarnings('error')

N_states = problem.N_states

# Validation or training data?
data_type = int_input('What kind of data? Enter 0 for validation, 1 for training:')
if data_type:
    data_type = 'train'
else:
    data_type = 'val'

Ns = config.Ns[data_type]
X0_pool = problem.sample_X0(Ns)

# Arrays to store generated data
t_OUT = np.empty((1, 0))
X_OUT = np.empty((2 * N_states, 0))
A_OUT = np.empty((4 * N_states, 0))
V_OUT = np.empty((2, 0))
V = np.zeros((2, 1))
A = np.zeros((4 * N_states, 1))

N_sol = 0
sol_time = []
step = 0
X0 = X0_pool[:, 0]
X0_orignal = copy.deepcopy(X0)

# ---------------------------------------------------------------------------- #
for i in range(8):  # segment for speed v2, separate into 8 segment
    for j in range(8):  # segment for speed v1, separate into 8 segment
        for m in range(11):  # segment for speed x2, separate into 11 segment
            for n in range(11):  # segment for speed x1, separate into 11 segment
                print('Solving BVP #', N_sol + 1, 'of', Ns, '...', end='\r')
                step += 1
                print(step)
                print(X0)
                bc = problem.make_bc(X0)

                start_time = time.time()
                tol = 1e-3  # 1e-01

                # Initial guess is zeros
                if N_sol == 0:
                    X_guess = np.vstack((X0.reshape(-1, 1),
                                         np.zeros((4 * N_states + 2, 1))))
                else:
                    X_guess = np.vstack((X0.reshape(-1, 1),
                                         A_OUT[:, -1].reshape(-1, 1),
                                         V_OUT[:, -1].reshape(-1, 1)))
                    print('stop')

                # Without time marching for BVP_solver
                t_guess1 = np.linspace(0., 5., 11)
                X_guess1 = X_guess
                for l in range(10):
                    X_guess1 = np.hstack((X_guess1, X_guess))

                SOL = solve_bvp(problem.aug_dynamics, bc, t_guess1, X_guess1,
                                verbose=0, tol=tol, max_nodes=2500)

                A_M1 = SOL.y[2 * N_states:6 * N_states, 0:1]
                V1_M1 = -SOL.y[-2:-1]
                V2_M1 = -SOL.y[-1:]
                V_M1 = np.vstack((V1_M1, V2_M1))[:, 0:1]

                # With time marching for BVP_solver
                t_guess2 = np.array([0.])
                X_guess2 = X_guess

                for k in range(config.tseq.shape[0]):
                    t_guess2 = np.concatenate((t_guess2, config.tseq[k:k + 1]))
                    X_guess2 = np.hstack((X_guess2, X_guess2[:, -1:]))

                    # try:
                    SOL = solve_bvp(problem.aug_dynamics, bc, t_guess2, X_guess2,
                                    verbose=0, tol=tol, max_nodes=2500)

                    t_guess2 = SOL.x
                    X_guess2 = SOL.y

                A_M2 = SOL.y[2 * N_states:6 * N_states, 0:1]
                V1_M2 = -SOL.y[-2:-1]
                V2_M2 = -SOL.y[-1:]
                V_M2 = np.vstack((V1_M2, V2_M2))[:, 0:1]

                sol_time.append(time.time() - start_time)

                # Justify which V1 and V2 is better to use in the next X_guess in the BVP solver
                # The lambda should be changed because it is corresponding to the V

                if V_M1[0, 0:1] > np.array([0]) or V_M2[0, 0:1] > np.array([0]):
                    V[0, 0:1] = min(V_M1[0, 0:1], V_M2[0, 0:1])
                    if V_M1[0, 0:1] < V_M2[0, 0:1]:
                        A[:2 * N_states, 0:1] = A_M1[:2 * N_states, 0:1]
                    else:
                        A[:2 * N_states, 0:1] = A_M2[:2 * N_states, 0:1]
                else:
                    V[0, 0:1] = max(V_M1[0, 0:1], V_M2[0, 0:1])
                    if V_M1[0, 0:1] > V_M2[0, 0:1]:
                        A[:2 * N_states, 0:1] = A_M1[:2 * N_states, 0:1]
                    else:
                        A[:2 * N_states, 0:1] = A_M2[:2 * N_states, 0:1]

                if V_M1[1, 0:1] > np.array([0]) or V_M2[1, 0:1] > np.array([0]):
                    V[1, 0:1] = min(V_M1[1, 0:1], V_M2[1, 0:1])
                    if V_M1[1, 0:1] < V_M2[1, 0:1]:
                        A[2 * N_states:4 * N_states, 0:1] = A_M1[2 * N_states:4 * N_states, 0:1]
                    else:
                        A[2 * N_states:4 * N_states, 0:1] = A_M2[2 * N_states:4 * N_states, 0:1]
                else:
                    V[1, 0:1] = max(V_M1[1, 0:1], V_M2[1, 0:1])
                    if V_M1[1, 0:1] > V_M2[1, 0:1]:
                        A[2 * N_states:4 * N_states, 0:1] = A_M1[2 * N_states:4 * N_states, 0:1]
                    else:
                        A[2 * N_states:4 * N_states, 0:1] = A_M2[2 * N_states:4 * N_states, 0:1]

                result1 = A_OUT
                result2 = V_OUT

                t_OUT = np.hstack((t_OUT, SOL.x.reshape(1, -1)[:, 0:1]))
                X_OUT = np.hstack((X_OUT, SOL.y[:2 * N_states, 0:1]))
                A_OUT = np.hstack((A_OUT, A))
                V_OUT = np.hstack((V_OUT, V))

                N_sol += 1

                X0[0] = X0[0] + 0.5  # step for x1 is 0.5 m
            X0[2] = X0[2] + 0.5  # step for x2 is 0.5 m
            X0[0] = X0_orignal[0]
        X0[1] = X0[1] + 1  # step for v1 is 1 m/s
        X0[2] = X0_orignal[2]
    X0[3] = X0[3] + 1  # step for v2 is 1 m/s
    X0[1] = X0_orignal[1]

# ---------------------------------------------------------------------------- #

sol_time = np.sum(sol_time)

print('')
print(step, '/', step, 'successful solution attempts:')
print('Average solution time: %1.1f' % (sol_time/step), 'sec')
print('Total solution time: %1.1f' % (sol_time), 'sec')

print('')
print('Total data generated:', X_OUT.shape[1])
print('')

# ---------------------------------------------------------------------------- #

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