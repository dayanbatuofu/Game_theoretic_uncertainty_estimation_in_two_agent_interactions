'''
This script generates data from scratch using time-marching.
'''

import numpy as np
from utilities.BVP_solver import solve_bvp
# from scipy.integrate import solve_bvp
import scipy.io
import time
import warnings
import copy

from utilities.other import int_input

from examples.choose_problem import system, problem, config

np.seterr(over='warn', divide='warn', invalid='warn')
warnings.filterwarnings('error')

N_states = problem.N_states
alpha = problem.alpha

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

# Validation or training data?
data_type = int_input('What kind of data? Enter 0 for validation, 1 for training:')
if data_type:
    data_type = 'train_800'
    Ns = config.Ns[data_type]
    X0_pool = problem.sample_X0(Ns)
    X0 = X0_pool[:, 0]
    X0_orignal = copy.deepcopy(X0)
    for m in range(11):  # segment for speed x2, separate into 11 segment
        for n in range(11):  # segment for speed x1, separate into 11 segment
            print('Solving BVP #', N_sol + 1, 'of', Ns, '...', end='\r')

            step += 1
            print(step)
            print(X0)
            bc = problem.make_bc(X0)

            start_time = time.time()
            tol = 5e-3  # 1e-01

            # Initial guess setting
            X_guess = np.vstack((X0.reshape(-1, 1),
                                 np.array([[alpha],
                                           [-alpha * 3.],
                                           [0.],
                                           [0.],
                                           [0.],
                                           [0.],
                                           [alpha],
                                           [-alpha * 3.],
                                           [0.],
                                           [0.]])))

            # Without time marching for BVP_solver
            collision_lower = problem.R1 / 2 - problem.theta2 * problem.W1 / 2
            collision_upper = problem.R1 / 2 + problem.W1 / 2 + problem.L1
            X_guess1 = X_guess

            n_sample = 800
            delta_t = 1.25e-5
            if X_guess[0, 0] >= X_guess[2, 0]:
                t1 = 2 * (collision_lower - X_guess[2, 0]) / (2 * X_guess[3, 0] - 5)
                t2 = (collision_upper - X_guess1[0, 0]) / X_guess1[1, 0]
                t3 = 3
                if t1 < t2:
                    t = np.zeros(n_sample + 1)
                    for i in range(1, int(n_sample / 2) + 1):
                        t[i] = t1 - (int(n_sample / 2) + 1 - i) * delta_t
                        t[-i] = t1 + (int(n_sample / 2) + 1 - i) * delta_t
                    t_guess1 = np.hstack(
                        (0, t[1:int(n_sample / 2) + 1], t1, t[int(n_sample / 2) + 1:n_sample + 1], t2, t3))
                    for i in range(n_sample + 3):
                        X_guess1 = np.hstack((X_guess1, X_guess))
                    for l in range(1, n_sample + 3):
                        X_guess1[5, l] = -alpha * (3 - t_guess1[l])
                        X_guess1[11, l] = -alpha * (3 - t_guess1[l])
                else:
                    t = np.zeros(n_sample + 1)
                    for i in range(1, int(n_sample / 2) + 1):
                        t[i] = t2 - (int(n_sample / 2) + 1 - i) * delta_t
                        t[-i] = t2 + (int(n_sample / 2) + 1 - i) * delta_t
                    t_guess1 = np.hstack((0, t[1:int(n_sample / 2) + 1], t2, t[int(n_sample / 2) + 1:n_sample + 1], t3))
                    for i in range(n_sample + 2):
                        X_guess1 = np.hstack((X_guess1, X_guess))
                    for l in range(1, n_sample + 2):
                        X_guess1[5, l] = -alpha * (3 - t_guess1[l])
                        X_guess1[11, l] = -alpha * (3 - t_guess1[l])
            else:
                t1 = 2 * (collision_lower - X_guess[0, 0]) / (2 * X_guess[1, 0] - 5)
                t2 = (collision_upper - X_guess1[2, 0]) / X_guess1[3, 0]
                t3 = 3
                if t1 < t2:
                    t = np.zeros(n_sample + 1)
                    for i in range(1, int(n_sample / 2) + 1):
                        t[i] = t1 - (int(n_sample / 2) + 1 - i) * delta_t
                        t[-i] = t1 + (int(n_sample / 2) + 1 - i) * delta_t
                    t_guess1 = np.hstack((0, t[1:int(n_sample / 2) + 1], t1, t[int(n_sample / 2) + 1:n_sample + 1], t2, t3))
                    for i in range(n_sample + 3):
                        X_guess1 = np.hstack((X_guess1, X_guess))
                    for l in range(1, n_sample + 3):
                        X_guess1[5, l] = -alpha * (3 - t_guess1[l])
                        X_guess1[11, l] = -alpha * (3 - t_guess1[l])
                else:
                    t = np.zeros(n_sample + 1)
                    for i in range(1, int(n_sample / 2) + 1):
                        t[i] = t2 - (int(n_sample / 2) + 1 - i) * delta_t
                        t[-i] = t2 + (int(n_sample / 2) + 1 - i) * delta_t
                    t_guess1 = np.hstack((0, t[1:int(n_sample / 2) + 1], t2, t[int(n_sample / 2) + 1:n_sample + 1], t3))
                    for i in range(n_sample + 2):
                        X_guess1 = np.hstack((X_guess1, X_guess))
                    for l in range(1, n_sample + 2):
                        X_guess1[5, l] = -alpha * (3 - t_guess1[l])
                        X_guess1[11, l] = -alpha * (3 - t_guess1[l])

            if X_guess[0, 0] >= X_guess[2, 0]:
                if t1 < t2:
                    for i in range(1, n_sample + 2):
                        X_guess1[0, i] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[i]
                        X_guess1[2, i] = X_guess1[2, 0] + t_guess1[i] / t1 * (collision_lower - X_guess1[2, 0])
                        X_guess1[3, i] = X_guess1[3, 0] - 5 * t_guess1[i]

                    X_guess1[0, -1] = collision_upper
                    X_guess1[2, -1] = X_guess1[2, 0] + X_guess1[3, 0] * t_guess1[-1] - 2.5 * t_guess1[-1] ** 2
                    X_guess1[3, -1] = X_guess1[3, 0] - 5 * t_guess1[-1]
                else:
                    for i in range(1, n_sample + 2):
                        X_guess1[0, i] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[i]
                        X_guess1[2, i] = X_guess1[2, 0] + X_guess1[3, 0] * t_guess1[i] - 2.5 * t_guess1[i] ** 2
                        X_guess1[3, i] = X_guess1[3, 0] - 5 * t_guess1[i]
            else:
                if t1 < t2:
                    for i in range(1, n_sample + 2):
                        X_guess1[0, i] = X_guess1[0, 0] + t_guess1[i] / t1 * (collision_lower - X_guess1[0, 0])
                        X_guess1[1, i] = X_guess1[1, 0] - 5 * t_guess1[i]
                        X_guess1[2, i] = X_guess1[2, 0] + X_guess1[3, 0] * t_guess1[i]

                    X_guess1[0, -1] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[-1] - 2.5 * t_guess1[-1] ** 2
                    X_guess1[1, -1] = X_guess1[1, 0] - 5 * t_guess1[-1]
                    X_guess1[2, -1] = collision_upper
                else:
                    for i in range(1, n_sample + 2):
                        X_guess1[0, i] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[i] - 2.5 * t_guess1[i] ** 2
                        X_guess1[1, i] = X_guess1[3, 0] - 5 * t_guess1[i]
                        X_guess1[2, i] = X_guess1[2, 0] + X_guess1[3, 0] * t_guess1[i]

            X_guess1[0, -1] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[-1]
            X_guess1[1, -1] = X_guess1[1, 0]
            X_guess1[2, -1] = X_guess1[2, 0] + X_guess1[2, 0] * t_guess1[-1]
            X_guess1[3, -1] = X_guess1[3, 0]
            X_guess1[5, -1] = 0
            X_guess1[11, -1] = 0

            SOL = solve_bvp(problem.aug_dynamics, bc, t_guess1, X_guess1,
                            verbose=2, tol=tol, max_nodes=2500)

            t_M1 = SOL.x
            X_M1 = SOL.y[:2 * N_states]
            A_M1 = SOL.y[2 * N_states:6 * N_states]
            V1_M1 = -SOL.y[-2:-1]
            V2_M1 = -SOL.y[-1:]
            V_M1 = np.vstack((V1_M1, V2_M1))
            print(V_M1[:, 0])

            sol_time.append(time.time() - start_time)

            t_OUT = np.hstack((t_OUT, t_M1.reshape(1, -1)))
            X_OUT = np.hstack((X_OUT, X_M1))
            A_OUT = np.hstack((A_OUT, A_M1))
            V_OUT = np.hstack((V_OUT, V_M1))

            N_sol += 1

            X0[0] = X0[0] + 0.5  # step for x1 is 0.5 m
        X0[2] = X0[2] + 0.5  # step for x2 is 0.5 m
        X0[0] = X0_orignal[0]
else:
    data_type = 'val_800'
    Ns = config.Ns[data_type]
    X0_pool = problem.sample_X0(Ns)
    X0 = X0_pool[:, 0]
    X0_orignal = copy.deepcopy(X0)
    for m in range(6):  # segment for speed x2, separate into 11 segment
        for n in range(6):  # segment for speed x1, separate into 11 segment
            print('Solving BVP #', N_sol + 1, 'of', Ns, '...', end='\r')

            step += 1
            print(step)
            print(X0)
            bc = problem.make_bc(X0)

            start_time = time.time()
            tol = 5e-3  # 1e-01

            # Initial guess setting
            X_guess = np.vstack((X0.reshape(-1, 1),
                                 np.array([[alpha],
                                           [-alpha * 3.],
                                           [0.],
                                           [0.],
                                           [0.],
                                           [0.],
                                           [alpha],
                                           [-alpha * 3.],
                                           [0.],
                                           [0.]])))

            # Without time marching for BVP_solver
            collision_lower = problem.R1 / 2 - problem.theta2 * problem.W1 / 2
            collision_upper = problem.R1 / 2 + problem.W1 / 2 + problem.L1
            X_guess1 = X_guess

            n_sample = 800
            delta_t = 1.25e-5
            if X_guess[0, 0] >= X_guess[2, 0]:
                t1 = 2 * (collision_lower - X_guess[2, 0]) / (2 * X_guess[3, 0] - 5)
                t2 = (collision_upper - X_guess1[0, 0]) / X_guess1[1, 0]
                t3 = 3
                if t1 < t2:
                    t = np.zeros(n_sample + 1)
                    for i in range(1, int(n_sample / 2) + 1):
                        t[i] = t1 - (int(n_sample / 2) + 1 - i) * delta_t
                        t[-i] = t1 + (int(n_sample / 2) + 1 - i) * delta_t
                    t_guess1 = np.hstack(
                        (0, t[1:int(n_sample / 2) + 1], t1, t[int(n_sample / 2) + 1:n_sample + 1], t2, t3))
                    for i in range(n_sample + 3):
                        X_guess1 = np.hstack((X_guess1, X_guess))
                    for l in range(1, n_sample + 3):
                        X_guess1[5, l] = -alpha * (3 - t_guess1[l])
                        X_guess1[11, l] = -alpha * (3 - t_guess1[l])
                else:
                    t = np.zeros(n_sample + 1)
                    for i in range(1, int(n_sample / 2) + 1):
                        t[i] = t2 - (int(n_sample / 2) + 1 - i) * delta_t
                        t[-i] = t2 + (int(n_sample / 2) + 1 - i) * delta_t
                    t_guess1 = np.hstack((0, t[1:int(n_sample / 2) + 1], t2, t[int(n_sample / 2) + 1:n_sample + 1], t3))
                    for i in range(n_sample + 2):
                        X_guess1 = np.hstack((X_guess1, X_guess))
                    for l in range(1, n_sample + 2):
                        X_guess1[5, l] = -alpha * (3 - t_guess1[l])
                        X_guess1[11, l] = -alpha * (3 - t_guess1[l])
            else:
                t1 = 2 * (collision_lower - X_guess[0, 0]) / (2 * X_guess[1, 0] - 5)
                t2 = (collision_upper - X_guess1[2, 0]) / X_guess1[3, 0]
                t3 = 3
                if t1 < t2:
                    t = np.zeros(n_sample + 1)
                    for i in range(1, int(n_sample / 2) + 1):
                        t[i] = t1 - (int(n_sample / 2) + 1 - i) * delta_t
                        t[-i] = t1 + (int(n_sample / 2) + 1 - i) * delta_t
                    t_guess1 = np.hstack(
                        (0, t[1:int(n_sample / 2) + 1], t1, t[int(n_sample / 2) + 1:n_sample + 1], t2, t3))
                    for i in range(n_sample + 3):
                        X_guess1 = np.hstack((X_guess1, X_guess))
                    for l in range(1, n_sample + 3):
                        X_guess1[5, l] = -alpha * (3 - t_guess1[l])
                        X_guess1[11, l] = -alpha * (3 - t_guess1[l])
                else:
                    t = np.zeros(n_sample + 1)
                    for i in range(1, int(n_sample / 2) + 1):
                        t[i] = t2 - (int(n_sample / 2) + 1 - i) * delta_t
                        t[-i] = t2 + (int(n_sample / 2) + 1 - i) * delta_t
                    t_guess1 = np.hstack((0, t[1:int(n_sample / 2) + 1], t2, t[int(n_sample / 2) + 1:n_sample + 1], t3))
                    for i in range(n_sample + 2):
                        X_guess1 = np.hstack((X_guess1, X_guess))
                    for l in range(1, n_sample + 2):
                        X_guess1[5, l] = -alpha * (3 - t_guess1[l])
                        X_guess1[11, l] = -alpha * (3 - t_guess1[l])

            if X_guess[0, 0] >= X_guess[2, 0]:
                if t1 < t2:
                    for i in range(1, n_sample + 2):
                        X_guess1[0, i] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[i]
                        X_guess1[2, i] = X_guess1[2, 0] + t_guess1[i] / t1 * (collision_lower - X_guess1[2, 0])
                        X_guess1[3, i] = X_guess1[3, 0] - 5 * t_guess1[i]

                    X_guess1[0, -1] = collision_upper
                    X_guess1[2, -1] = X_guess1[2, 0] + X_guess1[3, 0] * t_guess1[-1] - 2.5 * t_guess1[-1] ** 2
                    X_guess1[3, -1] = X_guess1[3, 0] - 5 * t_guess1[-1]
                else:
                    for i in range(1, n_sample + 2):
                        X_guess1[0, i] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[i]
                        X_guess1[2, i] = X_guess1[2, 0] + X_guess1[3, 0] * t_guess1[i] - 2.5 * t_guess1[i] ** 2
                        X_guess1[3, i] = X_guess1[3, 0] - 5 * t_guess1[i]
            else:
                if t1 < t2:
                    for i in range(1, n_sample + 2):
                        X_guess1[0, i] = X_guess1[0, 0] + t_guess1[i] / t1 * (collision_lower - X_guess1[0, 0])
                        X_guess1[1, i] = X_guess1[1, 0] - 5 * t_guess1[i]
                        X_guess1[2, i] = X_guess1[2, 0] + X_guess1[3, 0] * t_guess1[i]

                    X_guess1[0, -1] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[-1] - 2.5 * t_guess1[-1] ** 2
                    X_guess1[1, -1] = X_guess1[1, 0] - 5 * t_guess1[-1]
                    X_guess1[2, -1] = collision_upper
                else:
                    for i in range(1, n_sample + 2):
                        X_guess1[0, i] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[i] - 2.5 * t_guess1[i] ** 2
                        X_guess1[1, i] = X_guess1[3, 0] - 5 * t_guess1[i]
                        X_guess1[2, i] = X_guess1[2, 0] + X_guess1[3, 0] * t_guess1[i]

            X_guess1[0, -1] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[-1]
            X_guess1[1, -1] = X_guess1[1, 0]
            X_guess1[2, -1] = X_guess1[2, 0] + X_guess1[2, 0] * t_guess1[-1]
            X_guess1[3, -1] = X_guess1[3, 0]
            X_guess1[5, -1] = 0
            X_guess1[11, -1] = 0

            SOL = solve_bvp(problem.aug_dynamics, bc, t_guess1, X_guess1,
                            verbose=2, tol=tol, max_nodes=2500)

            t_M1 = SOL.x
            X_M1 = SOL.y[:2 * N_states]
            A_M1 = SOL.y[2 * N_states:6 * N_states]
            V1_M1 = -SOL.y[-2:-1]
            V2_M1 = -SOL.y[-1:]
            V_M1 = np.vstack((V1_M1, V2_M1))
            print(V_M1[:, 0])

            sol_time.append(time.time() - start_time)

            t_OUT = np.hstack((t_OUT, t_M1.reshape(1, -1)))
            X_OUT = np.hstack((X_OUT, X_M1))
            A_OUT = np.hstack((A_OUT, A_M1))
            V_OUT = np.hstack((V_OUT, V_M1))

            N_sol += 1

            X0[0] = X0[0] + 1  # step for x1 is 1 m
        X0[2] = X0[2] + 1  # step for x2 is 1 m
        X0[0] = X0_orignal[0]

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