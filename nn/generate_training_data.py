import numpy as np
from constants import CONSTANTS as C
import scipy
from scipy import optimize
from pyDOE import *

def main():

    fh = open("train_data.txt", "w")


    s_other_x          = 0
    s_other_y          = [0, 1]

    s_self_x           = [-2, 2]
    s_self_y           = [0, 1]

    s_desired_other_x  = [-2, 2]
    s_desired_other_y  = [0, 1]

    s_desired_self_x   = 2
    s_desired_self_y   = 0
    c_other            = [20, 100]
    c_self             = 13

    lhd = lhs(6, 50000)

    # fh.write("s_other_y\ts_self_x\ts_self_y\ts_desired_other_x\ts_desired_other_y\tc_other\n")


    for i in range(len(lhd)):

        s_other_y_inp           = lhd[i][0] * (s_other_y[1] - s_other_y[0]) + s_other_y[0]
        s_self_x_inp            = lhd[i][1] * (s_self_x[1] - s_self_x[0]) + s_self_x[0]
        s_self_y_inp            = lhd[i][2] * (s_self_y[1] - s_self_y[0]) + s_self_y[0]
        s_desired_other_x_inp   = lhd[i][3] * (s_desired_other_x[1] - s_desired_other_x[0]) + s_desired_other_x[0]
        s_desired_other_y_inp   = lhd[i][4] * (s_desired_other_y[1] - s_desired_other_y[0]) + s_desired_other_y[0]
        c_other_inp             = lhd[i][5] * (c_other[1] - c_other[0]) + c_other[0]

        [actions_self, actions_other] = get_actionsa([s_other_x, s_other_y_inp], [s_self_x_inp, s_self_y_inp], [s_desired_other_x_inp, s_desired_other_y_inp], [s_desired_self_x, s_desired_self_y], c_other_inp, c_self, C.T_FUTURE)

        fh.write("%f\t%f\t%f\t%f\t%f\t%f\t" % (s_other_y_inp, s_self_x_inp, s_self_y_inp, s_desired_other_x_inp, s_desired_other_y_inp, c_other_inp))

        # Write self actions
        for j in range(len(actions_self)):
            fh.write("%f\t%f\t" % (actions_self[j][0], actions_self[j][1]))

        # Write other actions
        for j in range(len(actions_other)):
            fh.write("%f\t%f\t" % (actions_other[j][0], actions_other[j][1]))

        fh.write("\n")

        print(i)

    pass


def get_actionsa(s_other, s_self, s_desired_other, s_desired_self, c_other, c_self, t_steps):
    """ Function that accepts 2 vehicles states, intents, criteria, and an amount of future steps
    and return the ideal actions based on the loss function"""

    # Error between desired position and current position
    error_other = np.array(s_desired_other) - np.array(s_other)
    error_self = np.array(s_desired_self) - np.array(s_self)

    # Define theta
    theta_other = np.clip(error_other, -C.THETA_LIMITER_X, C.THETA_LIMITER_X)
    theta_self = np.clip(error_self, -C.THETA_LIMITER_Y, C.THETA_LIMITER_Y)

    # Initialize actions
    actions_other = np.array([0 for _ in range(2 * t_steps)])
    actions_self = np.array([0 for _ in range(2 * t_steps)])

    bounds = tuple([(-C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER,
                     C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER) for _ in range(2 * t_steps)])

    A = np.zeros((t_steps, t_steps))
    A[np.tril_indices(t_steps, 0)] = 1

    cons_other = []
    for i in range(t_steps):
        cons_other.append({'type': 'ineq',
                           'fun': lambda x, i=i: s_other[1] + sum(x[t_steps:t_steps + i + 1]) - C.Y_MINIMUM})
        cons_other.append({'type': 'ineq',
                           'fun': lambda x, i=i: -s_other[1] - sum(x[t_steps:t_steps + i + 1]) + C.Y_MAXIMUM})

    cons_self = []
    for i in range(t_steps):
        cons_self.append({'type': 'ineq',
                          'fun': lambda x, i=i: s_self[1] + sum(x[t_steps:t_steps + i + 1]) - C.Y_MINIMUM})
        cons_self.append({'type': 'ineq',
                          'fun': lambda x, i=i: -s_self[1] - sum(x[t_steps:t_steps + i + 1]) + C.Y_MAXIMUM})

    loss_value = 0
    loss_value_old = loss_value + C.LOSS_THRESHOLD + 1
    iter_count = 0

    # Estimate machine actions
    optimization_results = scipy.optimize.minimize(loss_func, actions_self, bounds=bounds, constraints=cons_self,
                                                   args=(s_other, s_self, actions_other, theta_self, c_self))
    actions_self = optimization_results.x
    loss_value = optimization_results.fun

    while np.abs(loss_value - loss_value_old) > C.LOSS_THRESHOLD and iter_count < 1:
        loss_value_old = loss_value
        iter_count += 1

        # Estimate human actions
        optimization_results = scipy.optimize.minimize(loss_func, actions_other, bounds=bounds,
                                                       constraints=cons_other,
                                                       args=(s_self, s_other, actions_self, theta_other, c_other))
        actions_other = optimization_results.x

        # Estimate machine actions
        optimization_results = scipy.optimize.minimize(loss_func, actions_self, bounds=bounds,
                                                       constraints=cons_self,
                                                       args=(s_other, s_self, actions_other, theta_self, c_self))
        actions_self = optimization_results.x
        loss_value = optimization_results.fun

    # Normalize output for network training
    actions_other = (actions_other + C.VEHICLE_MOVEMENT_SPEED*C.ACTION_PREDICTION_MULTIPLIER) / (2*C.VEHICLE_MOVEMENT_SPEED*C.ACTION_PREDICTION_MULTIPLIER)
    actions_self = (actions_self + C.VEHICLE_MOVEMENT_SPEED*C.ACTION_PREDICTION_MULTIPLIER) / (2*C.VEHICLE_MOVEMENT_SPEED*C.ACTION_PREDICTION_MULTIPLIER)

    actions_other = np.transpose(np.vstack((actions_other[:t_steps], actions_other[t_steps:])))
    actions_self = np.transpose(np.vstack((actions_self[:t_steps], actions_self[t_steps:])))

    return actions_self, actions_other


def loss_func(actions, s_other, s_self, actions_other, theta_self, c):
    """ Loss function defined to be a combination of state_loss and intent_loss with a weighted factor c """

    t_steps = int(len(actions) / 2)

    actions = np.transpose(np.vstack((actions[:t_steps], actions[t_steps:])))
    actions_other = np.transpose(np.vstack((actions_other[:t_steps], actions_other[t_steps:])))

    theta_vectorized = np.tile(theta_self, (t_steps, 1))

    A = np.zeros((t_steps, t_steps))
    A[np.tril_indices(t_steps, 0)] = 1

    # Define state loss
    state_loss = np.reciprocal(
        np.linalg.norm(s_self + np.matmul(A, actions) - s_other - np.matmul(A, actions_other), axis=1))

    # Define action loss
    intent_loss = np.square(np.linalg.norm(actions - theta_vectorized))

    return np.sum(state_loss) + c * np.sum(intent_loss)  # Return sum with a weighted factor


main()