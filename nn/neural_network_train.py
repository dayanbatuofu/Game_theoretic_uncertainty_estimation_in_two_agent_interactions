import numpy as np
from constants import CONSTANTS as C
import scipy

def main():

    fh = open("train_data.txt", "w")


    fh.write("%f\t%f\t%f\t%f\t%f\t%f\n" % (input[0],input[0],input[0],input[0]))

    pass







def get_actions(s_other, s_self, s_desired_other, s_desired_self, c_other, c_self, t_steps):
    """ Function that accepts 2 vehicles states, intents, criteria, and an amount of future steps
    and return the ideal actions based on the loss function"""

    # Error between desired position and current position
    error_other = np.array(s_desired_other) - np.array(s_other)
    error_self = np.array(s_desired_self) - np.array(s_self)

    # Define theta
    theta_other = np.clip(error_other, -C.THETA_LIMITER_X, C.THETA_LIMITER_X)
    theta_self = np.clip(error_self, -C.THETA_LIMITER_Y, C.THETA_LIMITER_Y)

    actions_other = np.tile(theta_other, (t_steps, 1))

    a0 = np.array([0 for _ in range(2 * t_steps)])

    bounds = tuple([(-C.VEHICLE_MOVEMENT_SPEED, C.VEHICLE_MOVEMENT_SPEED) for _ in range(2 * t_steps)])

    cons_other = ({'type': 'ineq', 'fun': lambda x: s_other[1] + sum(x[t_steps:]) - C.Y_MINIMUM},
                  {'type': 'ineq', 'fun': lambda x: s_other[1] + sum(x[t_steps:]) + C.Y_MAXIMUM})

    cons_self = ({'type': 'ineq', 'fun': lambda x: s_self[1] + sum(x[t_steps:]) - C.Y_MINIMUM},
                 {'type': 'ineq', 'fun': lambda x: s_self[1] + sum(x[t_steps:]) + C.Y_MAXIMUM})

    loss_value = 0
    loss_value_old = loss_value + C.LOSS_THRESHOLD + 1
    iter_count = 0

    while np.abs(loss_value - loss_value_old) > C.LOSS_THRESHOLD and iter_count < 1:
        loss_value_old = loss_value
        iter_count += 1

        # Estimate machine actions
        optimization_results = scipy.optimize.minimize(loss_func, a0, bounds=bounds, constraints=cons_self,
                                                       args=(s_other, s_self, actions_other, theta_self, c_self))
        actions_self = np.transpose(np.vstack((optimization_results.x[:t_steps], optimization_results.x[t_steps:])))
        loss_value = optimization_results.fun

        # Estimate human actions
        optimization_results = scipy.optimize.minimize(loss_func, a0, bounds=bounds, constraints=cons_other,
                                                       args=(s_self, s_other, actions_self, theta_other, c_other))
        actions_other = np.transpose(np.vstack((optimization_results.x[:t_steps], optimization_results.x[t_steps:])))

    return actions_self, actions_other


def loss_func(actions, s_other, s_self, actions_other, theta_self, c):

    """ Loss function defined to be a combination of state_loss and intent_loss with a weighted factor c """

    t_steps = int(len(actions)/2)

    actions = np.transpose(np.vstack((actions[:t_steps], actions[t_steps:])))

    theta_vectorized = np.tile(theta_self, (t_steps, 1))

    A = np.zeros((t_steps, t_steps))
    A[np.tril_indices(t_steps, 0)] = 1

    # Define state loss
    state_loss = np.reciprocal(np.linalg.norm(s_self + np.matmul(A, actions) - s_other - np.matmul(A, actions_other), axis=1))

    # Define action loss
    intent_loss = np.square(np.linalg.norm(actions - theta_vectorized))

    return np.sum(state_loss) + c * np.sum(intent_loss)  # Return sum with a weighted factor


main()