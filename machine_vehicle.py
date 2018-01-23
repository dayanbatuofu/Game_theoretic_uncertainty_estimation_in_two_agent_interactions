from constants import CONSTANTS as C
import numpy as np
import scipy
from scipy import optimize


class MachineVehicle:

    '''
    States:
            X-Position
            Y-Position
    '''

    def __init__(self, machine_initial_state, human_initial_state):

        self.machine_criteria = C.MACHINE_CRITERIA
        self.machine_intent = C.MACHINE_INTENT

        self.machine_states = [machine_initial_state]
        self.machine_actions = [(0, 0)]

        self.human_states = [human_initial_state]
        self.human_actions = [(0, 0)]
        self.human_predicted_states = [human_initial_state]
        self.human_predicted_initial_intent = (0, -1)

        self.debug_1 = 0
        self.debug_2 = 0
        self.debug_3 = 0

    def get_state(self):
        return self.machine_states[-1]

    def update(self, human_state):

        human_predicted_intent = (0, 0) # Implement correction function here
        human_criteria = 0.9 # Implement correction function here
        #human_predicted_intent, human_criteria = self.get_human_predicted_intent()
        #human_predicted_intent = (0, human_predicted_intent)

        self.human_states.append(human_state)

        machine_new_action = self.get_action(self.human_states[-1], self.machine_states[-1],
                                             human_predicted_intent, self.machine_intent,
                                             self.machine_criteria)

        # Use Y Components (human is restricted to y axis actions)
        # human_predicted_action = self.get_human_action(self.human_states[-1][1], self.machine_states[-1][1],
        #                                      human_predicted_intent[1], machine_new_action[1],
        #                                      human_criteria)
        # human_predicted_action = (0, human_predicted_action)
        #
        # self.human_predicted_states.append(np.add(self.human_predicted_states[-1], human_predicted_action))

        self.update_state_action(machine_new_action)

    def update_state_action(self, action):

        self.machine_states.append(np.add(self.machine_states[-1], action))
        self.machine_actions.append(action)

    def get_action(self, s_other, s_self, s_desired_other, s_desired_self, c_self):

        # Error between desired position and current position
        error_other = np.array(s_desired_other) - np.array(s_other)
        error_self = np.array(s_desired_self) - np.array(s_self)

        # Define theta
        theta_other = np.clip(error_other, -C.THETA_LIMITER_X, C.THETA_LIMITER_X)
        theta_self = np.clip(error_self, -C.THETA_LIMITER_Y, C.THETA_LIMITER_Y)



        a0 = np.array([0 for _ in range(2 * C.T_FUTURE)])

        x_bounds = tuple([(-C.VEHICLE_MOVEMENT_SPEED, C.VEHICLE_MOVEMENT_SPEED) for i in range(C.T_FUTURE)])

        # Lane keeping
        if theta_self[1] > 0:
            y_bound = (0, C.VEHICLE_LATERAL_MOVEMENT_SPEED)
        elif theta_self[1] < 0:
            y_bound = (-C.VEHICLE_LATERAL_MOVEMENT_SPEED, 0)
        else:
            y_bound = (0, 0)

        y_bounds = tuple([y_bound for i in range(C.T_FUTURE)])

        bounds = x_bounds + y_bounds

        optimization_results = scipy.optimize.minimize(self.loss_func, a0, bounds=bounds,
                                                       args=(s_other, s_self, theta_other, theta_self, c_self))
        actions_x = optimization_results.x[:C.T_FUTURE]
        actions_y = optimization_results.x[C.T_FUTURE:]
        actions = list(zip(actions_x, actions_y))

        return actions[0]  # return first action

    def loss_func(self, actions, s_other, s_self, theta_other, theta_self, c_self):

        actions_x = actions[:C.T_FUTURE]
        actions_y = actions[C.T_FUTURE:]

        loss_sum = 0
        for i in range(C.T_FUTURE):

            state_difference = (np.array(s_other) + np.array(i * np.array(theta_other))) - (np.array(s_self) + np.array(sum(actions_x[:i]) + sum(actions_y[:i])))
            state_loss = 1 / np.linalg.norm(state_difference)
            intent_loss = np.linalg.norm(np.array(theta_self) - np.array((actions_x[i], actions_y[i]))) ** 2

            loss_sum += state_loss + c_self * intent_loss

        human_predicted_state = (np.array(s_other) + np.array(i * np.array(theta_other)))
        self.debug_1 = human_predicted_state[0]
        self.debug_2 = human_predicted_state[1]

        state_difference = np.array(s_other) - (np.array(s_self) + np.array(actions_x[0] + actions_y[0]))
        self.debug_3 = np.linalg.norm(state_difference)

        return loss_sum

    @staticmethod
    def get_human_predicted_intent():

        pass


