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

        self.state_loss = 0
        self.action_loss = 0

    def get_state(self):
        return self.machine_states[-1]

    def update(self, human_state):

        human_predicted_intent = (0, -1) # Implement correction function here
        human_criteria = 0.9 # Implement correction function here
        #human_predicted_intent, human_criteria = self.get_human_predicted_intent()
        #human_predicted_intent = (0, human_predicted_intent)

        self.human_states.append(human_state)

        # Use X Components (machine is restricted to x axis actions)
        machine_new_action = self.get_action(self.human_states[-1][0], self.machine_states[-1][0],
                                             human_predicted_intent[0], self.machine_intent[0],
                                             self.machine_criteria)
        machine_new_action = (machine_new_action, 0)

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

    def get_action(self, human_state, machine_state, human_intent, machine_intent, criteria):

        # Initial conditions
        a0 = [0 for _ in range(C.STEPS_FOR_CONSIDERATION)]

        # Bounded action space
        bounds = tuple([(-C.VEHICLE_MOVEMENT_LIMITER, C.VEHICLE_MOVEMENT_LIMITER) for _ in range(C.STEPS_FOR_CONSIDERATION)])

        optimization_results = scipy.optimize.minimize(self.loss_func, a0, bounds=bounds,
                                                       args=(human_state,
                                                             machine_state,
                                                             human_intent,
                                                             machine_intent,
                                                             criteria))

        actions = optimization_results.x

        return actions[0]  # Return first action

    def loss_func(self, actions, s_h, s_m, theta_h, theta_m, c_h):
        loss_sum = 0
        for i in range(len(actions)):
            state_loss = -((s_m + sum(actions[:(i + 1)])) - (s_h + theta_h)) ** 2
            action_loss = c_h * (theta_m - actions[i]) ** 2
            loss_sum += state_loss + action_loss

        self.state_loss = -((s_m + actions[0]) - (s_h + theta_h)) ** 2
        self.action_loss = c_h * (theta_m - actions[0]) ** 2

        return loss_sum

    @staticmethod
    def get_human_predicted_intent():

        pass


