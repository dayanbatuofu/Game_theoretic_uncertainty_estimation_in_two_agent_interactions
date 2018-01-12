from constants import CONSTANTS as C
import numpy as np
import copy


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

    def get_state(self):
        return self.machine_states[-1]

    def update(self, human_state):

        human_predicted_intent = (0, -1) # Implement correction function here
        human_criteria = 0.9 # Implement correction function here
        #human_predicted_intent, human_criteria = self.get_human_predicted_intent()
        #human_predicted_intent = (0, human_predicted_intent)

        self.human_states.append(human_state)

        # Use X Components (machine is restricted to x axis actions)
        machine_new_action = self.get_machine_action(10, self.human_states[-1][0], self.machine_states[-1][0],
                                             human_predicted_intent[0], self.machine_intent[0],
                                             self.machine_criteria)
        machine_new_action = (machine_new_action, 0)

        # Use Y Components (human is restricted to y axis actions)
        human_predicted_action = self.get_human_action(self.human_states[-1][1], self.machine_states[-1][1],
                                             human_predicted_intent[1], machine_new_action[1],
                                             human_criteria)
        human_predicted_action = (0, human_predicted_action)

        self.human_predicted_states.append(np.add(self.human_predicted_states[-1], human_predicted_action))

        self.update_state_action(machine_new_action)

    def update_state_action(self, action):

        self.machine_states.append(np.add(self.machine_states[-1], action))
        self.machine_actions.append(action)

    @staticmethod
    def get_machine_action(num_future_steps, human_state, machine_state, human_intent, machine_intent, criteria):

        T = num_future_steps

        # Define A
        A = np.zeros(shape=(T, T))
        for i in range(T):
            for j in range(i, T):
                A[i][j] = T-j

        # Define b
        # assume human actions
        a_h = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        c_t = np.zeros(shape=(T, 1))
        for i in range(T):
            c_t[i] = human_state + sum(a_h[:i+1]) - machine_state

        b = np.zeros(shape=(T, 1))
        for i in range(T):
            c_t_summed = copy.copy(c_t)
            c_t_summed[i+1:] = 0
            b[i] = 2*c_t[i]*np.sum(c_t_summed) - 2*criteria*machine_intent

        A_matrix = -2*A + 2*criteria*np.identity(T)
        b_matrix = -1*b

        a = np.multiply(np.linalg.pinv(A_matrix), b_matrix)

        # action_space = [-C.VEHICLE_MOVEMENT_SPEED, 0, C.VEHICLE_MOVEMENT_SPEED]
        # state_space = []
        #
        # for action in action_space:
        #     human_future_state = human_state + human_intent * C.STEPS_FOR_CONSIDERATION
        #     machine_future_state = machine_state + action * C.STEPS_FOR_CONSIDERATION
        #
        #     state_norm = 1/np.abs(human_future_state - machine_future_state + C.EPS)
        #     intent_norm = criteria*(machine_intent - action)**2
        #
        #     if np.abs(human_state-machine_state) >= C.VEHICLE_CLEARANCE:
        #         state_norm = 0
        #
        #     state_space.append(state_norm + intent_norm)

        #return action_space[np.argmin(state_space)]

        return a[0][0]

    @staticmethod
    def get_human_action(human_state, machine_state, human_intent, machine_intent, criteria):

        action_space = [-C.VEHICLE_MOVEMENT_SPEED, 0, C.VEHICLE_MOVEMENT_SPEED]
        state_space = []

        for action in action_space:

            human_future_state = human_state + human_intent * C.STEPS_FOR_CONSIDERATION
            machine_future_state = machine_state + action * C.STEPS_FOR_CONSIDERATION

            state_norm = 1/np.abs(human_future_state - machine_future_state + C.EPS)
            intent_norm = criteria*(machine_intent - action)**2
            state_space.append(state_norm + intent_norm)

        return action_space[np.argmin(state_space)]

    @staticmethod
    def get_human_predicted_intent():

        pass


