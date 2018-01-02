from constants import CONSTANTS as C
import numpy as np


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

    def get_state(self):
        return self.machine_states[-1]

    def update(self, human_state):

        #human_predicted_intent = (0, -1) # Implement correction function here
        #human_criteria = 0.9 # Implement correction function here
        human_predicted_intent, human_criteria = self.get_human_predicted_intent()

        self.human_states.append(human_state)

        # Use X Components (machine is restricted to x axis actions)
        machine_new_action = self.get_machine_action(self.human_states[-1][0], self.machine_states[-1][0],
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
    def get_machine_action(human_state, machine_state, human_intent, machine_intent, criteria):

        action_space = [-C.VEHICLE_MOVEMENT_SPEED, 0, C.VEHICLE_MOVEMENT_SPEED]
        state_space = []

        for action in action_space:
            human_future_state = human_state + human_intent * C.STEPS_FOR_CONSIDERATION
            machine_future_state = machine_state + action * C.STEPS_FOR_CONSIDERATION

            state_norm = 1/np.abs(human_future_state - machine_future_state)
            intent_norm = criteria*(machine_intent - action)**2

            if np.abs(human_state-machine_state) >= C.VEHICLE_CLEARANCE:
                state_norm = 0

            state_space.append(state_norm + intent_norm)

        return action_space[np.argmin(state_space)]

    @staticmethod
    def get_human_action(human_state, machine_state, human_intent, machine_intent, criteria):

        action_space = [-C.VEHICLE_MOVEMENT_SPEED, 0, C.VEHICLE_MOVEMENT_SPEED]
        state_space = []

        for action in action_space:

            human_future_state = human_state + human_intent * C.STEPS_FOR_CONSIDERATION
            machine_future_state = machine_state + action * C.STEPS_FOR_CONSIDERATION

            state_norm = 1/np.abs(human_future_state - machine_future_state)
            intent_norm = criteria*(machine_intent - action)**2
            state_space.append(state_norm + intent_norm)

        return action_space[np.argmin(state_space)]

    @staticmethod
    def get_human_predicted_intent(self):
        pass


