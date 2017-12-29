from constants import CONSTANTS as C
import numpy as np


class MachineVehicle:

    '''
    States:
            X-Position
            Y-Position
    '''

    def __init__(self, machine_initial_state, human_initial_state):

        self.machine_states = [machine_initial_state]
        self.machine_actions = [(0, 0)]

        self.human_states = [human_initial_state]
        self.human_actions = [(0, 0)]
        self.human_predicted_states = [human_initial_state]

    def get_state(self):
        return self.machine_states[-1]

    def update(self, human_state):


        human_predicted_intent = (0, -1) # Implement correction function here
        human_criteria = 1 # Implement correction function here

        self.human_states.append(human_state)

        human_predicted_action = human_predicted_intent
        machine_new_action = self.get_action(human_predicted_intent, human_criteria)


        self.human_predicted_states.append(self.human_predicted_states[-1] + human_predicted_action)
        self.machine_states.append(self.machine_states[-1] + machine_new_action)

        self.update_state_action(machine_new_action)

    def update_state_action(self, action):

        self.machine_states.append(np.add(self.machine_states[-1], action))
        self.machine_actions.append(action)

    def get_action(self, human_predicted_intent, human_criteria):

        state_norms = []
        for human_state, machine_state in zip(self.human_predicted_states, self.machine_states):
            state_norms.append(np.linalg.norm(np.subtract(human_state, machine_state)))

        pass

    def choose_action(self, human_predicted_action):
        # Implement loss function minimization here
        pass