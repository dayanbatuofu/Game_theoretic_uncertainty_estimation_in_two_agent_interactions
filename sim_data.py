import pickle

class Sim_Data():

    def __init__(self):

        self.human_state = []
        self.human_predicted_theta = []
        self.human_action_set = []

        self.machine_state = []
        self.machine_theta = []
        self.machine_predicted_theta = []
        self.machine_action_set = []
        self.machine_predicted_action_set = []

    def append(self, human_state, human_predicted_theta, human_action_set, machine_state, machine_theta, machine_predicted_theta, machine_previous_action_set, machine_predicted_action_set):

        self.human_state.append(human_state)
        self.human_predicted_theta.append(human_predicted_theta)
        self.human_action_set.append(human_action_set)

        self.machine_state.append(machine_state)
        self.machine_theta.append(machine_theta)
        self.machine_predicted_theta.append(machine_predicted_theta)
        self.machine_action_set.append(machine_previous_action_set)
        self.machine_predicted_action_set.append(machine_predicted_action_set)
