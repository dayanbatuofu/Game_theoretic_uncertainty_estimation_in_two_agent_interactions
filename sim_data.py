import pickle

class Sim_Data():

    def __init__(self):

        self.machine_states_set = [] #actual states
        self.machine_trajectory_set = [] #actual trajectory (control variables)
        self.machine_actions_set = []#actual actions (converted from trajectory)

        self.machine_theta_set = [] #this is constant for now

        self.human_predicted_theta_set = [] #my prediction of the agents theta
        self.machine_expected_actions_set = [] #my prediction of the agent's expectation of my actions
        self.machine_planed_actions_set = [] #my current plan of actions

        self.human_predicted_trajectory_set = [] #my prediction of the agent's trajectory
        self.human_predicted_actions_set = [] #converted from above

    def append(self, machine_states_set, machine_actions_set, machine_theta, human_predicted_theta, machine_expected_actions,
               machine_trajectory, machine_planed_actions_set, human_predicted_trajectory, human_predicted_actions):

        self.machine_states_set = machine_states_set
        self.machine_trajectory_set.append(machine_trajectory)
        self.machine_actions_set = machine_actions_set
        self.machine_planed_actions_set = machine_planed_actions_set

        self.machine_theta_set.append(machine_theta)

        self.human_predicted_theta_set.append(human_predicted_theta)
        self.machine_expected_actions_set.append(machine_expected_actions)

        self.human_predicted_trajectory_set.append(human_predicted_trajectory)
        self.human_predicted_actions_set.append(human_predicted_actions)
