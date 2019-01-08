import pickle

class Sim_Data():

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.states = [[]] * self.num_agents #actual states
        self.actions = [[]] * self.num_agents #actual actions (converted from trajectory)
        self.theta = [[]] * self.num_agents #this is constant for now
        self.planned_action_sets = [[]] * self.num_agents
        self.planned_trajectory_set = [[]] * self.num_agents
        self.predicted_theta_other = [[]] * self.num_agents #my prediction of the agents theta
        self.predicted_theta_self = [[]] * self.num_agents  # my prediction of the other's prediction of my theta
        self.predicted_actions_other = [[]] * self.num_agents  # converted from above
        self.predicted_others_prediction_of_my_actions = [[]] * self.num_agents
        self.wanted_trajectory_self = [[]] * self.num_agents
        self.wanted_trajectory_other = [[]] * self.num_agents
        self.wanted_states_other = [[]] * self.num_agents
        self.inference_probability = [[]] * self.num_agents
        self.inference_probability_proactive = [[]] * self.num_agents
        self.theta_probability = [[]] * self.num_agents
        self.gracefulness = [[]] * self.num_agents

    def record(self, agents):
        for i in range(self.num_agents):
            self.states[i] = agents[i].states
            self.actions[i] = agents[i].actions_set
            self.planned_trajectory_set[i] = agents[i].planned_trajectory_set
            self.planned_action_sets[i].append(agents[i].planned_actions_set)
            self.predicted_theta_other[i].append(agents[i].predicted_theta_other)
            self.predicted_theta_self[i].append(agents[i].predicted_theta_self)
            self.predicted_actions_other[i].append(agents[i].predicted_actions_other)
            self.predicted_others_prediction_of_my_actions[i].append(agents[i].predicted_others_prediction_of_my_actions)
            self.wanted_trajectory_self[i].append(agents[i].wanted_trajectory_self)
            self.wanted_trajectory_other[i].append(agents[i].wanted_trajectory_other)
            self.wanted_states_other[i].append(agents[i].wanted_states_other)
            self.inference_probability[i].append(agents[i].inference_probability)
            self.inference_probability_proactive[i].append(agents[i].inference_probability_proactive)
            self.theta_probability[i].append(agents[i].theta_probability)
            self.gracefulness[i] = agents[i].social_gracefulness