import pickle

class Sim_Data():

    def __init__(self):

        self.car1_states = [] #actual states
        self.car1_actions = []#actual actions (converted from trajectory)
        self.car1_theta = [] #this is constant for now
        self.car1_planned_action_sets = []
        self.car1_predicted_theta_other = [] #my prediction of the agents theta
        self.car1_predicted_theta_self = []  # my prediction of the other's prediction of my theta
        self.car1_predicted_actions_other = []  # converted from above
        self.car1_predicted_others_prediction_of_my_actions = []

        self.car2_states = []  # actual states
        self.car2_actions = []  # actual actions (converted from trajectory)
        self.car2_theta = []  # this is constant for now
        self.car2_planned_action_sets = []
        self.car2_predicted_theta_other = []  # my prediction of the agents theta
        self.car2_predicted_theta_self = []  # my prediction of the other's prediction of my theta
        self.car2_predicted_actions_other = []  # converted from above
        self.car2_predicted_others_prediction_of_my_actions = []

    def append_car1(self, states, actions, action_sets):
        # , predicted_theta_other, predicted_theta_self,
        #             predicted_actions_other, predicted_others_prediction_of_my_actions):

        self.car1_states = states
        self.car1_actions = actions
        self.car1_planned_action_sets.append(action_sets)
        # self.car1_predicted_theta_other.append(predicted_theta_other)
        # self.car1_predicted_theta_self.append(predicted_theta_self)
        # self.car1_predicted_actions_other.append(predicted_actions_other)
        # self.car1_predicted_others_prediction_of_my_actions.append(predicted_others_prediction_of_my_actions)

    def append_car2(self, states, actions, action_sets, predicted_theta_other, predicted_theta_self,
                    predicted_actions_other, predicted_others_prediction_of_my_actions):

        self.car2_states = states
        self.car2_actions = actions
        self.car2_planned_action_sets.append(action_sets)
        self.car2_predicted_theta_other.append(predicted_theta_other)
        self.car2_predicted_theta_self.append(predicted_theta_self)
        self.car2_predicted_actions_other.append(predicted_actions_other)
        self.car2_predicted_others_prediction_of_my_actions.append(predicted_others_prediction_of_my_actions)