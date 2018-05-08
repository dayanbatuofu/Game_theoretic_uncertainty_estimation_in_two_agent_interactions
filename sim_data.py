import pickle

class Sim_Data():

    def __init__(self):

        self.car1_states = [] #actual states
        self.car1_actions = []#actual actions (converted from trajectory)
        self.car1_theta = [] #this is constant for now
        self.car1_predicted_theta_of_other = [] #my prediction of the agents theta
        self.car1_prediction_of_actions_of_other = []  # converted from above
        self.car1_prediction_of_others_prediction_of_my_actions = []

        self.car2_states = []  # actual states
        self.car2_actions = []  # actual actions (converted from trajectory)
        self.car2_theta = []  # this is constant for now
        self.car2_predicted_theta_of_other = []  # my prediction of the agents theta
        self.car2_prediction_of_actions_of_other = []  # converted from above
        self.car2_prediction_of_others_prediction_of_my_actions = []

    def append_car1(self, states, actions, predicted_theta_of_other, prediction_of_actions_of_other, prediction_of_others_prediction_of_my_actions):

        self.car1_states = states
        self.car1_actions.append(actions)
        self.car1_predicted_theta_of_other.append(predicted_theta_of_other)
        self.car1_prediction_of_actions_of_other.append(prediction_of_actions_of_other)
        self.car1_prediction_of_others_prediction_of_my_actions.append(prediction_of_others_prediction_of_my_actions)

    def append_car2(self, states, actions, theta, predicted_theta_of_other, prediction_of_actions_of_other, prediction_of_others_prediction_of_my_actions):

        self.car2_states = states
        self.car2_actions.append(actions)
        self.car2_predicted_theta_of_other.append(predicted_theta_of_other)
        self.car2_prediction_of_actions_of_other.append(prediction_of_actions_of_other)
        self.car2_prediction_of_others_prediction_of_my_actions.append(prediction_of_others_prediction_of_my_actions)
