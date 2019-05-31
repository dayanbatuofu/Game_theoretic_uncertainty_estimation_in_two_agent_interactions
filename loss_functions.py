from numpy.core.multiarray import ndarray

from constants import CONSTANTS as C
from constants import MATRICES as M
import bezier
import numpy as np
import matplotlib.pyplot as plt
import time


class LossFunctions:
    """
    0 - aggressive
    1 - reactive (intersection)
    2 - passive aggressive
    """

    def __init__(self, characterization):

        self.characterization = characterization

    def loss(self, guess, autonomous_vehicle, guess_other):

        if self.characterization is "aggressive":
            return self.aggressive_loss(guess, autonomous_vehicle)

        elif self.characterization is "reactive":
            return self.reactive_multisearch(guess, guess_other, autonomous_vehicle)

        elif self.characterization is "passive_aggressive":
            return self.passive_aggressive_loss(guess, guess_other, autonomous_vehicle)

        else:
            raise ValueError('incorrect loss function characterization specified.')

    def aggressive_loss(self, trajectory, s):
        who = s.who
        o = s.other_car
        state_s = s.states[-1]
        state_o = s.states_o[-1]
        loss_total = 0
        for other_intent, other_intent_p in \
                zip(s.predicted_theta_other, s.inference_probability):
            ##############################################################################################
            # predict how others perceive your action
            trajectory_other_all, inference_probability = self.other_agent_response(agent=s, action=trajectory, other_agent_intent=other_intent)
            ##############################################################################################

        # ##############################################################################################
        # # predict how others perceive your action
        # trajectory_other_all, inference_probability = self.multi_search_intent(trajectory, s)
        # ##############################################################################################

            loss_all = []
            for trajectory_other in trajectory_other_all:
                # actions_self = s.interpolate_from_trajectory(trajectory)
                # actions_other = o.interpolate_from_trajectory(trajectory_other)
                #
                # s_other_predict = np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other)
                # s_self_predict = state_s + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self)

                s_other_predict, s_other_predict_vel = self.dynamic(trajectory_other, s)
                s_self_predict, s_self_predict_vel = self.dynamic(trajectory, s)
                D = s.collision_box.get_collision_loss(s_self_predict, s_other_predict, o.collision_box) + 1e-12
                gap = 1.05  # TODO: generalize this
                for i in range(s_self_predict.shape[0]):
                    if trajectory[1] == 0:
                        if s_self_predict[i, 0] <= -gap + 1e-12 or s_self_predict[i, 0] >= gap - 1e-12 or s_other_predict[
                            i, 1] >= gap - 1e-12 or s_other_predict[i, 1] <= -gap + 1e-12:
                            D[i] = np.inf
                    else:
                        if s_self_predict[i, 1] <= -gap + 1e-12 or s_self_predict[i, 1] >= gap - 1e-12 or s_other_predict[
                            i, 0] >= gap - 1e-12 or s_other_predict[i, 0] <= -gap + 1e-12:
                            D[i] = np.inf
                collision_loss = np.sum(np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5)))
                # collision_loss = autonomous_vehicle.collision_box_s.get_collision_loss(s_self_predict, s_other_predict, autonomous_vehicle.collision_box_o)

                if trajectory[1] == 0:
                    intent_loss = s.intent * np.exp(C.EXPTHETA * (s.P_CAR.DESIRED_POSITION[0] - s_self_predict[-1][0]))
                else:
                    intent_loss = s.intent * np.exp(C.EXPTHETA * (-s.P_CAR.DESIRED_POSITION[1] + s_self_predict[-1][1]))

                loss_all.append(collision_loss + intent_loss)
            loss_total += sum(np.array(loss_all) * np.array(inference_probability)) * other_intent_p
        return loss_total, trajectory_other_all, inference_probability  # Return weighted sum
        # return sum(np.array(loss_all) * np.array(
        #     inference_probability)), trajectory_other_all, inference_probability  # Return weighted sum

    def reactive_multisearch(self, guess_self, guess_other, s):
        who = s.who

        """ run multiple searches with different initial guesses """
        # predicted_trajectory_self = autonomous_vehicle.prediction_of_others_prediction_of_my_trajectory
        # theta_other = autonomous_vehicle.predicted_theta_of_other
        #
        # # FOR OTHER
        # trajectory_set = np.empty((0, 2))  # TODO: need to generalize
        # loss_value_set = []
        # for guess in guess_other:
        #
        #     loss = self.reactive_loss(guess, autonomous_vehicle.states_s[-1], autonomous_vehicle.states_o[-1],
        #                               predicted_trajectory_self, theta_other,
        #                               autonomous_vehicle.collision_box_s,
        #                               autonomous_vehicle.collision_box_o, autonomous_vehicle.P_CAR_S.ORIENTATION,
        #                               autonomous_vehicle.P_CAR_O.ORIENTATION, 1-who)
        #
        #     trajectory_set = np.append(trajectory_set, [guess], axis=0)
        #     loss_value_set = np.append(loss_value_set, loss)

        trajectory_other = s.predicted_trajectory_other

        # FOR SELF
        trajectory_set = np.empty((0, 2))  # TODO: need to generalize
        loss_value_set = []
        for guess in guess_self:
            if len(s.states) == 1:

                loss = self.reactive_loss(s.intent, [guess], trajectory_other, s.inference_probability, s.states[-1],
                                          [0, 0], [0, 0],
                                          s.states_o[-1], [0, 0], [0, 0], s)

            elif len(s.states) == 2:
                loss = self.reactive_loss(s.intent, [guess], trajectory_other, s.inference_probability,
                                          s.states[-1], s.states[-1] - s.states[-2], s.states[-1] - s.states[-2],
                                          s.states_o[-1], s.states_o[-1] - s.states_o[-2],
                                          s.states_o[-1] - s.states_o[-2], s)

            else:
                loss = self.reactive_loss(s.intent, [guess], trajectory_other, s.inference_probability,
                                          s.states[-1], s.states[-1] - s.states[-2],
                                          (s.states[-1] - s.states[-2]) - (s.states[-2] - s.states[-3]),
                                          s.states_o[-1], s.states_o[-1] - s.states_o[-2],
                                          (s.states_o[-1] - s.states_o[-2]) - (s.states_o[-2] - s.states_o[-3]),
                                          s)

            trajectory_set = np.append(trajectory_set, [guess], axis=0)
            loss_value_set = np.append(loss_value_set, loss)
        # if s.who == 0:
        #     print loss_value_set

        trajectory_self = trajectory_set[np.where(loss_value_set == np.min(loss_value_set))[0][0]]
        loss_self = np.min(loss_value_set)
        return trajectory_self

    def reactive_loss(self, theta_self, trajectory, trajectory_other, probability, s_self, s_self_vel, s_self_acc,
                      s_other, s_other_vel, s_other_acc, s):

        """ Loss function defined to be a combination of state_loss and intent_loss with a weighted factor c """
        loss_all = 0
        o = s.other_car
        box_self = s.collision_box
        box_other = o.collision_box
        s_who = s.who
        o_who = o.who
        #
        # s_ability = s.P_CAR.ABILITY
        # o_ability = o.P_CAR.ABILITY

        for t_s in trajectory:
            loss = []
            for t_o in trajectory_other:

                s_other_predict, s_other_predict_vel = self.dynamic(t_o, s)
                s_self_predict, s_self_predict_vel = self.dynamic(t_s, s)

                D = box_self.get_collision_loss(s_self_predict, s_other_predict, box_other) + 1e-12
                gap = 1.05  # TODO: generalize this
                for i in range(s_self_predict.shape[0]):
                    if t_s[1] == 0:
                        if s_self_predict[i, 0] <= -gap + 1e-12 or s_self_predict[i, 0] >= gap - 1e-12 or \
                                s_other_predict[i, 1] >= gap - 1e-12 or s_other_predict[i, 1] <= -gap + 1e-12:
                            D[i] = np.inf
                    else:
                        if s_self_predict[i, 1] <= -gap + 1e-12 or s_self_predict[i, 1] >= gap - 1e-12 or \
                                s_other_predict[i, 0] >= gap - 1e-12 or s_other_predict[i, 0] <= -gap + 1e-12:
                            D[i] = np.inf

                collision_loss = np.sum(np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5)))

                if t_s[1] == 0:
                    intent_loss = theta_self * np.exp(C.EXPTHETA * (- s_self_predict[-1][0] + 0.6))
                else:
                    intent_loss = theta_self * np.exp(C.EXPTHETA * (s_self_predict[-1][1] + 0.6))

                loss.append(collision_loss + intent_loss)
            loss_all += sum(np.array(loss) * np.array(probability))
        # print time.time()pychre
        return loss_all  # Return weighted sum

    def courteous_baseline_loss(self, agent, action, other_agent_intent):  # loss of other when self uses s.wanted_trajectory_self
        s = agent
        who = s.who
        o = s.other_car
        state_s = s.states[-1]
        state_o = s.states_o[-1]

        # predict how others perceive your action
        trajectory_other_all, inference_probability = \
            self.other_agent_response(action=action, agent=s, other_agent_intent=other_agent_intent)

        loss = 0.
        # what other will do if I did trajectory
        for trajectory_other, p in zip(trajectory_other_all, inference_probability):
            collision_loss = self.calculate_collision_loss(
                agent=s, action=action, other_agent=o, other_action=trajectory_other)
            intent_loss_other = self.calculate_intent_loss(
                agent=o, action=trajectory_other, agent_intent=other_agent_intent)
            other_loss = collision_loss + intent_loss_other
            loss += other_loss*p
        return loss  # Return weighted sum

    def passive_aggressive_loss(self, trajectory, guess_other, s):
        who = s.who
        o = s.other_car
        state_s = s.states[-1]
        state_o = s.states_o[-1]
        loss_total = 0
        for other_intent, other_intent_p in \
                zip(s.predicted_theta_other, s.inference_probability):
            ##############################################################################################
            # predict how others perceive your action
            trajectory_other_all, inference_probability = self.other_agent_response(agent=s, action=trajectory, other_agent_intent=other_intent)
            ##############################################################################################

            loss_all = []
            for trajectory_other in trajectory_other_all:  # what other will do if I did trajectory
                s_other_predict, s_other_predict_vel = self.dynamic(trajectory_other, s)
                s_self_predict, s_self_predict_vel = self.dynamic(trajectory, s)
                D = s.collision_box.get_collision_loss(s_self_predict, s_other_predict, o.collision_box) + 1e-12
                gap = 1.05  # TODO: generalize this
                for i in range(s_self_predict.shape[0]):
                    if trajectory[1] == 0:
                        if s_self_predict[i, 0] <= -gap + 1e-12 or s_self_predict[i, 0] >= gap - 1e-12 or s_other_predict[
                           i, 1] >= gap - 1e-12 or s_other_predict[i, 1] <= -gap + 1e-12:
                            D[i] = np.inf
                    else:
                        if s_self_predict[i, 1] <= -gap + 1e-12 or s_self_predict[i, 1] >= gap - 1e-12 or s_other_predict[
                           i, 0] >= gap - 1e-12 or s_other_predict[i, 0] <= -gap + 1e-12:
                            D[i] = np.inf
                collision_loss = np.sum(np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5)))

                if trajectory[1] == 0:
                    intent_loss = s.intent * np.exp(C.EXPTHETA * (s.P_CAR.DESIRED_POSITION[0] - s_self_predict[-1][0]))
                else:
                    intent_loss = s.intent * np.exp(C.EXPTHETA * (-s.P_CAR.DESIRED_POSITION[1] + s_self_predict[-1][1]))

                gracefulness_loss = []
                for wanted_trajectory_self in s.wanted_trajectory_self:  # what other want me to do right now
                    gracefulness_loss.append((trajectory[0] - wanted_trajectory_self[0]) ** 2)

                loss_all.append(collision_loss + intent_loss + 0.001 * sum(gracefulness_loss * s.inference_probability))
            loss_total += sum(np.array(loss_all) * np.array(inference_probability)) * other_intent_p
        return loss_total, trajectory_other_all, inference_probability  # Return weighted sum

    def berkeley_courtesy_loss(self, agent, action, baseline, beta):
        trajectory = action
        s = agent
        who = s.who
        o = s.other_car
        state_s = s.states[-1]
        state_o = s.states_o[-1]

        total_loss = 0.
        for other_intent, other_intent_p, baseline_loss in \
                zip(s.predicted_theta_other, s.inference_probability, baseline):
            # predict how others perceive your action
            trajectory_other_all, inference_probability = \
                self.other_agent_response(agent=s, action=trajectory, other_agent_intent=other_intent)

            loss = 0.
            # what other will do if I did trajectory
            for trajectory_other, p in zip(trajectory_other_all, inference_probability):
                collision_loss = self.calculate_collision_loss(
                    agent=s, action=trajectory, other_agent=o, other_action=trajectory_other)

                intent_loss = self.calculate_intent_loss(agent=s, action=trajectory, agent_intent=s.intent)
                intent_loss_other = self.calculate_intent_loss(
                    agent=o, action=trajectory_other, agent_intent=other_intent)
                gracefulness_loss = max([collision_loss + intent_loss_other - baseline_loss, 0])

                loss += (collision_loss + intent_loss + beta * gracefulness_loss) * p

            total_loss += loss * other_intent_p
        return total_loss  # Return weighted sum

    def other_agent_response(self, agent, action, other_agent_intent):
        """ run multiple searches with different initial guesses """
        s = agent
        trajectory = action

        who = s.who
        trials_theta = C.THETA_SET
        inference_set = []  # TODO: need to generalize
        loss_value_set = []

        if other_agent_intent is []:
            trials_theta_other = trials_theta
        else:
            trials_theta_other = [other_agent_intent]

        for theta_self in trials_theta:
            for theta_other in trials_theta_other:
                trajectory_other = self.best_trajectory(theta_self, theta_other, s, s.other_car, trajectory)

                inference_set.append([theta_self,
                                      theta_other,
                                      trajectory_other,
                                      1. / len(trajectory_other)])

        theta_self_out = []
        theta_other_out = []
        trajectory_other_out = []
        inference_probability_out = []

        for i in range(len(inference_set)):
            for j in range(len(inference_set[i][2])):
                theta_self_out.append(inference_set[i][0])
                theta_other_out.append(inference_set[i][1])
                trajectory_other_out.append(inference_set[i][2][j])
                inference_probability_out.append(1. / len(inference_set) * inference_set[i][3])

        inference_probability_out = np.array(inference_probability_out)
        # update inference probability accordingly
        for i in range(len(trials_theta_other)):
            id = np.where(theta_other_out == trials_theta[i])[0]
            inference_probability_out[id] = inference_probability_out[id] / \
                                            sum(inference_probability_out[id]) * s.theta_probability[i]
        inference_probability_out = inference_probability_out / sum(inference_probability_out)

        return trajectory_other_out, inference_probability_out

    def equilibrium(self, theta_self, theta_other, s, o):
        action_guess = C.TRAJECTORY_SET
        trials_trajectory_self = np.hstack((np.expand_dims(action_guess, axis=1),
                                            np.ones((action_guess.size, 1)) * s.P_CAR.ORIENTATION))
        trials_trajectory_other = np.hstack((np.expand_dims(action_guess, axis=1),
                                             np.ones((action_guess.size, 1)) * o.P_CAR.ORIENTATION))

        loss_matrix = np.zeros((trials_trajectory_self.shape[0], trials_trajectory_other.shape[0], 2))
        for i in range(trials_trajectory_self.shape[0]):
            for j in range(trials_trajectory_other.shape[0]):
                loss_matrix[i, j, :] = self.simulate_game([trials_trajectory_self[i]], [trials_trajectory_other[j]],
                                                          theta_self, theta_other, s, o)

        # find equilibrium
        my_loss_all = []
        other_loss_all = []
        eq_all = []
        for j in range(trials_trajectory_other.shape[0]):
            id_s = np.atleast_1d(np.argmin(loss_matrix[:, j, 0]))
            for i in range(id_s.size):
                id_o = np.atleast_1d(np.argmin(loss_matrix[id_s[i], :, 1]))
                if sum(np.isin(id_o, j)) > 0:
                    eq_all.append([id_s[i], j])
                    my_loss_all.append(loss_matrix[id_s[i], j, 0])
                    other_loss_all.append(loss_matrix[id_s[i], j, 1])

        # eq = [eq_all[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]

        if eq_all is not []:
            trajectory_self = [trials_trajectory_self[eq_all[i][0]] for i in range(len(eq_all))]
            trajectory_other = [trials_trajectory_other[eq_all[i][1]] for i in range(len(eq_all))]
        else:
            trajectory_self = []
            trajectory_other = []
        return trajectory_self, trajectory_other, my_loss_all, other_loss_all

    def best_trajectory(self, theta_self, theta_other, s, o, t):
        action_guess = C.TRAJECTORY_SET
        trials_trajectory_self = t
        trials_trajectory_other = np.hstack((np.expand_dims(action_guess, axis=1),
                                             np.ones((action_guess.size, 1)) * o.P_CAR.ORIENTATION))
        loss_matrix = np.zeros((trials_trajectory_other.shape[0], 2))
        for j in range(trials_trajectory_other.shape[0]):
            loss_matrix[j, :] = self.simulate_game([t], [trials_trajectory_other[j]], theta_self, theta_other, s, o)

        # find equilibrium
        my_loss_all = []
        other_loss_all = []
        eq_all = []
        id_o = np.atleast_1d(np.argmin(loss_matrix[:, 1]))

        if id_o is not []:
            trajectory_other = [trials_trajectory_other[id_o[i]] for i in range(len(id_o))]
        else:
            trajectory_other = []

        return trajectory_other

    def simulate_game(self, trajectory_self, trajectory_other, theta_self, theta_other, s,
                      o):  ## no velocity passed and code was working before :O????????
        if len(s.states) == 1:
            loss_s = self.reactive_loss(theta_self, trajectory_self, trajectory_other, [1],
                                        s.states[-1],
                                        [0, 0], [0, 0],
                                        s.states_o[-1],
                                        [0, 0], [0, 0],
                                        s)
            loss_o = self.reactive_loss(theta_other, trajectory_other, trajectory_self, [1],
                                        s.states_o[-1],
                                        [0, 0], [0, 0],
                                        s.states[-1],
                                        [0, 0], [0, 0],
                                        o)
        elif len(s.states) == 2:
            loss_s = self.reactive_loss(theta_self, trajectory_self, trajectory_other, [1],
                                        s.states[-1],
                                        s.states[-1] - s.states[-2], s.states[-1] - s.states[-2],
                                        s.states_o[-1],
                                        s.states_o[-1] - s.states_o[-2], s.states_o[-1] - s.states_o[-2],
                                        s)
            loss_o = self.reactive_loss(theta_other, trajectory_other, trajectory_self, [1],
                                        s.states_o[-1],
                                        s.states_o[-1] - s.states_o[-2], s.states_o[-1] - s.states_o[-2],
                                        s.states[-1],
                                        s.states[-1] - s.states[-2], s.states[-1] - s.states[-2],
                                        o)
        else:
            loss_s = self.reactive_loss(theta_self, trajectory_self, trajectory_other, [1],
                                        s.states[-1],
                                        s.states[-1] - s.states[-2],
                                        (s.states[-1] - s.states[-2]) - (s.states[-2] - s.states[-3]),
                                        s.states_o[-1],
                                        s.states_o[-1] - s.states_o[-2],
                                        (s.states_o[-1] - s.states_o[-2]) - (s.states_o[-2] - s.states_o[-3]),
                                        s)
            loss_o = self.reactive_loss(theta_other, trajectory_other, trajectory_self, [1],
                                        s.states_o[-1],
                                        s.states_o[-1] - s.states_o[-2],
                                        (s.states_o[-1] - s.states_o[-2]) - (s.states_o[-2] - s.states_o[-3]),
                                        s.states[-1],
                                        s.states[-1] - s.states[-2],
                                        (s.states[-1] - s.states[-2]) - (s.states[-2] - s.states[-3]),
                                        o)
        return loss_s, loss_o

    def intent_loss_func(self, trajectory, autonomous_vehicle, orientation_other, collision_box_self,
                         collision_box_other, state_self, state_other, action_other):
        who = (autonomous_vehicle.P_CAR_S.BOUND_X is None) + 0.0

        action_self = self.interpolate_from_trajectory(trajectory, state_other, orientation_other)
        trajectory_self = np.array(state_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, action_self))

        # actions and states of the agent
        action_other = np.array(C.ACTION_TIMESTEPS * [action_other])  # project current agent actions to future
        trajectory_other = np.array(state_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, action_other))

        # I expect the agent to be this much aggressive
        # theta_self = self.human_predicted_theta

        # calculate the gradient of the control objective
        A = M.LOWER_TRIANGULAR_MATRIX

        collision_loss = collision_box_other.get_collision_loss(trajectory_other, trajectory_self, collision_box_self)

        # dD/da
        # dDda_self = - np.dot(np.expand_dims(np.dot(A.transpose(), sigD**(-2)*dsigD),axis=1), np.expand_dims(ds, axis=0)) \
        #        - np.dot(np.dot(A.transpose(), np.diag(sigD**(-2)*dsigD)), np.dot(A, a_self - a_other))
        dDda_self = - 2 * C.EXPCOLLISION * np.dot(A.transpose(), (trajectory_other - trajectory_self) *
                                                  np.expand_dims(np.exp(
                                                      C.EXPCOLLISION * (-collision_loss + C.CAR_LENGTH ** 2 * 1.5)),
                                                                 axis=1))

        # update theta_hat_H
        w = - dDda_self  # negative gradient direction

        # if who == 0:
        #     if self.P.BOUND_HUMAN_X is not None: # intersection
        #         w[np.all([trajectory_other[:,0]<=1e-12, w[:,0] <= 1e-12], axis=0),0] = 0 #if against wall and push towards the wall, get a reaction force
        #         w[np.all([trajectory_other[:,0]>=-1e-12, w[:,0] >= -1e-12], axis=0),0] = 0 #TODO: these two lines are hard coded for intersection, need to check the interval
        #         # print(w)
        #     else: # lane changing
        #         w[np.all([trajectory_other[:,1]<=1e-12, w[:,1] <= 1e-12], axis=0),1] = 0 #if against wall and push towards the wall, get a reaction force
        #         w[np.all([trajectory_other[:,1]>=1-1e-12, w[:,1] >= -1e-12], axis=0),1] = 0 #TODO: these two lines are hard coded for lane changing
        # else:
        #     if self.P.BOUND_HUMAN_X is not None: # intersection
        #         w[np.all([trajectory_other[:,1]<=1e-12, w[:,1] <= 1e-12], axis=0),1] = 0 #if against wall and push towards the wall, get a reaction force
        #         w[np.all([trajectory_other[:,1]>=-1e-12, w[:,1] >= -1e-12], axis=0),1] = 0 #TODO: these two lines are hard coded for intersection, need to check the interval
        #     else: # lane changing
        #         w[np.all([trajectory_other[:,0]<=1e-12, w[:,0] <= 1e-12], axis=0),0] = 0 #if against wall and push towards the wall, get a reaction force
        #         w[np.all([trajectory_other[:,0]>=-1e-12, w[:,0] >= -1e-12], axis=0),0] = 0 #TODO: these two lines are hard coded for lane changing
        w = -w

        gap = 1.05
        if who == 1:
            l = np.array([- C.EXPTHETA * np.exp(C.EXPTHETA * (-trajectory_other[-1][0] + 0.4)), 0.])
            # don't take into account the time steps where one car has already passed
            decay = (((trajectory_other - trajectory_self)[:, 0] < gap) + 0.0) * (
                        (trajectory_other - trajectory_self)[:, 1] < gap + 0.0)
        else:
            l = np.array([0., C.EXPTHETA * np.exp(C.EXPTHETA * (trajectory_other[-1][1] + 0.4))])
            decay = (((trajectory_self - trajectory_other)[:, 0] < gap) + 0.0) * (
                        (trajectory_self - trajectory_other)[:, 1] < gap + 0.0)

        decay = decay * np.exp(np.linspace(0, -10, C.ACTION_TIMESTEPS))

        w = w * np.expand_dims(decay, axis=1)
        l = l * np.expand_dims(decay, axis=1)

        alpha = np.max((- np.trace(np.dot(np.transpose(w), l)) / (np.sum(l ** 2) + 1e-12), 0.1))
        x = w + alpha * l
        L = np.sum(x ** 2)
        return L, alpha

    def interpolate_from_trajectory(self, trajectory):

        nodes = np.array([[0, trajectory[0] * np.cos(np.deg2rad(trajectory[1])) / 2,
                           trajectory[0] * np.cos(np.deg2rad(trajectory[1]))],
                          [0, trajectory[0] * np.sin(np.deg2rad(trajectory[1])) / 2,
                           trajectory[0] * np.sin(np.deg2rad(trajectory[1]))]])

        curve = bezier.Curve(nodes, degree=2)

        positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_NUMPOINTS + 1)))
        # TODO: skip state?

        return np.diff(positions, n=1, axis=0)

    # def dynamic(self, action_self, vel_self, acc_self, state_self): # Dynamic with Quintic Polynomial Planning
    #     N = 100  # ??
    #     T = 1  # ??
    #     sf = np.asarray(action_self[-1]) + np.asarray(state_self)
    #     vel_0 = np.asarray(vel_self) / T  # initial vel??
    #     acc_0 = np.asarray(acc_self) / T / T  # initial acc??
    #     state_0 = np.asarray(state_self)  # initial state/position??
    #     # coefficient analytical calculations
    #     A = np.array([[1, 0, 0, 0, 0, 0],
    #                   [0, 1, 0, 0, 0, 0],
    #                   [0, 0, 2, 0, 0, 0],
    #                   [1, pow(T * N, 1), pow(T * N, 2), pow(T * N, 3), pow(T * N, 4), pow(T * N, 5)],
    #                   [0, 1, 2 * pow(T * N, 1), 3 * pow(T * N, 2),
    #                    4 * pow(T * N, 3), 5 * pow(T * N, 4)],
    #                   [0, 0, 2, 6 * pow(T * N, 1), 12 * pow(T * N, 2), 20 * pow(T * N, 3)]])
    #     b1 = np.array([state_0[0], vel_0[0], acc_0[0], sf[0], 0, 0])
    #     b2 = np.array([state_0[1], vel_0[1], acc_0[1], sf[1], 0, 0])
    #     coeffx = np.linalg.solve(A, b1)
    #     coeffy = np.linalg.solve(A, b2)
    #     a = A.dot(coeffx)
    #     b = np.polyval(coeffx, T * N)
    #     trajx = []
    #     trajy = []
    #     velx = []
    #     vely = []
    #     accx = []
    #     accy = []
    #     for t in range(0, N, 1):
    #         trajx.append(
    #             (coeffx[5] * (pow((T * t), 5))) + (coeffx[4] * (pow((T * t), 4))) + (coeffx[3] * (pow((T * t), 3))) + (
    #                     coeffx[2] * (pow((T * t), 2))) + (
    #                     coeffx[1] * (T * t)) + (coeffx[0]))
    #         trajy.append(
    #             (coeffy[5] * (pow((T * t), 5))) + (coeffy[4] * (pow((T * t), 4))) + (coeffy[3] * (pow((T * t), 3))) + (
    #                     coeffy[2] * (pow((T * t), 2))) + (
    #                     coeffy[1] * (T * t)) + (coeffy[0]))
    #
    #         velx.append(
    #             (5 * coeffx[5] * (pow((T * t), 4))) + (4 * coeffx[4] * (pow((T * t), 3))) + (
    #                     3 * coeffx[3] * (pow((T * t), 2))) + (
    #                     2 * coeffx[2] * (T * t)) + (coeffx[1]))
    #         vely.append(
    #             (5 * coeffy[5] * (pow((T * t), 4))) + (4 * coeffy[4] * (pow((T * t), 3))) + (
    #                     3 * coeffy[3] * (pow((T * t), 2))) + (
    #                     2 * coeffy[2] * (T * t)) + (coeffy[1]))
    #
    #         accx.append(
    #             (20 * coeffx[5] * pow((T * t), 3)) + (12 * coeffx[4] * (pow((T * t), 2))) + (
    #                         6 * coeffx[3] * (T * t)) + (
    #                     2 * coeffx[2]))
    #         accy.append(
    #             (20 * coeffy[5] * pow((T * t), 3)) + (12 * coeffy[4] * (pow((T * t), 2))) + (
    #                         6 * coeffy[3] * (T * t)) + (
    #                     2 * coeffy[2]))
    #
    #     predict_result_traj = np.column_stack((trajx, trajy))
    #     predict_result_vel = np.column_stack((velx, vely))
    #     predict_result_acc = np.column_stack((accx, accy))
    #     # print time.time()
    #     return predict_result_traj, predict_result_vel

    #
    #
    # def dynamic(self, action_self, vel_self, acc_self, state_self): # Dynamic of cubic polynomial acceleration as an action
    #      N = 100  # ??
    #      T = 1  # ??
    #
    #      vel_0 = np.asarray(vel_self) / T  # initial vel??
    # #     acc_0 = np.asarray(acc_self) / T / T  # initial acc??
    #      state_0 = np.asarray(state_self)  # initial state/position??
    #      sf = np.asarray(action_self[-1]) + state_0
    #      # print sf - state_0
    #      # coefficient analytical calculations
    #      A = np.array([[1, 0, 0, 0],
    #                    [0, 1, 0, 0],
    #                    [1, pow(T * N, 1), pow(T * N, 2), pow(T * N, 3)],
    #                    [0, 1, 2 * pow(T * N, 1), 3 * pow(T * N, 2)]])
    #      b1 = np.array([state_0[0], vel_0[0], sf[0], 0])
    #      b2 = np.array([state_0[1], vel_0[1], sf[1], 0])
    #      coeffx = np.linalg.solve(A, b1)
    #      coeffy = np.linalg.solve(A, b2)
    # #     a = A.dot(coeffx)
    # #     b = np.polyval(coeffx, T * N)
    #      trajx = []
    #      trajy = []
    #      velx = []
    #      vely = []
    # #     accx = []
    # #     accy = []
    #      for t in range(0, N, 1):
    #         trajx.append(
    #             ((coeffx[3] * (pow((T * t), 3))) + (coeffx[2] * (pow((T * t), 2))) + (coeffx[1] * (T * t)) + (coeffx[0])))
    #         trajy.append(
    #             ((coeffy[3] * (pow((T * t), 3))) + (coeffy[2] * (pow((T * t), 2))) + (coeffy[1] * (T * t)) + (coeffy[0])))
    #
    #         velx.append((3 * coeffx[3] * (pow((T * t), 2))) + (2 * coeffx[2] * (T * t)) + (coeffx[1]))
    #         vely.append((3 * coeffy[3] * (pow((T * t), 2))) + (2 * coeffy[2] * (T * t)) + (coeffy[1]))
    #
    #      predict_result_traj = np.column_stack((trajx, trajy))
    #      predict_result_vel = np.column_stack((velx, vely))
    #      return predict_result_traj, predict_result_vel

    # def dynamic(self, action_self, vel_self, acc_self, state_self, ability): # Dynamic of cubic polynomial on velocity
    #      N = 100  # ??
    #      T = 1  # ??
    #
    #      vel_0 = np.asarray(vel_self) / T  # type: ndarray # initial vel??
    #      state_0 = np.asarray(state_self)  # initial state/position??
    #      acci = np.asarray(action_self[-1]) * ability
    #      # vel_f = vel_0
    #      vel_f = np.array([0,0])
    #      # print sf - state_0
    #      # coefficient analytical calculations
    #      # print acci
    #      A = np.array([[1, 0, 0, 0],
    #                    [0, 1, 0, 0],
    #                    [1, pow(T * N, 1), pow(T * N, 2), pow(T * N, 3)],
    #                    [0, 1, 2 * pow(T * N, 1), 3 * pow(T * N, 2)]])
    #      b1 = np.array([vel_0[0],acci[0],vel_f[0],0])
    #      b2 = np.array([vel_0[1],acci[1],vel_f[1],0])
    #      coeffx = np.linalg.solve(A, b1)
    #      coeffy = np.linalg.solve(A, b2)
    # #     a = A.dot(coeffx)
    # #     b = np.polyval(coeffx, T * N)
    #      trajx = []
    #      trajy = []
    #      velx = []
    #      vely = []
    # #     accx = []
    # #     accy = []
    #      for t in range(0, N, 1):
    #         trajx.append(
    #             ((coeffx[3] * (pow((T * t), 4))/4) + (coeffx[2] * (pow((T * t), 3))/3) + (coeffx[1] * (pow((T * t), 2))/2) + (coeffx[0] * (T * t)) + state_0[0]))
    #         trajy.append(
    #             ((coeffy[3] * (pow((T * t), 4))/4) + (coeffy[2] * (pow((T * t), 3))/3) + (coeffy[1] * (pow((T * t), 2))/2) + (coeffy[0] * (T * t)) + state_0[1]))
    #
    #         velx.append(
    #             ((coeffx[3] * (pow((T * t), 3))) + (coeffx[2] * (pow((T * t), 2))) + (coeffx[1] * (T * t)) + (coeffx[0])))
    #         vely.append(
    #             ((coeffy[3] * (pow((T * t), 3))) + (coeffy[2] * (pow((T * t), 2))) + (coeffy[1] * (T * t)) + (coeffy[0])))
    #      predict_result_traj = np.column_stack((trajx, trajy))
    #      predict_result_vel = np.column_stack((velx, vely))
    #
    #      return predict_result_traj, predict_result_vel
    #
    # dynamic with acc as a linear function
    # def dynamic(self, action_self, vel_self, acc_self, state_self, ability,s): # Dynamic of cubic polynomial on velocity
    #      N = 100  # ??
    #      T = 1  # ??
    #
    #      vel_0 = np.asarray(vel_self) / T  # type: ndarray # initial vel??
    #      state_0 = np.asarray(state_self)  # initial state/position??
    #
    #      if s.who == 1:
    #          acci = np.array([action_self[0] * ability, 0])
    #      else:
    #          acci = np.array([0, -action_self[0] * ability])
    #      # print sf - state_0
    #      # coefficient analytical calculations
    #      # print acci
    #      A = np.array([[1, 0, 0, 0],
    #                    [0, 1, 0, 0],
    #                    [0, 0, 2, 0],
    #                    [0, 0, 2, 6 * pow(T * N, 1)]])
    #      b1 = np.array([state_0[0], vel_0[0], acci[0], 0])
    #      b2 = np.array([state_0[1], vel_0[1], acci[1], 0])
    #      coeffx = np.linalg.solve(A, b1)
    #      coeffy = np.linalg.solve(A, b2)
    # #     a = A.dot(coeffx)
    # #     b = np.polyval(coeffx, T * N)
    #      trajx = []
    #      trajy = []
    #      velx = []
    #      vely = []
    # #     accx = []
    # #     accy = []
    #      for t in range(0, N, 1):
    #         velx.append(
    #             ((coeffx[3] * 3 * (pow((T * t), 2))) + (coeffx[2] * 2 * (pow((T * t), 1))) + (coeffx[1])))
    #         vely.append(
    #             ((coeffy[3] * 3 * (pow((T * t), 2))) + (coeffy[2] * 2 * (pow((T * t), 1))) + (coeffy[1])))
    #      if s.who == 1:
    #          velx = np.clip(velx, -C.PARAMETERSET_2.VEHICLE_MAX_SPEED, C.PARAMETERSET_2.VEHICLE_MAX_SPEED)
    #          vely = np.clip(vely, 0, 0)
    #      else:
    #          vely = np.clip(vely, -C.PARAMETERSET_2.VEHICLE_MAX_SPEED, C.PARAMETERSET_2.VEHICLE_MAX_SPEED)
    #          velx = np.clip(velx, 0, 0)
    #      predict_result_vel = np.column_stack((velx, vely))
    #      A = np.zeros([N, N])
    #      A[np.tril_indices(N, 0)] = 1
    #      predict_result_traj = np.matmul(A, predict_result_vel) + state_0
    #      return predict_result_traj, predict_result_vel
    def dynamic(self, action_self, s):  # Dynamic of cubic polynomial on velocity
        ability = s.P_CAR.ABILITY
        ability_o = s.P_CAR.ABILITY_O
        N = C.ACTION_TIMESTEPS  # ??
        T = 1  # ??
        if s.who == 1:  # the car who conduct the prediction
            if action_self[1] == 0:  # the car dynamic it want to predict
                vel_self = s.actions_set[-1]
                state_0 = np.asarray(s.states[-1])
                acci = np.array([action_self[0] * ability, 0])
            else:
                vel_self = s.actions_set_o[-1]
                state_0 = np.array(s.states_o[-1])
                acci = np.array([0, -action_self[0] * ability_o])

        if s.who == 0:
            if action_self[1] == 0:  # the car dynamic it want to predict
                vel_self = s.actions_set_o[-1]
                state_0 = np.asarray(s.states_o[-1])
                acci = np.array([action_self[0] * ability_o, 0])
            else:
                vel_self = s.actions_set[-1]
                state_0 = np.array(s.states[-1])
                acci = np.array([0, -action_self[0] * ability])

        # vel_self = s.states[-s.track_back] - s.states[-s.track_back - 1]

        vel_0 = np.asarray(vel_self) / T  # type: ndarray # initial vel??

        A = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 2, 0],
                      [0, 0, 2, 6 * pow(T * N, 1)]])
        b1 = np.array([state_0[0], vel_0[0], acci[0], 0])
        b2 = np.array([state_0[1], vel_0[1], acci[1], 0])
        coeffx = np.linalg.solve(A, b1)
        coeffy = np.linalg.solve(A, b2)
        #     a = A.dot(coeffx)
        #     b = np.polyval(coeffx, T * N)
        velx = []
        vely = []
        for t in range(0, N, 1):
            velx.append(
                ((coeffx[3] * 3 * (pow((T * t), 2))) + (coeffx[2] * 2 * (pow((T * t), 1))) + (coeffx[1])))
            vely.append(
                ((coeffy[3] * 3 * (pow((T * t), 2))) + (coeffy[2] * 2 * (pow((T * t), 1))) + (coeffy[1])))
        if action_self[1] == 0:
            velx = np.clip(velx, 0, C.PARAMETERSET_2.VEHICLE_MAX_SPEED)
            vely = np.clip(vely, 0, 0)
        else:
            vely = np.clip(vely, -C.PARAMETERSET_2.VEHICLE_MAX_SPEED, 0)
            velx = np.clip(velx, 0, 0)
        predict_result_vel = np.column_stack((velx, vely))
        A = np.zeros([N, N])
        A[np.tril_indices(N, 0)] = 1
        predict_result_traj = np.matmul(A, predict_result_vel) + state_0
        return predict_result_traj, predict_result_vel

    def calculate_collision_loss(self, agent, action, other_agent, other_action):
        s = agent
        o = other_agent
        s_other_predict, s_other_predict_vel = self.dynamic(other_action, other_agent)
        s_self_predict, s_self_predict_vel = self.dynamic(action, agent)
        d = s.collision_box.get_collision_loss(s_self_predict, s_other_predict, o.collision_box) + 1e-12
        gap = 1.05  # TODO: generalize this
        for i in range(s_self_predict.shape[0]):
            if action[1] == 0:
                if s_self_predict[i, 0] <= -gap + 1e-12 or s_self_predict[i, 0] >= gap - 1e-12 or s_other_predict[
                   i, 1] >= gap - 1e-12 or s_other_predict[i, 1] <= -gap + 1e-12:
                    d[i] = np.inf
            else:
                if s_self_predict[i, 1] <= -gap + 1e-12 or s_self_predict[i, 1] >= gap - 1e-12 or s_other_predict[
                   i, 0] >= gap - 1e-12 or s_other_predict[i, 0] <= -gap + 1e-12:
                    d[i] = np.inf
        collision_loss = np.sum(np.exp(C.EXPCOLLISION * (-d + C.CAR_LENGTH ** 2 * 1.5)))
        return collision_loss

    def calculate_intent_loss(self, agent, action, agent_intent=None):
        if agent_intent is None:
            agent_intent = agent.intent  # use real intent

        s_self_predict, s_self_predict_vel = self.dynamic(action, agent)

        intent_loss = 0
        if action[1] == 0:
            intent_loss = agent_intent * np.exp(
                C.EXPTHETA * (agent.P_CAR.DESIRED_POSITION[0] - s_self_predict[-1][0]))
        else:
            intent_loss = agent_intent * np.exp(
                C.EXPTHETA * (-agent.P_CAR.DESIRED_POSITION[1] + s_self_predict[-1][1]))
        return intent_loss