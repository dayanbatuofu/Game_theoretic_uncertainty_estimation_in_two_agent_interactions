from constants import CONSTANTS as C
from constants import MATRICES as M
import bezier
import numpy as np

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
            return self.passive_aggressive_loss(guess, autonomous_vehicle)

        else:
            raise ValueError('incorrect loss function characterization specified.')

    def aggressive_loss(self, trajectory, s):
        who = s.who
        o = s.other_car
        state_s = s.states[-1]
        state_o = s.states_o[-1]

        ##############################################################################################
        # predict how others perceive your action
        action_self = s.interpolate_from_trajectory(trajectory)[:s.track_back]
        theta_other, theta_self, trajectory_other_all, trajectory_self_all = self.multi_search_intent(action_self, s)
        ##############################################################################################

        loss_all = []
        for trajectory_other in trajectory_other_all:
            actions_self = s.interpolate_from_trajectory(trajectory)
            actions_other = o.interpolate_from_trajectory(trajectory_other)

            s_other_predict = state_o + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other)
            s_self_predict = state_s + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self)

            D = s.collision_box.get_collision_loss(s_self_predict, s_other_predict, o.collision_box)+1e-12
            gap = 1.05 #TODO: generalize this
            for i in range(s_self_predict.shape[0]):
                if who == 1:
                    if s_self_predict[i,0]<=-gap+1e-12 or s_self_predict[i,0]>=gap-1e-12 or s_other_predict[i,1]>=gap-1e-12 or s_other_predict[i,1]<=-gap+1e-12:
                        D[i] = np.inf
                elif who == 0:
                    if s_self_predict[i,1]<=-gap+1e-12 or s_self_predict[i,1]>=gap-1e-12 or s_other_predict[i,0]>=gap-1e-12 or s_other_predict[i,0]<=-gap+1e-12:
                        D[i] = np.inf
            collision_loss = np.sum(np.exp(C.EXPCOLLISION *(-D + C.CAR_LENGTH**2*1.5)))
            # collision_loss = autonomous_vehicle.collision_box_s.get_collision_loss(s_self_predict, s_other_predict, autonomous_vehicle.collision_box_o)

            if who == 1:
                intent_loss = s.intent * np.exp(C.EXPTHETA * (s.P_CAR.DESIRED_POSITION[0] - s_self_predict[-1][0]))
            else:
                intent_loss = s.intent * np.exp(C.EXPTHETA * (-s.P_CAR.DESIRED_POSITION[1] + s_self_predict[-1][1]))

            loss_all.append(collision_loss + intent_loss)

        return np.mean(loss_all)  # Return weighted sum

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

            loss = self.reactive_loss(s.intent, [guess], trajectory_other, s.states[-1], s.states_o[-1], s)

            trajectory_set = np.append(trajectory_set, [guess], axis=0)
            loss_value_set = np.append(loss_value_set, loss)

        trajectory_self = trajectory_set[np.where(loss_value_set == np.min(loss_value_set))[0][0]]

        return trajectory_self

    def reactive_loss(self, theta_self, trajectory, trajectory_other, s_self, s_other, s):

        """ Loss function defined to be a combination of state_loss and intent_loss with a weighted factor c """
        loss = []
        for t_s in trajectory:
            for t_o in trajectory_other:
                o = s.other_car
                box_self = s.collision_box
                box_other = o.collision_box
                who = s.who

                actions_self    = s.interpolate_from_trajectory(t_s)
                actions_other   = o.interpolate_from_trajectory(t_o)

                s_other_predict = s_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other)
                s_self_predict = s_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self)
                D = box_self.get_collision_loss(s_self_predict, s_other_predict, box_other)+1e-12
                gap = 1.05 #TODO: generalize this
                for i in range(s_self_predict.shape[0]):
                    if who == 1:
                        if s_self_predict[i,0]<=-gap+1e-12 or s_self_predict[i,0]>=gap-1e-12 or s_other_predict[i,1]>=gap-1e-12 or s_other_predict[i,1]<=-gap+1e-12:
                            D[i] = np.inf
                    elif who == 0:
                        if s_self_predict[i,1]<=-gap+1e-12 or s_self_predict[i,1]>=gap-1e-12 or s_other_predict[i,0]>=gap-1e-12 or s_other_predict[i,0]<=-gap+1e-12:
                            D[i] = np.inf

                collision_loss = np.sum(np.exp(C.EXPCOLLISION *(-D + C.CAR_LENGTH**2*1.5)))

                if who == 1:
                    intent_loss = theta_self * np.exp(C.EXPTHETA * (- s_self_predict[-1][0] + 0.4))
                else:
                    intent_loss = theta_self * np.exp(C.EXPTHETA * (s_self_predict[-1][1] + 0.4))

                loss.append(collision_loss + intent_loss)

        return np.mean(loss) # Return weighted sum

    def passive_aggressive_loss(self, trajectory, s):
        who = s.who
        o = s.other_car
        state_s = s.states[-1]
        state_o = s.states_o[-1]

        ##############################################################################################
        # predict how others perceive your action
        action_self = s.interpolate_from_trajectory(trajectory)[:s.track_back]
        theta_other, theta_self, trajectory_other_all, trajectory_self_all = self.multi_search_intent(action_self, s)
        ##############################################################################################

        loss_all = []
        for trajectory_other  in trajectory_other_all:  # what other will do if I did trajectory
            actions_self = s.interpolate_from_trajectory(trajectory)
            actions_other = o.interpolate_from_trajectory(trajectory_other)

            s_other_predict = state_o + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other)
            s_self_predict = state_s + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self)

            D = s.collision_box.get_collision_loss(s_self_predict, s_other_predict, o.collision_box)+1e-12
            gap = 1.05 #TODO: generalize this
            for i in range(s_self_predict.shape[0]):
                if who == 1:
                    if s_self_predict[i,0]<=-gap+1e-12 or s_self_predict[i,0]>=gap-1e-12 or s_other_predict[i,1]>=gap-1e-12 or s_other_predict[i,1]<=-gap+1e-12:
                        D[i] = np.inf
                elif who == 0:
                    if s_self_predict[i,1]<=-gap+1e-12 or s_self_predict[i,1]>=gap-1e-12 or s_other_predict[i,0]>=gap-1e-12 or s_other_predict[i,0]<=-gap+1e-12:
                        D[i] = np.inf
            collision_loss = np.sum(np.exp(C.EXPCOLLISION *(-D + C.CAR_LENGTH**2*1.5)))
            # collision_loss = autonomous_vehicle.collision_box_s.get_collision_loss(s_self_predict, s_other_predict, autonomous_vehicle.collision_box_o)

            if who == 1:
                intent_loss = s.intent * np.exp(C.EXPTHETA * (s.P_CAR.DESIRED_POSITION[0] - s_self_predict[-1][0]))
            else:
                intent_loss = s.intent * np.exp(C.EXPTHETA * (-s.P_CAR.DESIRED_POSITION[1] + s_self_predict[-1][1]))

            gracefulness_loss = []
            for wanted_trajectory_self in s.wanted_trajectory_self:  # what other want me to do right now
                gracefulness_loss.append((trajectory[0] - wanted_trajectory_self[0]) ** 2)

            loss_all.append(collision_loss + intent_loss + 0.3*np.mean(gracefulness_loss))

        return np.mean(loss_all)  # Return weighted sum

    def multi_search_intent(self, trajectory, s):
        """ run multiple searches with different initial guesses """
        who = s.who
        trials_theta = C.THETA_SET
        inference_set = [] #TODO: need to generalize
        loss_value_set = []

        for theta_self in trials_theta:
            for k in range(len(s.predicted_theta_other)):
                theta_other = s.predicted_theta_other[k]
                trajectory_self, trajectory_other, my_loss_all, other_loss_all = self.equilibrium(theta_self, theta_other, s, s.other_car)

                my_trajectory = [trajectory_self[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
                other_trajectory = [trajectory_other[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]
                other_trajectory_conservative = \
                                    [trajectory_other[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]  #others move slow

                if trajectory_self is not []:
                    action_self = [self.interpolate_from_trajectory(my_trajectory[i])
                                   for i in range(len(my_trajectory))]
                    action_other = [self.interpolate_from_trajectory(other_trajectory[i])
                                    for i in range(len(other_trajectory))]
                    # fun_all = [
                    #            np.linalg.norm(action_self[i][:s.track_back]-trajectory)
                    #            +\
                    #            np.linalg.norm(action_other[i][:s.track_back]-
                    #                           s.predicted_actions_other[s.track_back:2*s.track_back])
                    #            for i in range(len(trajectory_self))]
                    # fun_all = np.round(fun_all, 6)
                    # fun = min(fun_all)
                    fun_self = [np.linalg.norm(action_self[i][:s.track_back]-trajectory)
                                for i in range(len(action_self))]
                    fun_other = [np.linalg.norm(action_other[i][:s.track_back]-
                                                s.predicted_actions_other[k][s.track_back:2*s.track_back])
                                for i in range(len(action_other))]
                    fun = min(fun_self) + min(fun_other)

                    trajectory_self = [my_trajectory[i] for i in np.where(fun_self == np.min(fun_self))[0]]
                    trajectory_other = [other_trajectory_conservative[i] for i in np.where(fun_self == np.min(fun_self))[0]]
                    # my_loss_all = [my_loss_all[i] for i in np.where(fun_all == np.min(fun_all))[0]]
                    #
                    # trajectory_self = [trajectory_self[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
                    # trajectory_other = [trajectory_other[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
                else:
                    fun = 1e32

                inference_set.append([theta_self,
                                      theta_other,
                                      trajectory_other,
                                      trajectory_self])
                loss_value_set.append(fun)

        candidate = np.where(loss_value_set == np.min(loss_value_set))[0]
        # inference = inference_set[candidate[np.random.randint(len(candidate))]]
        theta_self_out = []
        theta_other_out = []
        trajectory_self_out = []
        trajectory_other_out = []
        for i in range(len(candidate)):
            for j in range(len(inference_set[candidate[i]][2])):
                for k in range(len(inference_set[candidate[i]][3])):
                    theta_self_out.append(inference_set[candidate[i]][0])
                    theta_other_out.append(inference_set[candidate[i]][1])
                    trajectory_other_out.append(inference_set[candidate[i]][2][k])
                    trajectory_self_out.append(inference_set[candidate[i]][3][j])

        return theta_other_out, theta_self_out, trajectory_other_out, trajectory_self_out

    def equilibrium(self, theta_self, theta_other, s, o):
        action_guess = C.TRAJECTORY_SET
        trials_trajectory_self = np.hstack((np.expand_dims(action_guess, axis=1),
                               np.ones((action_guess.size,1)) * s.P_CAR.ORIENTATION))
        trials_trajectory_other = np.hstack((np.expand_dims(action_guess, axis=1),
                               np.ones((action_guess.size,1)) * o.P_CAR.ORIENTATION))
        loss_matrix = np.zeros((trials_trajectory_self.shape[0],trials_trajectory_other.shape[0],2))
        for i in range(trials_trajectory_self.shape[0]):
            for j in range(trials_trajectory_other.shape[0]):
                loss_matrix[i,j,:] = self.simulate_game([trials_trajectory_self[i]],[trials_trajectory_other[j]],
                                                        theta_self,theta_other,s,o)

        # find equilibrium
        my_loss_all = []
        other_loss_all = []
        eq_all = []
        for j in range(trials_trajectory_other.shape[0]):
            id_s = np.atleast_1d(np.argmin(loss_matrix[:,j,0]))
            for i in range(id_s.size):
                id_o = np.atleast_1d(np.argmin(loss_matrix[id_s[i],:,1]))
                if sum(np.isin(id_o,j))>0:
                    eq_all.append([id_s[i],j])
                    my_loss_all.append(loss_matrix[id_s[i],j,0])
                    other_loss_all.append(loss_matrix[id_s[i],j,1])

        # eq = [eq_all[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]

        if eq_all is not []:
            trajectory_self = [trials_trajectory_self[eq_all[i][0]] for i in range(len(eq_all))]
            trajectory_other = [trials_trajectory_other[eq_all[i][1]] for i in range(len(eq_all))]
        else:
            trajectory_self = []
            trajectory_other = []

        return trajectory_self, trajectory_other, my_loss_all, other_loss_all

    def simulate_game(self, trajectory_self, trajectory_other, theta_self, theta_other, s, o):
        loss_s = self.reactive_loss(theta_self, trajectory_self, trajectory_other, s.states[-1],
                                         s.states_o[-1], s)
        loss_o = self.reactive_loss(theta_other, trajectory_other, trajectory_self, s.states_o[-1],
                                         s.states[-1], o)

        return loss_s, loss_o

    def intent_loss_func(self, trajectory, autonomous_vehicle, orientation_other, collision_box_self,
                         collision_box_other, state_self, state_other, action_other):
        who = (autonomous_vehicle.P_CAR_S.BOUND_X is None) + 0.0

        action_self = self.interpolate_from_trajectory(trajectory, state_other, orientation_other)
        trajectory_self = np.array(state_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, action_self))

        # actions and states of the agent
        action_other = np.array(C.ACTION_TIMESTEPS * [action_other])#project current agent actions to future
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
                                                  np.expand_dims(np.exp(C.EXPCOLLISION *(-collision_loss + C.CAR_LENGTH**2*1.5)),
                                                                 axis=1))

        # update theta_hat_H
        w = - dDda_self # negative gradient direction

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
            l = np.array([- C.EXPTHETA * np.exp(C.EXPTHETA*(-trajectory_other[-1][0] + 0.4)), 0.])
            # don't take into account the time steps where one car has already passed
            decay = (((trajectory_other - trajectory_self)[:,0]<gap) + 0.0)  * ((trajectory_other - trajectory_self)[:,1]<gap + 0.0)
        else:
            l = np.array([0., C.EXPTHETA * np.exp(C.EXPTHETA*(trajectory_other[-1][1] + 0.4))])
            decay = (((trajectory_self - trajectory_other)[:,0]<gap) + 0.0)  * ((trajectory_self - trajectory_other)[:,1]<gap + 0.0)

        decay = decay * np.exp(np.linspace(0,-10,C.ACTION_TIMESTEPS))

        w = w*np.expand_dims(decay, axis=1)
        l = l*np.expand_dims(decay, axis=1)

        alpha = np.max((- np.trace(np.dot(np.transpose(w),l))/(np.sum(l**2)+1e-12),0.1))
        x = w + alpha * l
        L = np.sum(x**2)
        return L, alpha

    def interpolate_from_trajectory(self, trajectory):

        nodes = np.array([[0, trajectory[0]*np.cos(np.deg2rad(trajectory[1]))/2, trajectory[0]*np.cos(np.deg2rad(trajectory[1]))],
                          [0, trajectory[0]*np.sin(np.deg2rad(trajectory[1]))/2, trajectory[0]*np.sin(np.deg2rad(trajectory[1]))]])

        curve = bezier.Curve(nodes, degree=2)

        positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_NUMPOINTS + 1)))
        #TODO: skip state?
        return np.diff(positions, n=1, axis=0)