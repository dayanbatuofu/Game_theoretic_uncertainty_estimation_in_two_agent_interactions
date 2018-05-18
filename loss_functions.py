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

    def loss(self, guess, autonomous_vehicle):

        if self.characterization is "aggressive":
            return self.aggressive_loss(guess, autonomous_vehicle)

        elif self.characterization is "reactive":
            return self.reactive_loss(guess, autonomous_vehicle)

        elif self.characterization is "passive_aggressive":
            return self.passive_aggressive_loss(guess, autonomous_vehicle)

        else:
            raise ValueError('incorrect loss function characterization specified.')

    def aggressive_loss(self, trajectory, autonomous_vehicle):
        who = (autonomous_vehicle.P_CAR_S.BOUND_X is None) + 0.0
        state_s = autonomous_vehicle.states_s[-1]
        state_o = autonomous_vehicle.states_o[-1]

        ##############################################################################################
        # predict how others perceive your action
        trials = np.arange(5, -1.1, -0.1)
        guess_set = np.hstack((np.expand_dims(trials, axis=1), np.ones((trials.size, 1)) * autonomous_vehicle.P_CAR_O.ORIENTATION))

        action_self = self.interpolate_from_trajectory(trajectory, state_s, autonomous_vehicle.P_CAR_S.ORIENTATION)[0]

        intent_optimization_results = self.multi_search_intent(autonomous_vehicle, guess_set, action_self)
        alpha_me_by_other, r, rho = intent_optimization_results

        expected_trajectory_of_other = [r, rho]  # I expect you to understand that I expect you to do this
        ##############################################################################################

        actions_self = self.interpolate_from_trajectory(trajectory, state_s, autonomous_vehicle.P_CAR_S.ORIENTATION)
        actions_other = self.interpolate_from_trajectory(expected_trajectory_of_other, state_o, autonomous_vehicle.P_CAR_O.ORIENTATION)

        s_other_predict = state_o + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other)
        s_self_predict = state_s + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self)

        D = autonomous_vehicle.collision_box_s.get_collision_loss(s_self_predict, s_other_predict, autonomous_vehicle.collision_box_o)+1e-12
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

        intent_loss = autonomous_vehicle.intent_s[0] * np.exp(C.EXPTHETA * np.linalg.norm(autonomous_vehicle.P_CAR_S.DESIRED_POSITION - s_self_predict[-1]))

        loss = collision_loss + intent_loss
        return loss, expected_trajectory_of_other  # Return weighted sum

    def reactive_loss(self, trajectory, autonomous_vehicle):
        """ Loss function defined to be a combination of state_loss and intent_loss with a weighted factor c """
        who = (autonomous_vehicle.P_CAR_S.BOUND_X is None) + 0.0
        state_s = autonomous_vehicle.states_s[-1]
        state_o = autonomous_vehicle.states_o[-1]

        actions_self = self.interpolate_from_trajectory(trajectory, state_s, autonomous_vehicle.P_CAR_S.ORIENTATION)
        actions_other = self.interpolate_from_trajectory(trajectory_other, state_o, autonomous_vehicle.P_CAR_O.ORIENTATION)

        # Define state loss
        # state_loss = np.reciprocal(box_self.get_collision_distance(s_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self),
        #                                                            s_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other), box_other)+1e-12)

        s_other_predict = state_o + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other)
        s_self_predict = state_s + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self)
        D = autonomous_vehicle.collision_box_s.get_collision_loss(s_self_predict, s_other_predict, autonomous_vehicle.collision_box_o)+1e-12
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

        intent_loss = autonomous_vehicle.intent_s[0] * np.exp(C.EXPTHETA * np.linalg.norm(autonomous_vehicle.P_CAR_S.DESIRED_POSITION - s_self_predict[-1]))

        # return np.linalg.norm(np.reciprocal(sigD)) + theta_self[0] * np.linalg.norm(intent_loss) # Return weighted sum
        loss = collision_loss + intent_loss

        return loss  # Return weighted sum

    def passive_aggressive_loss(self, trajectory, autonomous_vehicle):
        who = (autonomous_vehicle.P_CAR_S.BOUND_X is None) + 0.0
        state_s = autonomous_vehicle.states_s[-1]
        state_o = autonomous_vehicle.states_o[-1]

        ##############################################################################################
        # predict how others perceive your action
        trials = np.arange(5, -1.1, -0.1)
        # guess_set = np.hstack((np.ones((trials.size,1)) * self.human_predicted_theta[0], np.expand_dims(trials, axis=1),
        #                        np.ones((trials.size,1)) * self.machine_orientation))
        guess_set = np.hstack((np.expand_dims(trials, axis=1), np.ones((trials.size, 1)) * autonomous_vehicle.P_CAR_O.ORIENTATION))

        action_self = self.interpolate_from_trajectory(trajectory, state_s, autonomous_vehicle.P_CAR_S.ORIENTATION)[0]

        intent_optimization_results = self.multi_search_intent(autonomous_vehicle, guess_set, action_self)
        alpha_me_by_other, r, rho = intent_optimization_results

        expected_trajectory_of_other = [r, rho]  # I expect you to understand that I expect you to do this
        ##############################################################################################

        actions_self = self.interpolate_from_trajectory(trajectory, state_s, autonomous_vehicle.P_CAR_S.ORIENTATION)
        actions_other = self.interpolate_from_trajectory(expected_trajectory_of_other, state_o,
                                                         autonomous_vehicle.P_CAR_O.ORIENTATION)

        s_other_predict = state_o + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other)
        s_self_predict = state_s + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self)
        D = autonomous_vehicle.collision_box_s.get_collision_loss(s_self_predict, s_other_predict, autonomous_vehicle.collision_box_o)+1e-12
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

        intent_loss = autonomous_vehicle.intent_s[0] * np.exp(C.EXPTHETA * np.linalg.norm(autonomous_vehicle.P_CAR_S.DESIRED_POSITION - s_self_predict[-1]))

        # return np.linalg.norm(np.reciprocal(sigD)) + theta_self[0] * np.linalg.norm(intent_loss) # Return weighted sum
        gracefulness_loss = (trajectory[0] - autonomous_vehicle.P_CAR_S.COMMON_THETA[0]) ** 2

        loss = collision_loss + intent_loss + gracefulness_loss
        return loss, expected_trajectory_of_other  # Return weighted sum

    def multi_search_intent(self, autonomous_vehicle, guess_set, action_self):

        """ run multiple searches with different initial guesses """

        trajectory_set = np.empty((0,3)) #TODO: need to generalize
        loss_value_set = []

        for guess in guess_set:
            # optimization_results = scipy.optimize.minimize(self.intent_loss_func, guess,
            #                                           bounds=intent_bounds, constraints=cons, args=(
            #                                           self.machine_orientation, self.human_predicted_theta[0], 1 - who))
            fun, alpha = self.intent_loss_func(trajectory=guess,
                                               autonomous_vehicle=autonomous_vehicle,
                                               orientation_other=autonomous_vehicle.P_CAR_O.ORIENTATION,
                                               collision_box_self=autonomous_vehicle.collision_box_s,
                                               collision_box_other=autonomous_vehicle.collision_box_o,
                                               state_self=autonomous_vehicle.states_s[-1],
                                               state_other=autonomous_vehicle.states_o[-1],
                                               action_other=action_self)

            # if np.isfinite(optimization_results.fun) and not np.isnan(optimization_results.fun):
            trajectory_set = np.vstack((trajectory_set, np.array([alpha, guess[0], guess[1]])))
            loss_value_set = np.append(loss_value_set, fun)

        trajectory = trajectory_set[np.where(loss_value_set == np.min(loss_value_set))[0][0]]
        return trajectory

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

    def interpolate_from_trajectory(self, trajectory, state, orientation):

        nodes = np.array([[state[0], state[0] + trajectory[0]*np.cos(np.deg2rad(trajectory[1]))/2, state[0] + trajectory[0]*np.cos(np.deg2rad(trajectory[1]))],
                          [state[1], state[1] + trajectory[0]*np.sin(np.deg2rad(trajectory[1]))/2, state[1] + trajectory[0]*np.sin(np.deg2rad(trajectory[1]))]])

        curve = bezier.Curve(nodes, degree=2)

        positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_NUMPOINTS + 1)))
        #TODO: skip state?
        return np.diff(positions, n=1, axis=0)