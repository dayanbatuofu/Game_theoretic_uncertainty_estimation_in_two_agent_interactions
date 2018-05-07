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

    def loss(self, trajectory, s_other, s_self, theta_self, theta_max, box_other, box_self, orientation_other, orientation_self):

        if self.characterization is "aggressive":
            pass
        elif self.characterization is "reactive":
            pass
        elif self.characterization is "passive_aggressive":
            pass
        else:
            raise ValueError('incorrect loss function characterization specified.')

    def aggressive_loss(self, trajectory, s_other, s_self, theta_self, theta_max, box_other, box_self,
                      orientation_other, orientation_self, who):

        ##############################################################################################
        # predict how others perceive your action
        trials = np.arange(5, -1.1, -0.1)
        # guess_set = np.hstack((np.ones((trials.size,1)) * self.human_predicted_theta[0], np.expand_dims(trials, axis=1),
        #                        np.ones((trials.size,1)) * self.machine_orientation))
        guess_set = np.hstack((np.expand_dims(trials, axis=1),
                               np.ones((trials.size, 1)) * orientation_other))
        action_self = self.interpolate_from_trajectory(trajectory, s_self, orientation_self)[0]
        intent_optimization_results = self.multi_search_intent(guess_set, [], [], orientation_other,
                                                               s_other, s_self, action_self, 1 - who)
        alpha_me_by_other, r, rho = intent_optimization_results

        expected_trajectory_other_by_me = [r, rho]  # I expect you to understand that I expect you to do this
        ##############################################################################################

        actions_self = self.interpolate_from_trajectory(trajectory, s_self, orientation_self)
        actions_other = self.interpolate_from_trajectory(expected_trajectory_other_by_me, s_other,
                                                         orientation_other)

        s_other_predict = s_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other)
        s_self_predict = s_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self)
        D = box_self.get_collision_distance(s_self_predict, s_other_predict, box_other) + 1e-12
        gap = 1.05  # TODO: generalize this
        for i in range(s_self_predict.shape[0]):
            if who == 1:
                if s_self_predict[i, 0] <= -gap + 1e-12 or s_self_predict[i, 0] >= gap - 1e-12 or s_other_predict[
                    i, 1] >= gap - 1e-12 or s_other_predict[i, 1] <= -gap + 1e-12:
                    D[i] = np.inf
            elif who == 0:
                if s_self_predict[i, 1] <= -gap + 1e-12 or s_self_predict[i, 1] >= gap - 1e-12 or s_other_predict[
                    i, 0] >= gap - 1e-12 or s_other_predict[i, 0] <= -gap + 1e-12:
                    D[i] = np.inf

        collision_loss = np.sum(np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5)))

        if who == 1:
            intent_loss = theta_self[0] * np.exp(C.EXPTHETA * (- s_self_predict[-1][0] + 0.4))
        else:
            intent_loss = theta_self[0] * np.exp(C.EXPTHETA * (s_self_predict[-1][1] + 0.4))

        # return np.linalg.norm(np.reciprocal(sigD)) + theta_self[0] * np.linalg.norm(intent_loss) # Return weighted sum
        loss = collision_loss + intent_loss

        return loss, expected_trajectory_other_by_me  # Return weighted sum

    def reactive_loss(self, intent, orientation, alpha0, who):
        # alpha = intent[0] #aggressiveness of the agent
        trajectory = intent  # what I was expected to do

        # what I could have done and been
        s_other = np.array(self.machine_states_set[-1])
        nodes = np.array([[s_other[0], s_other[0] + trajectory[0] * np.cos(np.deg2rad(orientation)) / 2,
                           s_other[0] + trajectory[0] * np.cos(np.deg2rad(trajectory[1]))],
                          [s_other[1], s_other[1] + trajectory[0] * np.sin(np.deg2rad(orientation)) / 2,
                           s_other[1] + trajectory[0] * np.sin(np.deg2rad(trajectory[1]))]])
        curve = bezier.Curve(nodes, degree=2)
        positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_TIMESTEPS + 1)))
        a_other = np.diff(positions, n=1, axis=0)
        s_other_traj = np.array(s_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, a_other))

        # actions and states of the agent
        s_self = np.array(self.human_states_set[-1])
        a_self = np.array(
            C.ACTION_TIMESTEPS * [self.human_actions_set[-1]])  # project current agent actions to future
        s_self_traj = np.array(s_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, a_self))

        # I expect the agent to be this much aggressive
        theta_self = self.human_predicted_theta

        # calculate the gradient of the control objective
        A = M.LOWER_TRIANGULAR_MATRIX
        D = np.sum((s_other_traj - s_self_traj) ** 2.,
                   axis=1) + 1e-12  # should be t_steps by 1, add small number for numerical stability
        # need to check if states are in the collision box
        gap = 1.05
        for i in range(s_self_traj.shape[0]):
            if who == 1:
                if s_self_traj[i, 0] <= -gap + 1e-12 or s_self_traj[i, 0] >= gap - 1e-12 or s_other_traj[
                    i, 1] >= gap - 1e-12 or s_other_traj[i, 1] <= -gap + 1e-12:
                    D[i] = np.inf
            elif who == 0:
                if s_self_traj[i, 1] <= -gap + 1e-12 or s_self_traj[i, 1] >= gap - 1e-12 or s_other_traj[
                    i, 0] >= gap - 1e-12 or s_other_traj[i, 0] <= -gap + 1e-12:
                    D[i] = np.inf

        # dD/da
        # dDda_self = - np.dot(np.expand_dims(np.dot(A.transpose(), sigD**(-2)*dsigD),axis=1), np.expand_dims(ds, axis=0)) \
        #        - np.dot(np.dot(A.transpose(), np.diag(sigD**(-2)*dsigD)), np.dot(A, a_self - a_other))
        dDda_self = - 2 * C.EXPCOLLISION * np.dot(A.transpose(), (s_self_traj - s_other_traj) *
                                                  np.expand_dims(
                                                      np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5)),
                                                      axis=1))

        # update theta_hat_H
        w = - dDda_self  # negative gradient direction

        if who == 0:
            if self.P.BOUND_HUMAN_X is not None:  # intersection
                w[np.all([s_self_traj[:, 0] <= 1e-12, w[:, 0] <= 1e-12],
                         axis=0), 0] = 0  # if against wall and push towards the wall, get a reaction force
                w[np.all([s_self_traj[:, 0] >= -1e-12, w[:, 0] >= -1e-12],
                         axis=0), 0] = 0  # TODO: these two lines are hard coded for intersection, need to check the interval
                # print(w)
            else:  # lane changing
                w[np.all([s_self_traj[:, 1] <= 1e-12, w[:, 1] <= 1e-12],
                         axis=0), 1] = 0  # if against wall and push towards the wall, get a reaction force
                w[np.all([s_self_traj[:, 1] >= 1 - 1e-12, w[:, 1] >= -1e-12],
                         axis=0), 1] = 0  # TODO: these two lines are hard coded for lane changing
        else:
            if self.P.BOUND_HUMAN_X is not None:  # intersection
                w[np.all([s_self_traj[:, 1] <= 1e-12, w[:, 1] <= 1e-12],
                         axis=0), 1] = 0  # if against wall and push towards the wall, get a reaction force
                w[np.all([s_self_traj[:, 1] >= -1e-12, w[:, 1] >= -1e-12],
                         axis=0), 1] = 0  # TODO: these two lines are hard coded for intersection, need to check the interval
            else:  # lane changing
                w[np.all([s_self_traj[:, 0] <= 1e-12, w[:, 0] <= 1e-12],
                         axis=0), 0] = 0  # if against wall and push towards the wall, get a reaction force
                w[np.all([s_self_traj[:, 0] >= -1e-12, w[:, 0] >= -1e-12],
                         axis=0), 0] = 0  # TODO: these two lines are hard coded for lane changing
        w = -w

        # calculate best alpha for the enumeration of trajectory

        if who == 1:
            l = np.array([- C.EXPTHETA * np.exp(C.EXPTHETA * (-s_self_traj[-1][0] + 0.4)), 0.])
            # don't take into account the time steps where one car has already passed
            decay = (((s_self_traj - s_other_traj)[:, 0] < gap) + 0.0) * (
            (s_self_traj - s_other_traj)[:, 1] < gap + 0.0)
        else:
            l = np.array([0., C.EXPTHETA * np.exp(C.EXPTHETA * (s_self_traj[-1][1] + 0.4))])
            decay = (((s_other_traj - s_self_traj)[:, 0] < gap) + 0.0) * (
            (s_other_traj - s_self_traj)[:, 1] < gap + 0.0)
        decay = decay * np.exp(np.linspace(0, -10, C.ACTION_TIMESTEPS))
        w = w * np.expand_dims(decay, axis=1)
        l = l * np.expand_dims(decay, axis=1)
        alpha = np.max((- np.trace(np.dot(np.transpose(w), l)) / (np.sum(l ** 2) + 1e-12), 0.1))
        x = w + alpha * l
        L = np.sum(x ** 2)
        return L, alpha

    def passive_aggressive_loss(self, trajectory, s_other, s_self, theta_self, theta_max, box_other, box_self,
                      orientation_other, orientation_self, expected_trajectory_self, who):

        ##############################################################################################
        # predict how others perceive your action
        trials = np.arange(5, -1.1, -0.1)
        # guess_set = np.hstack((np.ones((trials.size,1)) * self.human_predicted_theta[0], np.expand_dims(trials, axis=1),
        #                        np.ones((trials.size,1)) * self.machine_orientation))
        guess_set = np.hstack((np.expand_dims(trials, axis=1),
                               np.ones((trials.size, 1)) * orientation_other))
        action_self = self.interpolate_from_trajectory(trajectory, s_self, orientation_self)[0]
        intent_optimization_results = self.multi_search_intent(guess_set, [], [], orientation_other,
                                                               s_other, s_self, action_self, 1 - who)
        alpha_me_by_other, r, rho = intent_optimization_results

        expected_trajectory_other_by_me = [r, rho]  # I expect you to understand that I expect you to do this
        ##############################################################################################

        actions_self = self.interpolate_from_trajectory(trajectory, s_self, orientation_self)
        actions_other = self.interpolate_from_trajectory(expected_trajectory_other_by_me, s_other,
                                                         orientation_other)

        s_other_predict = s_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other)
        s_self_predict = s_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self)
        D = box_self.get_collision_distance(s_self_predict, s_other_predict, box_other) + 1e-12
        gap = 1.05  # TODO: generalize this
        for i in range(s_self_predict.shape[0]):
            if who == 1:
                if s_self_predict[i, 0] <= -gap + 1e-12 or s_self_predict[i, 0] >= gap - 1e-12 or s_other_predict[
                    i, 1] >= gap - 1e-12 or s_other_predict[i, 1] <= -gap + 1e-12:
                    D[i] = np.inf
            elif who == 0:
                if s_self_predict[i, 1] <= -gap + 1e-12 or s_self_predict[i, 1] >= gap - 1e-12 or s_other_predict[
                    i, 0] >= gap - 1e-12 or s_other_predict[i, 0] <= -gap + 1e-12:
                    D[i] = np.inf

        collision_loss = np.sum(np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5)))

        if who == 1:
            intent_loss = theta_self[0] * np.exp(C.EXPTHETA * (- s_self_predict[-1][0] + 0.4))
        else:
            intent_loss = theta_self[0] * np.exp(C.EXPTHETA * (s_self_predict[-1][1] + 0.4))

        # return np.linalg.norm(np.reciprocal(sigD)) + theta_self[0] * np.linalg.norm(intent_loss) # Return weighted sum
        gracefulness_loss = (trajectory[0] - expected_trajectory_self[0]) ** 2

        loss = collision_loss + intent_loss + gracefulness_loss

        return loss, expected_trajectory_other_by_me  # Return weighted sum


    def multi_search_intent(self, guess_set, intent_bounds, cons, orientation_s, state_s, state_o, action_o, who):

        """ run multiple searches with different initial guesses """

        trajectory_set = np.empty((0,3)) #TODO: need to generalize
        loss_value_set = []

        for guess in guess_set:
            # optimization_results = scipy.optimize.minimize(self.intent_loss_func, guess,
            #                                           bounds=intent_bounds, constraints=cons, args=(
            #                                           self.machine_orientation, self.human_predicted_theta[0], 1 - who))
            fun, alpha = self.intent_loss_func(guess, orientation_s, state_s, state_o, action_o, who)

            # if np.isfinite(optimization_results.fun) and not np.isnan(optimization_results.fun):
            trajectory_set = np.vstack((trajectory_set, np.array([alpha, guess[0], guess[1]])))
            loss_value_set = np.append(loss_value_set, fun)

        trajectory = trajectory_set[np.where(loss_value_set == np.min(loss_value_set))[0][0]]
        return trajectory

    def intent_loss_func(self, intent, orientation_self, state_self, state_other, action_other, who_self):
        who = 1-who_self

        # alpha = intent[0] #aggressiveness of the agent
        trajectory = intent #what I was expected to do

        # what I could have done and been
        s_other = np.array(state_self)
        nodes = np.array([[s_other[0], s_other[0] + trajectory[0]*np.cos(np.deg2rad(orientation_self))/2, s_other[0] + trajectory[0]*np.cos(np.deg2rad(trajectory[1]))],
                  [s_other[1], s_other[1] + trajectory[0]*np.sin(np.deg2rad(orientation_self))/2, s_other[1] + trajectory[0]*np.sin(np.deg2rad(trajectory[1]))]])
        curve = bezier.Curve(nodes, degree=2)
        positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_TIMESTEPS + 1)))
        a_other = np.diff(positions, n=1, axis=0)
        s_other_traj = np.array(s_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, a_other))

        # actions and states of the agent
        s_self = np.array(state_other) #self.human_states_set[-1]
        a_self = np.array(C.ACTION_TIMESTEPS * [action_other])#project current agent actions to future
        s_self_traj = np.array(s_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, a_self))

        # I expect the agent to be this much aggressive
        # theta_self = self.human_predicted_theta

        # calculate the gradient of the control objective
        A = M.LOWER_TRIANGULAR_MATRIX
        D = np.sum((s_other_traj-s_self_traj)**2., axis=1) + 1e-12 #should be t_steps by 1, add small number for numerical stability
        # need to check if states are in the collision box
        gap = 1.05
        for i in range(s_self_traj.shape[0]):
            if who == 1:
                if s_self_traj[i,0]<=-gap+1e-12 or s_self_traj[i,0]>=gap-1e-12 or s_other_traj[i,1]>=gap-1e-12 or s_other_traj[i,1]<=-gap+1e-12:
                    D[i] = np.inf
            elif who == 0:
                if s_self_traj[i,1]<=-gap+1e-12 or s_self_traj[i,1]>=gap-1e-12 or s_other_traj[i,0]>=gap-1e-12 or s_other_traj[i,0]<=-gap+1e-12:
                    D[i] = np.inf

        # dD/da
        # dDda_self = - np.dot(np.expand_dims(np.dot(A.transpose(), sigD**(-2)*dsigD),axis=1), np.expand_dims(ds, axis=0)) \
        #        - np.dot(np.dot(A.transpose(), np.diag(sigD**(-2)*dsigD)), np.dot(A, a_self - a_other))
        dDda_self = - 2 * C.EXPCOLLISION * np.dot(A.transpose(), (s_self_traj - s_other_traj) *
                                                  np.expand_dims(np.exp(C.EXPCOLLISION *(-D + C.CAR_LENGTH**2*1.5)),
                                                                 axis=1))

        # update theta_hat_H
        w = - dDda_self # negative gradient direction

        if who == 0:
            if self.P.BOUND_HUMAN_X is not None: # intersection
                w[np.all([s_self_traj[:,0]<=1e-12, w[:,0] <= 1e-12], axis=0),0] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_self_traj[:,0]>=-1e-12, w[:,0] >= -1e-12], axis=0),0] = 0 #TODO: these two lines are hard coded for intersection, need to check the interval
                # print(w)
            else: # lane changing
                w[np.all([s_self_traj[:,1]<=1e-12, w[:,1] <= 1e-12], axis=0),1] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_self_traj[:,1]>=1-1e-12, w[:,1] >= -1e-12], axis=0),1] = 0 #TODO: these two lines are hard coded for lane changing
        else:
            if self.P.BOUND_HUMAN_X is not None: # intersection
                w[np.all([s_self_traj[:,1]<=1e-12, w[:,1] <= 1e-12], axis=0),1] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_self_traj[:,1]>=-1e-12, w[:,1] >= -1e-12], axis=0),1] = 0 #TODO: these two lines are hard coded for intersection, need to check the interval
            else: # lane changing
                w[np.all([s_self_traj[:,0]<=1e-12, w[:,0] <= 1e-12], axis=0),0] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_self_traj[:,0]>=-1e-12, w[:,0] >= -1e-12], axis=0),0] = 0 #TODO: these two lines are hard coded for lane changing
        w = -w

        #calculate best alpha for the enumeration of trajectory

        if who == 1:
            l = np.array([- C.EXPTHETA *np.exp(C.EXPTHETA*(-s_self_traj[-1][0] + 0.4)), 0.])
            # don't take into account the time steps where one car has already passed
            decay = (((s_self_traj - s_other_traj)[:,0]<gap) + 0.0)  * ((s_self_traj - s_other_traj)[:,1]<gap + 0.0)
        else:
            l = np.array([0., C.EXPTHETA *np.exp(C.EXPTHETA*(s_self_traj[-1][1] + 0.4))])
            decay = (((s_other_traj - s_self_traj)[:,0]<gap) + 0.0)  * ((s_other_traj - s_self_traj)[:,1]<gap + 0.0)
        decay = decay * np.exp(np.linspace(0,-10,C.ACTION_TIMESTEPS))
        w = w*np.expand_dims(decay, axis=1)
        l = l*np.expand_dims(decay, axis=1)
        alpha = np.max((- np.trace(np.dot(np.transpose(w),l))/(np.sum(l**2)+1e-12),0.1))
        x = w + alpha * l
        L = np.sum(x**2)
        return L, alpha