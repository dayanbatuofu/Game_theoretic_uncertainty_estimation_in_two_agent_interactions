from constants import CONSTANTS as C
from constants import MATRICES as M
import numpy as np
import scipy
import bezier
from collision_box import Collision_Box
from loss_functions import LossFunctions
from scipy import optimize
import pygame as pg
from scipy.interpolate import spline


class AutonomousVehicle:

    """
    States:
            X-Position
            Y-Position
    """

    def __init__(self, scenario_parameters, car_parameters_self, loss_style, who):

        self.P = scenario_parameters
        self.P_CAR = car_parameters_self
        # self.P_CAR_O = car_parameters_other
        self.loss = LossFunctions(loss_style)
        self.who = who
        self.image = pg.transform.rotate(pg.transform.scale(pg.image.load(C.ASSET_LOCATION + self.P_CAR.SPRITE),
                                                                  (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                                   int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))), self.P_CAR.ORIENTATION)

        # self.image_o = pg.transform.rotate(pg.transform.scale(pg.image.load(C.ASSET_LOCATION + self.P_CAR_O.SPRITE),
        #                                                          (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
        #                                                           int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))), self.P_CAR_O.ORIENTATION)

        self.collision_box = Collision_Box(self.image.get_width() / C.COORDINATE_SCALE / C.ZOOM,
                                             self.image.get_height() / C.COORDINATE_SCALE / C.ZOOM, self.P)
        # self.collision_box_o = Collision_Box(self.image_o.get_width() / C.COORDINATE_SCALE / C.ZOOM,
        #                                      self.image_o.get_height() / C.COORDINATE_SCALE / C.ZOOM, self.P)

        # Initialize my space
        self.states = [self.P_CAR.INITIAL_POSITION]
        self.intent = self.P_CAR.INTENT
        self.actions_set = [self.interpolate_from_trajectory(self.P_CAR.COMMON_THETA)[0]]
        self.trajectory = []
        self.planned_actions_set = []
        self.track_back = 0

        # Initialize others space
        self.states_o = []
        self.actions_set_o = []
        self.other_car = []

        # Initialize prediction_variables
        self.predicted_theta_other = self.P_CAR.INTENT  # consider others as equally aggressive
        self.predicted_theta_self = self.P_CAR.INTENT
        self.predicted_trajectory_other = []
        self.predicted_actions_other    = np.tile((0, 0), (C.ACTION_TIMESTEPS, 1))
        self.predicted_others_prediction_of_my_actions = np.tile((0, 0), (C.ACTION_TIMESTEPS, 1))
        self.predicted_others_prediction_of_my_trajectory = self.P_CAR.COMMON_THETA

    def get_state(self, delay):
        return self.states[-1 * delay]

    def update(self, frame):
        who = self.who
        other = self.other_car
        """ Function ran on every frame of simulation"""

        ########## Update human characteristics here ########
        if who == 1: # 1 moves first
            self.states_o = np.array(other.states) #get other's states
            self.actions_set_o = np.array(other.actions_set) #get other's actions
        elif who == 0:
            self.states_o = np.array(other.states[:-1]) #get other's states
            self.actions_set_o = np.array(other.actions_set[:-1]) #get other's actions

        # if len(self.states_o) > C.TRACK_BACK and len(self.states) > C.TRACK_BACK:
        self.track_back = min(C.TRACK_BACK, len(self.states))
        theta_other, theta_self, predicted_trajectory_other, predicted_others_prediction_of_my_trajectory \
            = self.get_predicted_intent_of_other() #"self" inside prediction is human (who=0)
        self.predicted_theta_other = theta_other
        self.predicted_theta_self = theta_self
        self.predicted_trajectory_other = predicted_trajectory_other
        self.predicted_others_prediction_of_my_trajectory = predicted_others_prediction_of_my_trajectory

        ########## Calculate machine actions here ###########
        [planned_actions, predicted_actions_other, predicted_others_prediction_of_my_actions] = self.get_actions()

        planned_actions[np.where(np.abs(planned_actions)<1e-6)]=0.  #remove numerical errors
        self.predicted_actions_other = predicted_actions_other
        self.predicted_others_prediction_of_my_actions  = predicted_others_prediction_of_my_actions

        self.states.append(np.add(self.states[-1], (planned_actions[self.track_back][0],
                                                    planned_actions[self.track_back][1])))
        self.actions_set.append(planned_actions[0])
        self.planned_actions_set = planned_actions

    def get_actions(self):

        """ Function that accepts 2 vehicles states, intents, criteria, and an amount of future steps
        and return the ideal actions based on the loss function"""

        # if len(self.actions_set_o)>0:
        #     a0_other = self.actions_set_o
        #     initial_trajectory_other = [np.linalg.norm(a0_other[-1])*C.ACTION_TIMESTEPS,
        #                                 np.arctan2(a0_other[-1, 1], a0_other[-1, 0])*180./np.pi]
        #     initial_trajectory_self = self.trajectory_s
        # else:
        #     initial_trajectory_other = self.P_CAR_O.COMMON_THETA
        #     initial_trajectory_self = self.P_CAR_S.COMMON_THETA

        theta_other = self.predicted_theta_other
        theta_self = self.intent
        box_other = self.other_car.collision_box
        box_self = self.collision_box

        # initial_expected_trajectory_self = self.prediction_of_others_prediction_of_my_trajectory

        bounds_self = [(-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # Radius
                       (-C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION,
                        C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION)]  # Angle

        bounds_other = [(-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # Radius
                        (-C.ACTION_TURNANGLE + self.other_car.P_CAR.ORIENTATION,
                        C.ACTION_TURNANGLE + self.other_car.P_CAR.ORIENTATION)]  # Angle

        A = np.zeros((C.ACTION_TIMESTEPS, C.ACTION_TIMESTEPS))
        A[np.tril_indices(C.ACTION_TIMESTEPS, 0)] = 1

        cons_other = []
        cons_self = []

        if self.other_car.P_CAR.BOUND_X is not None:
            if self.other_car.P_CAR.BOUND_X[0] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: -self.states_o[-1][0] -
                                                                   (x[0]*scipy.cos(np.deg2rad(x[1]))) -
                                                                   box_other.height/2 +
                                                                   self.other_car.P_CAR.BOUND_X[1]})
            if self.other_car.P_CAR.BOUND_X[1] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: self.states_o[-1][0] +
                                                                   (x[0]*scipy.cos(np.deg2rad(x[1]))) -
                                                                   box_other.height/2 -
                                                                   self.other_car.P_CAR.BOUND_X[0]})

        if self.other_car.P_CAR.BOUND_Y is not None:
            if self.other_car.P_CAR.BOUND_Y[0] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: -self.states_o[-1][1] -
                                                                   (x[0]*scipy.sin(np.deg2rad(x[1]))) -
                                                                   box_other.width/2 +
                                                                   self.other_car.P_CAR.BOUND_Y[1]})
            if self.other_car.P_CAR.BOUND_Y[1] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: self.states_o[-1][1] +
                                                                   (x[0]*scipy.sin(np.deg2rad(x[1]))) -
                                                                   box_other.width/2 -
                                                                   self.other_car.P_CAR.BOUND_Y[0]})

        if self.P_CAR.BOUND_X is not None:
            if self.P_CAR.BOUND_X[0] is not None:
                cons_self.append({'type': 'ineq','fun': lambda x: -self.states[-1][0] -
                                                                  (x[0]*scipy.cos(np.deg2rad(x[1]))) -
                                                                  box_self.height/2 + self.P_CAR.BOUND_X[1]})
            if self.P_CAR.BOUND_X[1] is not None:
                cons_self.append({'type': 'ineq','fun': lambda x: self.states[-1][0] +
                                                                  (x[0]*scipy.cos(np.deg2rad(x[1]))) -
                                                                  box_self.height/2 - self.P_CAR.BOUND_X[0]})

        if self.P_CAR.BOUND_Y is not None:
            if self.P_CAR.BOUND_Y[0] is not None:
                    cons_self.append({'type': 'ineq','fun': lambda x: self.states[-1][1] +
                                                                      (x[0]*scipy.sin(np.deg2rad(x[1]))) -
                                                                      box_self.width/2 - self.P_CAR.BOUND_Y[0]})
            if self.P_CAR.BOUND_Y[1] is not None:
                    cons_self.append({'type': 'ineq','fun': lambda x: -self.states[-1][1] -
                                                                      (x[0]*scipy.sin(np.deg2rad(x[1]))) -
                                                                      box_self.width/2 + self.P_CAR.BOUND_Y[1]})

        loss_value = 0
        loss_value_old = loss_value + C.LOSS_THRESHOLD + 1
        iter_count = 0

        trajectory_other = self.predicted_trajectory_other
        # trajectory_self = initial_trajectory_self
        # predicted_trajectory_self = initial_expected_trajectory_self

        # guess_set = np.array([[0,0],[10,0]]) #TODO: need to generalize this

        trials = np.array([5.,1.,0.,-1.])
        guess_set = np.hstack((np.expand_dims(trials, axis=1), np.ones((trials.size, 1)) * self.P_CAR.ORIENTATION))
        guess_other = np.hstack((np.expand_dims(trials, axis=1), np.ones((trials.size, 1)) * self.other_car.P_CAR.ORIENTATION))

        if self.loss.characterization is 'reactive':
            trajectory_self, predicted_trajectory_other = self.loss.loss(guess_set, self, guess_other=guess_other)
        else:
            trajectory_self, predicted_trajectory_other = self.multi_search(guess_set)

        # Interpolate for output
        actions_self = self.interpolate_from_trajectory(trajectory_self)

        # if predicted_trajectory_other is None:
        #     predicted_actions_other = self.predicted_actions_of_other
        # else:
        predicted_actions_other = self.interpolate_from_trajectory(predicted_trajectory_other)

        predicted_actions_self = self.interpolate_from_trajectory(self.predicted_others_prediction_of_my_trajectory)

        return actions_self, predicted_actions_other, predicted_actions_self

    def multi_search(self, guess_set):

        s = self
        o = s.other_car
        theta_s = s.intent
        box_o = o.collision_box
        box_s = s.collision_box
        orientation_o = o.P_CAR.ORIENTATION
        orientation_s = s.P_CAR.ORIENTATION

        """ run multiple searches with different initial guesses """
        trajectory_set = np.empty((0,2)) #TODO: need to generalize
        predicted_trajectory_other_set = np.empty((0,2))
        loss_value_set = []

        for guess in guess_set:
            fun, predicted_trajectory_other = self.loss.loss(guess, self, [])
            trajectory_set = np.append(trajectory_set, [guess], axis=0)

            if predicted_trajectory_other is not None:
                predicted_trajectory_other_set = np.append(predicted_trajectory_other_set, [predicted_trajectory_other], axis=0)

            loss_value_set = np.append(loss_value_set, fun)

        candidates = np.where(loss_value_set == np.min(loss_value_set))[0]
        trajectory = trajectory_set[candidates[np.random.randint(candidates.size)]]

        if len(predicted_trajectory_other_set) == 0:
            predicted_trajectory_other = None
        else:
            predicted_trajectory_other = predicted_trajectory_other_set[np.where(loss_value_set == np.min(loss_value_set))[0][0]]

        return trajectory, predicted_trajectory_other

    def get_predicted_intent_of_other(self):
        """ predict the aggressiveness of the agent and what the agent expect me to do """

        who = self.who

        cons = []
        if who == 1: # machine looking at human
            if self.P.BOUND_HUMAN_X is not None: #intersection
                intent_bounds = [(0.1, None), # alpha
                                 (0 * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED), # radius
                                 (-C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION, C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION)] # angle, to accommodate crazy behavior
                cons.append({'type': 'ineq','fun': lambda x: self.states[-1][1] + (x[0] * scipy.sin(np.deg2rad(x[1]))) - (0.4 - 0.33)})
                cons.append({'type': 'ineq','fun': lambda x: - self.states[-1][1] - (x[0] * scipy.sin(np.deg2rad(x[1]))) + (-0.4 + 0.33)})

            else: #TODO: update this part
                intent_bounds = [(0.1, None), # alpha
                                 (-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED), # radius
                                 (-90, 90)] # angle, to accommodate crazy behavior
        else: # human looking at machine
            if self.P.BOUND_HUMAN_X is not None: #intersection
                intent_bounds = [(0.1, None), # alpha
                                 (0 * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED), # radius
                                 (-C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION, C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION)] # angle, to accommodate crazy behavior
                cons.append({'type': 'ineq','fun': lambda x: self.states[-1][0] + (x[0] * scipy.cos(np.deg2rad(x[1]))) - (0.4 - 0.33)})
                cons.append({'type': 'ineq','fun': lambda x: - self.states[-1][0] - (x[0] * scipy.cos(np.deg2rad(x[1]))) + (-0.4 + 0.33)})

            else: #TODO: update this part
                intent_bounds = [(0.1, None), # alpha
                                 (-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED), # radius
                                 (-90, 90)] # angle, to accommodate crazy behavior

        #TODO: damn...I don't know why the solver is not working, insert a valid solution, output nan...
        # trials_trajectory_other = np.arange(5,-0.1,-0.1)
        # trials_trajectory_self = np.arange(5,-0.1,-0.1)

        # guess_set = np.hstack((np.ones((trials.size,1)) * self.human_predicted_theta[0], np.expand_dims(trials, axis=1),
        #                        np.ones((trials.size,1)) * self.machine_orientation))

        intent_optimization_results = self.multi_search_intent()
        theta_other, theta_self, trajectory_other, trajectory_self = intent_optimization_results

        return theta_other, theta_self, trajectory_other, trajectory_self

    def multi_search_intent(self):
        """ run multiple searches with different initial guesses """
        s = self
        who = self.who
        trials_theta = np.array([1.,1e3,1e6])
        inference_set = [] #TODO: need to generalize
        loss_value_set = []

        for theta_self in trials_theta:
            for theta_other in trials_theta:
            # for theta_other in [1.]:
                trajectory_self, trajectory_other = self.equilibrium(theta_self, theta_other, s, s.other_car)
                if trajectory_self is not []:
                    action_self = [self.interpolate_from_trajectory(trajectory_self[i])
                                   for i in range(len(trajectory_self))]
                    action_other = [self.interpolate_from_trajectory(trajectory_other[i])
                                    for i in range(len(trajectory_other))]
                    fun_all = [
                               # np.linalg.norm(action_self[i][:s.track_back]-s.actions_set[-s.track_back:])
                               # +\
                               np.linalg.norm(action_other[i][:s.track_back]-s.actions_set_o[-s.track_back:])
                               for i in range(len(trajectory_self))]
                    fun = min(fun_all)

                    trajectory_self = [trajectory_self[i] for i in np.where(fun_all == np.min(fun_all))[0]]
                    trajectory_other = [trajectory_other[i] for i in np.where(fun_all == np.min(fun_all))[0]]
                else:
                    fun = 1e32

                id = np.random.randint(len(trajectory_other))
                inference_set.append([theta_other, theta_self,
                                      trajectory_other[id],
                                      trajectory_self[id]])
                loss_value_set.append(fun)

        candidate = np.where(loss_value_set == np.min(loss_value_set))[0]
        inference = inference_set[candidate[np.random.randint(len(candidate))]]
        return inference

    def equilibrium(self, theta_self, theta_other, s, o):
        action_guess = np.array([5.,1.,0.,-1.])
        trials_trajectory_self = np.hstack((np.expand_dims(action_guess, axis=1),
                               np.ones((action_guess.size,1)) * s.P_CAR.ORIENTATION))
        trials_trajectory_other = np.hstack((np.expand_dims(action_guess, axis=1),
                               np.ones((action_guess.size,1)) * o.P_CAR.ORIENTATION))
        loss_matrix = np.zeros((trials_trajectory_self.shape[0],trials_trajectory_other.shape[0],2))
        for i in range(trials_trajectory_self.shape[0]):
            for j in range(trials_trajectory_other.shape[0]):
                loss_matrix[i,j,:] = self.simulate_game(trials_trajectory_self[i],trials_trajectory_other[j],
                                                        theta_self,theta_other,s,o)

        # find equilibrium
        sum_loss_all = []
        eq_all = []
        for j in range(trials_trajectory_other.shape[0]):
            id_s = np.atleast_1d(np.argmin(loss_matrix[:,j,0]))
            for i in range(id_s.size):
                id_o = np.atleast_1d(np.argmin(loss_matrix[id_s[i],:,1]))
                if sum(np.isin(id_o,j))>0:
                    eq_all.append([id_s[i],j])
                    sum_loss_all.append(sum(loss_matrix[id_s[i],j,:]))

        eq = [eq_all[i] for i in np.where(sum_loss_all == np.min(sum_loss_all))[0]]

        if eq is not []:
            trajectory_self = [trials_trajectory_self[eq[i][0]] for i in range(len(eq))]
            trajectory_other = [trials_trajectory_other[eq[i][1]] for i in range(len(eq))]
        else:
            trajectory_self = []
            trajectory_other = []

        return trajectory_self, trajectory_other

    def simulate_game(self, trajectory_self, trajectory_other, theta_self, theta_other, s, o):
        loss_s = self.loss.reactive_loss(theta_self, trajectory_self, trajectory_other, s.states[-s.track_back],
                                         s.states_o[-s.track_back], s)
        loss_o = self.loss.reactive_loss(theta_other, trajectory_other, trajectory_self, s.states_o[-s.track_back],
                                         s.states[-s.track_back], o)

        return loss_s, loss_o




    def intent_loss_func(self, intent):
        orientation_self = self.P_CAR_S.ORIENTATION
        state_self = self.states_s[-C.TRACK_BACK]
        state_other = self.states_o[-C.TRACK_BACK]
        action_other = self.actions_set_o[-C.TRACK_BACK]
        who = 1 - (self.P_CAR_S.BOUND_X is None)

        # alpha = intent[0] #aggressiveness of the agent
        trajectory = intent #what I was expected to do

        # what I could have done and been
        s_other = np.array(state_self)
        nodes = np.array([[s_other[0], s_other[0] + trajectory[0]*np.cos(np.deg2rad(self.P_CAR_S.ORIENTATION))/2, s_other[0] + trajectory[0]*np.cos(np.deg2rad(trajectory[1]))],
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

        # if who == 0:
        # alpha = 1.

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