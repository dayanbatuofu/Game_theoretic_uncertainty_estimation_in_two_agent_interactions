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
from scipy import stats

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
                                                             int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))),
                                         self.P_CAR.ORIENTATION)

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
        self.predicted_theta_self = self.P_CAR.INTENT  # what other think about me
        self.predicted_trajectory_other = []  # what other will do
        self.predicted_actions_other = np.tile((0, 0), (C.ACTION_TIMESTEPS, 1))
        self.predicted_others_prediction_of_my_trajectory = self.P_CAR.COMMON_THETA  # what other believe I will do
        self.predicted_others_prediction_of_my_actions = np.tile((0, 0), (C.ACTION_TIMESTEPS, 1))
        self.wanted_trajectory_self = []  # what other wants me to do
        self.wanted_trajectory_other = [] # what I want other to do
        self.inference_probability = [] # probability density of the inference vectors
        self.inference_probability_proactive = [] # for proactive and socially aware actions
        self.theta_probability = np.ones(C.THETA_SET.shape)/C.THETA_SET.size
        self.social_gracefulness = []  # collect difference between action and what other wants

    def get_state(self, delay):
        return self.states[-1 * delay]

    def update(self, frame):
        who = self.who
        other = self.other_car
        self.frame = frame
        """ Function ran on every frame of simulation"""

        ########## Update human characteristics here ########
        if who == 1:  # 1 moves first
            self.states_o = np.array(other.states)  # get other's states
            self.actions_set_o = np.array(other.actions_set)  # get other's actions
        elif who == 0:
            self.states_o = np.array(other.states[:-1])  # get other's states
            self.actions_set_o = np.array(other.actions_set[:-1])  # get other's actions

        # if len(self.states_o) > C.TRACK_BACK and len(self.states) > C.TRACK_BACK:
        self.track_back = min(C.TRACK_BACK, len(self.states))
        theta_other, theta_self, predicted_trajectory_other, predicted_others_prediction_of_my_trajectory, \
        wanted_others_prediction_of_my_trajectory, other_wanted_trajectory, inference_probability, theta_probability = \
            self.get_predicted_intent_of_other()

        self.wanted_trajectory_self = wanted_others_prediction_of_my_trajectory
        self.wanted_trajectory_other = other_wanted_trajectory
        self.inference_probability = inference_probability
        self.inference_probability_proactive = inference_probability
        self.theta_probability = theta_probability

        self.predicted_theta_other = theta_other
        self.predicted_theta_self = theta_self
        self.predicted_trajectory_other = predicted_trajectory_other
        self.predicted_others_prediction_of_my_trajectory = predicted_others_prediction_of_my_trajectory
        self.predicted_actions_other = [self.interpolate_from_trajectory(predicted_trajectory_other[i])
                                        for i in range(len(predicted_trajectory_other))]
        self.predicted_others_prediction_of_my_actions = [
            self.interpolate_from_trajectory(predicted_others_prediction_of_my_trajectory[i])
            for i in range(len(predicted_others_prediction_of_my_trajectory))]

        ########## Calculate machine actions here ###########
        planned_trajectory, planned_actions = self.get_actions()

        planned_actions[np.where(np.abs(planned_actions) < 1e-6)] = 0.  # remove numerical errors

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

        bounds_self = [(-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),
                       # Radius
                       (-C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION,
                        C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION)]  # Angle

        bounds_other = [(-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),
                        # Radius
                        (-C.ACTION_TURNANGLE + self.other_car.P_CAR.ORIENTATION,
                         C.ACTION_TURNANGLE + self.other_car.P_CAR.ORIENTATION)]  # Angle

        A = np.zeros((C.ACTION_TIMESTEPS, C.ACTION_TIMESTEPS))
        A[np.tril_indices(C.ACTION_TIMESTEPS, 0)] = 1

        cons_other = []
        cons_self = []

        if self.other_car.P_CAR.BOUND_X is not None:
            if self.other_car.P_CAR.BOUND_X[0] is not None:
                cons_other.append({'type': 'ineq', 'fun': lambda x: -self.states_o[-1][0] -
                                                                    (x[0] * scipy.cos(np.deg2rad(x[1]))) -
                                                                    box_other.height / 2 +
                                                                    self.other_car.P_CAR.BOUND_X[1]})
            if self.other_car.P_CAR.BOUND_X[1] is not None:
                cons_other.append({'type': 'ineq', 'fun': lambda x: self.states_o[-1][0] +
                                                                    (x[0] * scipy.cos(np.deg2rad(x[1]))) -
                                                                    box_other.height / 2 -
                                                                    self.other_car.P_CAR.BOUND_X[0]})

        if self.other_car.P_CAR.BOUND_Y is not None:
            if self.other_car.P_CAR.BOUND_Y[0] is not None:
                cons_other.append({'type': 'ineq', 'fun': lambda x: -self.states_o[-1][1] -
                                                                    (x[0] * scipy.sin(np.deg2rad(x[1]))) -
                                                                    box_other.width / 2 +
                                                                    self.other_car.P_CAR.BOUND_Y[1]})
            if self.other_car.P_CAR.BOUND_Y[1] is not None:
                cons_other.append({'type': 'ineq', 'fun': lambda x: self.states_o[-1][1] +
                                                                    (x[0] * scipy.sin(np.deg2rad(x[1]))) -
                                                                    box_other.width / 2 -
                                                                    self.other_car.P_CAR.BOUND_Y[0]})

        if self.P_CAR.BOUND_X is not None:
            if self.P_CAR.BOUND_X[0] is not None:
                cons_self.append({'type': 'ineq', 'fun': lambda x: -self.states[-1][0] -
                                                                   (x[0] * scipy.cos(np.deg2rad(x[1]))) -
                                                                   box_self.height / 2 + self.P_CAR.BOUND_X[1]})
            if self.P_CAR.BOUND_X[1] is not None:
                cons_self.append({'type': 'ineq', 'fun': lambda x: self.states[-1][0] +
                                                                   (x[0] * scipy.cos(np.deg2rad(x[1]))) -
                                                                   box_self.height / 2 - self.P_CAR.BOUND_X[0]})

        if self.P_CAR.BOUND_Y is not None:
            if self.P_CAR.BOUND_Y[0] is not None:
                cons_self.append({'type': 'ineq', 'fun': lambda x: self.states[-1][1] +
                                                                   (x[0] * scipy.sin(np.deg2rad(x[1]))) -
                                                                   box_self.width / 2 - self.P_CAR.BOUND_Y[0]})
            if self.P_CAR.BOUND_Y[1] is not None:
                cons_self.append({'type': 'ineq', 'fun': lambda x: -self.states[-1][1] -
                                                                   (x[0] * scipy.sin(np.deg2rad(x[1]))) -
                                                                   box_self.width / 2 + self.P_CAR.BOUND_Y[1]})

        loss_value = 0
        loss_value_old = loss_value + C.LOSS_THRESHOLD + 1
        iter_count = 0

        trajectory_other = self.predicted_trajectory_other
        # trajectory_self = initial_trajectory_self
        # predicted_trajectory_self = initial_expected_trajectory_self

        # guess_set = np.array([[0,0],[10,0]]) #TODO: need to generalize this

        trials = C.TRAJECTORY_SET
        guess_set = np.hstack((np.expand_dims(trials, axis=1), np.ones((trials.size, 1)) * self.P_CAR.ORIENTATION))
        guess_other = np.hstack(
            (np.expand_dims(trials, axis=1), np.ones((trials.size, 1)) * self.other_car.P_CAR.ORIENTATION))

        if self.loss.characterization is 'basic':
            trajectory_self = self.basic_motion()
        elif self.loss.characterization is 'reactive':
            trajectory_self = self.loss.loss(guess_set, self, guess_other=guess_other)
        else:
            trajectory_self = self.multi_search(guess_set)

        # Interpolate for output
        actions_self = self.interpolate_from_trajectory(trajectory_self)

        return trajectory_self, actions_self

    def basic_motion(self):
        theta_self = self.intent
        trials_theta = C.THETA_SET
        trajectory_probability = np.zeros(C.TRAJECTORY_SET.shape)
        for j in range(len(trials_theta)):
            theta_other = trials_theta[j]
            trajectory_self, trajectory_other, my_loss_all, other_loss_all = self.equilibrium(theta_self,
                                                                                              theta_other, self,
                                                                                              self.other_car)
            for i in range(len(C.TRAJECTORY_SET)):
                t = C.TRAJECTORY_SET[i]
                trajectory_probability[i] += len(np.where(trajectory_self == t)[0])*self.theta_probability[j]
        trajectory_probability = trajectory_probability/sum(trajectory_probability)

        var = stats.rv_discrete(name='var', values=(C.TRAJECTORY_SET, trajectory_probability))
        trajectory = np.hstack((var.rvs(size=1), [self.P_CAR.ORIENTATION]))
        return trajectory

    def multi_search(self, guess_set):
        s = self
        o = s.other_car
        theta_s = s.intent
        box_o = o.collision_box
        box_s = s.collision_box
        orientation_o = o.P_CAR.ORIENTATION
        orientation_s = s.P_CAR.ORIENTATION

        """ run multiple searches with different initial guesses """
        trajectory_set = np.empty((0, 2))  # TODO: need to generalize
        trajectory_other_set = []
        loss_value_set = []
        inference_probability_set = []

        for guess in guess_set:
            fun, trajectory_other, inference_probability = self.loss.loss(guess, self, [])
            trajectory_set = np.append(trajectory_set, [guess], axis=0)
            trajectory_other_set.append(trajectory_other)
            inference_probability_set.append(inference_probability)
            loss_value_set = np.append(loss_value_set, fun)

        candidates = np.where(loss_value_set == np.min(loss_value_set))[0][0]
        self.predicted_trajectory_other = trajectory_other_set[candidates]
        self.predicted_actions_other = [self.interpolate_from_trajectory(self.predicted_trajectory_other[i])
                                        for i in range(len(self.predicted_trajectory_other))]
        self.inference_probability_proactive = inference_probability_set[candidates]

        trajectory = trajectory_set[candidates]

        return trajectory

    def get_predicted_intent_of_other(self):
        """ predict the aggressiveness of the agent and what the agent expect me to do """
        who = self.who
        cons = []
        if who == 1:  # machine looking at human
            if self.P.BOUND_HUMAN_X is not None:  # intersection
                intent_bounds = [(0.1, None),  # alpha
                                 (0 * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),
                                 # radius
                                 (-C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION,
                                  C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION)]  # angle, to accommodate crazy behavior
                cons.append({'type': 'ineq',
                             'fun': lambda x: self.states[-1][1] + (x[0] * scipy.sin(np.deg2rad(x[1]))) - (0.4 - 0.33)})
                cons.append({'type': 'ineq',
                             'fun': lambda x: - self.states[-1][1] - (x[0] * scipy.sin(np.deg2rad(x[1]))) + (
                             -0.4 + 0.33)})

            else:  # TODO: update this part
                intent_bounds = [(0.1, None),  # alpha
                                 (-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED,
                                  C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # radius
                                 (-90, 90)]  # angle, to accommodate crazy behavior
        else:  # human looking at machine
            if self.P.BOUND_HUMAN_X is not None:  # intersection
                intent_bounds = [(0.1, None),  # alpha
                                 (0 * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),
                                 # radius
                                 (-C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION,
                                  C.ACTION_TURNANGLE + self.P_CAR.ORIENTATION)]  # angle, to accommodate crazy behavior
                cons.append({'type': 'ineq',
                             'fun': lambda x: self.states[-1][0] + (x[0] * scipy.cos(np.deg2rad(x[1]))) - (0.4 - 0.33)})
                cons.append({'type': 'ineq',
                             'fun': lambda x: - self.states[-1][0] - (x[0] * scipy.cos(np.deg2rad(x[1]))) + (
                             -0.4 + 0.33)})

            else:  # TODO: update this part
                intent_bounds = [(0.1, None),  # alpha
                                 (-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED,
                                  C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # radius
                                 (-90, 90)]  # angle, to accommodate crazy behavior

        # TODO: damn...I don't know why the solver is not working, insert a valid solution, output nan...
        # trials_trajectory_other = np.arange(5,-0.1,-0.1)
        # trials_trajectory_self = np.arange(5,-0.1,-0.1)

        # guess_set = np.hstack((np.ones((trials.size,1)) * self.human_predicted_theta[0], np.expand_dims(trials, axis=1),
        #                        np.ones((trials.size,1)) * self.machine_orientation))

        # if self.loss.characterization is 'reactive':
        intent_optimization_results = self.multi_search_intent()  # believes none is aggressive
        # elif self.loss.characterization is 'aggressive':
        #     intent_optimization_results = self.multi_search_intent_aggressive()  # believes both are aggressive
        # elif self.loss.characterization is 'passive_aggressive':
        #     intent_optimization_results = self.multi_search_intent_passive_aggressive()  # believes both are aggressive

        # theta_other, theta_self, trajectory_other, trajectory_self = intent_optimization_results

        return intent_optimization_results

    def multi_search_intent(self):
        """ run multiple searches with different initial guesses """
        s = self
        who = self.who
        trials_theta = C.THETA_SET
        inference_set = []  # T0poODO: need to generalize
        loss_value_set = []

        for theta_self in trials_theta:
        # for theta_self in [s.intent]:
        #     if s.who == 1:
        #         theta_self = s.intent
            for theta_other in trials_theta:
                # for theta_other in [1.]:
                trajectory_self, trajectory_other, my_loss_all, other_loss_all = self.equilibrium(theta_self,
                                                                                                  theta_other, s,
                                                                                                  s.other_car)

                # my_trajectory = [trajectory_self[i] for i in
                #                  np.where(other_loss_all == np.min(other_loss_all))[0]]  # I believe others move fast
                # other_trajectory = [trajectory_other[i] for i in
                #                     np.where(my_loss_all == np.min(my_loss_all))[0]]  # others believe I move fast
                my_trajectory = trajectory_self
                other_trajectory = trajectory_other
                # other_trajectory_conservative = \
                #     [trajectory_other[i] for i in
                #      np.where(other_loss_all == np.min(other_loss_all))[0]]  # others move fast

                trajectory_self_wanted_other = []
                other_trajectory_wanted = []

                if trajectory_self is not []:
                    action_self = [self.interpolate_from_trajectory(my_trajectory[i])
                                   for i in range(len(my_trajectory))]
                    action_other = [self.interpolate_from_trajectory(other_trajectory[i])
                                    for i in range(len(other_trajectory))]

                    fun_self = [np.linalg.norm(action_self[i][:s.track_back] - s.actions_set[-s.track_back:])
                                for i in range(len(action_self))]
                    fun_other = [np.linalg.norm(action_other[i][:s.track_back] - s.actions_set_o[-s.track_back:])
                                 for i in range(len(action_other))]

                    fun = min(fun_other)

                    # what I think other want me to do if he wants to take the benefit
                    trajectory_self_wanted_other = \
                        [trajectory_self[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]

                    # what I want other to do
                    other_trajectory_wanted = \
                        [trajectory_other[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]

                    # what I think others expect me to do
                    trajectory_self = np.atleast_2d(
                        [my_trajectory[i] for i in np.where(fun_self == np.min(fun_self))[0]])

                    # what I think others will do
                    trajectory_other = np.atleast_2d(
                        [other_trajectory[i] for i in np.where(fun_other == fun)[0]])
                else:
                    fun = 1e32

                inference_set.append([theta_self,
                                      theta_other,
                                      trajectory_other,
                                      trajectory_self,
                                      trajectory_self_wanted_other,
                                      other_trajectory_wanted,
                                      1./len(trajectory_other)*len(trajectory_self_wanted_other)*len(other_trajectory_wanted)])
                loss_value_set.append(round(fun*1000)/1000)

        candidate = np.where(loss_value_set == np.min(loss_value_set))[0]
        theta_self_out = []
        theta_other_out = []
        trajectory_self_out = []
        trajectory_other_out = []
        trajectory_self_wanted_other_out = []
        other_trajectory_wanted_out = []
        inference_probability_out = []
        theta_probability = []

        for i in range(len(candidate)):
            for j in range(len(inference_set[candidate[i]][2])):
                for k in range(len(inference_set[candidate[i]][3])):
                    for l in range(len(inference_set[candidate[i]][4])):
                        for p in range(len(inference_set[candidate[i]][5])):
                            theta_self_out.append(inference_set[candidate[i]][0])
                            theta_other_out.append(inference_set[candidate[i]][1])
                            trajectory_other_out.append(inference_set[candidate[i]][2][j])
                            trajectory_self_out.append(inference_set[candidate[i]][3][k])
                            trajectory_self_wanted_other_out.append(inference_set[candidate[i]][4][l])
                            other_trajectory_wanted_out.append(inference_set[candidate[i]][5][p])
                            inference_probability_out.append(1./len(candidate)*inference_set[candidate[i]][6])

        inference_probability_out = np.array(inference_probability_out)
        # update theta probability
        for theta_other in trials_theta:
            theta_probability.append(sum(inference_probability_out[np.where(theta_other_out==theta_other)[0]]))
        # theta_probability = (self.theta_probability * self.frame + theta_probability) / (self.frame + 1)
        theta_probability = self.theta_probability * theta_probability
        if sum(theta_probability) > 0:
            theta_probability = theta_probability/sum(theta_probability)
        else:
            theta_probability = np.ones(C.THETA_SET.shape)/C.THETA_SET.size

        # update inference probability accordingly
        for i in range(len(trials_theta)):
            id = np.where(theta_other_out == trials_theta[i])[0]
            inference_probability_out[id] = inference_probability_out[id]/\
                                             sum(inference_probability_out[id]) * theta_probability[i]
        inference_probability_out = inference_probability_out/sum(inference_probability_out)

        return theta_other_out, theta_self_out, trajectory_other_out, trajectory_self_out, \
               trajectory_self_wanted_other_out, other_trajectory_wanted_out, inference_probability_out, \
               theta_probability

    def multi_search_intent_aggressive(self):
        """ run multiple searches with different initial guesses """
        s = self
        who = self.who
        trials_theta = C.THETA_SET
        inference_set = []  # TODO: need to generalize
        loss_value_set = []

        for theta_self in trials_theta:
            for theta_other in trials_theta:
                # for theta_other in [1.]:
                trajectory_self, trajectory_other, my_loss_all, other_loss_all = self.equilibrium(theta_self,
                                                                                                  theta_other, s,
                                                                                                  s.other_car)

                my_trajectory = [trajectory_self[i] for i in
                                 np.where(other_loss_all == np.min(other_loss_all))[0]]  # I believe others move fast
                other_trajectory = [trajectory_other[i] for i in
                                    np.where(my_loss_all == np.min(my_loss_all))[0]]  # I believe others move fast
                other_trajectory_conservative = \
                    [trajectory_other[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]  # others move slow

                if trajectory_self is not []:
                    action_self = [self.interpolate_from_trajectory(my_trajectory[i])
                                   for i in range(len(my_trajectory))]
                    action_other = [self.interpolate_from_trajectory(other_trajectory[i])
                                    for i in range(len(other_trajectory))]

                    fun_self = [np.linalg.norm(action_self[i][:s.track_back] - s.actions_set[-s.track_back:])
                                for i in range(len(action_self))]
                    fun_other = [np.linalg.norm(action_other[i][:s.track_back] - s.actions_set_o[-s.track_back:])
                                 for i in range(len(action_other))]

                    fun = min(fun_other)

                    # what I think other want me to do if he wants to take the benefit
                    trajectory_self_wanted_other = \
                        [trajectory_self[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]
                    trajectory_self_wanted_other = \
                        [trajectory_self_wanted_other[i] for i in np.where(fun_other == fun)[0]]

                    trajectory_self = np.atleast_2d(
                        [my_trajectory[i] for i in np.where(fun_self == np.min(fun_self))[0]])
                    trajectory_other = np.atleast_2d(other_trajectory_conservative)
                    # my_loss_all = [my_loss_all[i] for i in np.where(fun_self == np.min(fun_self))[0]]
                    #
                    # trajectory_self = [trajectory_self[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
                    # trajectory_other = [trajectory_other[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
                else:
                    fun = 1e32

                inference_set.append([theta_self,
                                      theta_other,
                                      trajectory_other,
                                      trajectory_self,
                                      trajectory_self_wanted_other])
                loss_value_set.append(fun)

        candidate = np.where(loss_value_set == np.min(loss_value_set))[0]
        # inference = inference_set[candidate[np.random.randint(len(candidate))]]
        theta_self_out = []
        theta_other_out = []
        trajectory_self_out = []
        trajectory_other_out = []
        trajectory_self_wanted_other_out = []
        for i in range(len(candidate)):
            for j in range(len(inference_set[candidate[i]][2])):
                for k in range(len(inference_set[candidate[i]][3])):
                    for q in range(len(inference_set[candidate[i]][4])):
                        theta_self_out.append(inference_set[candidate[i]][0])
                        theta_other_out.append(inference_set[candidate[i]][1])
                        trajectory_other_out.append(inference_set[candidate[i]][2][k])
                        trajectory_self_out.append(inference_set[candidate[i]][3][j])
                        trajectory_self_wanted_other_out.append(inference_set[candidate[i]][4][q])
        return theta_other_out, theta_self_out, trajectory_other_out, trajectory_self_out, \
               trajectory_self_wanted_other_out

    def multi_search_intent_passive_aggressive(self):
        """ run multiple searches with different initial guesses """
        s = self
        who = self.who
        trials_theta = C.THETA_SET
        inference_set = []  # TODO: need to generalize
        loss_value_set = []

        for theta_self in trials_theta:
            for theta_other in trials_theta:
                trajectory_self, trajectory_other, my_loss_all, other_loss_all = self.equilibrium(theta_self,
                                                                                                  theta_other, s,
                                                                                                  s.other_car)

                my_trajectory = [trajectory_self[i] for i in
                                 np.where(other_loss_all == np.min(other_loss_all))[0]]  # I believe others move fast
                other_trajectory = [trajectory_other[i] for i in
                                    np.where(my_loss_all == np.min(my_loss_all))[0]]  # I believe others move fast
                other_trajectory_conservative = \
                    [trajectory_other[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]  # others move slow

                if trajectory_self is not []:
                    action_self = [self.interpolate_from_trajectory(my_trajectory[i])
                                   for i in range(len(my_trajectory))]
                    action_other = [self.interpolate_from_trajectory(other_trajectory[i])
                                    for i in range(len(other_trajectory))]

                    fun_self = [np.linalg.norm(action_self[i][:s.track_back] - s.actions_set[-s.track_back:])
                                for i in range(len(action_self))]
                    fun_other = [np.linalg.norm(action_other[i][:s.track_back] - s.actions_set_o[-s.track_back:])
                                 for i in range(len(action_other))]

                    fun = min(fun_other)

                    # what I think other want me to do if he wants to take the benefit
                    trajectory_self_wanted_other = \
                        [trajectory_self[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]
                    trajectory_self_wanted_other = \
                        [trajectory_self_wanted_other[i] for i in np.where(fun_other == fun)[0]]

                    trajectory_self = np.atleast_2d(
                        [my_trajectory[i] for i in np.where(fun_self == np.min(fun_self))[0]])
                    trajectory_other = np.atleast_2d(other_trajectory_conservative)
                    # my_loss_all = [my_loss_all[i] for i in np.where(fun_self == np.min(fun_self))[0]]
                    #
                    # trajectory_self = [trajectory_self[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
                    # trajectory_other = [trajectory_other[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
                else:
                    fun = 1e32

                inference_set.append([theta_self,
                                      theta_other,
                                      trajectory_other,
                                      trajectory_self,
                                      trajectory_self_wanted_other])
                loss_value_set.append(fun)

        candidate = np.where(loss_value_set == np.min(loss_value_set))[0]
        # inference = inference_set[candidate[np.random.randint(len(candidate))]]
        theta_self_out = []
        theta_other_out = []
        trajectory_self_out = []
        trajectory_other_out = []
        trajectory_self_wanted_other_out = []
        for i in range(len(candidate)):
            for j in range(len(inference_set[candidate[i]][2])):
                for k in range(len(inference_set[candidate[i]][3])):
                    for q in range(len(inference_set[candidate[i]][4])):
                        theta_self_out.append(inference_set[candidate[i]][0])
                        theta_other_out.append(inference_set[candidate[i]][1])
                        trajectory_other_out.append(inference_set[candidate[i]][2][k])
                        trajectory_self_out.append(inference_set[candidate[i]][3][j])
                        trajectory_self_wanted_other_out.append(inference_set[candidate[i]][4][q])
        return theta_other_out, theta_self_out, trajectory_other_out, trajectory_self_out, \
               trajectory_self_wanted_other_out

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
                    other_loss_all.append(loss_matrix[id_s[i], j, 1])  # put self in the other's shoes

        # eq = [eq_all[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]

        if eq_all is not []:
            trajectory_self = [trials_trajectory_self[eq_all[i][0]] for i in range(len(eq_all))]
            trajectory_other = [trials_trajectory_other[eq_all[i][1]] for i in range(len(eq_all))]
        else:
            trajectory_self = []
            trajectory_other = []

        return trajectory_self, trajectory_other, my_loss_all, other_loss_all

    def simulate_game(self, trajectory_self, trajectory_other, theta_self, theta_other, s, o):
        loss_s = self.loss.reactive_loss(theta_self, trajectory_self, trajectory_other, [1], s.states[-s.track_back],
                                         s.states_o[-s.track_back], s)
        loss_o = self.loss.reactive_loss(theta_other, trajectory_other, trajectory_self, [1], s.states_o[-s.track_back],
                                         s.states[-s.track_back], o)

        return loss_s, loss_o

    def intent_loss_func(self, intent):
        orientation_self = self.P_CAR_S.ORIENTATION
        state_self = self.states_s[-C.TRACK_BACK]
        state_other = self.states_o[-C.TRACK_BACK]
        action_other = self.actions_set_o[-C.TRACK_BACK]
        who = 1 - (self.P_CAR_S.BOUND_X is None)

        # alpha = intent[0] #aggressiveness of the agent
        trajectory = intent  # what I was expected to do

        # what I could have done and been
        s_other = np.array(state_self)
        nodes = np.array([[s_other[0], s_other[0] + trajectory[0] * np.cos(np.deg2rad(self.P_CAR_S.ORIENTATION)) / 2,
                           s_other[0] + trajectory[0] * np.cos(np.deg2rad(trajectory[1]))],
                          [s_other[1], s_other[1] + trajectory[0] * np.sin(np.deg2rad(orientation_self)) / 2,
                           s_other[1] + trajectory[0] * np.sin(np.deg2rad(trajectory[1]))]])
        curve = bezier.Curve(nodes, degree=2)
        positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_TIMESTEPS + 1)))
        a_other = np.diff(positions, n=1, axis=0)
        s_other_traj = np.array(s_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, a_other))

        # actions and states of the agent
        s_self = np.array(state_other)  # self.human_states_set[-1]
        a_self = np.array(C.ACTION_TIMESTEPS * [action_other])  # project current agent actions to future
        s_self_traj = np.array(s_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, a_self))

        # I expect the agent to be this much aggressive
        # theta_self = self.human_predicted_theta

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

        # if who == 0:
        # alpha = 1.

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
