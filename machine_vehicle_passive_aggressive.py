#this is the active vehicle for intersection

from constants import CONSTANTS as C
from constants import MATRICES as M
import numpy as np
import scipy
import bezier
from scipy import optimize
import pygame as pg
from scipy.interpolate import spline


class MachineVehicle:

    """
    States:
            X-Position
            Y-Position
    """

    def __init__(self, P, ot_box, my_box, my_intial_state, ot_intent, ot_my_intent, my_intent,
                 ot_orientation, my_orientation, who, script):

        self.P = P  # Scenario parameters
        self.other_collision_box = ot_box
        self.my_collision_box = my_box

        # Initialize machine space
        if script is None:
            self.machine_states_set = [my_intial_state]
        self.machine_theta = my_intent
        self.machine_actions_set = []
        self.machine_trajectory = []
        self.machine_planed_actions_set = []

        self.human_predicted_trajectory = []
        self.human_predicted_actions    = []
        self.machine_expected_actions  = np.tile((0, 0), (C.ACTION_TIMESTEPS, 1))
        self.who = who
        self.machine_orientation = my_orientation
        # Initialize predicted human predicted machine space
        if who == 1:
            self.machine_expected_trajectory = self.P.COMMON_THETA_MACHINE
        else:
            self.machine_expected_trajectory = self.P.COMMON_THETA_HUMAN

        # Initialize human space
        self.human_states_set = []
        self.human_predicted_theta = ot_intent
        self.human_actions_set = []
        self.human_orientation = ot_orientation

    def get_state(self, delay):
        return self.machine_states_set[-1*delay]

    def update(self, human, frame):

        """ Function ran on every frame of simulation"""

        ########## Update human characteristics here ########
        if self.who == 0:
            self.human_states_set = np.array(human.machine_states_set)[:-1] #get other's states
            self.human_actions_set = np.array(human.machine_actions_set)[:-1] #get other's actions
        else:
            self.human_states_set = np.array(human.machine_states_set) #get other's states
            self.human_actions_set = np.array(human.machine_actions_set) #get other's actions

        if len(self.human_states_set) > 1 and len(self.machine_states_set) > 1: # human will not repeat this
            theta_human, machine_expected_trajectory = self.get_human_predicted_intent(self.who) #"self" inside prediction is human (who=0)
            self.human_predicted_theta = [theta_human]
            self.machine_expected_trajectory = machine_expected_trajectory

        ########## Calculate machine actions here ###########
        [machine_trajectory, human_predicted_trajectory, machine_actions, human_predicted_actions,
         machine_expected_actions] = self.get_actions()

        self.machine_trajectory = machine_trajectory
        self.human_predicted_trajectory = human_predicted_trajectory
        self.human_predicted_actions    = human_predicted_actions
        self.machine_expected_actions  = machine_expected_actions

        # Update self state
        # if self.scripted_state is not None: #if action scripted
        #     self.machine_states_set.append(self.scripted_state[frame+1]) # get the NEXT state
        #     machine_actions = np.subtract(self.machine_states_set[-1], self.machine_states_set[-2])
        #     self.machine_actions_set.append(machine_actions)
        # else:
        self.machine_states_set.append(np.add(self.machine_states_set[-1], (machine_actions[0][0], machine_actions[0][1])))
        self.machine_actions_set.append(machine_actions[0])
        self.machine_planed_actions_set.append(machine_actions)

    def get_actions(self):

        """ Function that accepts 2 vehicles states, intents, criteria, and an amount of future steps
        and return the ideal actions based on the loss function

        Identifier = 0 for human call
        Identifier = 1 for machine call"""

        identifier = self.who
        if len(self.human_actions_set)>0:
            a0_other = self.human_actions_set
            initial_trajectory_other = [np.linalg.norm(a0_other[-1])*C.ACTION_TIMESTEPS,
                                        np.arctan2(a0_other[-1, 1], a0_other[-1, 0])*180./np.pi]
            initial_trajectory_self = self.machine_trajectory
        else:
            if self.who == 1:
                initial_trajectory_other = self.P.COMMON_THETA_HUMAN
                initial_trajectory_self = self.P.COMMON_THETA_MACHINE
            else:
                initial_trajectory_other = self.P.COMMON_THETA_MACHINE
                initial_trajectory_self = self.P.COMMON_THETA_HUMAN

        s_other = self.human_states_set[-1]
        s_self = self.machine_states_set[-1]
        theta_other = self.human_predicted_theta
        theta_self = self.machine_theta
        box_other = self.other_collision_box
        box_self = self.my_collision_box

        # Initialize actions
        # initial_trajectory_self = (np.linalg.norm(a0_self[-1]), np.arctan2(a0_self[-1, 0], a0_self[-1, 1]))
        # initial_predicted_trajectory_self = (np.linalg.norm(a0_predicted_self[-1]), np.arctan2(a0_predicted_self[-1, 0], a0_predicted_self[-1, 1]))
        # initial_trajectory_other = theta_other[1:3]
        initial_expected_trajectory_self = self.machine_expected_trajectory

        if identifier == 0:  # If human is calling
            defcon_other_x = self.P.BOUND_MACHINE_X
            defcon_other_y = self.P.BOUND_MACHINE_Y
            orientation_other = self.P.MACHINE_ORIENTATION
            bounds_other = [(0 * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # Radius
                            (-C.ACTION_TURNANGLE + self.P.MACHINE_ORIENTATION,
                             C.ACTION_TURNANGLE + self.P.MACHINE_ORIENTATION)]  # Angle
            defcon_self_x = self.P.BOUND_HUMAN_X
            defcon_self_y = self.P.BOUND_HUMAN_Y
            orientation_self = self.P.HUMAN_ORIENTATION
            bounds_self = [(-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # Radius
                           (-C.ACTION_TURNANGLE + self.P.HUMAN_ORIENTATION,
                            C.ACTION_TURNANGLE + self.P.HUMAN_ORIENTATION)]  # Angle

        if identifier == 1:  # If machine is calling
            defcon_other_x = self.P.BOUND_HUMAN_X
            defcon_other_y = self.P.BOUND_HUMAN_Y
            orientation_other = self.P.HUMAN_ORIENTATION

            bounds_other = [(-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # Radius
                           (-C.ACTION_TURNANGLE + self.P.HUMAN_ORIENTATION,
                            C.ACTION_TURNANGLE + self.P.HUMAN_ORIENTATION)]  # Angle

            defcon_self_x = self.P.BOUND_MACHINE_X
            defcon_self_y = self.P.BOUND_MACHINE_Y
            orientation_self = self.P.MACHINE_ORIENTATION
            bounds_self = [(-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # Radius
                       (-C.ACTION_TURNANGLE + self.P.MACHINE_ORIENTATION,
                        C.ACTION_TURNANGLE + self.P.MACHINE_ORIENTATION)]  # Angle

        A = np.zeros((C.ACTION_TIMESTEPS, C.ACTION_TIMESTEPS))
        A[np.tril_indices(C.ACTION_TIMESTEPS, 0)] = 1

        cons_other = []
        cons_self = []

        if defcon_other_x is not None:
            if defcon_other_x[0] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: -s_other[0] - (x[0]*scipy.cos(np.deg2rad(x[1]))) - box_other.height/2 + defcon_other_x[1]})
            if defcon_other_x[1] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: s_other[0] + (x[0]*scipy.cos(np.deg2rad(x[1]))) - box_other.height/2 - defcon_other_x[0]})

        if defcon_other_y is not None:
            if defcon_other_y[0] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: -s_other[1] - (x[0]*scipy.sin(np.deg2rad(x[1]))) - box_other.width/2 + defcon_other_y[1]})
            if defcon_other_y[1] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: s_other[1] + (x[0]*scipy.sin(np.deg2rad(x[1]))) - box_other.width/2 - defcon_other_y[0]})

        if defcon_self_x is not None:
            if defcon_self_x[0] is not None:
                cons_self.append({'type': 'ineq','fun': lambda x: -s_self[0] - (x[0]*scipy.cos(np.deg2rad(x[1]))) - box_self.height/2 + defcon_self_x[1]})
            if defcon_self_x[1] is not None:
                cons_self.append({'type': 'ineq','fun': lambda x: s_self[0] + (x[0]*scipy.cos(np.deg2rad(x[1]))) - box_self.height/2 - defcon_self_x[0]})

        if defcon_self_y is not None:
            if defcon_self_y[0] is not None:
                    cons_self.append({'type': 'ineq','fun': lambda x: s_self[1] + (x[0]*scipy.sin(np.deg2rad(x[1]))) - box_self.width/2 - defcon_self_y[0]})
            if defcon_self_y[1] is not None:
                    cons_self.append({'type': 'ineq','fun': lambda x: -s_self[1] - (x[0]*scipy.sin(np.deg2rad(x[1]))) - box_self.width/2 + defcon_self_y[1]})

        loss_value = 0
        loss_value_old = loss_value + C.LOSS_THRESHOLD + 1
        iter_count = 0

        trajectory_other = initial_trajectory_other
        trajectory_self = initial_trajectory_self
        predicted_trajectory_self = initial_expected_trajectory_self

        # guess_set = np.array([[0,0],[10,0]]) #TODO: need to generalize this

        trials = np.arange(5,-1.1,-0.1)
        guess_set = np.hstack((np.expand_dims(trials, axis=1),
                    np.ones((trials.size,1)) * self.machine_orientation))
        guess_other = np.hstack((np.expand_dims(trials, axis=1),
                    np.ones((trials.size,1)) * self.human_orientation))

        # Estimate machine actions
        trajectory_self, predicted_trajectory_other = self.multi_search(guess_set, bounds_self,
                                                                        cons_self, s_other, s_self, theta_self,
                                                                        box_other, box_self,
                                                                        orientation_other, orientation_self,
                                                                        self.machine_expected_trajectory, self.who)

        # Interpolate for output
        actions_self = self.interpolate_from_trajectory(trajectory_self, s_self, orientation_self)
        predicted_actions_other = self.interpolate_from_trajectory(predicted_trajectory_other, s_other, orientation_other)
        predicted_actions_self = self.interpolate_from_trajectory(predicted_trajectory_self, s_self, orientation_self)

        return trajectory_self, predicted_trajectory_other, actions_self, predicted_actions_other, predicted_actions_self

    def multi_search(self, guess_set, bounds, cons, s_o, s_s, theta_s, box_o, box_s, orientation_o,
                     orientation_s, expected_trajectory_s, who):

        """ run multiple searches with different initial guesses """

        trajectory_set = np.empty((0,2)) #TODO: need to generalize
        predicted_trajectory_other_set = np.empty((0,2))
        loss_value_set = []

        for guess in guess_set:
            # optimization_results = scipy.optimize.minimize(self.loss_func, guess, bounds=bounds, #constraints=cons,
            #                                                args=(s_o, s_s, traj_o, theta_s,
            #                                                      self.P.VEHICLE_MAX_SPEED * C.ACTION_TIMESTEPS, box_o,
            #                                                      box_s, orientation_o, orientation_s, who),
            #                                                )

            fun, predicted_trajectory_other = self.loss_func(guess, s_o, s_s, theta_s,
                                                             self.P.VEHICLE_MAX_SPEED * C.ACTION_TIMESTEPS, box_o,
                                                             box_s, orientation_o, orientation_s,
                                                             expected_trajectory_s, who)

            # if np.isfinite(optimization_results.fun) and not np.isnan(optimization_results.fun) \
            #         and optimization_results.success:
            # trajectory_set = np.append(trajectory_set, [optimization_results.x], axis=0)
            # loss_value_set = np.append(loss_value_set, optimization_results.fun)
            trajectory_set = np.append(trajectory_set, [guess], axis=0)
            predicted_trajectory_other_set = np.append(predicted_trajectory_other_set, [predicted_trajectory_other],
                                                       axis=0)
            loss_value_set = np.append(loss_value_set, fun)

        trajectory = trajectory_set[np.where(loss_value_set == np.min(loss_value_set))[0][0]]
        predicted_trajectory_other = predicted_trajectory_other_set[np.where(loss_value_set ==
                                                                             np.min(loss_value_set))[0][0]]

        return trajectory, predicted_trajectory_other

    def loss_func(self, trajectory, s_other, s_self, theta_self, theta_max, box_other, box_self,
                  orientation_other, orientation_self, expected_trajectory_self, who):

        ##############################################################################################
        # predict how others perceive your action
        trials = np.arange(5,-1.1,-0.1)
        # guess_set = np.hstack((np.ones((trials.size,1)) * self.human_predicted_theta[0], np.expand_dims(trials, axis=1),
        #                        np.ones((trials.size,1)) * self.machine_orientation))
        guess_set = np.hstack((np.expand_dims(trials, axis=1),
                               np.ones((trials.size,1)) * orientation_other))
        action_self = self.interpolate_from_trajectory(trajectory, s_self, orientation_self)[0]
        intent_optimization_results = self.multi_search_intent(guess_set, [], [], orientation_other,
                                                               s_other, s_self, action_self, 1-who)
        alpha_me_by_other, r, rho = intent_optimization_results

        expected_trajectory_other_by_me = [r, rho] # I expect you to understand that I expect you to do this
        ##############################################################################################

        actions_self    = self.interpolate_from_trajectory(trajectory, s_self, orientation_self)
        actions_other   = self.interpolate_from_trajectory(expected_trajectory_other_by_me, s_other, orientation_other)

        s_other_predict = s_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other)
        s_self_predict = s_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self)
        D = box_self.get_collision_distance(s_self_predict, s_other_predict, box_other)+1e-12
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
            intent_loss = theta_self[0] * np.exp(C.EXPTHETA * (- s_self_predict[-1][0] + 0.4))
        else:
            intent_loss = theta_self[0] * np.exp(C.EXPTHETA * (s_self_predict[-1][1] + 0.4))

        # return np.linalg.norm(np.reciprocal(sigD)) + theta_self[0] * np.linalg.norm(intent_loss) # Return weighted sum
        gracefulness_loss = (trajectory[0] - expected_trajectory_self[0])**2

        loss = collision_loss + intent_loss + gracefulness_loss

        return loss, expected_trajectory_other_by_me # Return weighted sum

    def get_human_predicted_intent(self, who):
        """ predict the aggressiveness of the agent and what the agent expect me to do """

        cons = []
        if who == 1: # machine looking at human
            if self.P.BOUND_HUMAN_X is not None: #intersection
                intent_bounds = [(0.1, None), # alpha
                                 (0 * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED), # radius
                                 (-C.ACTION_TURNANGLE + self.machine_orientation, C.ACTION_TURNANGLE + self.machine_orientation)] # angle, to accommodate crazy behavior
                cons.append({'type': 'ineq','fun': lambda x: self.machine_states_set[-1][1] + (x[0]*scipy.sin(np.deg2rad(x[1]))) - (0.4 - 0.33)})
                cons.append({'type': 'ineq','fun': lambda x: - self.machine_states_set[-1][1] - (x[0]*scipy.sin(np.deg2rad(x[1]))) + (-0.4 + 0.33)})

            else: #TODO: update this part
                intent_bounds = [(0.1, None), # alpha
                                 (-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED), # radius
                                 (-90, 90)] # angle, to accommodate crazy behavior
        else: # human looking at machine
            if self.P.BOUND_HUMAN_X is not None: #intersection
                intent_bounds = [(0.1, None), # alpha
                                 (0 * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED), # radius
                                 (-C.ACTION_TURNANGLE + self.machine_orientation, C.ACTION_TURNANGLE + self.machine_orientation)] # angle, to accommodate crazy behavior
                cons.append({'type': 'ineq','fun': lambda x: self.machine_states_set[-1][0] + (x[0]*scipy.cos(np.deg2rad(x[1]))) - (0.4 - 0.33)})
                cons.append({'type': 'ineq','fun': lambda x: - self.machine_states_set[-1][0] - (x[0]*scipy.cos(np.deg2rad(x[1]))) + (-0.4 + 0.33)})

            else: #TODO: update this part
                intent_bounds = [(0.1, None), # alpha
                                 (-C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED), # radius
                                 (-90, 90)] # angle, to accommodate crazy behavior

        #TODO: damn...I don't know why the solver is not working, insert a valid solution, output nan...
        trials = np.arange(5,-0.1,-0.1)
        # guess_set = np.hstack((np.ones((trials.size,1)) * self.human_predicted_theta[0], np.expand_dims(trials, axis=1),
        #                        np.ones((trials.size,1)) * self.machine_orientation))
        guess_set = np.hstack((np.expand_dims(trials, axis=1),
                               np.ones((trials.size,1)) * self.machine_orientation))

        intent_optimization_results = self.multi_search_intent(guess_set, intent_bounds, cons,
                                                               self.machine_orientation, self.machine_states_set[-1],
                                                               self.human_states_set[-1], self.human_actions_set[-1], who)
        alpha_other, r, rho = intent_optimization_results

        # what the agent expected me to do
        expected_trajectory = [r, rho] # scale the radius
        return alpha_other, expected_trajectory

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

    def interpolate_from_trajectory(self, trajectory, state, orientation):

        nodes = np.array([[state[0], state[0] + trajectory[0]*np.cos(np.deg2rad(trajectory[1]))/2, state[0] + trajectory[0]*np.cos(np.deg2rad(trajectory[1]))],
                          [state[1], state[1] + trajectory[0]*np.sin(np.deg2rad(trajectory[1]))/2, state[1] + trajectory[0]*np.sin(np.deg2rad(trajectory[1]))]])

        curve = bezier.Curve(nodes, degree=2)

        positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_NUMPOINTS + 1)))
        #TODO: skip state?
        return np.diff(positions, n=1, axis=0)