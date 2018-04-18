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

    def __init__(self, P, ot_box, my_box, ot_initial_state, my_intial_state, ot_intent, my_intent,
                 ot_orientation, my_orientation, who):

        self.P = P  # Scenario parameters
        self.other_collision_box = ot_box
        self.my_collision_box = my_box

        # Initialize machine space
        self.machine_states = [my_intial_state]
        self.machine_theta = my_intent
        self.machine_actions = []

        # Initialize human space
        self.human_states = [ot_initial_state]
        self.human_predicted_theta = ot_intent
        self.human_actions = []

        # Initialize predicted human predicted machine space
        self.machine_predicted_theta = my_intent


        self.debug_1 = 0
        self.debug_2 = 0
        self.debug_3 = 0

        self.machine_action_set = np.tile((0, 0), (C.ACTION_TIMESTEPS, 1))
        self.machine_predicted_action_set = np.tile((0, 0), (C.ACTION_TIMESTEPS, 1))
        self.human_action_set = np.tile((0, 0), (C.ACTION_TIMESTEPS, 1))

        self.who = who
        self.human_orientation = ot_orientation
        self.machine_orientation = my_orientation

    def get_state(self, delay):
        return self.machine_states[-1*delay]

    def update(self, human_state):

        """ Function ran on every frame of simulation"""

        ########## Update human characteristics here ########

        if len(self.human_states) > C.T_PAST:
            human_predicted_theta, machine_estimated_theta = self.get_human_predicted_intent()

            # ''' DEBUG ONLY '''
            # human_predicted_theta = C.PARAMETERSET_1.HUMAN_INTENT
            # machine_estimated_theta = C.PARAMETERSET_1.MACHINE_INTENT
            # ''' DEBUG ONLY '''

            self.human_predicted_theta = human_predicted_theta
            self.machine_predicted_theta = machine_estimated_theta


        ########## Calculate machine actions here ###########


        [machine_actions, human_predicted_actions, predicted_actions_self] = self.get_actions(self.who, self.human_action_set, self.machine_action_set,
                                                                                              self.machine_predicted_action_set,
                                                                                              self.human_states[-1], self.machine_states[-1],
                                                                                              self.human_predicted_theta, self.machine_theta,
                                                                                              self.other_collision_box, self.my_collision_box)
        self.human_predicted_action_set    = human_predicted_actions
        self.machine_action_set            = machine_actions
        self.machine_predicted_action_set  = predicted_actions_self

        # Update machine state
        self.machine_states.append(np.add(self.machine_states[-1], (machine_actions[0][0], machine_actions[0][1])))
        self.machine_actions.append(machine_actions[0])

        # Update human state
        last_human_state = self.human_states[-1]
        self.human_states.append(human_state)
        self.human_actions.append(np.array(human_state)-np.array(last_human_state))


    def get_actions(self, identifier, a0_other, a0_self, a0_predicted_self, s_other, s_self, theta_other, theta_self, box_other, box_self):

        """ Function that accepts 2 vehicles states, intents, criteria, and an amount of future steps
        and return the ideal actions based on the loss function

        Identifier = 0 for human call
        Identifier = 1 for machine call"""

        # Initialize actions
        # initial_trajectory_other = (np.linalg.norm(a0_other[-1]), np.arctan2(a0_other[-1, 0], a0_other[-1, 1]))
        # initial_trajectory_self = (np.linalg.norm(a0_self[-1]), np.arctan2(a0_self[-1, 0], a0_self[-1, 1]))
        # initial_predicted_trajectory_self = (np.linalg.norm(a0_predicted_self[-1]), np.arctan2(a0_predicted_self[-1, 0], a0_predicted_self[-1, 1]))
        initial_trajectory_other = theta_other[1:3]
        initial_trajectory_self = theta_self[1:3]
        initial_predicted_trajectory_self = self.machine_predicted_theta[1:3]

        if identifier == 0:  # If human is calling
            defcon_other_x = self.P.BOUND_MACHINE_X
            defcon_other_y = self.P.BOUND_MACHINE_Y
            orientation_other = self.P.MACHINE_ORIENTATION
            bounds_other = [(0, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # Radius
                            (-C.ACTION_TURNANGLE + self.P.MACHINE_ORIENTATION,
                             C.ACTION_TURNANGLE + self.P.MACHINE_ORIENTATION)]  # Angle
            defcon_self_x = self.P.BOUND_HUMAN_X
            defcon_self_y = self.P.BOUND_HUMAN_Y
            orientation_self = self.P.HUMAN_ORIENTATION
            bounds_self = [(0, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # Radius
                           (-C.ACTION_TURNANGLE + self.P.HUMAN_ORIENTATION,
                            C.ACTION_TURNANGLE + self.P.HUMAN_ORIENTATION)]  # Angle

        if identifier == 1:  # If machine is calling
            defcon_other_x = self.P.BOUND_HUMAN_X
            defcon_other_y = self.P.BOUND_HUMAN_Y
            orientation_other = self.P.HUMAN_ORIENTATION

            bounds_other = [(0, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # Radius
                           (-C.ACTION_TURNANGLE + self.P.HUMAN_ORIENTATION,
                            C.ACTION_TURNANGLE + self.P.HUMAN_ORIENTATION)]  # Angle

            defcon_self_x = self.P.BOUND_MACHINE_X
            defcon_self_y = self.P.BOUND_MACHINE_Y
            orientation_self = self.P.MACHINE_ORIENTATION
            bounds_self = [(0, C.ACTION_TIMESTEPS * self.P.VEHICLE_MAX_SPEED),  # Radius
                       (-C.ACTION_TURNANGLE + self.P.MACHINE_ORIENTATION,
                        C.ACTION_TURNANGLE + self.P.MACHINE_ORIENTATION)]  # Angle


        A = np.zeros((C.ACTION_TIMESTEPS, C.ACTION_TIMESTEPS))
        A[np.tril_indices(C.ACTION_TIMESTEPS, 0)] = 1

        cons_other = []
        cons_self = []

        if defcon_other_x is not None:
            if defcon_other_x[0] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: s_other[0] + (x[0]*scipy.cos(np.deg2rad(x[1]))) - box_other.height/2 - defcon_other_x[0]})
            if defcon_other_x[1] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: -s_other[0] - (x[0]*scipy.cos(np.deg2rad(x[1]))) - box_other.height/2 + defcon_other_x[1]})

        if defcon_other_y is not None:
            if defcon_other_y[0] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: s_other[1] + (x[0]*scipy.sin(np.deg2rad(x[1]))) - box_other.width/2 - defcon_other_y[0]})
            if defcon_other_y[1] is not None:
                cons_other.append({'type': 'ineq','fun': lambda x: -s_other[1] - (x[0]*scipy.sin(np.deg2rad(x[1]))) - box_other.width/2 + defcon_other_y[1]})

        if defcon_self_x is not None:
            if defcon_self_x[0] is not None:
                cons_self.append({'type': 'ineq','fun': lambda x: s_self[0] + (x[0]*scipy.cos(np.deg2rad(x[1]))) - box_self.height/2 - defcon_self_x[0]})
            if defcon_self_x[1] is not None:
                cons_self.append({'type': 'ineq','fun': lambda x: -s_self[0] - (x[0]*scipy.cos(np.deg2rad(x[1]))) - box_self.height/2 + defcon_self_x[1]})

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
        predicted_trajectory_self = initial_predicted_trajectory_self

        # guess_set = np.array([[0,0],[10,0]]) #TODO: need to generalize this
        guess_set = [(0,0),(0,-90)] #TODO: max added (0,-90) to address a nan issue that appears in the intersection case for human

        while np.abs(loss_value-loss_value_old) > C.LOSS_THRESHOLD and iter_count < 2:
            loss_value_old = loss_value
            iter_count += 1

            # Estimate human's estimated machine actions
            predicted_trajectory_self, _ = self.multi_search(np.append(guess_set, [predicted_trajectory_self], axis=0), bounds_self,
                                                         cons_self, s_other, s_self, trajectory_other,
                                                         self.machine_predicted_theta, box_other, box_self, orientation_other, orientation_self)

            # Estimate human actions
            trajectory_other, loss_value = self.multi_search(np.append(guess_set, [trajectory_other], axis=0), bounds_other, cons_other, s_self,
                                                        s_other, predicted_trajectory_self, theta_other, box_self,
                                                        box_other, orientation_self, orientation_other)

        # Estimate machine actions
        trajectory_self, _ = self.multi_search(np.append(guess_set, [trajectory_self], axis=0), bounds_self,
                                             cons_self, s_other, s_self, trajectory_other,
                                             theta_self, box_other, box_self, orientation_other, orientation_self)


        # Interpolate for output
        actions_self = self.interpolate_from_trajectory(trajectory_self, s_self, orientation_self)
        actions_other = self.interpolate_from_trajectory(trajectory_other, s_other, orientation_other)
        predicted_actions_self = self.interpolate_from_trajectory(predicted_trajectory_self, s_self, orientation_self)

        return actions_self, actions_other, predicted_actions_self

    def multi_search(self, guess_set, bounds, cons, s_o, s_s, traj_o, theta_s, box_o, box_s, orientation_o, orientation_s):

        """ run multiple searches with different initial guesses """

        trajectory_set = np.empty((0,2)) #TODO: need to generalize
        loss_value_set = []

        for guess in guess_set:
            optimization_results = scipy.optimize.minimize(self.loss_func, guess, bounds=bounds, constraints=cons,
                                                           args=(s_o, s_s, traj_o, theta_s, self.P.VEHICLE_MAX_SPEED * C.ACTION_TIMESTEPS, box_o, box_s, orientation_o, orientation_s))
            if np.isfinite(optimization_results.fun):
                trajectory_set = np.append(trajectory_set, [optimization_results.x], axis=0)
                loss_value_set = np.append(loss_value_set, optimization_results.fun)

        trajectory = trajectory_set[np.where(loss_value_set == np.min(loss_value_set))[0][0]]


        # self.loss_func((0,0), s_o, s_s, traj_o, theta_s, self.P.VEHICLE_MAX_SPEED * C.ACTION_TIMESTEPS, box_o, box_s, orientation_o, orientation_s)

        return trajectory, np.min(loss_value_set)


    def loss_func(self, trajectory, s_other, s_self, trajectory_other, theta_self, theta_max, box_other, box_self, orientation_other, orientation_self):

        """ Loss function defined to be a combination of state_loss and intent_loss with a weighted factor c """

        actions_self    = self.interpolate_from_trajectory(trajectory, s_self, orientation_self)
        actions_other   = self.interpolate_from_trajectory(trajectory_other, s_other, orientation_other)

        # Define state loss
        state_loss = np.reciprocal(box_self.get_collision_distance(s_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_self),
                                                                   s_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, actions_other), box_other)+1e-12)

        # Define action loss
        intended_trajectory = self.interpolate_from_trajectory(theta_self[1:3], s_self, orientation_self)
        intent_loss = np.square(np.linalg.norm(actions_self - intended_trajectory, axis=1))

        return np.linalg.norm(state_loss) + theta_self[0] * np.linalg.norm(intent_loss) # Return weighted sum

    def get_human_predicted_intent(self):
        """ Function accepts initial conditions and a time period for which to correct the
        attributes of the human car """

        t_steps = C.T_PAST
        s_self = np.array(self.human_states[-t_steps:])
        s_other = np.array(self.machine_states[-t_steps:])
        a_self = np.array(self.human_actions[-t_steps:])
        a_other = np.array(self.machine_actions[-t_steps:])
        theta_self = self.human_predicted_theta
        theta_other = self.machine_predicted_theta
        # A = np.zeros((t_steps, t_steps))
        # A[np.tril_indices(t_steps, 0)] = 1 #lower tri 1s
        A = M.LOWER_TRIANGULAR_SMALL

        # B = np.zeros((t_steps, t_steps))
        # for i in range(t_steps-1):
        #     B[i,range(i+1,t_steps)]= np.arange(t_steps-1-i,0,-1)
        # B = B + np.transpose(B) + np.diag(np.arange(t_steps,0,-1))
        # b = np.arange(t_steps,0,-1)

        D = np.sum((np.array(s_self)-np.array(s_other))**2, axis=1) + 1e-12 #should be t_steps by 1, add small number for numerical stability
        # need to check if states are in the collision box
        for i in range(s_self.shape[0]):
            if s_self[i,1]<=-1.5 or s_self[i,1]>=1.5 or s_other[i,0]>=1.5 or s_other[i,0]<=-1.5:
                D[i] = np.inf

        sigD = 1000. / (1 + np.exp(10.*(-D + C.CAR_LENGTH**2*5)))+0.01
        dsigD = 10.*sigD / (1 + np.exp(10.*(D - C.CAR_LENGTH**2*5)))
        ds = s_self[-1,:] - s_other[-1,:]

        # dD/da
        dDda_self = - np.dot(np.expand_dims(np.dot(A.transpose(), sigD**(-2)*dsigD),axis=1), np.expand_dims(ds, axis=0)) \
               -  np.dot(np.dot(A.transpose(), np.diag(sigD**(-2)*dsigD)), np.dot(A, a_self - a_other))
        dDda_other = - np.dot(np.expand_dims(np.dot(A.transpose(), sigD**(-2)*dsigD),axis=1), np.expand_dims(-ds, axis=0)) \
               - np.dot(np.dot(A.transpose(), np.diag(sigD**(-2)*dsigD)), np.dot(A, a_other - a_self))
        # K_self = -2/(D**3)*B
        # K_other = K_self
        # ds = np.array(s_self)[0]-np.array(s_other)[0]
        # c_self = np.dot(np.diag(-2/(D**3)), np.dot(np.expand_dims(b, axis=1), np.expand_dims(ds, axis=0))) + \
        #          np.dot(np.diag(2/(D**3)), np.dot(B, a_other))
        # c_other = np.dot(np.diag(-2/(D**3)), np.dot(np.expand_dims(b, axis=1), np.expand_dims(-ds, axis=0))) + \
        #          np.dot(np.diag(2/(D**3)), np.dot(B, a_self))


        # update theta_hat_H
        w = - dDda_self # negative gradient direction

        if self.who == 1: #machine
            if self.P.BOUND_HUMAN_X is not None: # intersection
                w[np.all([s_self[:,0]<=0, w[:,0] <= 0], axis=0),0] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_self[:,0]>=0, w[:,0] >= 0], axis=0),0] = 0 #TODO: these two lines are hard coded for intersection, need to check the interval
            else: # lane changing
                w[np.all([s_self[:,1]<=0, w[:,1] <= 0], axis=0),1] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_self[:,1]>=1, w[:,1] >= 0], axis=0),1] = 0 #TODO: these two lines are hard coded for lane changing
        else: #human
            if self.P.BOUND_HUMAN_X is not None: # intersection
                w[np.all([s_self[:,1]<=0, w[:,1] <= 0], axis=0),1] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_self[:,1]>=0, w[:,1] >= 0], axis=0),1] = 0 #TODO: these two lines are hard coded for intersection, need to check the interval
            else: # lane changing
                w[np.all([s_self[:,0]<=0, w[:,0] <= 0], axis=0),0] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_self[:,0]>=1, w[:,0] >= 0], axis=0),0] = 0 #TODO: these two lines are hard coded for lane changing

        w = -w

        # A = np.sum(a_self,axis=0)
        # W = np.sum(w,axis=0)
        # AW = np.diag(np.dot(np.transpose(a_self),w))
        # AA = np.sum(np.array(a_self)**2,axis=0)
        # # theta = (AW*A+W*AA)/(-W*A+AW*t_steps+1e-6)
        # # bound_y = [0,1] - np.array(s_self)[-1,1]
        # # theta[1] = np.clip(theta[1], bound_y[0], bound_y[1])
        # # alpha = W/(t_steps*theta-A)
        # # alpha = np.mean(np.clip(alpha,0.,100.))
        #
        # #Max: found a bug in the derivation, redo as follows
        # numerator = np.dot(W,A)-t_steps*(np.sum(AW))
        # denominator = t_steps*np.sum(AA)-np.dot(A,A)
        # if np.abs(numerator) < 1e-6 and np.abs(denominator) < 1e-6: # alpha = 0/0
        #     alpha = C.INTENT_LIMIT
        #     theta = A
        # else:
        #     alpha = numerator/denominator
        #     alpha = np.mean(np.clip(alpha,0.01,C.INTENT_LIMIT))
        #     theta = A + W/alpha

        if self.P.BOUND_HUMAN_X is not None: #intersection
            intent_bounds = [(0.1, None), # alpha
                             (0, C.T_PAST * self.P.VEHICLE_MAX_SPEED), # radius
                             (-180, 0)] # angle, to accommodate crazy behavior
        else:
            intent_bounds = [(0.1, None), # alpha
                             (0, C.T_PAST * self.P.VEHICLE_MAX_SPEED), # radius
                             (-90, 90)] # angle, to accommodate crazy behavior

        intent_optimization_results = scipy.optimize.minimize(self.intent_loss_func, self.human_predicted_theta,
                                                              bounds=intent_bounds, args=(w, a_self,
                                                              self.human_orientation, self.human_predicted_theta[0]))
        alpha, r, rho = intent_optimization_results.x
        theta = [r / t_steps * C.ACTION_TIMESTEPS, rho] # scale the radius

        # theta = theta / t_steps * C.ACTION_TIMESTEPS
        # bound_x = [-1, 1]
        # bound_y = [0,1] - np.array(s_self)[-1,1]
        # theta[0] = np.clip(theta[0], bound_x[0], bound_x[1])
        # theta[1] = np.clip(theta[1], bound_y[0], bound_y[1])

        current_theta_point = [self.human_predicted_theta[1] * scipy.cos(np.deg2rad(self.human_predicted_theta[2])),
                               self.human_predicted_theta[1] * scipy.sin(np.deg2rad(self.human_predicted_theta[2]))]
        intent_theta_point = [theta[0] * scipy.cos(np.deg2rad(theta[1])),
                               theta[0] * scipy.sin(np.deg2rad(theta[1]))]
        theta_point = (1-C.LEARNING_RATE)*np.array(current_theta_point) + C.LEARNING_RATE*np.array(intent_theta_point)
        bound_y = [0,1] - np.array(s_self)[-1,1]
        # theta_point[1] = np.clip(theta_point[1], bound_y[0], bound_y[1])

        if self.who == 1:
            if self.P.BOUND_HUMAN_X is not None:
                _bound = [self.P.BOUND_HUMAN_X[0], self.P.BOUND_HUMAN_X[1]] - np.array(s_self)[-1, 0]
                theta_point[0] = np.clip(theta_point[0], _bound[0], _bound[1])

            if self.P.BOUND_HUMAN_Y is not None:
                _bound = [self.P.BOUND_HUMAN_Y[0], self.P.BOUND_HUMAN_Y[1]] - np.array(s_self)[-1, 1]
                theta_point[1] = np.clip(theta_point[1], _bound[0], _bound[1])
        else:
            if self.P.BOUND_MACHINE_X is not None:
                _bound = [self.P.BOUND_MACHINE_X[0], self.P.BOUND_MACHINE_X[1]] - np.array(s_self)[-1, 0]
                theta_point[0] = np.clip(theta_point[0], _bound[0], _bound[1])

            if self.P.BOUND_MACHINE_Y is not None:
                _bound = [self.P.BOUND_MACHINE_Y[0], self.P.BOUND_MACHINE_Y[1]] - np.array(s_self)[-1, 1]
                theta_point[1] = np.clip(theta_point[1], _bound[0], _bound[1])

        dist, angle = np.linalg.norm(theta_point), scipy.arctan2(theta_point[1],theta_point[0])/np.pi*180
        human_theta = [(1-C.LEARNING_RATE)*self.human_predicted_theta[0] + C.LEARNING_RATE*alpha, dist, angle]

        # human_theta = self.P.HUMAN_INTENT


        # update theta_tilde_M
        w = - dDda_other # negative gradient direction
        if self.who == 1: #machine
            if self.P.BOUND_HUMAN_X is not None: #intersection
                w[np.all([s_other[:,1]<=0, w[:,1] <= 0], axis=0),1] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_other[:,1]>=0, w[:,1] >= 0], axis=0),1] = 0 #TODO: these two lines are hard coded for lane changing
            else:
                w[np.all([s_other[:,1]<=0, w[:,1] <= 0], axis=0),1] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_other[:,1]>=1, w[:,1] >= 0], axis=0),1] = 0 #TODO: these two lines are hard coded for lane changing
        else: #human
             if self.P.BOUND_HUMAN_X is not None: #intersection
                w[np.all([s_other[:,0]<=0, w[:,0] <= 0], axis=0),0] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_other[:,0]>=0, w[:,0] >= 0], axis=0),0] = 0 #TODO: these two lines are hard coded for lane changing
             else:
                w[np.all([s_other[:,0]<=0, w[:,0] <= 0], axis=0),0] = 0 #if against wall and push towards the wall, get a reaction force
                w[np.all([s_other[:,0]>=1, w[:,0] >= 0], axis=0),0] = 0 #TODO: these two lines are hard coded for lane changing

        w = -w

        # A = np.sum(a_other,axis=0)
        # W = np.sum(w,axis=0)
        # AW = np.diag(np.dot(np.transpose(a_other),w))
        # AA = np.sum(np.array(a_other)**2,axis=0)
        # numerator = np.dot(W,A)-t_steps*(np.sum(AW))
        # denominator = t_steps*np.sum(AA)-np.dot(A,A)
        # if np.abs(numerator) < 1e-6 and np.abs(denominator) < 1e-6: # alpha = 0/0
        #     alpha = C.INTENT_LIMIT
        #     theta = A
        # else:
        #     alpha = numerator/denominator
        #     alpha = np.mean(np.clip(alpha,0.01,C.INTENT_LIMIT))
        #     theta = A + W/alpha

        intent_bounds = [(0.1, None), # alpha
                         (0, C.T_PAST * self.P.VEHICLE_MAX_SPEED), # radius
                         (-C.ACTION_TURNANGLE, C.ACTION_TURNANGLE)] # angle

        intent_optimization_results = scipy.optimize.minimize(self.intent_loss_func, self.machine_predicted_theta,
                                                              bounds=intent_bounds, args=(w, a_other,
                                                              self.machine_actions, self.machine_predicted_theta[0]))
        alpha, r, rho = intent_optimization_results.x
        theta = [r / t_steps * C.ACTION_TIMESTEPS, rho] # scale the radius

        # theta = theta / t_steps * C.ACTION_TIMESTEPS
        current_theta_point = [self.machine_predicted_theta[1] * scipy.cos(np.deg2rad(self.machine_predicted_theta[2])),
                               self.machine_predicted_theta[1] * scipy.sin(np.deg2rad(self.machine_predicted_theta[2]))]
        intent_theta_point = [theta[0] * scipy.cos(np.deg2rad(theta[1])), theta[0] * scipy.sin(np.deg2rad(theta[1]))]
        theta_point = (1-C.LEARNING_RATE)*np.array(current_theta_point) + C.LEARNING_RATE*np.array(intent_theta_point)

        if self.who == 1:
            if self.P.BOUND_MACHINE_X is not None:
                _bound = [self.P.BOUND_MACHINE_X[0], self.P.BOUND_MACHINE_X[1]] - np.array(s_other)[-1, 0]
                theta_point[0] = np.clip(theta_point[0], _bound[0], _bound[1])

            if self.P.BOUND_MACHINE_Y is not None:
                _bound = [self.P.BOUND_MACHINE_Y[0], self.P.BOUND_MACHINE_Y[1]] - np.array(s_other)[-1, 1]
                theta_point[1] = np.clip(theta_point[1], _bound[0], _bound[1])

        else:
            if self.P.BOUND_HUMAN_X is not None:
                _bound = [self.P.BOUND_HUMAN_X[0], self.P.BOUND_HUMAN_X[1]] - np.array(s_other)[-1, 0]
                theta_point[0] = np.clip(theta_point[0], _bound[0], _bound[1])

            if self.P.BOUND_HUMAN_Y is not None:
                _bound = [self.P.BOUND_HUMAN_Y[0], self.P.BOUND_HUMAN_Y[1]] - np.array(s_other)[-1, 1]
                theta_point[1] = np.clip(theta_point[1], _bound[0], _bound[1])

        dist, angle = np.linalg.norm(theta_point), scipy.arctan2(theta_point[1],theta_point[0])/np.pi*180
        machine_predicted_theta = [(1-C.LEARNING_RATE)*self.machine_predicted_theta[0] + C.LEARNING_RATE*alpha, dist, angle]

        # machine_predicted_theta = C.PARAMETERSET_1.MACHINE_INTENT

        # # Clip thetas
        # if self.P.BOUND_HUMAN_X is not None:
        #     predicted_theta[1] = np.clip(predicted_theta[1], self.P.BOUND_HUMAN_X[0], self.P.BOUND_HUMAN_X[1])
        #
        # if self.P.BOUND_HUMAN_Y is not None:
        #     predicted_theta[2] = np.clip(predicted_theta[2], self.P.BOUND_HUMAN_Y[0], self.P.BOUND_HUMAN_Y[1])
        #
        # if self.P.BOUND_MACHINE_X is not None:
        #     machine_estimated_theta[11] = np.clip(machine_estimated_theta[1], self.P.BOUND_MACHINE_X[0], self.P.BOUND_MACHINE_X[1])
        #
        # if self.P.BOUND_MACHINE_Y is not None:
        #     machine_estimated_theta[2] = np.clip(machine_estimated_theta[2], self.P.BOUND_MACHINE_Y[0], self.P.BOUND_MACHINE_Y[1])

        return human_theta, machine_predicted_theta

    def interpolate_from_trajectory(self, trajectory, state, orientation):

        nodes = np.array([[state[0], state[0] + trajectory[0]*np.cos(np.deg2rad(trajectory[1]))/2, state[0] + trajectory[0]*np.cos(np.deg2rad(trajectory[1]))],
                          [state[1], state[1] + trajectory[0]*np.sin(np.deg2rad(trajectory[1]))/2, state[1] + trajectory[0]*np.sin(np.deg2rad(trajectory[1]))]])

        curve = bezier.Curve(nodes, degree=2)

        positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_NUMPOINTS + 1)))
        #TODO: skip state?
        return np.diff(positions, n=1, axis=0)

    def intent_loss_func(self, intent, w, a, orientation, alpha0):
        alpha, r, rho = intent
        # theta = [r*np.cos(np.deg2rad(a)), r*np.sin(np.deg2rad(a))]

        state = [0,0]
        trajectory = [r,rho]
        nodes = np.array([[state[0], state[0] + trajectory[0]*np.cos(np.deg2rad(orientation))/2, state[0] + trajectory[0]*np.cos(np.deg2rad(trajectory[1]))],
                  [state[1], state[1] + trajectory[0]*np.sin(np.deg2rad(orientation))/2, state[1] + trajectory[0]*np.sin(np.deg2rad(trajectory[1]))]])
        curve = bezier.Curve(nodes, degree=2)
        positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.T_PAST + 1)))
        intent_a = np.diff(positions, n=1, axis=0)

        x = w + alpha*(a - intent_a)
        L = np.sum(x**2) + 0.001*(alpha-alpha0)**2
        #TODO: added a small penalty on moving alpha to avoid abitrary alpha when a-intent_a = 0
        return L