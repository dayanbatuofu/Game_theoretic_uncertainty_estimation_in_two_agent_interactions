from constants import CONSTANTS as C
import numpy as np
import scipy
from scipy import optimize
import pygame as pg


class MachineVehicle:

    """
    States:
            X-Position
            Y-Position
    """

    def __init__(self, P, ot_box, my_box, human_initial_state):

        self.P = P  # Scenario parameters
        self.other_collision_box = ot_box
        self.my_collision_box = my_box

        self.machine_theta = P.MACHINE_INTENT
        self.machine_predicted_theta = P.MACHINE_INTENT

        self.machine_states = [P.MACHINE_INITIAL_POSITION]
        self.machine_actions = []

        self.human_states = [human_initial_state]
        self.human_actions = []
        self.human_predicted_states = [human_initial_state]

        self.human_predicted_theta = P.HUMAN_INTENT

        self.human_predicted_state = human_initial_state

        # self.action_prediction_model = load_model('nn/action_prediction_model.h5')

        self.debug_1 = 0
        self.debug_2 = 0
        self.debug_3 = 0

        t_steps = C.T_FUTURE

        self.machine_previous_action_set = np.tile((P.MACHINE_INTENT[1] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER,
                                                    P.MACHINE_INTENT[2] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER),
                                                   (t_steps, 1))
        self.machine_previous_predicted_action_set = np.tile((P.MACHINE_INTENT[1] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER,
                                                              P.MACHINE_INTENT[2] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER),
                                                             (t_steps, 1))
        self.human_previous_action_set = np.tile((P.HUMAN_INTENT[1] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER,
                                                  P.HUMAN_INTENT[2] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER),
                                                 (t_steps, 1))

    def get_state(self):
        return self.machine_states[-1]

    def update(self, human_state):

        """ Function ran on every frame of simulation"""

        ########## Update human characteristics here ########

        if len(self.human_states) > C.T_PAST:
            human_predicted_theta, machine_estimated_theta = self.get_human_predicted_intent()

            self.human_predicted_theta = human_predicted_theta
            self.machine_predicted_theta = machine_estimated_theta


        ########## Calculate machine actions here ###########


        [machine_actions, human_predicted_actions, predicted_actions_self] = self.get_actions(1, self.human_previous_action_set, self.machine_previous_action_set,
                                                                             self.machine_previous_predicted_action_set,
                                                                             self.human_states[-1], self.machine_states[-1],
                                                                             self.human_predicted_theta, self.machine_theta,
                                                                             self.other_collision_box, self.my_collision_box,
                                                                             C.T_FUTURE)
        self.human_previous_action_set              = human_predicted_actions
        self.machine_previous_action_set            = machine_actions
        self.machine_previous_predicted_action_set  = predicted_actions_self



        self.human_predicted_state = human_state + sum(human_predicted_actions)

        self.update_state_action(machine_actions)

        last_human_state = self.human_states[-1]
        self.human_states.append(human_state)
        self.human_actions.append(np.array(human_state)-np.array(last_human_state))

    def update_state_action(self, actions):

        # Restrict speed
        action_x = np.clip(actions[0][0], -C.VEHICLE_MOVEMENT_SPEED, C.VEHICLE_MOVEMENT_SPEED)
        action_y = np.clip(actions[0][1], -C.VEHICLE_MOVEMENT_SPEED, C.VEHICLE_MOVEMENT_SPEED)

        self.machine_states.append(np.add(self.machine_states[-1], (action_x, action_y)))
        self.machine_actions.append((action_x, action_y))

    def get_actions(self, identifier, a0_other, a0_self, a0_predicted_self, s_other, s_self, theta_other, theta_self, box_other, box_self, t_steps):

        """ Function that accepts 2 vehicles states, intents, criteria, and an amount of future steps
        and return the ideal actions based on the loss function

        Identifier = 0 for human call
        Identifier = 1 for machine call"""

        # Initialize actions
        initial_actions_other = a0_other
        initial_actions_self = a0_self
        initial_predicted_actions_self = a0_predicted_self

        bounds = []
        for _ in range(t_steps):
            bounds.append((-C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER, C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER))
        for _ in range(t_steps):
            bounds.append((-C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER, C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER))

        A = np.zeros((t_steps, t_steps))
        A[np.tril_indices(t_steps, 0)] = 1

        if identifier == 0: # If human is calling
            other_bound_x = self.P.BOUND_MACHINE_X
            other_bound_y = self.P.BOUND_MACHINE_Y
            self_bound_x = self.P.BOUND_HUMAN_X
            self_bound_y = self.P.BOUND_HUMAN_Y

        if identifier == 1: # If machine is calling
            other_bound_x = self.P.BOUND_HUMAN_X
            other_bound_y = self.P.BOUND_HUMAN_Y
            self_bound_x = self.P.BOUND_MACHINE_X
            self_bound_y = self.P.BOUND_MACHINE_Y

        cons_other = []
        cons_self = []


        if other_bound_x is not None:
            if other_bound_x[0] is not None:
                for i in range(t_steps):
                    cons_other.append({'type': 'ineq','fun': lambda x, i=i: s_other[0] - box_other.height/2 + sum(x[0:i+1]) - other_bound_x[0]})
            if other_bound_x[1] is not None:
                for i in range(t_steps):
                    cons_other.append({'type': 'ineq','fun': lambda x, i=i: -s_other[0] - box_other.height/2 - sum(x[0:i+1]) + other_bound_x[1]})

        if other_bound_y is not None:
            if other_bound_y[0] is not None:
                for i in range(t_steps):
                    cons_other.append({'type': 'ineq','fun': lambda x, i=i: s_other[1]  - box_other.width/2 + sum(x[t_steps:t_steps+i+1]) - other_bound_y[0]})
            if other_bound_y[1] is not None:
                for i in range(t_steps):
                    cons_other.append({'type': 'ineq','fun': lambda x, i=i: -s_other[1] - box_other.width/2 - sum(x[t_steps:t_steps+i+1]) + other_bound_y[1]})

        if self_bound_x is not None:
            if self_bound_x[0] is not None:
                for i in range(t_steps):
                    cons_self.append({'type': 'ineq','fun': lambda x, i=i: s_self[0]  - box_self.height/2 + sum(x[0:i+1]) - self_bound_x[0]})
            if self_bound_x[1] is not None:
                for i in range(t_steps):
                    cons_self.append({'type': 'ineq','fun': lambda x, i=i: -s_self[0] - box_self.height/2 - sum(x[0:i+1]) + self_bound_x[1]})

        if self_bound_y is not None:
            if self_bound_y[0] is not None:
                for i in range(t_steps):
                    cons_self.append({'type': 'ineq','fun': lambda x, i=i: s_self[1] - box_self.width/2 + sum(x[t_steps:t_steps+i+1]) - self_bound_y[0]})
            if self_bound_y[1] is not None:
                for i in range(t_steps):
                    cons_self.append({'type': 'ineq','fun': lambda x, i=i: -s_self[1] - box_self.width/2 - sum(x[t_steps:t_steps+i+1]) + self_bound_y[1]})


        loss_value = 0
        loss_value_old = loss_value + C.LOSS_THRESHOLD + 1
        iter_count = 0

        actions_other = initial_actions_other
        actions_self = initial_actions_self
        predicted_actions_self = initial_predicted_actions_self

        # # Choose machine actions
        # optimization_results = scipy.optimize.minimize(self.loss_func, actions_self, bounds=bounds, constraints=cons_self,
        #                                                args=(self.P, s_other, s_self, actions_other, theta_self))
        # actions_self = np.column_stack((optimization_results.x[:t_steps],optimization_results.x[t_steps:]))
        # loss_value = optimization_results.fun

        while np.abs(loss_value-loss_value_old) > C.LOSS_THRESHOLD and iter_count < 1:
            loss_value_old = loss_value
            iter_count += 1

            # Estimate human's estimated machine actions
            optimization_results = scipy.optimize.minimize(self.loss_func, predicted_actions_self, bounds=bounds, constraints=cons_self,
                                                           args=(self.P, s_other, s_self, initial_actions_other, self.machine_predicted_theta, box_other, box_self))
            predicted_actions_self = np.column_stack((optimization_results.x[:t_steps], optimization_results.x[t_steps:]))

            # Estimate human actions
            optimization_results = scipy.optimize.minimize(self.loss_func, actions_other, bounds=bounds, constraints=cons_other,
                                                           args=(self.P, s_self, s_other, predicted_actions_self, theta_other, box_self, box_other))
            actions_other = np.column_stack((optimization_results.x[:t_steps], optimization_results.x[t_steps:]))
            loss_value = optimization_results.fun

        # Estimate machine actions
        optimization_results = scipy.optimize.minimize(self.loss_func, actions_self, bounds=bounds, constraints=cons_self,
                                                       args=(self.P, s_other, s_self, actions_other, theta_self, box_other, box_self))
        actions_self = np.column_stack((optimization_results.x[:t_steps], optimization_results.x[t_steps:]))

        return actions_self, actions_other, predicted_actions_self

    @staticmethod
    def loss_func(actions, P, s_other, s_self, actions_other, theta_self, box_other, box_self):

        """ Loss function defined to be a combination of state_loss and intent_loss with a weighted factor c """

        t_steps = int(len(actions)/2)

        action_factor = C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER

        actions             = np.column_stack((actions[:t_steps], actions[t_steps:]))

        theta_vector        = np.tile((theta_self[1] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER,
                                       theta_self[2] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER),
                                      (t_steps, 1))

        A = np.zeros((t_steps, t_steps))
        A[np.tril_indices(t_steps, 0)] = 1

        # Define state loss
        difference = s_self + np.matmul(A, actions) - s_other - np.matmul(A, actions_other)
        difference[:, 1] *= P.Y_CLEARANCE_WEIGHT # Weight y differences
        state_loss = np.reciprocal(box_self.get_minimum_distance(s_self + np.matmul(A, actions), s_other - np.matmul(A, actions_other), box_other)+1e-12)

        # Define action loss
        intent_loss = np.square(np.linalg.norm(actions - theta_vector, axis=1))

        # Define effort loss
        # effort_loss = 10 * theta_self[3] * np.sum(np.square(actions))

        # return np.sum(state_loss) + theta_self[0] * np.sum(intent_loss) + effort_loss  # Return weighted sum
        return np.sum(state_loss) + theta_self[0] * np.sum(intent_loss) # Return weighted sum

    def get_human_predicted_intent(self):
        """ Function accepts initial conditions and a time period for which to correct the
        attributes of the human car """

        t_steps = C.T_PAST
        s_self = self.human_states[-t_steps:]
        s_other = self.machine_states[-t_steps:]
        a_self = self.human_actions[-t_steps:]
        a_other = self.machine_actions[-t_steps:]
        theta_self = self.human_predicted_theta
        theta_other = self.machine_predicted_theta
        # nstate = len(s_other) #number of states
        # alpha_self = theta_self[0]
        # alpha_other = theta_other[0]
        A = np.zeros((t_steps, t_steps))
        A[np.tril_indices(t_steps, 0)] = 1 #lower tri 1s
        B = np.zeros((t_steps, t_steps))
        for i in range(t_steps-1):
            B[i,range(i+1,t_steps)]= np.arange(t_steps-1-i,0,-1)
        B = B + np.transpose(B) + np.diag(np.arange(t_steps,0,-1))
        b = np.arange(t_steps,0,-1)
        # phi_self = np.tile((theta_self[1] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER,
        #                        theta_self[2] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER),
        #                       (t_steps, 1))
        # phi_other = np.tile((theta_other[1] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER,
        #                        theta_other[2] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER),
        #                       (t_steps, 1))

        D = np.sqrt(np.sum((np.array(s_self)-np.array(s_other))**2, axis=1)) + 1e-3 #should be t_steps by 1, add small number for numerical stability

        # compute big K
        K_self = -2/(D**3)*B
        K_other = K_self
        ds = np.array(s_self)[0]-np.array(s_other)[0]
        c_self = np.dot(np.diag(-2/(D**3)), np.dot(np.expand_dims(b, axis=1), np.expand_dims(ds, axis=0))) + \
                 np.dot(np.diag(2/(D**3)), np.dot(B, a_other))
        c_other = np.dot(np.diag(-2/(D**3)), np.dot(np.expand_dims(b, axis=1), np.expand_dims(-ds, axis=0))) + \
                 np.dot(np.diag(2/(D**3)), np.dot(B, a_self))


        # update theta_hat_H
        w = np.dot(K_self, a_self)+c_self
        A = np.sum(a_self,axis=0)
        W = np.sum(w,axis=0)
        AW = np.diag(np.dot(np.transpose(a_self),w))
        AA = np.sum(np.array(a_self)**2,axis=0)
        # theta = (AW*A+W*AA)/(-W*A+AW*t_steps+1e-6)
        # bound_y = [0,1] - np.array(s_self)[-1,1]
        # theta[1] = np.clip(theta[1], bound_y[0], bound_y[1])
        # alpha = W/(t_steps*theta-A)
        # alpha = np.mean(np.clip(alpha,0.,100.))

        #Max: found a bug in the derivation, redo as follows
        numerator = np.dot(W,A)-t_steps*(np.sum(AW))
        denominator = t_steps*np.sum(AA)-np.dot(A,A)
        if np.abs(numerator) < 1e-6 and  np.abs(denominator) < 1e-6: # alpha = 0/0
            alpha = 100.
            theta = A
        else:
            alpha = numerator/denominator
            alpha = np.mean(np.clip(alpha,0.01,100.))
            theta = A + W/alpha

        theta = theta / C.VEHICLE_MOVEMENT_SPEED / C.ACTION_PREDICTION_MULTIPLIER
        bound_x = [-1, 1]
        bound_y = [0,1] - np.array(s_self)[-1,1]
        theta[0] = np.clip(theta[0], bound_x[0], bound_x[1])
        theta[1] = np.clip(theta[1], bound_y[0], bound_y[1])
        human_theta = (1-C.LEARNING_RATE)*self.human_predicted_theta + C.LEARNING_RATE*np.hstack((alpha,theta))

        # update theta_tilde_M
        w = np.dot(K_other, a_other)+c_other
        A = np.sum(a_other,axis=0)
        W = np.sum(w,axis=0)
        AW = np.diag(np.dot(np.transpose(a_other),w))
        AA = np.sum(np.array(a_other)**2,axis=0)
        numerator = np.dot(W,A)-t_steps*(np.sum(AW))
        denominator = t_steps*np.sum(AA)-np.dot(A,A)
        if np.abs(numerator) < 1e-6 and  np.abs(denominator) < 1e-6: # alpha = 0/0
            alpha = 100.
            theta = A
        else:
            alpha = numerator/denominator
            alpha = np.mean(np.clip(alpha,0.01,100.))
            theta = A + W/alpha

        theta = theta / C.VEHICLE_MOVEMENT_SPEED / C.ACTION_PREDICTION_MULTIPLIER
        bound_x = [-1, 1]
        bound_y = [0,1] - np.array(s_other)[-1,1]
        theta[0] = np.clip(theta[0], bound_x[0], bound_x[1])
        theta[1] = np.clip(theta[1], bound_y[0], bound_y[1])

        machine_estimated_theta = (1-C.LEARNING_RATE)*self.machine_predicted_theta + C.LEARNING_RATE*np.hstack((alpha,theta))

        predicted_theta = human_theta


        # Clip thetas
        if self.P.BOUND_HUMAN_X is not None:
            predicted_theta[1] = np.clip(predicted_theta[1], self.P.BOUND_HUMAN_X[0], self.P.BOUND_HUMAN_X[1])

        if self.P.BOUND_HUMAN_Y is not None:
            predicted_theta[2] = np.clip(predicted_theta[2], self.P.BOUND_HUMAN_Y[0], self.P.BOUND_HUMAN_Y[1])

        if self.P.BOUND_MACHINE_X is not None:
            machine_estimated_theta[11] = np.clip(machine_estimated_theta[1], self.P.BOUND_MACHINE_X[0], self.P.BOUND_MACHINE_X[1])

        if self.P.BOUND_MACHINE_Y is not None:
            machine_estimated_theta[2] = np.clip(machine_estimated_theta[2], self.P.BOUND_MACHINE_Y[0], self.P.BOUND_MACHINE_Y[1])

        return predicted_theta, machine_estimated_theta

    def human_loss_func(self, human_theta, machine_states, human_states, machine_theta):

        """ Loss function for the human correction defined to be the norm of the difference between actual actions and
        predicted actions"""

        t_steps = int(len(machine_states))

        actual_actions = np.diff(human_states, axis=0)
        predicted_actions = self.get_actions(machine_states[0], human_states[0], machine_theta, human_theta, t_steps - 1)
        predicted_actions_self = predicted_actions[0]

        difference = np.array(actual_actions) - np.array(predicted_actions_self)

        return np.linalg.norm(difference)