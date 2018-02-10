from constants import CONSTANTS as C
import numpy as np
import scipy
from scipy import optimize


class MachineVehicle:

    """
    States:
            X-Position
            Y-Position
    """

    def __init__(self, machine_initial_state, human_initial_state):

        self.machine_criteria = C.MACHINE_CRITERIA
        self.machine_intent = C.MACHINE_INTENT

        self.machine_states = [machine_initial_state]
        self.machine_actions = [(0, 0)]

        self.human_states = [human_initial_state]
        self.human_actions = [(0, 0)]
        self.human_predicted_states = [human_initial_state]

        self.human_predicted_intent = (0, 0)
        self.human_predicted_criteria = 20
        self.human_predicted_state = human_initial_state

        self.debug_1 = 0
        self.debug_2 = 0
        self.debug_3 = 0

    def get_state(self):
        return self.machine_states[-1]

    def update(self, human_state):

        """ Function ran on every frame of simulation"""

        self.human_states.append(human_state)

        ########## Update human characteristics here ########

        # machine_intent = C.MACHINE_INTENT #?????
        # machine_criteria = C.MACHINE_CRITERIA  # ?????
        #
        # if len(self.human_states) > C.T_PAST:
        #     human_predicted_intent, human_predicted_criteria = self.get_human_predicted_intent(self.human_predicted_intent,
        #                                                                              self.human_predicted_criteria,
        #                                                                              self.machine_states,
        #                                                                              self.human_states,
        #                                                                              machine_intent,
        #                                                                              machine_criteria,
        #                                                                              C.T_PAST)
        #     self.human_predicted_intent = human_predicted_intent
        #     self.human_predicted_criteria = human_predicted_criteria


        ########## Calculate machine actions here ###########

        [machine_actions, human_predicted_actions] = self.get_actions(self.human_states[-1], self.machine_states[-1],
                                                                      self.human_predicted_intent, self.machine_intent,
                                                                      self.human_predicted_criteria, self.machine_criteria, C.T_FUTURE)

        self.human_predicted_state = human_state + sum(human_predicted_actions)


        machine_new_action = np.clip(machine_actions[0], -C.VEHICLE_MOVEMENT_SPEED, C.VEHICLE_MOVEMENT_SPEED) # Restrict speed
        self.update_state_action(machine_new_action)

        self.human_predicted_state = human_state + sum(human_predicted_actions)

    def update_state_action(self, action):

        self.machine_states.append(np.add(self.machine_states[-1], action))
        self.machine_actions.append(action)

    def get_actions(self, s_other, s_self, s_desired_other, s_desired_self, c_other, c_self, t_steps):

        """ Function that accepts 2 vehicles states, intents, criteria, and an amount of future steps
        and return the ideal actions based on the loss function"""

        # Error between desired position and current position
        error_other = np.array(s_desired_other) - np.array(s_other)
        error_self = np.array(s_desired_self) - np.array(s_self)

        # Define theta
        theta_other = np.clip(error_other, -C.THETA_LIMITER_X, C.THETA_LIMITER_X)
        theta_self = np.clip(error_self, -C.THETA_LIMITER_Y, C.THETA_LIMITER_Y)


        # Initialize actions
        actions_other = np.array([0 for _ in range(2 * t_steps)])
        actions_self = np.array([0 for _ in range(2 * t_steps)])


        bounds = tuple([(-C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER, C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER) for _ in range(2 * t_steps)])

        A = np.zeros((t_steps, t_steps))
        A[np.tril_indices(t_steps, 0)] = 1

        cons_other = ({'type': 'ineq', 'fun': lambda x: s_other[1] + x[10] - C.Y_MINIMUM},
                      {'type': 'ineq', 'fun': lambda x: -s_other[1] - x[10] + C.Y_MAXIMUM})

        cons_self = ({'type': 'ineq', 'fun': lambda x: s_self[1] + x[10] - C.Y_MINIMUM},
                     {'type': 'ineq', 'fun': lambda x: -s_self[1] - x[10] + C.Y_MAXIMUM})

        loss_value = 0
        loss_value_old = loss_value + C.LOSS_THRESHOLD + 1
        iter_count = 0

        # Estimate machine actions
        optimization_results = scipy.optimize.minimize(self.loss_func, actions_self, bounds=bounds, constraints=cons_self,
                                                       args=(s_other, s_self, actions_other, theta_self, c_self))
        actions_self = optimization_results.x
        loss_value = optimization_results.fun

        while np.abs(loss_value-loss_value_old) > C.LOSS_THRESHOLD and iter_count < 1:
            loss_value_old = loss_value
            iter_count += 1

            # Estimate human actions
            optimization_results = scipy.optimize.minimize(self.loss_func, actions_other, bounds=bounds, constraints=cons_other,
                                                           args=(s_self, s_other, actions_self, theta_other, c_other))
            actions_other = optimization_results.x

            # Estimate machine actions
            optimization_results = scipy.optimize.minimize(self.loss_func, actions_self, bounds=bounds, constraints=cons_self,
                                                           args=(s_other, s_self, actions_other, theta_self, c_self))
            actions_self = optimization_results.x
            loss_value = optimization_results.fun


        actions_other = np.transpose(np.vstack((actions_other[:t_steps], actions_other[t_steps:])))
        actions_self = np.transpose(np.vstack((actions_self[:t_steps], actions_self[t_steps:])))

        return actions_self, actions_other


    @staticmethod
    def loss_func(actions, s_other, s_self, actions_other, theta_self, c):

        """ Loss function defined to be a combination of state_loss and intent_loss with a weighted factor c """

        t_steps = int(len(actions)/2)

        actions = np.transpose(np.vstack((actions[:t_steps], actions[t_steps:])))
        actions_other = np.transpose(np.vstack((actions_other[:t_steps], actions_other[t_steps:])))

        theta_vectorized = np.tile(theta_self, (t_steps, 1))

        A = np.zeros((t_steps, t_steps))
        A[np.tril_indices(t_steps, 0)] = 1

        # Define state loss
        state_loss = np.reciprocal(np.linalg.norm(s_self + np.matmul(A, actions) - s_other - np.matmul(A, actions_other), axis=1))

        # Define action loss
        intent_loss = np.square(np.linalg.norm(actions - theta_vectorized))

        return np.sum(state_loss) + c * np.sum(intent_loss)  # Return sum with a weighted factor


    def get_human_predicted_intent(self, old_human_intent, old_human_criteria, machine_states, human_states,  machine_intent, machine_criteria, t_steps):

        """ Function accepts initial conditions and a time period for which to correct the
        attributes of the human car """

        machine_states = machine_states[-t_steps:]
        human_states = human_states[-t_steps:]

        optimization_results = scipy.optimize.minimize(self.human_loss_func,
                                                       np.array([old_human_intent[0], old_human_intent[1], old_human_criteria]),
                                                       args=(machine_states, human_states, machine_intent, machine_criteria))

        predicted_intent_x = optimization_results.x[0]
        predicted_intent_y = optimization_results.x[1]
        predicted_intent = [predicted_intent_x, predicted_intent_y]

        predicted_criteria = optimization_results.x[2]

        return [predicted_intent, predicted_criteria]

    def human_loss_func(self, optimized_characteristics, machine_states, human_states, machine_intent, machine_criteria):

        """ Loss function for the human correction defined to be the norm of the difference between actual actions and
        predicted actions"""

        t_steps = int(len(machine_states))

        intent = optimized_characteristics[0:2]  # 2D
        criteria = optimized_characteristics[2]  #1D

        actual_actions = np.diff(human_states, axis=0)
        predicted_actions = self.get_actions(machine_states[0], human_states[0], machine_intent, intent, machine_criteria, criteria, t_steps - 1)
        predicted_actions_self = predicted_actions[0]

        difference = np.array(actual_actions) - np.array(predicted_actions_self)

        return np.linalg.norm(difference)
