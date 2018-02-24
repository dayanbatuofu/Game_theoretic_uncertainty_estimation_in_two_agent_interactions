from constants import CONSTANTS as C
import numpy as np
import scipy
from scipy import optimize
# from keras.models import load_model


class MachineVehicle:

    """
    States:
            X-Position
            Y-Position
    """

    def __init__(self, machine_initial_state, human_initial_state):

        self.machine_theta = C.MACHINE_INTENT

        self.machine_states = [machine_initial_state]
        self.machine_actions = [(0, 0)]

        self.human_states = [human_initial_state]
        self.human_actions = [(0, 0)]
        self.human_predicted_states = [human_initial_state]

        self.human_predicted_theta = C.HUMAN_INTENT

        self.human_predicted_state = human_initial_state

        # self.action_prediction_model = load_model('nn/action_prediction_model.h5')

        self.debug_1 = 0
        self.debug_2 = 0
        self.debug_3 = 0

    def get_state(self):
        return self.machine_states[-1]

    def update(self, human_state):

        """ Function ran on every frame of simulation"""

        self.human_states.append(human_state)

        ########## Update human characteristics here ########

        machine_theta = C.MACHINE_INTENT  # ?????

        if len(self.human_states) > C.T_PAST:
            human_predicted_theta = self.get_human_predicted_intent(self.human_predicted_theta,
                                                                    self.machine_states,
                                                                    self.human_states,
                                                                    machine_theta,
                                                                    C.T_PAST)

            self.human_predicted_theta = human_predicted_theta


        ########## Calculate machine actions here ###########

        # Use prediction function
        [machine_actions, human_predicted_actions] = self.get_actions(self.human_states[-1], self.machine_states[-1],
                                                                        self.human_predicted_theta, self.machine_theta, C.T_FUTURE)

        # Use prediction model
        # [machine_actions, human_predicted_actions] = self.get_learned_action(self.human_states[-1], self.machine_states[-1],
        #                                                                self.human_theta, self.machine_theta, C.T_FUTURE)


        self.human_predicted_state = human_state + sum(human_predicted_actions)

        self.update_state_action(machine_actions)


    def update_state_action(self, actions):

        # Restrict speed
        action_x = np.clip(actions[0][0], -C.VEHICLE_MOVEMENT_SPEED, C.VEHICLE_MOVEMENT_SPEED)
        action_y = np.clip(actions[0][1], -C.VEHICLE_MOVEMENT_SPEED, C.VEHICLE_MOVEMENT_SPEED)

        self.machine_states.append(np.add(self.machine_states[-1], (action_x, action_y)))
        self.machine_actions.append((action_x, action_y))

    def get_actions(self, s_other, s_self, theta_other, theta_self, t_steps):

        """ Function that accepts 2 vehicles states, intents, criteria, and an amount of future steps
        and return the ideal actions based on the loss function"""

        # Initialize actions
        actions_other = np.array([0 for _ in range(2 * t_steps)])
        actions_self = np.array([0 for _ in range(2 * t_steps)])

        bounds = []
        for _ in range(t_steps):
            bounds.append((-C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER, C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER))
        for _ in range(t_steps):
            bounds.append((-C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER, C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER))

        A = np.zeros((t_steps, t_steps))
        A[np.tril_indices(t_steps, 0)] = 1

        cons_other = []
        for i in range(t_steps):
            cons_other.append({'type': 'ineq',
                               'fun': lambda x, i=i: s_other[1] + sum(x[t_steps:t_steps+i+1]) - C.Y_MINIMUM})
            cons_other.append({'type': 'ineq',
                               'fun': lambda x, i=i: -s_other[1] - sum(x[t_steps:t_steps+i+1]) + C.Y_MAXIMUM})

        cons_self = []
        for i in range(t_steps):
            cons_self.append({'type': 'ineq',
                              'fun': lambda x, i=i: s_self[1] + sum(x[t_steps:t_steps+i+1]) - C.Y_MINIMUM})
            cons_self.append({'type': 'ineq',
                              'fun': lambda x, i=i: -s_self[1] - sum(x[t_steps:t_steps+i+1]) + C.Y_MAXIMUM})

        loss_value = 0
        loss_value_old = loss_value + C.LOSS_THRESHOLD + 1
        iter_count = 0

        # Estimate machine actions
        optimization_results = scipy.optimize.minimize(self.loss_func, actions_self, bounds=bounds, constraints=cons_self,
                                                       args=(s_other, s_self, actions_other, theta_self))
        actions_self = optimization_results.x
        loss_value = optimization_results.fun

        while np.abs(loss_value-loss_value_old) > C.LOSS_THRESHOLD and iter_count < 1:
            loss_value_old = loss_value
            iter_count += 1

            # Estimate human actions
            optimization_results = scipy.optimize.minimize(self.loss_func, actions_other, bounds=bounds, constraints=cons_other,
                                                           args=(s_self, s_other, actions_self, theta_other))
            actions_other = optimization_results.x

            # Estimate machine actions
            optimization_results = scipy.optimize.minimize(self.loss_func, actions_self, bounds=bounds, constraints=cons_self,
                                                           args=(s_other, s_self, actions_other, theta_self))
            actions_self = optimization_results.x
            loss_value = optimization_results.fun

        actions_other = np.transpose(np.vstack((actions_other[:t_steps], actions_other[t_steps:])))
        actions_self = np.transpose(np.vstack((actions_self[:t_steps], actions_self[t_steps:])))

        return actions_self, actions_other

    def get_learned_action(self, s_other, s_self, s_desired_other, s_desired_self, t_steps):

        """ Function that predicts actions based upon loaded neural network """

        s_other_y_range = [0, 1]
        s_self_x_range = [-2, 2]
        s_self_y_range = [0, 1]
        s_desired_other_x_range = [-2, 2]
        s_desired_other_y_range = [0, 1]
        c_other_range = [20, 100]

        #  Normalize inputs
        s_other_y_norm = (s_other[1] - s_other_y_range[0]) / (s_other_y_range[1] - s_other_y_range[0])
        s_self_x_norm = (s_self[0] - s_self_x_range[0]) / (s_self_x_range[1] - s_self_x_range[0])
        s_self_y_norm = (s_self[1] - s_self_y_range[0]) / (s_self_y_range[1] - s_self_y_range[0])
        s_desired_other_x_norm = (s_desired_other[0] - s_desired_other_x_range[0]) / (s_desired_other_x_range[1] - s_desired_other_x_range[0])
        s_desired_other_y_norm = (s_desired_other[1] - s_desired_other_y_range[0]) / (s_desired_other_y_range[1] - s_desired_other_y_range[0])

        network_output = self.action_prediction_model.predict(np.array([[s_other_y_norm,
                                                                         s_self_x_norm,
                                                                         s_self_y_norm,
                                                                         s_desired_other_x_norm,
                                                                         s_desired_other_y_norm]]))

        actions_self = np.array(network_output[0:t_steps])
        actions_other = np.array(network_output[t_steps:])

        # Scale outputs
        actions_self = actions_self * (2 * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER) - (C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER)
        actions_other = actions_other * (2 * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER) - (C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER)

        return actions_self, actions_other

    @staticmethod
    def loss_func(actions, s_other, s_self, actions_other, theta_self):

        """ Loss function defined to be a combination of state_loss and intent_loss with a weighted factor c """

        t_steps = int(len(actions)/2)

        action_factor = C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER

        actions             = np.transpose(np.vstack((actions[:t_steps], actions[t_steps:])))
        actions_other       = np.transpose(np.vstack((actions_other[:t_steps], actions_other[t_steps:])))
        theta_vector        = np.tile((theta_self[1] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER,
                                       theta_self[2] * C.VEHICLE_MOVEMENT_SPEED * C.ACTION_PREDICTION_MULTIPLIER),
                                      (t_steps, 1))

        A = np.zeros((t_steps, t_steps))
        A[np.tril_indices(t_steps, 0)] = 1

        # Define state loss
        state_loss = np.reciprocal(np.linalg.norm(s_self + np.matmul(A, actions) - s_other - np.matmul(A, actions_other), axis=1))

        # Define action loss
        intent_loss = np.square(np.linalg.norm(actions - theta_vector, axis=1))

        return np.sum(state_loss) + theta_self[0] * np.sum(intent_loss)  # Return weighted sum

    def get_human_predicted_intent(self, old_human_theta, machine_states, human_states, machine_theta, t_steps):

        """ Function accepts initial conditions and a time period for which to correct the
        attributes of the human car """

        machine_states = machine_states[-t_steps:]
        human_states = human_states[-t_steps:]

        bounds = [(-1, 1), (-1, 1)]

        old_human_theta_multiplier = old_human_theta[0]
        old_human_theta_vector = old_human_theta[1:3]

        optimization_results = scipy.optimize.minimize(self.human_loss_func,
                                                       old_human_theta_vector,
                                                       bounds=bounds,
                                                       args=(old_human_theta_multiplier, machine_states, human_states, machine_theta))
        predicted_theta_vector = optimization_results.x

        predicted_theta = (old_human_theta_multiplier, predicted_theta_vector[0], predicted_theta_vector[1])

        return predicted_theta

    def human_loss_func(self, human_theta_vector, human_theta_multiplier, machine_states, human_states, machine_theta):

        """ Loss function for the human correction defined to be the norm of the difference between actual actions and
        predicted actions"""

        human_theta = (human_theta_multiplier, human_theta_vector[0], human_theta_vector[1])

        t_steps = int(len(machine_states))

        actual_actions = np.diff(human_states, axis=0)
        predicted_actions = self.get_actions(machine_states[0], human_states[0], machine_theta, human_theta, t_steps - 1)
        predicted_actions_self = predicted_actions[0]

        difference = np.array(actual_actions) - np.array(predicted_actions_self)

        return np.linalg.norm(difference)
