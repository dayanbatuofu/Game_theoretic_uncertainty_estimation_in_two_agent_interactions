from constants import CONSTANTS as C


class MachineVehicle:

    '''
    States:
            X-Position
            Y-Position
            X-Velocity
            Y-Velocity
    '''

    def __init__(self, initial_state):

        self.states = [initial_state]
        self.actions = [0]

        self.human_states = []
        self.human_actions = []

    def get_state(self):
        return self.states[-1]

    def update(self, human_state):

        human_predicted_action = self.predict_human_action(human_state)
        new_action = self.optimize_loss(human_state, human_predicted_action)

        self.update_state_action(new_action)

    def update_state_action(self, action):

        x, y, x_vel, y_vel = self.states[-1]

        if action == 1:
            x_vel += C.VEHICLE_ACCELERATION
        if action == 2:
            x_vel -= C.VEHICLE_ACCELERATION

        x += x_vel
        y += y_vel

        self.states.append((x, y, x_vel, y_vel))
        self.actions.append(action)

    def predict_human_action(self, human_state):
        # Implement prediction of human state here
        pass

    def optimize_loss(self, human_state, human_predicted_action):
        # Implement loss function minimization here
        pass