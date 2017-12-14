class MachineVehicle:

    '''
    States:
            X-Position
            Y-Position
            X-Velocity
            Y-Velocity
    '''

    def __init__(self, initial_state):
        self.state = initial_state

    def get_position(self):
        return self.state[0:2]

    def update(self):
        pass