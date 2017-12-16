class HumanVehicle:

    '''
    States:
            X-Position
            Y-Position
            X-Velocity
            Y-Velocity
    '''

    def __init__(self):
        input_file = open('human_state_files/human_change_lane_immediately.txt')

        self.states = []
        for line in input_file:
            line = line.split()  # to deal with blank
            if line:  # lines (ie skip them)
                line = [float(i) for i in line]
                self.states.append(line)

    def get_state(self, time_step):
        return self.states[time_step]