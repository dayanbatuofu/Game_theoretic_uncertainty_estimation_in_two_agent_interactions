
class TrackVehicle:

    '''
    States:
            X-Position
            Y-Position
    '''

    def __init__(self, state_set):

        input_file = open(state_set)

        self.states = []
        for line in input_file:
            line = line.split()  # to deal with blank
            if line:  # lines (ie skip them)
                line = tuple([float(i) for i in line])
                self.states.append(line)

    def get_state(self, time_step):
        return list(self.states[time_step])
