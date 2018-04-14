import numpy as np

class CONSTANTS:

    # DISPLAY
    FPS = 10
    AXES_SHOW = 0.5
    COORDINATE_SCALE = 150

    CAR_WIDTH = 0.66
    CAR_LENGTH = 1.33

    ZOOM = 0.3


    # POSITION BOUNDS
    Y_MINIMUM = 0
    Y_MAXIMUM = 1


    # OPTIMIZATION
    ACTION_TIMESTEPS = 100  # 5 seconds
    ACTION_TURNANGLE = 30  # degrees
    ACTION_NUMPOINTS = 100

    T_PAST = 10

    THETA_LIMITER_X = 0.01
    THETA_LIMITER_Y = 0.01

    LOSS_THRESHOLD = 0.01

    LEARNING_RATE = 0.5
    INTENT_LIMIT = 1000. #TODO: this is the max alpha, need to explain what this means

    class PARAMETERSET_1:

        # DISPLAY
        SCREEN_WIDTH = 4
        SCREEN_HEIGHT = 5

        # SPEED
        VEHICLE_MAX_SPEED = 0.1

        # INITIAL CONDITIONS
        MACHINE_INITIAL_POSITION = np.array([-1, 0])

        # INTENTS
        HUMAN_INTENT = np.array([10., VEHICLE_MAX_SPEED*0.75*100, -7.7])
        MACHINE_INTENT = np.array([10, VEHICLE_MAX_SPEED*100, 0.])

        # VEHICLE ORIENTATIONS
        HUMAN_ORIENTATION = 0
        MACHINE_ORIENTATION = 0

        # BOUNDS
        BOUND_HUMAN_X = None
        BOUND_HUMAN_Y = np.array([-0.5, 1.5])

        BOUND_MACHINE_X = None
        BOUND_MACHINE_Y = np.array([-0.5, 1.5])

        # COLLISION BOXES
        COLLISION_BOXES = np.array([(-np.inf, np.inf, -0.5, 0.5), (-np.inf, np.inf, 0.5, 1.5)])  # List of separate collision boxes (-x, x, -y, y)

        # LOSS WEIGHT
        Y_CLEARANCE_WEIGHT = 0.3

    class PARAMETERSET_2:

        # DISPLAY
        SCREEN_WIDTH = 5
        SCREEN_HEIGHT = 5

        # SPEED
        VEHICLE_MAX_SPEED = 0.05

        # INITIAL CONDITIONS
        MACHINE_INITIAL_POSITION = np.array([-3.0, 0])

        # INTENTS
        HUMAN_INTENT = np.array([10, VEHICLE_MAX_SPEED*1.0*100, -90])
        MACHINE_INTENT = np.array([100., VEHICLE_MAX_SPEED*100, 0])

        # VEHICLE ORIENTATIONS
        HUMAN_ORIENTATION = -90
        MACHINE_ORIENTATION = 0

        # BOUNDS
        BOUND_HUMAN_X = np.array([-0.4, 0.4])
        BOUND_HUMAN_Y = None

        BOUND_MACHINE_X = None
        BOUND_MACHINE_Y = np.array([-0.4, 0.4])

        # COLLISION BOXES
        COLLISION_BOXES = np.array([(-0.4, 0.4, -0.4, 0.4)])  # List of separate collision boxes (-x, x, -y, y)

        # LOSS WEIGHT
        Y_CLEARANCE_WEIGHT = 1

class MATRICES:

    LOWER_TRIANGULAR_MATRIX = np.zeros((CONSTANTS.ACTION_NUMPOINTS, CONSTANTS.ACTION_NUMPOINTS))
    LOWER_TRIANGULAR_MATRIX[np.tril_indices(CONSTANTS.ACTION_NUMPOINTS, 0)] = 1

    LOWER_TRIANGULAR_SMALL = np.zeros((CONSTANTS.T_PAST, CONSTANTS.T_PAST))
    LOWER_TRIANGULAR_SMALL[np.tril_indices(CONSTANTS.T_PAST, 0)] = 1




