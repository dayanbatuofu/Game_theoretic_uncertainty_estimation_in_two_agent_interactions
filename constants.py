import numpy as np

class CONSTANTS:

    # DISPLAY
    FPS = 10

    CAR_WIDTH = 100
    CAR_LENGTH = 200
    LANE_WIDTH = 150

    # SPEEDS
    VEHICLE_MOVEMENT_SPEED = 0.02

    # POSITION BOUNDS
    Y_MINIMUM = 0
    Y_MAXIMUM = 1


    # OPTIMIZATION

    ACTION_PREDICTION_MULTIPLIER = 10

    T_FUTURE = 10
    T_PAST = 10

    THETA_LIMITER_X = 0.1
    THETA_LIMITER_Y = 0.1

    LOSS_THRESHOLD = 0.01

    LEARNING_RATE = 0.1

    # DIVIDE BY ZERO CATCH
    EPS = 0.0000001

    class PARAMETERSET_1:

        # DISPLAY
        SCREEN_WIDTH = 300
        SCREEN_HEIGHT = 800

        # ORIGIN
        ORIGIN = np.array([SCREEN_WIDTH/2 - 75, SCREEN_HEIGHT/2])
        COORDINATE_SCALE = 150


        # INITIAL CONDITIONS
        MACHINE_INITIAL_POSITION = np.array([-1, 0])

        # INTENTS
        HUMAN_INTENT = np.array([30, 0, 0])
        MACHINE_INTENT = np.array([10, 1, 0])

        # VEHICLE ORIENTATIONS
        HUMAN_ORIENTATION = 0
        MACHINE_ORIENTATION = 0

        # BOUNDS
        BOUND_HUMAN_X = None
        BOUND_HUMAN_Y = np.array([0, 1])

        BOUND_MACHINE_X = None
        BOUND_MACHINE_Y = np.array([0, 1])

        # LOSS WEIGHT
        Y_CLEARANCE_WEIGHT = 0.3

    class PARAMETERSET_2:
        # DISPLAY
        SCREEN_WIDTH = 1000
        SCREEN_HEIGHT = 1000

        # ORIGIN
        ORIGIN = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2])
        COORDINATE_SCALE = 150

        # INITIAL CONDITIONS
        MACHINE_INITIAL_POSITION = np.array([-2.5, 0])

        # INTENTS
        HUMAN_INTENT = np.array([30, 0, -1])
        MACHINE_INTENT = np.array([10, 1, 0])

        # VEHICLE ORIENTATIONS
        HUMAN_ORIENTATION = 90
        MACHINE_ORIENTATION = 0

        # BOUNDS
        BOUND_HUMAN_X = np.array([0, 0])
        BOUND_HUMAN_Y = None

        BOUND_MACHINE_X = None
        BOUND_MACHINE_Y = np.array([0, 0])

        # LOSS WEIGHT
        Y_CLEARANCE_WEIGHT = 0.3



