import numpy as np


class CarParameters:

    def __init__(self, SPRITE, INITIAL_STATE, aggressiveness, gracefulness, ORIENTATION, MAX_SPEED):

        self.SPRITE = SPRITE
        self.INITIAL_STATE = INITIAL_STATE
        self.aggressiveness = aggressiveness
        self.gracefulness = gracefulness
        self.ORIENTATION = ORIENTATION
        self.MAX_SPEED = MAX_SPEED


class CONSTANTS:

    # DISPLAY
    DRAW = True
    ASSET_LOCATION = "../assets/"
    FPS = 1
    AXES_SHOW = 0.5
    COORDINATE_SCALE = 20

    CAR_WIDTH = 2.0  # unit: m
    CAR_LENGTH = 4.0  # unit: m

    ZOOM = 0.7

    # POSITION BOUNDS
    Y_MINIMUM = 0
    Y_MAXIMUM = 1

    FRAME_TIME = 0.05  # each time step takes 0.05 sec. OBSOLETE

    # OPTIMIZATION
    ACTION_TIMESTEPS = 100  # 5 seconds
    ACTION_TURNANGLE = 0  # degrees #TODO: not sure why slsqp does not work with larger angles
    ACTION_NUMPOINTS = 100

    TRACK_BACK = 1

    THETA_LIMITER_X = 0.01
    THETA_LIMITER_Y = 0.01

    LOSS_THRESHOLD = 0.01

    LEARNING_RATE = 0.5
    INTENT_LIMIT = 1000. #TODO: this is the max alpha, need to explain what this means

    EXPTHETA = 1
    EXPCOLLISION = 3

    np.random.seed(1)
    THETA_SET = np.array([1, 1e3]) #TODO: CHANGE THETA_SET
    TRAJECTORY_SET = np.array([3., 2., 1., 0., -1., -2.])

    COURTESY_CONSTANT = 10.

    # class LaneChanging:
    #
    #     # DISPLAY
    #     SCREEN_WIDTH = 5
    #     SCREEN_HEIGHT = 5
    #
    #     # BOUNDS
    #     BOUND_HUMAN_X = None
    #     BOUND_HUMAN_Y = np.array([-0.5, 1.5])
    #
    #     BOUND_MACHINE_X = None
    #     BOUND_MACHINE_Y = np.array([-0.5, 1.5])
    #
    #     # COLLISION BOXES
    #     COLLISION_BOXES = np.array([(-np.inf, np.inf, -0.5, 0.5), (-np.inf, np.inf, 0.5, 1.5)])  # List of separate collision boxes (-x, x, -y, y)
    #
    #     VEHICLE_MAX_SPEED = 0.1
    #
    #     # Left Car
    #     CAR_1 = CarParameters(SPRITE="grey_car_sized.png",
    #                           INITIAL_POSITION=np.array([-1, -1]),
    #                           DESIRED_POSITION=np.array([3, 1]),  # Maybe change to be further down the road?
    #                           BOUND_X=None,
    #                           BOUND_Y=np.array([-0.5, 1.5]),
    #                           INTENT=1,
    #                           COMMON_THETA=np.array([5., 0]),
    #                           ORIENTATION=0,
    #                           ABILITY=0.02,
    #                           ABILITY_O=0.02)
    #
    #     # Right Car
    #     CAR_2 = CarParameters(SPRITE="white_car_sized.png",
    #                           INITIAL_POSITION=np.array([0, 0]),
    #                           DESIRED_POSITION=np.array([3, 0]),  # Maybe change to be further down the road?
    #                           BOUND_X=None,
    #                           BOUND_Y=np.array([-0.5, 1.5]),
    #                           INTENT=1e3,
    #                           COMMON_THETA=np.array([5., 0]),
    #                           ORIENTATION=0,
    #                           ABILITY=0.02,
    #                           ABILITY_O=0.02)

    class Intersection:

        # DISPLAY
        SCREEN_WIDTH = 50  # meter
        SCREEN_HEIGHT = 50  # meter

        # BOUNDS
        BOUND_HUMAN_X = np.array([-1., 1.])
        BOUND_HUMAN_Y = None

        BOUND_MACHINE_X = None
        BOUND_MACHINE_Y = np.array([-1., 1.])

        # COLLISION BOXES
        COLLISION_BOXES = np.array([(-1., 1., -1., 1.)])  # List of separate collision boxes (-x, x, -y, y)

        VEHICLE_MAX_SPEED = 40.2  # m/s
        INITIAL_SPEED = 13.4  # m/s
        VEHICLE_MIN_SPEED = 0.0  # m/s
        MAX_ACCELERATION = 8  # m/s^2
        MAX_DECELERATION = -8  # m/s^2

        MAX_TIME = 7.0  # seconds
        MIN_TIME_INTERVAL = 0.05  # seconds, this is the min time interval to be used by RL for this environment

        # Left Car
        CAR_1 = CarParameters(SPRITE="grey_car_sized.png",
                              INITIAL_STATE=[20., INITIAL_SPEED],  # initial distance in m and speed in m/s
                              aggressiveness=1,
                              gracefulness=0,
                              ORIENTATION=0,  # moving from bottom to top
                              MAX_SPEED=[VEHICLE_MAX_SPEED, VEHICLE_MIN_SPEED])

        # Right Car
        CAR_2 = CarParameters(SPRITE="white_car_sized.png",
                              INITIAL_STATE=[20., INITIAL_SPEED],
                              aggressiveness=1,
                              gracefulness=0,
                              ORIENTATION=-90,  # moving from right to left
                              MAX_SPEED=[VEHICLE_MAX_SPEED, VEHICLE_MIN_SPEED])

class MATRICES:

    LOWER_TRIANGULAR_MATRIX = np.zeros((CONSTANTS.ACTION_NUMPOINTS, CONSTANTS.ACTION_NUMPOINTS))
    LOWER_TRIANGULAR_MATRIX[np.tril_indices(CONSTANTS.ACTION_NUMPOINTS, 0)] = 1
    #
    # LOWER_TRIANGULAR_SMALL = np.zeros((CONSTANTS.T_PAST, CONSTANTS.T_PAST))
    # LOWER_TRIANGULAR_SMALL[np.tril_indices(CONSTANTS.T_PAST, 0)] = 1




