import numpy as np

class CarParameters:

    def __init__(self, SPRITE, INITIAL_POSITION, INITIAL_SPEED, BOUND_X, BOUND_Y, INTENT, COMMON_THETA, ORIENTATION, ABILITY, ABILITY_O):

        self.SPRITE = SPRITE
        self.INITIAL_POSITION = INITIAL_POSITION
        self.INITIAL_SPEED = INITIAL_SPEED
        self.INTENT = INTENT
        self.COMMON_THETA = COMMON_THETA
        self.ORIENTATION = ORIENTATION
        self.BOUND_X = BOUND_X
        self.BOUND_Y = BOUND_Y
        self.ABILITY = ABILITY
        self.ABILITY_O = ABILITY_O

class CONSTANTS:

    # DISPLAY
    DRAW = True
    ASSET_LOCATION = "assets/"
    FPS = 1
    AXES_SHOW = 0.5
    COORDINATE_SCALE = 150

    CAR_WIDTH = 0.66
    CAR_LENGTH = 1.33

    ZOOM = 0.6


    # POSITION BOUNDS
    Y_MINIMUM = 0
    Y_MAXIMUM = 1


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
    VEHICLE_MAX_SPEED = 0.05
    EXPTHETA = 1
    EXPCOLLISION = 3

    np.random.seed(1)
    THETA_SET = np.array([1, 1e3]) #TODO: CHANGE THETA_SET
    TRAJECTORY_SET = np.array([3., 2., 1., 0., -1., -2.])

    # class PARAMETERSET_1:
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
    #                           INTENT=1,
    #                           COMMON_THETA=np.array([5., 0]),
    #                           ORIENTATION=0,
    #                           ABILITY=0.02,
    #                           ABILITY_O=0.02)
    #
    # class PARAMETERSET_2:
    #
    #     # DISPLAY
    #     SCREEN_WIDTH = 5
    #     SCREEN_HEIGHT = 5
    #
    #     # BOUNDS
    #     BOUND_HUMAN_X = np.array([-0.4, 0.4])
    #     BOUND_HUMAN_Y = None
    #
    #     BOUND_MACHINE_X = None
    #     BOUND_MACHINE_Y = np.array([-0.4, 0.4])
    #
    #     # COLLISION BOXES
    #     COLLISION_BOXES = np.array([(-0.4, 0.4, -0.4, 0.4)])  # List of separate collision boxes (-x, x, -y, y)
    #
    #     VEHICLE_MAX_SPEED = 0.05
    #     INITIAL_SPEED = 0.025
    #
    #     # Left Car
    #     CAR_1 = CarParameters(SPRITE="grey_car_sized.png",
    #                           INITIAL_POSITION=np.array([-2, 0]),
    #                           DESIRED_POSITION=np.array([0.4, 0]),
    #                           BOUND_X=None,
    #                           BOUND_Y=np.array([-0.4, 0.4]),
    #                           INTENT=1,
    #                           COMMON_THETA=np.array([0., 0]),
    #                           ORIENTATION=0,
    #                           ABILITY=0.002,
    #                           ABILITY_O=0.002)
    #
    #     # Right Car
    #     CAR_2 = CarParameters(SPRITE="white_car_sized.png",
    #                           INITIAL_POSITION=np.array([0, 2.0]),
    #                           DESIRED_POSITION=np.array([0, -0.4]),
    #                           BOUND_X=np.array([-0.4, 0.4]),
    #                           BOUND_Y=None,
    #                           INTENT=1e3,
    #                           COMMON_THETA=np.array([0., -90]),
    #                           ORIENTATION=-90,
    #                           ABILITY=0.002,
    #                           ABILITY_O=0.002)

    class PARAMETERSET_3:
        # DISPLAY
        SCREEN_WIDTH = 5
        SCREEN_HEIGHT = 5

        # BOUNDS
        BOUND_HUMAN_X = np.array([-0.4, 0.4])
        BOUND_HUMAN_Y = None

        BOUND_MACHINE_X = None
        BOUND_MACHINE_Y = np.array([-0.4, 0.4])

        # COLLISION BOXES
        COLLISION_BOXES = np.array([(-0.4, 0.4, -0.4, 0.4)])  # List of separate collision boxes (-x, x, -y, y)

        VEHICLE_MAX_SPEED = 0.03
        INITIAL_SPEED = 0.025

        # Car 1
        CAR_1 = CarParameters(SPRITE="grey_car_sized.png",
                              INITIAL_POSITION=np.array([-1.46, 0.4]),
                              INITIAL_SPEED=0.025,
                              BOUND_X=None,
                              BOUND_Y=np.array([-0.4, 0.4]),
                              INTENT=1,
                              COMMON_THETA=np.array([0., 0]),
                              ORIENTATION=0,
                              ABILITY=0.001,
                              ABILITY_O=0.002)
        # Car 2
        CAR_2 = CarParameters(SPRITE="white_car_sized.png",
                              INITIAL_POSITION=np.array([4, -1.2]),
                              INITIAL_SPEED=0.025,
                              BOUND_X=np.array([-0.4, 0.4]),
                              BOUND_Y=None,
                              INTENT=1e3,
                              COMMON_THETA=np.array([0., -90]),
                              ORIENTATION=180,
                              ABILITY=0.002,
                              ABILITY_O=0.002)
        # Car 3
        CAR_3 = CarParameters(SPRITE="white_car_sized.png",
                              INITIAL_POSITION=np.array([1.6, -0.4]),
                              INITIAL_SPEED=0,
                              BOUND_X=np.array([-0.4, 0.4]),
                              BOUND_Y=None,
                              INTENT=1e3,
                              COMMON_THETA=np.array([0., -90]),
                              ORIENTATION=180,
                              ABILITY=0.002,
                              ABILITY_O=0.002)


class MATRICES:

    LOWER_TRIANGULAR_MATRIX = np.zeros((CONSTANTS.ACTION_NUMPOINTS, CONSTANTS.ACTION_NUMPOINTS))
    LOWER_TRIANGULAR_MATRIX[np.tril_indices(CONSTANTS.ACTION_NUMPOINTS, 0)] = 1




