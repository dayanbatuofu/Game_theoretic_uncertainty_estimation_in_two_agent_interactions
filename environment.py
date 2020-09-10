"""
Environment class
"""
import numpy as np
from models import constants as C

class Environment:

    def __init__(self, env_name):

        self.name = env_name

        # TODO: unify units for all parameters


        if self.name == 'intersection':
            self.car_width = 0.66
            self.car_length = 1.33
            self.vehicle_max_speed = 0.05
            self.initial_speed = 0.025

            self.n_agents = 2

            # BOUNDS: [agent1, agent2, ...], agent: [bounds along x, bounds along y], bounds: [min, max]
            self.bounds = [[[-0.4, 0.4], None], [None, [-0.4, 0.4]]]

            # first car moves bottom up, second car right to left
            self.car_par = [{"sprite": "grey_car_sized.png",
                             "initial_state": [[0, -2.0, 0, 0.1]],  # pos_x, pos_y, vel_x, vel_y
                             "desired_state": [0, 0.4],  # pos_x, pos_y
                             "initial_action": [0.],  # accel  #TODO: add steering angle
                             "par": 1,  # aggressiveness
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[2.0, 0, -0.1, 0]],
                             "desired_state": [-0.4, 0],
                             "initial_action": [0.],
                             "par": 1,
                             "orientation": -90.},
                            ]

        elif self.name == 'trained_intersection':
            #TODO: modify initial state to match with trained model
            self.n_agents = 2

            self.car_width = 2  # m
            self.car_length = 4  # m
            # VEHICLE_MAX_SPEED = 40.2  # m/s
            # INITIAL_SPEED = 13.4  # m/s
            # VEHICLE_MIN_SPEED = 0.0  # m/s
            # MAX_ACCELERATION = 8  # m/s^2
            # MAX_DECELERATION = -8  # m/s^2
            self.vehicle_max_speed = 40.2
            self.initial_speed = 13.4

            intersection = C.CONSTANTS.Intersection
            # # BOUNDS: [agent1, agent2, ...], agent: [bounds along x, bounds along y], bounds: [min, max]
            # boundx = intersection.SCREEN_WIDTH
            # boundy = intersection.SCREEN_HEIGHT
            #self.bounds = [[[-boundx, boundx], None], [None, [-boundy, boundy]]]
            self.bounds = [[[-self.car_width/2, self.car_width/2], None], [None, [-self.car_width/2, self.car_width/2]]]
            # first car moves bottom up, second car right to left
            "randomly pick initial states:"
            sy_M = np.random.uniform(intersection.CAR_1.INITIAL_STATE[0] * 0.5,
                                     intersection.CAR_1.INITIAL_STATE[0] * 1.0)
            max_speed = np.sqrt((sy_M - 1 - C.CONSTANTS.CAR_LENGTH * 0.5) * 2.
                                * abs(intersection.MAX_DECELERATION))
            vy_M = np.random.uniform(max_speed * 0.1, max_speed * 0.5)

            sx_H = np.random.uniform(intersection.CAR_2.INITIAL_STATE[0] * 0.5,
                                     intersection.CAR_2.INITIAL_STATE[0] * 1.0)
            max_speed = np.sqrt((sx_H - 1 - C.CONSTANTS.CAR_LENGTH * 0.5) * 2.
                                * abs(intersection.MAX_DECELERATION))
            vx_H = np.random.uniform(max_speed * 0.1, max_speed * 0.5)
            print("initial vel:", vy_M, -vx_H, "initial pos:", -sy_M, sx_H)
            self.car_par = [{"sprite": "grey_car_sized.png",
                             "initial_state": [[0, -sy_M, 0, vy_M]],  # pos_x, pos_y, vel_x, vel_y, positive vel
                             "desired_state": [0, 0.4],  # pos_x, pos_y
                             "initial_action": [0.],  # accel  #TODO: add steering angle
                             "par": 1,  # aggressiveness
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[sx_H, 0, -vx_H, 0]], #should be having negative velocity
                             "desired_state": [-0.4, 0],
                             "initial_action": [0.],
                             "par": 1,
                             "orientation": -90.},
                            ]

        elif self.name == 'merger':
            #TODO: modify initial state to match with trained model
            self.n_agents = 2

            self.car_width = 2  # m
            self.car_length = 4  # m
            # VEHICLE_MAX_SPEED = 40.2  # m/s
            # INITIAL_SPEED = 13.4  # m/s
            # VEHICLE_MIN_SPEED = 0.0  # m/s
            # MAX_ACCELERATION = 8  # m/s^2
            # MAX_DECELERATION = -8  # m/s^2
            self.vehicle_max_speed = 40.2
            self.initial_speed = 13.4

            merger = C.CONSTANTS.Merger
            # # BOUNDS: [agent1, agent2, ...], agent: [bounds along x, bounds along y], bounds: [min, max]
            # boundx = intersection.SCREEN_WIDTH
            # boundy = intersection.SCREEN_HEIGHT
            #self.bounds = [[[-boundx, boundx], None], [None, [-boundy, boundy]]]
            self.bounds = [[[-self.car_width/2, self.car_width/2], None], [None, [-self.car_width/2, self.car_width/2]]]
            # first car moves bottom up, second car right to left
            "randomly pick initial states:"
            sy_M = np.random.uniform(merger.CAR_1.INITIAL_STATE[0] * 0.5,
                                     merger.CAR_1.INITIAL_STATE[0] * 1.0)
            max_speed = np.sqrt((sy_M - 1 - C.CONSTANTS.CAR_LENGTH * 0.5) * 2.
                                * abs(merger.MAX_DECELERATION))
            vy_M = np.random.uniform(max_speed * 0.1, max_speed * 0.5)

            sy_H = np.random.uniform(merger.CAR_2.INITIAL_STATE[0] * 0.5,
                                     merger.CAR_2.INITIAL_STATE[0] * 1.0)
            max_speed = np.sqrt((sy_H - 1 - C.CONSTANTS.CAR_LENGTH * 0.5) * 2.
                                * abs(merger.MAX_DECELERATION))
            vy_H = np.random.uniform(max_speed * 0.1, max_speed * 0.5)
            print("merging ", "initial vel:", vy_M, -vy_H, "initial pos:", -sy_M, -sy_H)
            self.car_par = [{"sprite": "grey_car_sized.png",
                             "initial_state": [[0, -sy_M, 0, vy_M]],  # pos_x, pos_y, vel_x, vel_y, positive vel
                             "desired_state": [0, 0.4],  # pos_x, pos_y
                             "initial_action": [0.],  # accel  #TODO: add steering angle
                             "par": 1,  # aggressiveness
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[self.car_width+0.2, -sy_H, 0, vy_H]], #should be having negative velocity
                             "desired_state": [0, 0.4],
                             "initial_action": [0.],
                             "par": 1,
                             "orientation": 0.},
                            ]
        elif self.name == 'single_agent':
            # TODO: implement Fridovich-Keil et al. "Confidence-aware motion prediction for real-time collision avoidance"
            self.n_agents = 2  # one agent is observer
            self.bounds = [[[-0.4, 0.4], None], [None, [-0.4, 0.4]]]

            # first car moves bottom up, second car right to left
            self.car_par = [{"sprite": "grey_car_sized.png",
                             "initial_state": [[0, -2.0, 0, 0.1]],  # pos_x, pos_y, vel_x, vel_y
                             "desired_state": [0, 0.4],  # pos_x, pos_y
                             "initial_action": [0.],  # acc  #TODO: add steering angle
                             "par": 1,  # aggressiveness
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[2.0, 0, 0, 0]],
                             "desired_state": [-0.4, 0],
                             "initial_action": [0.],
                             "par": 1,
                             "orientation": -90.},
                            ]

            pass

        elif self.name == 'lane_change':
            pass
        elif self.name == 'random':
            #TODO: add randomized initial conditions
            pass
        else:

            pass





