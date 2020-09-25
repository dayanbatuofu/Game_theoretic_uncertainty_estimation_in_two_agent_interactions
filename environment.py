"""
Environment class
"""
import numpy as np
from models import constants as C
#from savi_simulation import Simulation as sim
import savi_simulation as sim
from models.rainbow.arguments import get_args
import models.rainbow.arguments
from models.rainbow.set_nfsp_models import get_models
import torch as t

class Environment:

    def __init__(self, env_name):

        self.name = env_name
        self.sim = sim
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
                             "par": 1,  # aggressiveness: check sim.theta_list
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[2.0, 0, -0.1, 0]],
                             "desired_state": [-0.4, 0],
                             "initial_action": [0.],
                             "par": 1,  # aggressiveness: check sim.theta_list
                             "orientation": -90.},
                            ]

        elif self.name == 'trained_intersection':
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
            # theta_list = [1, 1000]
            # lambda_list = [0.001, 0.005, 0.01, 0.05]
            self.car_par = [{"sprite": "grey_car_sized.png",
                             "initial_state": [[0, -sy_M, 0, vy_M]],  # pos_x, pos_y, vel_x, vel_y, positive vel
                             "desired_state": [0, 0.4],  # pos_x, pos_y
                             "initial_action": [0.],  # accel  #TODO: add steering angle
                             "par": (1, 0.001),  # aggressiveness: check sim.theta_list
                             "belief": (1000, 0.001),  # belief of other's params (beta: (theta, lambda))
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[sx_H, 0, -vx_H, 0]],  # should be having negative velocity
                             "desired_state": [-0.4, 0],
                             "initial_action": [0.],
                             "par": (1, 0.001),  # aggressiveness: check sim.theta_list
                             "belief": (1000, 0.001),  # belief of other's params (beta: (theta, lambda))
                             "orientation": -90.},
                            ]

            # TODO: choose action base on decision type and intent
            p1_state = self.car_par[0]["initial_state"][0]
            p2_state = self.car_par[1]["initial_state"][0]
            p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
            p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))
            pi_state = [p1_state, p2_state]
            # (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2)
            q_sets = get_models()[0]
            args = get_args()
            # TODO: define these somewhere else for general uses
            lambda_list = [0.001, 0.005, 0.01, 0.05]
            action_set = [-8, -4, 0, 4, 8]
            for i in range(len(self.car_par)):
                # TODO: assume other is NA?
                if self.car_par[i]["par"] == 0:  # NA
                    qi = q_sets[0]
                else:
                    qi = q_sets[3]
                q_vals_i = qi.forward(t.FloatTensor(pi_state[i]).to(t.device("cpu")))
                p_a = self.action_prob(q_vals_i, lambda_list[-1])
                action_i = action_set[np.argmax(p_a)]
                self.car_par[i]["initial_action"] = [action_i]
            print("initial params: ", self.car_par)

        elif self.name == 'bvp_intersection':
            pass
        
        elif self.name == 'merger':
            # TODO: modify initial state to match with trained model
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
                             "initial_state": [[0, -sy_M, 0,0, vy_M]],  # pos_x, pos_y, theta, delta, positive vel
                             "desired_state": [0, 0.4],  # pos_x, pos_y
                             "initial_action": [[0., 0.]],  # accel  #TODO: add steering angle
                             "par": 1,  # aggressiveness
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[self.car_width+0.2, -sy_H, 0,0, vy_H]], #should be having negative velocity
                             "desired_state": [0, 0.4],
                             "initial_action": [[0., 0.]],
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

    def action_prob(self, q_vals, _lambda):
        """
        Equation 1
        Noisy-rational model
        calculates probability distribution of action given hardmax Q values
        Uses:
        1. Softmax algorithm
        2. Q-value given state and theta(intent)
        3. lambda: "rationality coefficient"
        => P(uH|xH;beta,theta) = exp(beta*QH(xH,uH;theta))/sum_u_tilde[exp(beta*QH(xH,u_tilde;theta))]
        :return: Normalized probability distributions of available actions at a given state and lambda
        """
        # q_vals = q_values(state_h, state_m, intent=intent)
        exp_Q = []
        "Q*lambda"
        q_vals = q_vals.detach().numpy()  # detaching tensor
        Q = [q * _lambda for q in q_vals]
        "Q*lambda/(sum(Q*lambda))"

        for q in Q:
            exp_Q.append(np.exp(q))

        "normalizing"
        exp_Q /= sum(exp_Q)
        # print("exp_Q normalized:", exp_Q)
        return exp_Q




