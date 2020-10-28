"""
Environment class
"""
import numpy as np
from models import constants as C
import savi_simulation as sim
from models.rainbow.arguments import get_args
from HJI_Vehicle.utilities.neural_networks import HJB_network_t0 as HJB_network
from HJI_Vehicle.NN_output import get_Q_value
import models.rainbow.arguments
from models.rainbow.set_nfsp_models import get_models
from HJI_Vehicle.NN_output import get_Q_value
import dynamics
import torch as t
import random
from scipy.special import logsumexp


class Environment:
    # add: intent, noise, intent belief, noise belief
    def __init__(self, env_name, sim_par, sim_dt, agent_intent, agent_noise, agent_intent_belief, agent_noise_belief):

        self.name = env_name
        self.sim = sim
        self.sim_par = sim_par
        self.dt = sim_dt
        self.agent_intent = []
        self.agent_noise = []
        self.agent_intent_belief = []
        self.agent_noise_belief = []

        for i in range(len(agent_intent)):
            'check agent theta'
            if agent_intent[i] == 'NA':
                self.agent_intent.append(sim_par['theta'][0])
            elif agent_intent[i] == 'A':
                self.agent_intent.append(sim_par['theta'][1])
            'check agent theta belief'
            if agent_intent_belief[i] == 'NA':
                self.agent_intent_belief.append(sim_par['theta'][0])
            elif agent_intent_belief[i] == 'A':
                self.agent_intent_belief.append(sim_par['theta'][1])
            'check agent lambda'
            if agent_noise[i] == 'N':
                self.agent_noise.append(sim_par['lambda'][0])
            elif agent_noise[i] == 'NN':
                self.agent_noise.append(sim_par['lambda'][1])
            'check agent lambda belief'
            if agent_noise_belief[i] == 'N':
                self.agent_noise_belief.append(sim_par['lambda'][0])
            elif agent_noise_belief[i] == 'NN':
                self.agent_noise_belief.append(sim_par['lambda'][1])

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
                             "initial_action": [0.],  # accel
                             "par": 1,  # aggressiveness: check sim.theta_list
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[2.0, 0, -0.1, 0]],
                             "desired_state": [-0.4, 0],
                             "initial_action": [0.],
                             "par": 1,  # aggressiveness: check sim.theta_list
                             "orientation": -90.},
                            ]

        elif self.name == 'trained_intersection':  # NFSP Q function is used in this env
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
            # self.bounds = [[[-boundx, boundx], None], [None, [-boundy, boundy]]]
            self.bounds = [[[-self.car_width/2, self.car_width/2], None], [None, [-self.car_width/2, self.car_width/2]]]
            # first car moves bottom up, second car right to left
            "randomly pick initial states:"
            sy_H = np.random.uniform(intersection.CAR_1.INITIAL_STATE[0] * 0.5,
                                     intersection.CAR_1.INITIAL_STATE[0] * 1.0)
            max_speed = np.sqrt((sy_H - 1 - C.CONSTANTS.CAR_LENGTH * 0.5) * 2.
                                * abs(intersection.MAX_DECELERATION))
            vy_H = np.random.uniform(max_speed * 0.1, max_speed * 0.5)

            sx_M = np.random.uniform(intersection.CAR_2.INITIAL_STATE[0] * 0.5,
                                     intersection.CAR_2.INITIAL_STATE[0] * 1.0)
            max_speed = np.sqrt((sx_M - 1 - C.CONSTANTS.CAR_LENGTH * 0.5) * 2.
                                * abs(intersection.MAX_DECELERATION))
            vx_M = np.random.uniform(max_speed * 0.1, max_speed * 0.5)
            # theta_list = [1, 1000]
            # lambda_list = [0.001, 0.005, 0.01, 0.05]
            self.car_par = [{"sprite": "grey_car_sized.png",
                             "initial_state": [[0, -sy_H, 0, vy_H]],  # pos_x, pos_y, vel_x, vel_y, positive vel
                             "desired_state": [0, 0.4],  # pos_x, pos_y
                             "initial_action": [0.],  # accel
                             "par": (self.agent_intent[0], self.agent_noise[0]),  # aggressiveness: check sim.theta_list
                             "belief": (self.agent_intent_belief[0], self.agent_noise_belief[0]),  # belief of other's params (beta: (theta, lambda))
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[sx_M, 0, -vx_M, 0]],  # should be having negative velocity
                             "desired_state": [-0.4, 0],
                             "initial_action": [0.],
                             "par": (self.agent_intent[1], self.agent_noise[1]),  # aggressiveness: check sim.theta_list
                             "belief": (self.agent_intent_belief[1], self.agent_noise_belief[1]),  # belief of other's params (beta: (theta, lambda))
                             "orientation": -90.},
                            ]

            "choose action base on decision type and intent"
            p1_state = self.car_par[0]["initial_state"][0]
            p2_state = self.car_par[1]["initial_state"][0]
            p1_state_nn = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
            p2_state_nn = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))
            pi_state = [p1_state_nn, p2_state_nn]
            # (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2)
            q_sets = get_models()[0]
            args = get_args()

            lambda_list = self.sim_par["lambda"]
            action_set = self.sim_par["action_set"]
            theta_list = self.sim_par["theta"]
            for i in range(len(self.car_par)):
                if self.car_par[i]["par"][0] == theta_list[0]:  # NA_NA
                    if self.car_par[i]["belief"][0] == theta_list[0]:  # TODO: check if i-1 works
                        qi = q_sets[0]
                    elif self.car_par[i]["belief"][0] == theta_list[1]:  # NA_A
                        qi = q_sets[2]
                    else:
                        print("WARNING: NO CORRESPONDING THETA FOUND")
                elif self.car_par[i]["par"][0] == theta_list[1]:
                    if self.car_par[i]["belief"][0] == theta_list[0]:  # A_NA
                        qi = q_sets[3]
                    elif self.car_par[i]["belief"][0] == theta_list[1]:  # A_A
                        qi = q_sets[4]  # use a_a
                    else:
                        print("WARNING: NO CORRESPONDING THETA FOUND")
                else:
                    print("WARNING: NO CORRESPONDING THETA FOUND")
                q_vals_i = qi.forward(t.FloatTensor(pi_state[i]).to(t.device("cpu")))
                p_a = self.action_prob(q_vals_i, self.car_par[i]["par"][1])
                action_i = random.choices(action_set, weights=p_a, k=1)  # draw action using the distribution
                self.car_par[i]["initial_action"] = [action_i[0]]
            print("initial params: ", self.car_par)

        elif self.name == 'bvp_intersection':
            # TODO: the below is not done yet, need to change the variables
            self.n_agents = 2
            self.car_width = 1.5  # m
            self.car_length = 3  # m

            # # BOUNDS: [agent1, agent2, ...], agent: [bounds along x, bounds along y], bounds: [min, max]
            # self.bounds = [[[-boundx, boundx], None], [None, [-boundy, boundy]]]
            self.bounds = [[[-self.car_width / 2, self.car_width / 2], None],
                           [None, [-self.car_width / 2, self.car_width / 2]]]
            # first car (H) moves bottom up, second car (M) right to left

            "randomly pick initial states:"
            # initial state range: x: 15 to 20, v: 18 to 25
            # u range: [-5 10]
            # sy_H = np.random.uniform(15, 20)
            # vy_H = np.random.uniform(18, 25)
            # sx_M = np.random.uniform(15, 20)
            # vx_M = np.random.uniform(18, 25)
            sy_H = 15  # P1
            vy_H = 18
            sx_M = 16  # P2
            vx_M = 18

            assert 20 >= sy_H >= 15
            assert 20 >= sx_M >= 15
            self.car_par = [{"sprite": "grey_car_sized.png",
                             "initial_state": [[0, sy_H, 0, vy_H]],  # pos_x, pos_y, vel_x, vel_y
                             "desired_state": [0, 0.4],  # pos_x, pos_y
                             "initial_action": [0.],  # accel
                             "par": [self.agent_intent[0], self.agent_noise[0]],  # DON'T CHANGE; par is defined in main
                             "belief": [self.agent_intent_belief[0], self.agent_noise_belief[0]],
                             # belief of other's params (beta: (theta, lambda))
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[sx_M, 0, vx_M, 0]],
                             "desired_state": [-0.4, 0],
                             "initial_action": [0.],
                             "par": [self.agent_intent[1], self.agent_noise[1]],  # aggressiveness: check main
                             "belief": [self.agent_intent_belief[1], self.agent_noise_belief[1]],
                             # belief of other's params (beta: (theta, lambda))
                             "orientation": -90.},
                            ]

            "choose action base on decision type and intent"  # TODO: check this
            action_set = self.sim_par["action_set"]
            "METHOD 1: Get the whole p_action table using true param of self and belief of other's param"
            p1_state = self.car_par[0]["initial_state"][0]
            p2_state = self.car_par[1]["initial_state"][0]
            true_beta_h = self.car_par[0]["par"]
            true_beta_m = self.car_par[1]["par"]
            belief_beta_h = self.car_par[1]["belief"]
            belief_beta_m = self.car_par[0]["belief"]
            p_action1, p_action2_n = self.bvp_action_prob(p1_state, p2_state, true_beta_h, belief_beta_m)
            p_action1_n, p_action2 = self.bvp_action_prob(p1_state, p2_state, belief_beta_h, true_beta_m)

            actions = []
            for i, p_a in enumerate([p_action1, p_action2]):
                action_id = np.unravel_index(p_a.argmax(), p_a.shape)
                self.car_par[i]["initial_action"] = [action_set[action_id[i]]]

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
            # TODO: add randomized initial conditions
            pass
        else:

            pass

    # TODO: implement this in another file for general usage
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

    def bvp_action_prob(self, state_h, state_m, beta_h, beta_m):
        """
        Equation 1
        calculate action prob for both agents
        :param state_h:
        :param state_m:
        :param _lambda:
        :param theta:
        :return: [p_action_H, p_action_M], where p_action = [p_a1, ..., p_a5]
        """

        theta_h, lambda_h = beta_h
        theta_m, lambda_m = beta_m
        action_set = self.sim_par["action_set"]

        _lambda = [lambda_h, lambda_m]

        "Need state for agent H: xH, vH, xM, vM"
        p1_state_nn = np.array([[state_h[1]], [state_h[3]], [state_m[0]], [state_m[2]]])
        p2_state_nn = np.array([[state_m[0]], [state_m[2]], [state_h[1]], [state_h[3]]])

        # TODO: math needs to be checked
        _p_action_1 = np.zeros((len(action_set), len(action_set)))
        _p_action_2 = np.zeros((len(action_set), len(action_set)))
        time = np.array([[0]])
        dt = self.dt
        for i, p_a_h in enumerate(_p_action_1):
            for j, p_a_m in enumerate(_p_action_1[i]):
                new_p2_s = dynamics.bvp_dynamics_1d(state_m, action_set[j], dt)
                new_p1_s = dynamics.bvp_dynamics_1d(state_h, action_set[i], dt)
                if (theta_h, theta_m) == (1, 5):  # Flip A_AN to NA_A
                    new_p2_state_nn = np.array([[new_p2_s[0]], [new_p2_s[2]], [new_p1_s[1]], [new_p1_s[3]]])
                    q2, q1 = get_Q_value(new_p2_state_nn, time, np.array([[action_set[j]], [action_set[i]]]),
                                         (theta_m, theta_h))  # NA_A
                else:  # for A_A, NA_NA, NA_A
                    new_p1_state_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s[0]], [new_p2_s[2]]])
                    q1, q2 = get_Q_value(new_p1_state_nn, time, np.array([[action_set[i]], [action_set[j]]]),
                                         (theta_h, theta_m))
                lamb_Q1 = q1 * lambda_h
                _p_action_1[i][j] = lamb_Q1
                lamb_Q2 = q2 * lambda_m
                _p_action_2[i][j] = lamb_Q2

        "using logsumexp to prevent nan"
        Q1_logsumexp = logsumexp(_p_action_1)
        Q2_logsumexp = logsumexp(_p_action_2)
        "normalizing"  # TODO: check if this works
        _p_action_1 -= Q1_logsumexp
        _p_action_2 -= Q2_logsumexp
        _p_action_1 = np.exp(_p_action_1)
        _p_action_2 = np.exp(_p_action_2)

        print('p1 state:', p1_state_nn)
        print("action prob 1 from bvp:", _p_action_1)
        print("action prob 2 from bvp:", _p_action_2)
        assert round(np.sum(_p_action_1)) == 1
        assert round(np.sum(_p_action_2)) == 1

        return [_p_action_1, _p_action_2]  # [exp_Q_h, exp_Q_m]



