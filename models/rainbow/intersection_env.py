from __future__ import print_function

import random
import time
import numpy as np
import pygame as pg
import torch

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from autonomous_vehicle import AutonomousVehicle
from set_nfsp_models import get_models
from autograd_function import DynamicsMain
from sim_draw import Sim_Draw
from constants import CONSTANTS as C
import logging
logging.basicConfig(level=logging.WARN)

class IntersectionEnv:
    def __init__(self, control_style_ego, control_style_other, time_interval, max_time_steps):

        # Setup
        self.duration = 100
        self.parameters = C.Intersection  # Scenario parameters choice
        # Time handling
        self.clock = pg.time.Clock()
        self.fps = C.FPS
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 0
        self.car_num_display = 0
        self.time_interval = time_interval
        self.min_time_interval = C.Intersection.MIN_TIME_INTERVAL
        self.max_time_steps = max_time_steps
        self.inference_states = []
        self.inference_actions = []
        self.nfsp_models = get_models()[1]
        self.args = None
        self.ego_crossed_first = False

        self.state = []  # This is the state used in RL
        self.action = []  # collection of current actions from the agents
        self.collision = 0  # number of min time steps with collision

        self.ego_car = AutonomousVehicle(car_parameters=self.parameters.CAR_1,
                                         control_style=control_style_ego, who='M')  # autonomous car
        self.other_car = AutonomousVehicle(car_parameters=self.parameters.CAR_2,
                                           control_style=control_style_other, who='H')  # human car
        self.ego_car.other_car = self.other_car
        self.other_car.other_car = self.ego_car
        self.theta_set = [(1, 1), (1, 1), (1, 1000), (1000, 1), (1000, 1000), (1000, 1000)]
        self.likelihood_values = []
        self.likelihood1_values = []
        self.seed()
        self.renderer = Sim_Draw(self.parameters, C.ASSET_LOCATION)
        self.done = False
        self.paused = False
        self.isCollision = False
        self.intent_probs = []

    def loglikelihood(self, Q):
        obs_time_step = len(self.inference_actions)
        likelihood = 0
        for t in range(obs_time_step):
            state = self.inference_states[t]
            action = self.inference_actions[t]
            if torch.cuda.is_available():
                Q_vals = Q.forward(torch.FloatTensor(state).to(torch.device("cuda")))
            else:
                Q_vals = Q.forward(torch.FloatTensor(state).to(torch.device("cpu")))
            Q_v = Q_vals.tolist()
            # for action, ego chose
            L1 = Q_v[action]
            L2 = 0
            ctr = 0
            # todo: check if large positive values
            # for all actions
            for val in Q_v:
                if val > 1.0:
                    ctr += 1
                L2 += np.exp(val)
            # if ctr > 0:
            #     print('ctr: {}'.format(ctr))
            likelihood += -L1 + np.log(L2)
        # print('likelihood: {}'.format(likelihood))
        return likelihood

    def loglikelihood_fast(self, Q): #Q = one of the Q from q_set (ie Q_na_na)
        obs_time_step = len(self.inference_actions)

        if obs_time_step > 1:
            state = self.inference_states[-1]
            action = self.inference_actions[-1]
        else:
            state = self.inference_states[0]
            action = self.inference_actions[0]

        if torch.cuda.is_available():
            Q_vals = Q.forward(torch.FloatTensor(state).to(torch.device("cuda")))
        else:
            Q_vals = Q.forward(torch.FloatTensor(state).to(torch.device("cpu")))
        Q_v = Q_vals.tolist()
        # for action, ego chose
        L1 = Q_v[action]
        L2 = 0
        # todo: check if large positive values
        # for all actions
        for val in Q_v:
            L2 += np.exp(val)
        # if ctr > 0:
        #     print('ctr: {}'.format(ctr))
        likelihood = -L1 + np.log(L2)
        return likelihood


    def loglikelihood1_fast(self, Q):
        obs_time_step = len(self.inference_actions)

        if obs_time_step > 1:
            state = self.inference_states[-1]
            action = self.inference_actions[-1]
        else:
            state = self.inference_states[0]
            action = self.inference_actions[0]

        if torch.cuda.is_available():
            dist = Q.act_dist(torch.FloatTensor(state).to(torch.device("cuda")))
        else:
            dist = Q.act_dist(torch.FloatTensor(state).to(torch.device("cpu")))

        dist_a = dist.tolist()[0]
        # print('dist_a: {}'.format(dist_a))
        # print('dist_a_act: {}'.format(dist_a[action]))
        # print(np.log(dist_a[action]))
        return np.log(dist_a[action])

    # todo: speed-up
    def best_theta_set(self, Q_set):
        theta_set = []
        L_set = []

        for q in Q_set:
            L_set.append(self.loglikelihood(q))

        return L_set.index(min(L_set))

    def best_theta_set1_fast(self, Q_set):
        ctr = 0
        for q in Q_set:
            if len(self.likelihood1_values) == len(self.nfsp_models):
                self.likelihood1_values[ctr] += self.loglikelihood_fast(q)
            else:
                self.likelihood1_values.append(self.loglikelihood_fast(q))
            ctr += 1
        fast_idx = np.argmin(self.likelihood1_values)
        return fast_idx


    def best_theta_set_fast(self, Q_set):
        ctr = 0
        for q in Q_set:
            if len(self.likelihood_values) == len(self.nfsp_models):
                self.likelihood_values[ctr] += self.loglikelihood1_fast(q)
            else:
                self.likelihood_values.append(self.loglikelihood1_fast(q))
            ctr += 1

        fast_idx = np.argmax(self.likelihood_values)
        return fast_idx

    def seed(self, seed=None):
        random.seed(seed)
        return [seed]

    def step(self, action):
        if action['1'] == 0:
            a = self.parameters.MAX_DECELERATION
        elif action['1'] == 1:
            a = self.parameters.MAX_DECELERATION * 0.5
        elif action['1'] == 2:
            a = 0.0
        elif action['1'] == 3:
            a = self.parameters.MAX_ACCELERATION * 0.5
        else:
            a = self.parameters.MAX_ACCELERATION
        action_self = a
        if action['2'] == 0:
            a = self.parameters.MAX_DECELERATION
        elif action['2'] == 1:
            a = self.parameters.MAX_DECELERATION * 0.5
        elif action['2'] == 2:
            a = 0.0
        elif action['2'] == 3:
            a = self.parameters.MAX_ACCELERATION * 0.5
        else:
            a = self.parameters.MAX_ACCELERATION
        action_other = a  # the other car take one step

        self.ego_car.action = action_self
        self.other_car.action = action_other

        # show what action was taken
        # self.render()

        # get current states
        x_ego = x_ego_new = self.ego_car.state[0]
        x_other = x_other_new = self.other_car.state[0]
        v_ego = v_ego_new = self.ego_car.state[1]
        v_other = v_other_new = self.other_car.state[1]

        max_speed_ego = self.ego_car.car_parameters.MAX_SPEED[0]
        min_speed_ego = self.ego_car.car_parameters.MAX_SPEED[1]
        max_speed_other = self.other_car.car_parameters.MAX_SPEED[0]
        min_speed_other = self.other_car.car_parameters.MAX_SPEED[1]

        # update state and check for collision
        l = C.CAR_LENGTH
        w = C.CAR_WIDTH
        self.collision = 0
        for t in range(int(self.time_interval / self.min_time_interval + 1)):
            v_ego_new = max(min(max_speed_ego, action_self * t * self.min_time_interval + v_ego), min_speed_ego)
            v_other_new = max(min(max_speed_other, action_other * t * self.min_time_interval + v_other),
                              min_speed_other)
            x_ego_new = x_ego - t * 0.5 * (v_ego_new + v_ego) * self.min_time_interval
            x_other_new = x_other - t * 0.5 * (v_other_new + v_other) * self.min_time_interval
            collision_box1 = [[x_ego_new - 0.5 * l, -0.5 * w],
                              [x_ego_new - 0.5 * l, 0.5 * w],
                              [x_ego_new + 0.5 * l, 0.5 * w],
                              [x_ego_new + 0.5 * l, -0.5 * w]]
            collision_box2 = [[0.5 * w, x_other_new - 0.5 * l],
                              [-0.5 * w, x_other_new - 0.5 * l],
                              [-0.5 * w, x_other_new + 0.5 * l],
                              [0.5 * w, x_other_new + 0.5 * l]]
            c = 0
            polygon = Polygon(collision_box2)
            for p in collision_box1:
                point = Point(p[0], p[1])
                c += polygon.contains(point)
                if c > 0:
                    break
            self.collision += float(c > 0)  # number of times steps of collision

        if self.collision > 0:
            self.isCollision = True

        v_ego_new = max(min(max_speed_ego, action_self * self.time_interval + v_ego), min_speed_ego)
        x_ego -= 0.5 * (v_ego_new + v_ego) * self.time_interval  # start from positive distance to the center,
        # reduce to 0 when at the center
        v_ego = v_ego_new
        v_other_new = max(min(max_speed_other, action_other * self.time_interval + v_other), min_speed_other)
        x_other -= 0.5 * (v_other_new + v_other) * self.time_interval
        v_other = v_other_new

        if x_ego <= -0.5 * C.CAR_LENGTH - 1.:
            self.ego_car.isReached = True
        if x_other <= -0.5 * C.CAR_LENGTH - 1.:
            self.other_car.isReached = True

        if self.ego_car.isReached and not self.other_car.isReached:
            self.ego_crossed_first = True
        elif self.other_car.isReached and not self.ego_car.isReached:
            self.ego_crossed_first = False

        self.ego_car.state = [x_ego, v_ego]
        self.other_car.state = [x_other, v_other]

        if (x_ego <= -0.5 * C.CAR_LENGTH - 1. and x_other <= -0.5 * C.CAR_LENGTH - 1.) \
                or self.frame >= self.max_time_steps:  # road width = 2.0 m
            self.done = True

        self.state = (x_ego, v_ego, x_other, v_other)

        if self.args.acc_loss:
            loss_self = self.ego_car.self_loss_acc(self, action_self)
            loss_other = self.other_car.self_loss_acc(self, action_other)

        else:
            loss_self = self.ego_car.self_loss(self)
            loss_other = self.other_car.self_loss(self)

        loss = np.array([loss_self + self.ego_car.gracefulness * loss_other, loss_other,
                         loss_self + self.ego_car.gracefulness * loss_other, loss_other])

        # loss = np.array([(1 - self.ego_car.gracefulness) * loss_self + self.ego_car.gracefulness * loss_other, loss_other,
        #                  (1 - self.ego_car.gracefulness) * loss_self + self.ego_car.gracefulness * loss_other, loss_other])

        self.frame += 1
        return np.array(self.state), loss, self.done

    def step_inference(self, action, q_set):
        # print('state: {}'.format(self.state))
        self.inference_states.append([self.state[2], self.state[3], self.state[0], self.state[1]])
        self.inference_actions.append(action['2'])

        if action['1'] == 0:
            a = self.parameters.MAX_DECELERATION
        elif action['1'] == 1:
            a = self.parameters.MAX_DECELERATION * 0.5
        elif action['1'] == 2:
            a = 0.0
        elif action['1'] == 3:
            a = self.parameters.MAX_ACCELERATION * 0.5
        else:
            a = self.parameters.MAX_ACCELERATION
        action_self = a
        if action['2'] == 0:
            a = self.parameters.MAX_DECELERATION
        elif action['2'] == 1:
            a = self.parameters.MAX_DECELERATION * 0.5
        elif action['2'] == 2:
            a = 0.0
        elif action['2'] == 3:
            a = self.parameters.MAX_ACCELERATION * 0.5
        else:
            a = self.parameters.MAX_ACCELERATION
        action_other = a  # the other car take one step

        self.ego_car.action = action_self
        self.other_car.action = action_other

        # show what action was taken
        # self.render()

        # get current states
        x_ego = x_ego_new = self.ego_car.state[0]
        x_other = x_other_new = self.other_car.state[0]
        v_ego = v_ego_new = self.ego_car.state[1]
        v_other = v_other_new = self.other_car.state[1]

        max_speed_ego = self.ego_car.car_parameters.MAX_SPEED[0]
        min_speed_ego = self.ego_car.car_parameters.MAX_SPEED[1]
        max_speed_other = self.other_car.car_parameters.MAX_SPEED[0]
        min_speed_other = self.other_car.car_parameters.MAX_SPEED[1]

        # update state and check for collision
        l = C.CAR_LENGTH
        w = C.CAR_WIDTH
        self.collision = 0
        for t in range(int(self.time_interval / self.min_time_interval + 1)):
            v_ego_new = max(min(max_speed_ego, action_self * t * self.min_time_interval + v_ego), min_speed_ego)
            v_other_new = max(min(max_speed_other, action_other * t * self.min_time_interval + v_other),
                              min_speed_other)
            x_ego_new = x_ego - t * 0.5 * (v_ego_new + v_ego) * self.min_time_interval
            x_other_new = x_other - t * 0.5 * (v_other_new + v_other) * self.min_time_interval
            collision_box1 = [[x_ego_new - 0.5 * l, -0.5 * w],
                              [x_ego_new - 0.5 * l, 0.5 * w],
                              [x_ego_new + 0.5 * l, 0.5 * w],
                              [x_ego_new + 0.5 * l, -0.5 * w]]
            collision_box2 = [[0.5 * w, x_other_new - 0.5 * l],
                              [-0.5 * w, x_other_new - 0.5 * l],
                              [-0.5 * w, x_other_new + 0.5 * l],
                              [0.5 * w, x_other_new + 0.5 * l]]
            c = 0
            polygon = Polygon(collision_box2)
            for p in collision_box1:
                point = Point(p[0], p[1])
                c += polygon.contains(point)
                if c > 0:
                    break
            self.collision += float(c > 0)  # number of times steps of collision

        if self.collision > 0:
            self.isCollision = True

        v_ego_new = max(min(max_speed_ego, action_self * self.time_interval + v_ego), min_speed_ego)
        x_ego -= 0.5 * (v_ego_new + v_ego) * self.time_interval  # start from positive distance to the center,
        # reduce to 0 when at the center
        v_ego = v_ego_new
        v_other_new = max(min(max_speed_other, action_other * self.time_interval + v_other), min_speed_other)
        x_other -= 0.5 * (v_other_new + v_other) * self.time_interval
        v_other = v_other_new

        if x_ego <= -0.5 * C.CAR_LENGTH - 1.:
            self.ego_car.isReached = True
        if x_other <= -0.5 * C.CAR_LENGTH - 1.:
            self.other_car.isReached = True

        if self.ego_car.isReached and not self.other_car.isReached:
            self.ego_crossed_first = True
        elif self.other_car.isReached and not self.ego_car.isReached:
            self.ego_crossed_first = False

        self.ego_car.state = [x_ego, v_ego]
        self.other_car.state = [x_other, v_other]

        # if crossed the intersection, done or max time reached
        if (x_ego <= -0.5 * C.CAR_LENGTH - 1. and x_other <= -0.5 * C.CAR_LENGTH - 1.) \
                or self.frame >= self.max_time_steps:  # road width = 2.0 m
            self.done = True

        estimated_theta = self.best_theta_set1_fast(q_set)
        # estimated_theta = self.best_theta_set_fast(self.nfsp_models)
        # print(estimated_theta)

        self.state = (x_ego, v_ego, x_other, v_other, estimated_theta)

        # To compute inferred loss, we change theta_j temporarily

        actual_intent_other = self.other_car.aggressiveness
        if self.args.inferred_loss:
            self.other_car.aggressiveness = self.theta_set[estimated_theta][0]

        if self.args.acc_loss:
            loss_self = self.ego_car.self_loss_acc(self, action_self)
            loss_other = self.other_car.self_loss_acc(self, action_other)

        else:
            loss_self = self.ego_car.self_loss(self)
            loss_other = self.other_car.self_loss(self)

        # revert back to actual theta
        self.other_car.aggressiveness = actual_intent_other

        loss = np.array([loss_self + self.ego_car.gracefulness * loss_other, loss_other,
                         loss_self + self.ego_car.gracefulness * loss_other, loss_other])
        self.frame += 1
        return np.array(self.state), loss, self.done

    # def mpc(self, args, state, q_set, pos_trajectories, T):
    #     p2_state = [state[2], state[3], state[0], state[1]]
    #     p1_state = [state[0], state[1], state[2], state[3]]
    #
    #     self.inf_idx = int(state[4])
    #     self.args = args
    #     t1 = (self.ego_car.aggressiveness, self.theta_set[self.inf_idx][0])
    #     t2 = (t1[1], t1[0])
    #
    #     if t1[0] == t1[1] and t1[1] == 1 and self.inf_idx == 0:
    #         t1_idx = 1
    #     elif t1[0] == t1[1] and t1[1] == 1 and self.inf_idx == 1:
    #         t1_idx = 0
    #     elif t1[0] == t1[1] and t1[1] == 1000 and self.inf_idx == 4:
    #         t1_idx = 5
    #     elif t1[0] == t1[1] and t1[1] == 1000 and self.inf_idx == 5:
    #         t1_idx = 4
    #     else:
    #         t1_idx = self.theta_set.index(t1)
    #
    #     if t2[0] == t2[1] and t2[1] == 1:
    #         t2_idx = self.inf_idx
    #     elif t2[0] == t2[1] and t2[1] == 1000:
    #         t2_idx = self.inf_idx
    #     else:
    #         t2_idx = self.theta_set.index(t2)
    #
    #     self.t1_idx = t1_idx
    #     self.t2_idx = t2_idx
    #
    #     # print(args.device)
    #     Q_i_t_vals = q_set[t1_idx].forward(torch.FloatTensor(p1_state).to(args.device))
    #     Q_j_t_vals = q_set[t1_idx].forward(torch.FloatTensor(p2_state).to(args.device))
    #     Q_i_t_max = np.max(Q_i_t_vals.tolist())
    #     Q_j_t_max = np.max(Q_j_t_vals.tolist())
    #
    #     init_state = state[:4]
    #     Q_g_vals = np.zeros(len(pos_trajectories))
    #     start_time = time.time()
    #     i = 0
    #     for trajectory in pos_trajectories:
    #         self.get_chain_mpc(i, Q_g_vals, trajectory, init_state, self.inf_idx)
    #         i += 1
    #     end_time = time.time()
    #     print('Computation took:{}'.format(end_time - start_time))
    #     # print(pos_trajectories)
    #     # print(Q_g_vals)
    #     # max_index = np.argmax(opt_val)
    #     max_index = np.argmax(Q_g_vals)
    #     print('max_val: {}, max_index: {}'.format(np.max(Q_g_vals), max_index))
    #     # print('max_val: {}'.format(np.max(opt_val)))
    #
    #     if (1 - self.ego_car.gracefulness) * Q_i_t_max + self.ego_car.gracefulness * Q_j_t_max >= np.max(Q_g_vals):
    #         return None
    #     else:
    #         print(pos_trajectories[max_index])
    #         return pos_trajectories[max_index][0]
    #
    # def get_chain_mpc(self, index, Q_g_vals, traj, state, theta_idx):
    #     p1_r = []
    #     p2_r = []
    #     exp = []
    #
    #     for action in traj:
    #         # print('action: {}'.format(action))
    #         done = False
    #         if action == 0:
    #             a = self.parameters.MAX_DECELERATION
    #         elif action == 1:
    #             a = self.parameters.MAX_DECELERATION * 0.5
    #         elif action == 2:
    #             a = 0.0
    #         elif action == 3:
    #             a = self.parameters.MAX_ACCELERATION * 0.5
    #         else:
    #             a = self.parameters.MAX_ACCELERATION
    #         action_self = a
    #
    #         p2_state = [state[2], state[3], state[0], state[1]]
    #
    #         if torch.cuda.is_available():
    #             dist = self.nfsp_models[theta_idx].act_dist(torch.FloatTensor(p2_state).to(torch.device("cuda")))
    #         else:
    #             dist = self.nfsp_models[theta_idx].act_dist(torch.FloatTensor(p2_state).to(torch.device("cpu")))
    #         dist_a = dist.tolist()[0]
    #         action_p2 = np.argmax(dist_a)
    #         if action_p2 == 0:
    #             a = self.parameters.MAX_DECELERATION
    #         elif action_p2 == 1:
    #             a = self.parameters.MAX_DECELERATION * 0.5
    #         elif action_p2 == 2:
    #             a = 0.0
    #         elif action_p2 == 3:
    #             a = self.parameters.MAX_ACCELERATION * 0.5
    #         else:
    #             a = self.parameters.MAX_ACCELERATION
    #         action_other = a
    #
    #         # get current states
    #         x_ego = x_ego_new = state[0]
    #         x_other = x_other_new = state[2]
    #         v_ego = v_ego_new = state[1]
    #         v_other = v_other_new = state[3]
    #
    #         max_speed_ego = self.ego_car.car_parameters.MAX_SPEED[
    #             0]  # Variable(torch.tensor(self.ego_car.car_parameters.MAX_SPEED[0], dtype=float), requires_grad=True)
    #         min_speed_ego = self.ego_car.car_parameters.MAX_SPEED[
    #             1]  # Variable(torch.tensor(self.ego_car.car_parameters.MAX_SPEED[1], dtype=float), requires_grad=True)
    #         max_speed_other = self.other_car.car_parameters.MAX_SPEED[
    #             0]  # Variable(torch.tensor(self.other_car.car_parameters.MAX_SPEED[0], dtype=float), requires_grad=True)
    #         min_speed_other = self.other_car.car_parameters.MAX_SPEED[
    #             1]  # Variable(torch.tensor(self.other_car.car_parameters.MAX_SPEED[1], dtype=float), requires_grad=True)
    #
    #         # update state and check for collision
    #         l = C.CAR_LENGTH
    #         w = C.CAR_WIDTH
    #         collision = 0
    #         for t in range(int(self.time_interval / self.min_time_interval + 1)):
    #             v_ego_new = max(min(max_speed_ego, action_self * t * self.min_time_interval + v_ego), min_speed_ego)
    #             v_other_new = max(min(max_speed_other, action_other * t * self.min_time_interval + v_other),
    #                               min_speed_other)
    #             x_ego_new = x_ego - t * 0.5 * (v_ego_new + v_ego) * self.min_time_interval
    #             x_other_new = x_other - t * 0.5 * (v_other_new + v_other) * self.min_time_interval
    #             collision_box1 = [[x_ego_new - 0.5 * l, -0.5 * w],
    #                               [x_ego_new - 0.5 * l, 0.5 * w],
    #                               [x_ego_new + 0.5 * l, 0.5 * w],
    #                               [x_ego_new + 0.5 * l, -0.5 * w]]
    #             collision_box2 = [[0.5 * w, x_other_new - 0.5 * l],
    #                               [-0.5 * w, x_other_new - 0.5 * l],
    #                               [-0.5 * w, x_other_new + 0.5 * l],
    #                               [0.5 * w, x_other_new + 0.5 * l]]
    #             c = 0
    #             polygon = Polygon(collision_box2)
    #             for p in collision_box1:
    #                 point = Point(p[0], p[1])
    #                 c += polygon.contains(point)
    #                 if c > 0:
    #                     break
    #             collision += float(c > 0)  # number of times steps of collision
    #
    #         v_ego_new = max(min(max_speed_ego, action_self * self.time_interval + v_ego), min_speed_ego)
    #         x_ego_new -= 0.5 * (v_ego_new + v_ego) * self.time_interval  # start from positive distance to the center,
    #         # reduce to 0 when at the center
    #         v_ego = v_ego_new
    #         x_ego = x_ego_new
    #
    #         v_other_new = max(min(max_speed_other, action_other * self.time_interval + v_other), min_speed_other)
    #         x_other_new -= 0.5 * (v_other_new + v_other) * self.time_interval
    #         v_other = v_other_new
    #         x_other = x_other_new
    #         next_state = [x_ego, v_ego, x_other, v_other]
    #         # print(next_state)
    #
    #         # if crossed the intersection, done or max time reached
    #         if (x_ego <= -0.5 * C.CAR_LENGTH - 1. and x_other <= -0.5 * C.CAR_LENGTH - 1.) \
    #                 or self.frame >= self.max_time_steps:  # road width = 2.0 m
    #             done = True
    #
    #         loss_self = self.ego_car.loss(self, [x_ego, v_ego], done, collision, action_self, self.args)
    #         loss_other = self.other_car.loss(self, [x_other, v_other], done, collision, action_other, self.args)
    #
    #         p1_r.append(loss_self)
    #         p2_r.append(loss_other)
    #         # exp.append([state, action_self, next_state])
    #         state = next_state
    #         # print(state)
    #
    #     Q_i_t = np.sum(p1_r)
    #     Q_j_t = np.sum(p2_r)
    #     s_t_T = state
    #     # print(s_t_T)
    #     Q_i_vals = (get_models()[0])[self.t1_idx].forward(torch.FloatTensor(s_t_T).to(self.args.device))
    #     Q_j_vals = (get_models()[0])[self.t2_idx].forward(
    #         torch.FloatTensor([s_t_T[2], s_t_T[3], s_t_T[0], s_t_T[1]]).to(self.args.device))
    #
    #     Q_i_t_T_max = np.max(Q_i_vals.tolist())
    #     Q_j_t_T_max = np.max(Q_j_vals.tolist())
    #     Q_i_t += Q_i_t_T_max
    #     Q_j_t += Q_j_t_T_max
    #     # print(Q_i_t, Q_j_t)
    #     Q_g = self.ego_car.gracefulness * Q_j_t + (1 - self.ego_car.gracefulness) * Q_i_t
    #     Q_g_vals[index] = Q_g
    #
    #     # return p1_r, p2_r, exp, Q_g

    def mpc_psuedo_batch(self, args, state, q_set, pos_trajectories, T):

        self.inf_idx = int(state[4])
        self.args = args
        t1 = (self.ego_car.aggressiveness, self.theta_set[self.inf_idx][0])
        t2 = (t1[1], t1[0])

        if t1[0] == t1[1] and t1[1] == 1 and self.inf_idx == 0:
            t1_idx = 1
        elif t1[0] == t1[1] and t1[1] == 1 and self.inf_idx == 1:
            t1_idx = 0
        elif t1[0] == t1[1] and t1[1] == 1000 and self.inf_idx == 4:
            t1_idx = 5
        elif t1[0] == t1[1] and t1[1] == 1000 and self.inf_idx == 5:
            t1_idx = 4
        else:
            t1_idx = self.theta_set.index(t1)

        if t2[0] == t2[1] and t2[1] == 1:
            t2_idx = self.inf_idx
        elif t2[0] == t2[1] and t2[1] == 1000:
            t2_idx = self.inf_idx
        else:
            t2_idx = self.theta_set.index(t2)

        self.t1_idx = t1_idx #torch.tensor(t1_idx, dtype=torch.float, device=args.device)
        self.t2_idx = t2_idx #torch.tensor(t2_idx, dtype=torch.float, device=args.device)

        # initial state is leaf
        if args.requires_grad:
            init_state = torch.tensor(state[:4], dtype=torch.float, device=args.device, requires_grad=True)
            traj_tensor = torch.tensor(pos_trajectories, dtype=float, device=args.device, requires_grad=True)
            Q_g_vals = torch.zeros(len(pos_trajectories), dtype=torch.float, device=args.device, requires_grad=True)
        else:
            init_state = torch.tensor(state[:4], dtype=torch.float, device=args.device)
            traj_tensor = torch.tensor(pos_trajectories, dtype=float, device=args.device)
            Q_g_vals = torch.zeros(len(pos_trajectories), dtype=torch.float, device=args.device)
        idx = 0
        Q_i_t_vals = q_set[t1_idx].forward(init_state) #.to(args.device))
        Q_j_t_vals = q_set[t2_idx].forward(torch.FloatTensor([state[2], state[3], state[0], state[1]]).to(args.device))
        Q_i_t_max = torch.max(Q_i_t_vals)
        Q_j_t_max = torch.max(Q_j_t_vals)

        start_time = time.time()
        Q_g_max, Q_g_max_index = self.get_psuedo_q(init_state, traj_tensor, args, q_set)
        end_time = time.time()
        # print('Computation took:{}'.format(end_time - start_time))
        if ((1 - self.ego_car.gracefulness) * Q_i_t_max + self.ego_car.gracefulness * Q_j_t_max) >= Q_g_max:
            return None
        else:
            # print(pos_trajectories[Q_g_max_index][0])
            action_mpc = pos_trajectories[Q_g_max_index][0]
            return action_mpc

    def get_psuedo_q(self, init_state, trajectories, args, q_set):
        ns = init_state
        dynamics_main = DynamicsMain.apply
        ns_batch = init_state.repeat((trajectories.size(0), 1))
        # print(ns_batch)
        # print(ns_batch.shape)
        # print(trajectories.shape)
        return dynamics_main(ns_batch, trajectories, self, args, q_set)

    def reset_inference(self):
        self.ego_car.state[0] = np.random.uniform(C.Intersection.CAR_1.INITIAL_STATE[0] * 0.5,
                                                  C.Intersection.CAR_1.INITIAL_STATE[0] * 1.0)
        # self.ego_car.state[0] = 3.0
        max_speed = np.sqrt(
            (self.ego_car.state[0] - 1 - C.CAR_LENGTH * 0.5) * 2. * abs(C.Intersection.MAX_DECELERATION))
        self.ego_car.state[1] = np.random.uniform(max_speed * 0.1, max_speed * 0.5)
        self.other_car.state[0] = np.random.uniform(C.Intersection.CAR_2.INITIAL_STATE[0] * 0.5,
                                                    C.Intersection.CAR_2.INITIAL_STATE[0] * 1.0)
        max_speed = np.sqrt(
            (self.other_car.state[0] - 1 - C.CAR_LENGTH * 0.5) * 2. * abs(C.Intersection.MAX_DECELERATION))
        self.other_car.state[1] = np.random.uniform(max_speed * 0.1, max_speed * 0.5)

        self.done = False
        self.frame = 0
        self.isCollision = False
        self.ego_car.isReached = False
        self.other_car.isReached = False
        self.ego_crossed_first = False

        self.inference_actions = []
        self.inference_states = []
        self.likelihood_values = []
        self.likelihood1_values = []

        theta_j = np.random.choice([1000, 1], 1, p=[0.5, 0.5])[0]
        theta_i = np.random.choice([1000, 1], 1, p=[0.5, 0.5])[0]
        theta_mode = -1
        if theta_i == 1 and theta_j == 1:
            theta_mode = 0
        elif theta_i == 1 and theta_j == 1000:
            theta_mode = 2
        elif theta_i == 1000 and theta_j == 1:
            theta_mode = 3
        elif theta_i == 1000 and theta_j == 1000:
            theta_mode = 4
        self.state = (self.ego_car.state[0], self.ego_car.state[1],
                      self.other_car.state[0], self.other_car.state[1], theta_mode)

        return np.array(self.state)

    def reset_inference_state(self, state):
        self.ego_car.state[0] = state[0]
        self.ego_car.state[1] = state[1]
        self.other_car.state[0] = state[2]
        self.other_car.state[1] = state[3]

        self.done = False
        self.frame = 0
        self.isCollision = False
        self.ego_car.isReached = False
        self.other_car.isReached = False
        self.ego_crossed_first = False

        self.inference_actions = []
        self.inference_states = []
        self.likelihood_values = []
        self.likelihood1_values = []
        self.state = state
        return np.array(self.state)

    def reset_state(self, state):
        self.ego_car.state[0] = state[0]
        self.ego_car.state[1] = state[1]
        self.other_car.state[0] = state[2]
        self.other_car.state[1] = state[3]

        self.done = False
        self.frame = 0
        self.isCollision = False
        self.ego_car.isReached = False
        self.other_car.isReached = False
        self.ego_crossed_first = False

        self.inference_actions = []
        self.inference_states = []
        self.likelihood_values = []
        self.likelihood1_values = []
        self.state = state[:-1]
        return np.array(self.state)

    def reset(self):
        self.ego_car.state[0] = np.random.uniform(C.Intersection.CAR_1.INITIAL_STATE[0] * 0.5,
                                                  C.Intersection.CAR_1.INITIAL_STATE[0] * 1.0)
        # self.ego_car.state[0] = 3.0
        max_speed = np.sqrt(
            (self.ego_car.state[0] - 1 - C.CAR_LENGTH * 0.5) * 2. * abs(C.Intersection.MAX_DECELERATION))
        self.ego_car.state[1] = np.random.uniform(max_speed * 0.1, max_speed * 0.5)
        self.other_car.state[0] = np.random.uniform(C.Intersection.CAR_2.INITIAL_STATE[0] * 0.5,
                                                    C.Intersection.CAR_2.INITIAL_STATE[0] * 1.0)
        max_speed = np.sqrt(
            (self.other_car.state[0] - 1 - C.CAR_LENGTH * 0.5) * 2. * abs(C.Intersection.MAX_DECELERATION))
        self.other_car.state[1] = np.random.uniform(max_speed * 0.1, max_speed * 0.5)

        self.done = False
        self.frame = 0
        self.isCollision = False
        self.ego_car.isReached = False
        self.other_car.isReached = False
        self.ego_crossed_first = False

        self.inference_actions = []
        self.inference_states = []
        self.likelihood_values = []
        self.likelihood1_values = []
        self.state = (self.ego_car.state[0], self.ego_car.state[1],
                      self.other_car.state[0], self.other_car.state[1])

        return np.array(self.state)

    def render(self):
        # TODO: render one step
        self.renderer.draw_frame(self)

    def render_fullcycle(self, state_1_history, state_2_history):
        self.frame = 0
        FRAME_TIME = self.time_interval
        frame = FRAME_TIME

        for state1, state2 in zip(state_1_history, state_2_history):

            if frame < FRAME_TIME:
                time.sleep(FRAME_TIME - frame)

            start = time.time()

            self.ego_car.state = [state1[0], state1[1]]
            self.other_car.state = [state1[0], state1[1]]
            self.frame += 1

            frame = time.time() - start

            self.render()

    def simulate(self):
        FRAME_TIME = self.time_interval
        frame = FRAME_TIME

        while not self.done:
            if frame < FRAME_TIME:
                time.sleep(FRAME_TIME - frame)

            start = time.time()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    self.done = True

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_p:
                        self.paused = not self.paused

                    if event.key == pg.K_q:
                        pg.quit()
                        self.done = True

                    if event.key == pg.K_d:
                        self.car_num_display = ~self.car_num_display

            if not self.paused:
                a = self.ego_car.get_action()
                self.step(a)

                self.render()

            frame = time.time() - start
