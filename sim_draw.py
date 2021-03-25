# TODO: add uncertainty visualization
"""
Draws simulation and the results
1. draw_frame: update simulation
2. draw_axes: draw intersection
3. draw_dist: show results of car distances
4. draw_intent: show results of inferred intent
5. draw_prob: draw distribution of future states on intersection
6. c2p: transformation from states to pixels
"""
import pygame as pg
import pygame.gfxdraw
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import math
import time
import glob
import imageio
import os
import pdb

LIGHT_GREY = (230, 230, 230)
RED = (230, 0, 0)


class VisUtils:

    def __init__(self, sim):
        self.sim = sim
        self.env = sim.env
        self.drawing_prob = sim.drawing_prob
        self.drawing_intent = sim.drawing_intent
        self.beta_set = self.sim.beta_set
        self.theta_list = self.sim.theta_list
        self.lambda_list = self.sim.lambda_list
        # TODO: organize this!
        if self.drawing_intent:
            self.p_state_1 = sim.agents[0].predicted_states_self  # get the last prediction
            self.p_state_2 = sim.agents[0].predicted_states_other
            self.past_state_1 = sim.agents[1].state[-1]
            self.past_state_2 = sim.agents[0].state[-1]
            self.intent_1 = []
            self.intent_2 = []
            self.theta_distri_1 = [[], []]  # theta1, theta2
            self.theta_distri_2 = [[], []]  # theta1, theta2
            self.lambda_distri_1 = [[], []]  # lambda1, lambda2
            self.lambda_distri_2 = [[], []]  # lambda1, lambda2
            self.lambda_1 = []
            self.lambda_2 = []
            "conditional prob of p(theta, lambda)/p(lambda)"
            self.true_intent_prob_1 = []  # theta1, theta2
            self.true_intent_prob_2 = []  # theta1, theta2
            "getting ground truth parameters and record for each step"
            self.true_params = self.sim.true_params
            self.true_intent_1 = []
            self.true_intent_2 = []
            self.true_noise_1 = []
            self.true_noise_2 = []
            "get id for true (theta, lambda)"
            self.true_id = []
            for i, par in enumerate(self.true_params):
                theta_id = self.theta_list.index(par[0])
                lambda_id = self.lambda_list.index(par[1])
                self.true_id.append([theta_id, lambda_id])
        if self.drawing_prob:
            # TODO: fix this for non-BVP inference
            self.p_state_1 = sim.agents[1].predicted_states_other  # get the last prediction
            self.p_state_2 = sim.agents[1].predicted_states_self
            self.past_state_1 = sim.agents[1].state[-1]
            self.past_state_2 = sim.agents[0].state[-1]

        self.frame = sim.frame
        self.dist = []
        self.sleep_between_step = False

        if self.env.name == 'trained_intersection':
            self.sleep_between_step = True
            self.screen_width = 10  # 50
            self.screen_height = 10  # 50
            self.coordinate_scale = 80
            self.zoom = 0.16
            self.asset_location = "assets/"
            self.fps = 24  # max framework

            img_width = int(self.env.CAR_WIDTH * self.coordinate_scale * self.zoom)
            img_height = int(self.env.CAR_LENGTH * self.coordinate_scale * self.zoom)

            "initialize pygame"
            pg.init()
            self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                               self.screen_height * self.coordinate_scale))
            # self.car1_image = pg.transform.rotate(pg.image.load(self.asset_location + self.sim.agents[0].car_par["sprite"]),
            #                                       -self.sim.agents[0].car_par["orientation"])
            #
            # self.car2_image = pg.transform.rotate(pg.image.load(self.asset_location+ self.sim.agents[1].car_par["sprite"]),
            #                                       -self.sim.agents[1].car_par["orientation"])
            self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location
                                                                                   + self.sim.agents[i].car_par["sprite"]),
                                                                     (img_width, img_height)),
                                                  -self.sim.agents[i].car_par["orientation"])
                              for i in range(self.sim.N_AGENTS)]

            #self.origin = np.array([-15.0, 15.0])
            self.origin = np.array([0, 0])

        elif self.env.name == 'bvp_intersection':
            self.sleep_between_step = True
            self.screen_width = 10  # 50
            self.screen_height = 10  # 50
            self.coordinate_scale = 80
            self.zoom = 0.25
            self.asset_location = 'assets/'
            self.fps = 24  # max framework

            self.car_width = 1.5
            self.car_length = 3

            img_width = int(self.car_width * self.coordinate_scale * self.zoom)
            img_height = int(self.car_length * self.coordinate_scale * self.zoom)

            "initialize pygame"
            pg.init()
            self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                               self.screen_height * self.coordinate_scale))

            # self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location
            #                                                                        + self.car_par[i]['sprite']),
            #                                                          (img_width, img_height)),
            #                                       - self.car_par[i]['orientation']) for i in range(len(self.car_par))]
            self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location
                                                                                   + self.sim.agents[i].car_par[
                                                                                       "sprite"]),
                                                                     (img_width, img_height)),
                                                  -self.sim.agents[i].car_par["orientation"])
                              for i in range(self.sim.n_agents)]
            self.origin = np.array([35, 35])

        elif self.env.name == 'trained_intersection':  # TODO: check name
            self.screen_width = 5
            self.screen_height = 5
            self.asset_location = "assets/"
            self.fps = 24  # max framework
            self.coordinate_scale = 100
            self.zoom = 0.3

            "initialize pygame"
            pg.init()
            self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                               self.screen_height * self.coordinate_scale))

            img_width = int(self.env.CAR_WIDTH * self.coordinate_scale * self.zoom)
            img_height = int(self.env.CAR_LENGTH * self.coordinate_scale * self.zoom)

            "loading car image into pygame"
            self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location
                                                                                   + self.sim.agents[i].car_par[
                                                                                       "sprite"]),
                                                                     (img_width, img_height)),
                                                  -self.sim.agents[i].car_par["orientation"])
                              for i in range(self.sim.N_AGENTS)]

            self.origin = np.array([1.0, -1.0])

        else:
            print("WARNING: NO INTERSECTION NAME FOUND")

        "Draw Axis Lines"
        self.screen.fill((255, 255, 255))
        if self.env.name == 'bvp_intersection':
            self.bvp_draw_axes()
        else:
            self.draw_axes()
        pg.display.flip()
        pg.display.update()

    def draw_frame(self):
        # Draw the current frame
        self.frame = self.sim.frame
        frame = self.sim.frame

        # render 10 times for each step
        steps = 5

        for k in range(1, steps + 1):
            self.screen.fill((255, 255, 255))
            if self.env.name == 'bvp_intersection':
                self.bvp_draw_axes()
            else:
                self.draw_axes()
            # Draw Images
            for i in range(self.sim.n_agents):
                "getting pos of agent"
                pos_old = np.array(self.sim.agents[i].state[frame][:2])
                pos_new = np.array(self.sim.agents[i].state[frame+1][:2])  # get 0th and 1st element (not include 2)

                "smooth out the movement between each step"
                pos = pos_old * (1 - k * 1. / steps) + pos_new * (k * 1. / steps)
                "transform pos"
                if self.env.name == 'bvp_intersection':
                    if i == 0:
                        pixel_pos_car = self.bvp_c2p((35, pos[1]))
                    elif i == 1:
                        pixel_pos_car = self.bvp_c2p((pos[0], 35))
                    else:
                        print("AGENT EXCEEDS 2!")
                else:
                    pixel_pos_car = self.c2p(pos)
                size_car = self.car_image[i].get_size()
                "update car display"
                self.screen.blit(self.car_image[i],
                                 (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
                if self.sleep_between_step:
                    time.sleep(0.03)
            # Annotations
            # font = pg.font.SysFont("Arial", 30)
            font = pg.font.SysFont("Arial", 15)
            screen_w, screen_h = self.screen.get_size()
            label_x = screen_w - 800
            label_y = 260
            label_y_offset = 30
            "Labeling new states and actions"
            pos_h, speed_h = self.sim.agents[0].state[frame+1][1], self.sim.agents[0].state[frame+1][3]
            label = font.render("Car 1 position and speed: (%5.4f , %5.4f)" % (pos_h, speed_h), 1,
                                (0, 0, 0))
            self.screen.blit(label, (label_x, label_y))
            pos_m, speed_m = self.sim.agents[1].state[frame+1][0], self.sim.agents[1].state[frame+1][2]
            label = font.render("Car 2 position and speed: (%5.4f , %5.4f)" % (pos_m, speed_m), 1,
                                (0, 0, 0))
            self.screen.blit(label, (label_x, label_y + label_y_offset))
            action1, action2 = self.sim.agents[0].action[frame+1], self.sim.agents[1].action[frame+1]
            if self.env.name == 'merger':
                label = font.render("Car 1 action: " % action1, 1, (0, 0, 0))
            else:
                label = font.render("Car 1 action: (%5.4f)" % action1, 1, (0, 0, 0))

            self.screen.blit(label, (label_x, label_y + 2 * label_y_offset))
            if self.env.name == 'merger':
                label = font.render("Car 2 action: " % action2, 1, (0, 0, 0))
            else:
                label = font.render("Car 2 action: (%5.4f)" % action2, 1, (0, 0, 0))

            self.screen.blit(label, (label_x, label_y + 3 * label_y_offset))
            label = font.render("Frame: %i" % self.sim.frame, 1, (0, 0, 0))
            self.screen.blit(label, (10, 10))

            "drawing the map of state distribution"
            # pg.draw.circle(self.screen, (255, 255, 255), pos2, 10)  # surface,  color, (x, y),radius>=1  # test
            if self.drawing_prob:
                self.draw_prob()  # calling function to draw with data from inference

            pg.display.flip()
            pg.display.update()

        self.calc_dist()
        if self.drawing_intent:
            # if not self.frame == 0:
            self.calc_intent()

    def draw_axes(self):
        # draw lanes based on environment TODO: lanes are defined as bounds of agent state spaces, need to generalize
        for a in self.env.bounds:
            # pg.draw.line(self.screen, RED, (0,0), (10,10), 3)
            bound_x, bound_y = a[0], a[1]

            if bound_x:
                b_min, b_max = bound_x[0], bound_x[1]
                _bound1 = self.c2p((b_min, 0))
                _bound2 = self.c2p((b_max, 0))

                bounds = np.array([_bound1[0], _bound2[0]])
                road_width = bounds[1]-bounds[0]+ 2 # offset for road
                pg.draw.line(self.screen,LIGHT_GREY , (((bounds[1] + bounds[0])/2), 1),
                             (((bounds[1] + bounds[0])/2), self.screen_height * self.coordinate_scale,
                              ), bounds[1] - bounds[0])

                if self.env.name == 'merger':
                    pg.draw.line(self.screen,LIGHT_GREY , (((bounds[1] + bounds[0])/2+ road_width), 1),
                             (((bounds[1] + bounds[0])/2+road_width), self.screen_height * self.coordinate_scale,
                              ), bounds[1] - bounds[0])
            if bound_y:
                b_min, b_max = bound_y[0], bound_y[1]
                _bound1 = self.c2p((0, b_min))
                _bound2 = self.c2p((0, b_max))
                bounds = np.array([_bound1[1], _bound2[1]])
                pg.draw.line(self.screen, LIGHT_GREY, (1, (bounds[1] + bounds[0]) / 2),
                             (self.screen_width * self.coordinate_scale,
                              (bounds[1] + bounds[0]) / 2), bounds[0] - bounds[1])
                # pg.draw.line(self.screen, RED, ((((bounds[1] + bounds[0]) / 2)+ bounds[1]- bounds[0]),1),
                #              ((((bounds[1] + bounds[0])/2)+ bounds[1]- bounds[0]) , self.screen_height * self.coordinate_scale,
                #               ), bounds[1] - bounds[0])

    def bvp_draw_axes(self):
        # draw lanes based on environment
        pg.draw.line(self.screen, LIGHT_GREY, self.c2p((35, -50)), self.c2p((35, 100)), 35)
        pg.draw.line(self.screen, LIGHT_GREY, self.c2p((100, 35)), self.c2p((-50, 35)), 35)

    def draw_axes_lanes(self):
        # draw lanes based on environment TODO: lanes are defined as bounds of agent state spaces, need to generalize
        #pg.draw.line(self.screen, LIGHT_GREY, self.c2p((-10, 10)), self.c2p((10, -10)), 1)  # testing
        for a in self.env.bounds:
            bound_x, bound_y = a[0], a[1]
            if bound_x:
                b_min, b_max = bound_x[0], bound_x[1]
                _bound1 = self.c2p((b_min, 0))
                _bound2 = self.c2p((b_max, 0))
                bounds = np.array([_bound1[0], _bound2[0]])
                pg.draw.line(self.screen,LIGHT_GREY , ((bounds[1] + bounds[0])/2, 1),
                             ((bounds[1] + bounds[0])/2, self.screen_height * self.coordinate_scale,
                              ), bounds[1] - bounds[0])
            if bound_y:
                b_min, b_max = bound_y[0], bound_y[1]
                _bound1 = self.c2p((0, b_min))
                _bound2 = self.c2p((0, b_max))
                bounds = np.array([_bound1[1], _bound2[1]])
                pg.draw.line(self.screen, LIGHT_GREY, (1, (bounds[1] + bounds[0]) / 2),
                             (self.screen_width * self.coordinate_scale,
                              (bounds[1] + bounds[0]) / 2), bounds[0] - bounds[1])

    def draw_dist_n_action(self):
        """
        plotting distance between cars and action taken
        :return:
        """
        fig1, (ax1, ax2, ax3) = pyplot.subplots(3) #3 rows
        fig1.suptitle('Euclidean distance and Agent Actions')
        ax1.plot(self.dist, label='car dist')
        ax1.legend()
        ax1.set(xlabel='frame', ylabel='distance')

        ax2.plot(self.sim.agents[0].action, label='actual')
        ax2.plot(self.sim.agents[0].predicted_actions_self, label='predicted', linestyle='--')
        ax2.set_ylim([-10, 10])
        ax2.set_yticks([-8, -4, 0, 4, 8])
        if self.env.name == 'bvp_intersection':
            ax2.set_ylim([-7, 12])
            ax2.set_yticks([-5, 0, 5, 10])
        ax2.legend()
        ax2.set(xlabel='frame', ylabel='H actions')

        ax3.plot(self.sim.agents[1].action, label='actual')
        ax3.plot(self.sim.agents[0].predicted_actions_other, label='predicted', linestyle='--')
        ax3.set_ylim([-10, 10])
        ax3.set_yticks([-8, -4, 0, 4, 8])
        if self.env.name == 'bvp_intersection':
            ax3.set_ylim([-7, 12])
            ax3.set_yticks([-5, 0, 5, 10])
        ax3.legend()
        ax3.set(xlabel='frame', ylabel='M actions')

        pyplot.show()

    def calc_dist(self):
        """
        recording distance between two cars
        :return:
        """
        past_state_h = self.sim.agents[0].state[self.frame]
        past_state_m = self.sim.agents[1].state[self.frame]
        x_h = past_state_h[1]
        x_m = past_state_m[0]
        if self.env.name == 'bvp_intersection':
            x_h = 35 - x_h
            x_m = 35 - x_m
            dist = np.sqrt(x_h * x_h + x_m * x_m)
        else:
            dist = np.sqrt(x_h * x_h + x_m * x_m)
        self.dist.append(dist)

    def calc_intent(self):
        # TODO: store distribution of lambda (NN, N)
        # TODO: differentiate empathetic and non-empathetic cases
        # p_beta_d, beta_pair = self.sim.agents[1].predicted_intent_all[self.frame]
        theta_list = self.sim.theta_list
        lambda_list = self.sim.lambda_list
        if self.sim.sharing_belief:  # both agents uses same belief (common belief)
            self.true_intent_1.append(self.true_params[0][0])
            self.true_noise_1.append(self.true_params[0][1])
            self.true_intent_2.append(self.true_params[1][0])
            self.true_noise_2.append(self.true_params[1][1])
            p_beta_d, [beta_h, beta_m] = self.sim.agents[0].predicted_intent_all[-1]
            if self.sim.decision_type[0] == self.sim.decision_type[1] == 'bvp_empathetic':
                self.intent_1.append(beta_h[0])
                self.intent_2.append(beta_m[0])
                self.lambda_1.append(beta_h[1])
                self.lambda_2.append(beta_m[1])

                "get highest row/col -> compare distribution of that col/row!!"
                # =========================
                # beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
                # p_beta_dt = p_beta_d.transpose()
                # p_b_m = p_beta_d[beta_pair_id[1]]  # []
                # p_b_h = p_beta_dt[beta_pair_id[1]]
                # =========================
                "getting marginal beta belief table"
                joint_infer_m = self.sim.agents[0].predicted_intent_other
                p_joint_h, lambda_h = self.sim.agents[0].predicted_intent_self[-1]
                p_joint_m, lambda_m = joint_infer_m[-1]

                "NOTE THAT P_JOINT IS LAMBDA X THETA"
                "calculating the marginal for theta"
                sum_h = p_joint_h.sum(axis=0)
                sum_h = np.ndarray.tolist(sum_h)
                sum_m = p_joint_m.sum(axis=0)
                sum_m = np.ndarray.tolist(sum_m)  # [theta1, theta2]
                for i in range(len(sum_h)):
                    if not len(self.theta_distri_1) == len(sum_h):  # create 2D array if not already
                        for j in range(len(sum_h)):
                            self.theta_distri_1.append([])
                            self.theta_distri_2.append([])
                    self.theta_distri_1[i].append(sum_h[i])
                    self.theta_distri_2[i].append(sum_m[i])

                "calculating the marginal for lambda"
                sum_lamb_h = p_joint_h.sum(axis=1)
                sum_lamb_h = np.ndarray.tolist(sum_lamb_h)
                sum_lamb_m = p_joint_m.sum(axis=1)
                sum_lamb_m = np.ndarray.tolist(sum_lamb_m)  # [theta1, theta2]
                for i in range(len(sum_lamb_h)):
                    if not len(self.theta_distri_1) == len(sum_lamb_h):  # create 2D array if not already
                        for j in range(len(sum_lamb_h)):
                            self.lambda_distri_1.append([])
                            self.lambda_distri_2.append([])
                    self.lambda_distri_1[i].append(sum_lamb_h[i])
                    self.lambda_distri_2[i].append(sum_lamb_m[i])
                "getting the point estimated p(theta, lambda)"  # NOT IN USE
                # # for i in range(len(sum_lamb_h)):
                # p_beta_true_par_h = p_joint_h[self.true_id[0][1]][self.true_id[0][0]]  # for p_joint lambdas are the rows
                # p_beta_true_par_m = p_joint_m[self.true_id[1][1]][self.true_id[1][0]]
                # p_beta_true_par_h /= sum_lamb_h[self.true_id[0][1]]
                # p_beta_true_par_m /= sum_lamb_m[self.true_id[1][1]]
                # self.true_intent_prob_h.append(p_beta_true_par_h)
                # self.true_intent_prob_m.append(p_beta_true_par_m)
            elif self.sim.decision_type[0] == self.sim.decision_type[1] == 'bvp_non_empathetic':
                # TODO: record p_lambda
                true_beta_1, true_beta_2 = self.true_params
                b_id_1 = self.beta_set.index(true_beta_1)
                b_id_2 = self.beta_set.index(true_beta_2)
                p_b_1 = np.transpose(p_beta_d)[b_id_2]  # get col p_beta
                p_b_2 = p_beta_d[b_id_1]
                beta_1 = self.beta_set[np.argmax(p_b_1)]
                beta_2 = self.beta_set[np.argmax(p_b_2)]

                self.intent_1.append(beta_1[0])
                self.intent_2.append(beta_2[0])
                self.lambda_1.append(beta_1[1])
                self.lambda_2.append(beta_2[1])

                p_theta_1 = np.zeros((len(theta_list)))
                p_theta_2 = np.zeros((len(theta_list)))
                p_lambda_1 = np.zeros((len(lambda_list)))
                p_lambda_2 = np.zeros((len(lambda_list)))
                "get p_theta marginal"
                # TODO: generalize this calculation to different param sizes
                p_theta_1[0] = p_b_1[0] + p_b_1[1]  # NA
                p_theta_1[1] = p_b_1[2] + p_b_1[3]  # A
                p_theta_2[0] = p_b_2[0] + p_b_2[1]
                p_theta_2[1] = p_b_2[2] + p_b_2[3]
                p_lambda_1[0] = p_b_1[0] + p_b_1[2]  # noisy (refer to main.py)
                p_lambda_1[1] = p_b_1[1] + p_b_1[3]  # non-noisy
                p_lambda_2[0] = p_b_2[0] + p_b_2[2]
                p_lambda_2[1] = p_b_2[1] + p_b_2[3]
                p_theta_1 /= np.sum(p_theta_1)
                p_theta_2 /= np.sum(p_theta_2)
                p_lambda_1 /= np.sum(p_lambda_1)
                p_lambda_2 /= np.sum(p_lambda_2)
                assert round(np.sum(p_theta_1)) == 1
                self.theta_distri_1[0].append(p_theta_1[0])
                self.theta_distri_1[1].append(p_theta_1[1])
                self.theta_distri_2[0].append(p_theta_2[0])
                self.theta_distri_2[1].append(p_theta_2[1])

            elif self.sim.decision_type[0] == 'bvp_empathetic':
                self.intent_1.append(beta_h[0])
                self.intent_2.append(beta_m[0])
                self.lambda_1.append(beta_h[1])
                self.lambda_2.append(beta_m[1])

                "get highest row/col -> compare distribution of that col/row!!"
                # =========================
                # beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
                # p_beta_dt = p_beta_d.transpose()
                # p_b_m = p_beta_d[beta_pair_id[1]]  # []
                # p_b_h = p_beta_dt[beta_pair_id[1]]
                # =========================
                "getting marginal beta belief table"
                joint_infer_m = self.sim.agents[0].predicted_intent_other
                p_joint_h, lambda_h = self.sim.agents[0].predicted_intent_self[-1]
                p_joint_m, lambda_m = joint_infer_m[-1]

                "NOTE THAT P_JOINT IS LAMBDA X THETA"
                "calculating the marginal for theta"
                sum_h = p_joint_h.sum(axis=0)
                sum_h = np.ndarray.tolist(sum_h)
                sum_m = p_joint_m.sum(axis=0)
                sum_m = np.ndarray.tolist(sum_m)  # [theta1, theta2]
                for i in range(len(sum_h)):
                    if not len(self.theta_distri_1) == len(sum_h):  # create 2D array if not already
                        for j in range(len(sum_h)):
                            self.theta_distri_1.append([])
                            self.theta_distri_2.append([])
                    self.theta_distri_1[i].append(sum_h[i])
                    self.theta_distri_2[i].append(sum_m[i])

                "calculating the marginal for lambda"
                sum_lamb_h = p_joint_h.sum(axis=1)
                sum_lamb_h = np.ndarray.tolist(sum_lamb_h)
                sum_lamb_m = p_joint_m.sum(axis=1)
                sum_lamb_m = np.ndarray.tolist(sum_lamb_m)  # [theta1, theta2]
                for i in range(len(sum_lamb_h)):
                    if not len(self.theta_distri_1) == len(sum_lamb_h):  # create 2D array if not already
                        for j in range(len(sum_lamb_h)):
                            self.lambda_distri_1.append([])
                            self.lambda_distri_2.append([])
                    self.lambda_distri_1[i].append(sum_lamb_h[i])
                    self.lambda_distri_2[i].append(sum_lamb_m[i])

            elif self.sim.decision_type[0] == 'bvp_non_empathetic':
                # TODO: record p_lambda
                true_beta_1, true_beta_2 = self.true_params
                b_id_1 = self.beta_set.index(true_beta_1)
                b_id_2 = self.beta_set.index(true_beta_2)
                p_b_1 = np.transpose(p_beta_d)[b_id_2]  # get col p_beta
                p_b_2 = p_beta_d[b_id_1]
                beta_1 = self.beta_set[np.argmax(p_b_1)]
                beta_2 = self.beta_set[np.argmax(p_b_2)]

                self.intent_1.append(beta_1[0])
                self.intent_2.append(beta_2[0])
                self.lambda_1.append(beta_1[1])
                self.lambda_2.append(beta_2[1])

                p_theta_1 = np.zeros((len(theta_list)))
                p_theta_2 = np.zeros((len(theta_list)))
                p_lambda_1 = np.zeros((len(lambda_list)))
                p_lambda_2 = np.zeros((len(lambda_list)))
                "get p_theta marginal"
                # TODO: generalize this calculation to different param sizes
                p_theta_1[0] = p_b_1[0] + p_b_1[1]  # NA
                p_theta_1[1] = p_b_1[2] + p_b_1[3]  # A
                p_theta_2[0] = p_b_2[0] + p_b_2[1]
                p_theta_2[1] = p_b_2[2] + p_b_2[3]
                p_lambda_1[0] = p_b_1[0] + p_b_1[2]  # noisy (refer to main.py)
                p_lambda_1[1] = p_b_1[1] + p_b_1[3]  # non-noisy
                p_lambda_2[0] = p_b_2[0] + p_b_2[2]
                p_lambda_2[1] = p_b_2[1] + p_b_2[3]
                p_theta_1 /= np.sum(p_theta_1)
                p_theta_2 /= np.sum(p_theta_2)
                p_lambda_1 /= np.sum(p_lambda_1)
                p_lambda_2 /= np.sum(p_lambda_2)
                assert round(np.sum(p_theta_1)) == 1
                self.theta_distri_1[0].append(p_theta_1[0])
                self.theta_distri_1[1].append(p_theta_1[1])
                self.theta_distri_2[0].append(p_theta_2[0])
                self.theta_distri_2[1].append(p_theta_2[1])

            else:
                print("WARNING! DECISION MODEL CHOICE NOT SUPPORTED!")

        else:  # single agent inference
            joint_infer_m = self.sim.agents[1].predicted_intent_self
            theta_list = self.sim.theta_list
            lambda_list = self.sim.lambda_list
            if not len(joint_infer_m) == 0:
                p_joint_h, lambda_h = self.sim.agents[1].predicted_intent_other[-1]
                p_joint_m, lambda_m = joint_infer_m[-1]
                # TODO: process the lambda
                sum_h = p_joint_h.sum(axis=0)
                sum_h = np.ndarray.tolist(sum_h)
                sum_m = p_joint_m.sum(axis=0)
                sum_m = np.ndarray.tolist(sum_m)  # [theta1, theta2]
                # TODO: add sum to list
                idx_h = sum_h.index(max(sum_h))
                idx_m = sum_m.index(max(sum_m))
                for i in range(len(sum_h)):
                    if not len(self.theta_distri_1) == len(sum_h):  # create 2D array
                        j = 0
                        while j in range(len(sum_h)):
                            self.theta_distri_1.append([])
                            self.theta_distri_2.append([])
                    self.theta_distri_1[i].append(sum_h[i])
                    self.theta_distri_2[i].append(sum_m[i])
                H_intent = theta_list[idx_h]
                M_intent = theta_list[idx_m]
                self.intent_1.append(H_intent)
                self.intent_2.append(M_intent)
                self.lambda_1.append(lambda_h)
                self.lambda_2.append(lambda_m)
            else:
                p_joint_h, lambda_h = self.sim.agents[1].predicted_intent_other[-1]
                # print("-draw- p_joint_h: ", p_joint_h)
                sum_h = p_joint_h.sum(axis=0)
                sum_h = np.ndarray.tolist(sum_h)
                for i in range(len(sum_h)):
                    if not len(self.theta_distri_1) == len(sum_h):  # create 2D array
                        j = 0
                        while j in range(len(sum_h)):
                            self.theta_distri_1.append([])
                    self.theta_distri_1[i].append(sum_h[i])

                # print('sum of theta prob:', sum_h)
                idx_h = sum_h.index(max(sum_h))

                H_intent = theta_list[idx_h]
                print('probability of thetas H:', sum_h, 'H intent:', H_intent)
                self.intent_1.append(H_intent)
                self.lambda_1.append(lambda_h)
                self.true_intent_1.append(self.true_params[0][0])
                self.true_noise_1.append(self.true_params[0][1])

    def draw_intent(self):
        # TODO: need to revise the if condition
        joint_infer_m = self.sim.agents[0].predicted_intent_other
        theta_list = self.sim.theta_list
        lambda_list = self.sim.lambda_list
        if not len(joint_infer_m) == 0:
            # print("Lambdas:", self.lambda_h, self.lambda_m)
            # print('true intent:', self.true_intent_h, self.true_intent_m)
            # print('predicted intent:', self.intent_h, self.intent_m)
            # print('intent distribution:', self.intent_distri_h, self.intent_distri_m)

            fig2, (ax1, ax2, ax3, ax4, ax5) = pyplot.subplots(5, figsize=(5, 8))
            fig2.suptitle('Predicted intent and noise of agents')

            ax1.plot(self.intent_1, label='predicted P1 intent')
            ax1.plot(self.true_intent_1, label='true P1 intent', linestyle='--')
            ax1.legend()
            ax1.set_yticks(theta_list)
            ax1.set_yticklabels(['na', 'a'])
            ax1.set(xlabel='frame', ylabel='P1 intent')

            ax2.plot(self.intent_2, label='predicted P2 intent')
            ax2.plot(self.true_intent_2, label='true P2 intent', linestyle='--')
            ax2.legend()
            ax2.set_yticks(theta_list)
            ax2.set_yticklabels(['na', 'a'])
            ax2.set(xlabel='frame', ylabel='P2 intent')

            w = 0.15
            # TODO: generalize for more than two thetas
            x = list(range(0, len(self.intent_1)))
            x1 = [i - w for i in x]
            x2 = [i + w for i in x]
            ax3.bar(x1, self.theta_distri_1[0], width=0.15, label='NA')
            ax3.bar(x2, self.theta_distri_1[1], width=0.15, label='A')
            ax3.legend(loc="lower right")
            ax3.set_yticks([0.25, 0.5, 0.75])
            ax3.set(xlabel='frame', ylabel='P1 distri')

            w = 0.15
            x = list(range(0, len(self.intent_2)))
            x1 = [i - w for i in x]
            x2 = [i + w for i in x]
            ax4.bar(x1, self.theta_distri_2[0], width=0.15, label='NA')
            ax4.bar(x2, self.theta_distri_2[1], width=0.15, label='A')
            ax4.legend(loc='lower right')
            ax4.set_yticks([0.25, 0.5, 0.75])
            ax4.set(xlabel='frame', ylabel='P2 distri')

            "plotting lambdas"
            print('lambdas', self.lambda_1, self.lambda_2)
            ax5.plot(self.lambda_1, label='P1 ration')
            ax5.plot(self.lambda_2, label='P2 ration', linestyle='--')
            ax5.legend()
            ax5.set_yticks(lambda_list)
            ax5.set(xlabel='frame', ylabel='noise')
            print('Predicted P1 intent distri', self.theta_distri_1)
            print('Predicted P2 intent distri', self.theta_distri_2)
        else:
            print("predicted intent H", self.intent_1)
            print("predicted intent for H from AV:", self.sim.agents[0].predicted_intent_self)
            fig2, (ax1, ax2, ax3) = pyplot.subplots(3)
            fig2.suptitle('Predicted intent of H agent')
            ax1.plot(self.intent_1, label='predicted H intent')
            ax1.plot(self.true_intent_1, label='true H intent', linestyle='--')
            ax1.legend()
            # TODO: get actual intent from decision model/ autonomous vehicle
            ax1.set_yticks([1, 1000])
            ax1.set_yticklabels(['na', 'a'])
            ax1.set(xlabel='frame', ylabel='intent')

            w = 0.15
            x = list(range(0, len(self.intent_1)))
            x1 = [i-w for i in x]
            x2 = [i+w for i in x]
            ax2.bar(x1, self.theta_distri_1[0], width=0.15, label='theta 1')
            ax2.bar(x2, self.theta_distri_1[1], width=0.15, label='theta 2')
            ax2.legend()
            ax2.set_yticks([0.15, 0.5, 0.85])
            ax2.set(xlabel='frame', ylabel='probability')

            "plotting lambda h"
            ax3.plot(self.lambda_1, label='predtd H rationality')
            ax3.legend()
            ax3.set_yticks(self.sim.lambda_list)
            ax3.set(xlabel='frame', ylabel='intent')

        # TODO: plot actual distributions
        #pyplot.tight_layout()
        pyplot.show()

    def draw_prob(self):
        """
        drawing probability distribution of future state on pygame surface
        :params:

        :return:
        """

        "colors"
        red = (255, 0, 0)
        orange = (255, 165, 0)
        yellow = (255, 255, 51)
        green = (204, 255, 153)
        blue = (100, 178, 255)
        purple = (0, 100, 255)

        "get state distribution"
        #p_state1 = (0.25, [0, 0, 0, 0])  # [p_state, (sx, sy, vx, vy)]
        #print(self.p_state_H[-1])
        # if self.frame == 0: # or self.frame == 1:
        #     p_state_D, state_list  = self.p_state_H[0]
        # else:
        #     p_state_D, state_list = self.p_state_H[-1]
        self.frame = self.sim.frame
        p_state_D, state_list = self.p_state_1[self.frame]
        #print("PLOTTING: ", state_list, "and ", p_state_D)
        "checking if predicted states are actually reached"
        if not self.frame == 0:
            # TODO: figure out how predicted state and actual state align
            past_predicted_state = self.p_state_1[self.frame - 1][1][0]  # time -> state_list -> agent
            #print("-draw- Last state:" , self.sim.agents[0].state[-1])
            #print("-draw- past predicted states:", past_predicted_state)
            print(past_predicted_state, self.sim.agents[0].state[self.frame])
            assert self.sim.agents[0].state[self.frame] in past_predicted_state

        "unpacking the info"
        for k in range(len(state_list)):  # time steps
            states_k = state_list[k]
            p_state_Dk = p_state_D[k]

            for i in range(len(state_list[0])):
                x, y = states_k[i][0], states_k[i][1]
                #print("X, Y: ", x, y)
                nx, ny = self.c2p((x, y))
                p_s = p_state_Dk[i]
                #TODO: change the range of color! (we will have different distribution)
                #TODO: continuous distribution of color?
                "plot different colors base on their probabilities"
                if p_s > 0.22:
                    pg.draw.circle(self.screen, red, (nx, ny), 6) #(surface, color, pos, radius)
                elif 0.21 < p_s <= 0.22:
                    pg.draw.circle(self.screen, orange, (nx, ny), 6)
                elif 0.20 < p_s <= 0.21:
                    pg.draw.circle(self.screen, yellow, (nx, ny), 6)
                elif 0.19 < p_s <= 0.2:
                    pg.draw.circle(self.screen, green, (nx, ny), 6)
                elif 0.18 < p_s <= 0.19:
                    pg.draw.circle(self.screen, blue, (nx, ny), 6)
                else:
                    pg.draw.circle(self.screen, purple, (nx, ny), 6)

        # pg.draw.circle()

    def plot_loss(self):
        loss = self.sim.past_loss1
        states_1 = self.sim.agents[0].state
        states_2 = self.sim.agents[1].state
        x1 = []
        x2 = []
        for i in range(len(states_1)-1):
            x1.append(states_1[i][1])
            x2.append(states_2[i][0])
        assert len(x1) == len(loss)

        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, loss)
        ax.set_xticks([15, 20, 25, 30, 35, 40, 45])
        ax.set_yticks([15, 20, 25, 30, 35, 40, 45])
        ax.invert_xaxis()
        ax.set_xlabel('P1 location')
        ax.set_ylabel('P2 location')
        ax.set_zlabel('Loss')

        pyplot.show()

    def c2p(self, coordinates):
        x = self.coordinate_scale * (coordinates[0] - self.origin[0] + self.screen_width / 2)
        y = self.coordinate_scale * (- coordinates[1] + self.origin[1] + self.screen_height / 2)
        x = int(
            (x - self.screen_width * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_width * self.coordinate_scale * 0.5)
        y = int(
            (y - self.screen_height * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_height * self.coordinate_scale * 0.5)

        return np.array([x, y])

    def bvp_c2p(self, coordinates):
        """
        coordinates = x, y position in your environment(vehicle position)
        """
        # TODO: fix this
        x = self.coordinate_scale * (- coordinates[0] + self.origin[0] + self.screen_width / 2)
        y = self.coordinate_scale * (- coordinates[1] + self.origin[1] + self.screen_height / 2)
        x = int(
            (x - self.screen_width * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_width * self.coordinate_scale * 0.5)
        y = int(
            (y - self.screen_height * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_height * self.coordinate_scale * 0.5)
        'returns x, y for the pygame window'
        return np.array([x, y])

    # TODO: implement this
    def make_gif(self):
        """
        Saving png and creating gif
        :return:
        """
        path = 'sim_outputs/'

        image = glob.glob(path + "*.png")
        # print(image)
        # episode_step_count = len(image)
        img_list = image  # [path + "img" + str(i).zfill(3) + ".png" for i in range(episode_step_count)]

        images = []
        for filename in img_list:
            images.append(imageio.imread(filename))
        tag = 'theta1' + '=' + str(problem.theta1) + '_' + 'theta2' + '=' + str(problem.theta2) + '_' + 'time horizon' + '=' + str(config.t1)
        imageio.mimsave(path + 'movie_' + tag + '.gif', images, 'GIF', duration=0.2)
        # Delete images
        [os.remove(path + file) for file in os.listdir(path) if ".png" in file]
        return
