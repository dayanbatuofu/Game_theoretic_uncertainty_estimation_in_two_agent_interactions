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
import math
from inference_model import InferenceModel
from autonomous_vehicle import AutonomousVehicle
import time
import pdb

LIGHT_GREY = (230, 230, 230)
RED = (230, 0 ,0)


class VisUtils:

    def __init__(self, sim):
        "for drawing state distribution"

        self.sim = sim
        self.env = sim.env
        self.drawing_prob = sim.drawing_prob
        if self.drawing_prob:
            self.p_state_H = sim.agents[1].predicted_states_other  # get the last prediction
            self.p_state_M = sim.agents[1].predicted_states_self
            self.past_state_h = sim.agents[1].state[-1]
            self.past_state_m = sim.agents[0].state[-1]
            self.intent_h = []
            self.intent_m = []
            self.intent_distri_h = [[], []]  # theta1, theta2
            self.intent_distri_m = [[], []]  # theta1, theta2
            self.lambda_h = []
            self.lambda_m = []
            self.true_params = self.sim.true_params
            self.true_intent_h = []
            self.true_intent_m = []
            self.true_noise_h = []
            self.true_noise_m = []
        self.frame = sim.frame
        self.dist = []
        self.sleep_between_step = False


        if not sim.decision_type_h == 'constant_speed' and not sim.decision_type_m == 'constant_speed':
            self.sleep_between_step = True
            self.screen_width = 10  # 50
            self.screen_height = 10  # 50
            self.coordinate_scale = 80
            self.zoom = 0.16
            self.asset_location = "assets/"
            self.fps = 24  # max framework

            img_width = int(self.env.car_width * self.coordinate_scale * self.zoom)
            img_height = int(self.env.car_length * self.coordinate_scale * self.zoom)

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
                              for i in range(self.sim.n_agents)]

            #self.origin = np.array([-15.0, 15.0])
            self.origin = np.array([0, 0])

        else:
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

            img_width = int(self.env.car_width * self.coordinate_scale * self.zoom)
            img_height = int(self.env.car_length * self.coordinate_scale * self.zoom)

            "loading car image into pygame"
            self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location
                                                                                   + self.sim.agents[i].car_par[
                                                                                       "sprite"]),
                                                                     (img_width, img_height)),
                                                  -self.sim.agents[i].car_par["orientation"])
                              for i in range(self.sim.n_agents)]

            self.origin = np.array([1.0, -1.0])

        "Draw Axis Lines"
        self.screen.fill((255, 255, 255))
        self.draw_axes() #calling draw axis function
        pg.display.flip()
        pg.display.update()

    def draw_frame(self):
        # Draw the current frame
        self.frame = self.sim.frame
        frame = self.sim.frame

        # render 10 times for each step
        steps = 20

        for k in range(1, steps + 1):
            self.screen.fill((255, 255, 255))
            self.draw_axes()
            # Draw Images
            for i in range(self.sim.n_agents):
                "getting pos of agent"
                pos_old = np.array(self.sim.agents[i].state[frame][:2])
                pos_new = np.array(self.sim.agents[i].state[frame+1][:2])  # get 0 and 1 element (not include 2)
                "smooth out the movement between each step"
                pos = pos_old * (1 - k * 1. / steps) + pos_new * (k * 1. / steps)
                "transform pos"
                pixel_pos_car = self.c2p(pos)
                size_car = self.car_image[i].get_size()
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
            pos_h, speed_h = self.sim.agents[0].state[-1][1], self.sim.agents[0].state[-1][3]
            label = font.render("Car 1 position and speed: (%5.4f , %5.4f)" % (pos_h, speed_h), 1,
                                (0, 0, 0))
            self.screen.blit(label, (label_x, label_y))
            pos_m, speed_m = self.sim.agents[1].state[-1][0], self.sim.agents[1].state[-1][2]
            label = font.render("Car 2 position and speed: (%5.4f , %5.4f)" % (pos_m, speed_m), 1,
                                (0, 0, 0))
            self.screen.blit(label, (label_x, label_y + label_y_offset))
            action1, action2 = self.sim.agents[0].action[-1], self.sim.agents[1].action[-1]
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
        # if frame == 0:
        #     for i in range(self.sim.n_agents):
        #         pos = np.array(self.sim.agents[i].state[frame][:2])  # get 0 and 1 element (not include 2)
        #         "smooth out the movement between each step"
        #         #pos = pos_old * (1 - k * 1. / steps) + pos_new * (k * 1. / steps)
        #         "transform pos"
        #         pixel_pos_car = self.c2p(pos)
        #         size_car = self.car_image[i].get_size()
        #         self.screen.blit(self.car_image[i],
        #                         (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
        #     "drawing the map of state distribution"
        #     pg.draw.circle(self.screen, (255, 255, 255), pos2, 10)  # surface,  color, (x, y),radius>=1
        #     if self.drawing_prob:
        #         self.draw_prob()  # calling function to draw with data from inference
        #
        #     # Annotations
        #     # font = pg.font.SysFont("Arial", 30)
        #     font = pg.font.SysFont("Arial", 15)
        #     screen_w, screen_h = self.screen.get_size()
        #     label_x = screen_w - 800
        #     label_y = 260
        #     label_y_offset = 30
        #     #TODO: length of state/action is mismatched from frame: frame behind by 1
        #     pos_h, speed_h = self.sim.agents[0].state[-1][1], self.sim.agents[0].state[-1][3]
        #     label = font.render("Car 1 position and speed: (%5.4f , %5.4f)" % (pos_h, speed_h), 1,
        #                         (0, 0, 0))
        #     self.screen.blit(label, (label_x, label_y))
        #     pos_m, speed_m = self.sim.agents[1].state[-1][0], self.sim.agents[1].state[-1][2]
        #     label = font.render("Car 2 position and speed: (%5.4f , %5.4f)" % (pos_m, speed_m), 1,
        #                         (0, 0, 0))
        #     self.screen.blit(label, (label_x, label_y+label_y_offset))
        #     action1, action2 = self.sim.agents[0].action[-1], self.sim.agents[1].action[-1]
        #     label = font.render("Car 1 action: (%5.4f)" % action1, 1, (0, 0, 0))
        #     self.screen.blit(label, (label_x, label_y + 2*label_y_offset))
        #     label = font.render("Car 2 action: (%5.4f)" % action2, 1, (0, 0, 0))
        #     self.screen.blit(label, (label_x, label_y + 3*label_y_offset))
        #     label = font.render("Frame: %i" % self.sim.frame, 1, (0, 0, 0))
        #     self.screen.blit(label, (10, 10))
        #     pg.display.flip()
        #     pg.display.update()
        # else:
        #     for k in range(1, steps + 1):
        #         self.screen.fill((255, 255, 255))
        #         self.draw_axes()
        #         # Draw Images
        #         for i in range(self.sim.n_agents):
        #             "getting pos of agent"
        #             pos_old = np.array(self.sim.agents[i].state[frame - 1][:2])
        #             pos_new = np.array(self.sim.agents[i].state[frame][:2])  # get 0 and 1 element (not include 2)
        #             "smooth out the movement between each step"
        #             pos = pos_old * (1 - k * 1. / steps) + pos_new * (k * 1. / steps)
        #             "transform pos"
        #             pixel_pos_car = self.c2p(pos)
        #             size_car = self.car_image[i].get_size()
        #             self.screen.blit(self.car_image[i],
        #                              (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
        #             if self.sleep_between_step:
        #                 time.sleep(0.03)
        #         # Annotations
        #         # font = pg.font.SysFont("Arial", 30)
        #         font = pg.font.SysFont("Arial", 15)
        #         screen_w, screen_h = self.screen.get_size()
        #         label_x = screen_w - 800
        #         label_y = 260
        #         label_y_offset = 30
        #         pos_h, speed_h = self.sim.agents[0].state[-1][1], self.sim.agents[0].state[-1][3]
        #         label = font.render("Car 1 position and speed: (%5.4f , %5.4f)" % (pos_h, speed_h), 1,
        #                             (0, 0, 0))
        #         self.screen.blit(label, (label_x, label_y))
        #         pos_m, speed_m = self.sim.agents[1].state[-1][0], self.sim.agents[1].state[-1][2]
        #         label = font.render("Car 2 position and speed: (%5.4f , %5.4f)" % (pos_m, speed_m), 1,
        #                             (0, 0, 0))
        #         self.screen.blit(label, (label_x, label_y+ label_y_offset))
        #         action1, action2 = self.sim.agents[0].action[-1], self.sim.agents[1].action[-1]
        #         label = font.render("Car 1 action: (%5.4f)" % action1, 1, (0, 0, 0))
        #         self.screen.blit(label, (label_x, label_y + 2*label_y_offset))
        #         label = font.render("Car 2 action: (%5.4f)" % action2, 1, (0, 0, 0))
        #         self.screen.blit(label, (label_x, label_y + 3*label_y_offset))
        #         label = font.render("Frame: %i" % self.sim.frame, 1, (0, 0, 0))
        #         self.screen.blit(label, (10, 10))
        #
        #         "drawing the map of state distribution"
        #         #pg.draw.circle(self.screen, (255, 255, 255), pos2, 10)  # surface,  color, (x, y),radius>=1  # test
        #         if self.drawing_prob:
        #             self.draw_prob() #calling function to draw with data from inference
        #
        #         pg.display.flip()
        #         pg.display.update()

        self.calc_dist()
        if self.drawing_prob:
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

    def draw_dist(self):
        # TODO: implement plotting of distance between cars over time
        # pyplot.plot(self.dist)
        # pyplot.plot(self.sim.agents[0].action)
        fig1, (ax1, ax2, ax3) = pyplot.subplots(3) #3 rows
        fig1.suptitle('Euclidean distance and Agent Actions')
        ax1.plot(self.dist, label='car dist')
        ax1.legend()
        ax1.set(xlabel='time', ylabel='distance')

        #fig1, (ax2) = pyplot.subplots(1)
        #fig1.suptitle('Actions of H at each time')
        ax2.plot(self.sim.agents[0].action, label='actual')
        ax2.plot(self.sim.agents[1].predicted_actions_other, label='predicted', linestyle='--')
        ax2.set_ylim([-10, 10])
        ax2.set_yticks([-8, -4, 0, 4, 8])
        ax2.legend()
        ax2.set(xlabel='time', ylabel='H actions')

        #fig1, (ax3) = pyplot.subplots(1)
        #fig1.suptitle('Actions of M at each time')
        ax3.plot(self.sim.agents[1].action, label='actual')
        ax3.plot(self.sim.agents[1].predicted_actions_self, label='predicted', linestyle='--')
        ax3.set_ylim([-10, 10])
        ax3.set_yticks([-8, -4, 0, 4, 8])
        ax3.legend()
        ax3.set(xlabel='time', ylabel='M actions')
        # pyplot.ylabel("distance")
        # pyplot.xlabel("time")

        pyplot.show()

    def calc_dist(self):
        past_state_h = self.sim.agents[0].state[-1]
        past_state_m = self.sim.agents[1].state[-1]
        dist_h = past_state_h[1]
        dist_m = past_state_m[0]
        dist = np.sqrt(dist_h * dist_h + dist_m * dist_m)
        self.dist.append(dist)

    def calc_intent(self):
        # TODO: use common knowledge instead
        common_belief = self.sim.agents[1].predicted_intent_all
        theta_list = self.sim.theta_list
        lambda_list = self.sim.lambda_list
        if self.sim.sharing_belief:  # both agents uses same belief from empathetic inference
            beta_h, beta_m = self.sim.agents[1].predicted_intent_all[-1][1]
            self.intent_h.append(beta_h[0])
            self.intent_m.append(beta_m[0])
            self.lambda_h.append(beta_h[1])
            self.lambda_m.append(beta_m[1])

            joint_infer_m = self.sim.agents[1].predicted_intent_self
            p_joint_h, lambda_h = self.sim.agents[1].predicted_intent_other[-1]
            p_joint_m, lambda_m = joint_infer_m[-1]
            sum_h = p_joint_h.sum(axis=0)
            sum_h = np.ndarray.tolist(sum_h)
            sum_m = p_joint_m.sum(axis=0)
            sum_m = np.ndarray.tolist(sum_m)  # [theta1, theta2]
            for i in range(len(sum_h)):
                if not len(self.intent_distri_h) == len(sum_h):  # create 2D array
                    j = 0
                    while j in range(len(sum_h)):
                        self.intent_distri_h.append([])
                        self.intent_distri_m.append([])
                self.intent_distri_h[i].append(sum_h[i])
                self.intent_distri_m[i].append(sum_m[i])
            self.true_intent_h.append(self.true_params[0][0])
            self.true_noise_h.append(self.true_params[0][1])
            self.true_intent_m.append(self.true_params[1][0])
            self.true_noise_m.append(self.true_params[1][1])

        else:
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
                    if not len(self.intent_distri_h) == len(sum_h):  # create 2D array
                        j = 0
                        while j in range(len(sum_h)):
                            self.intent_distri_h.append([])
                            self.intent_distri_m.append([])
                    self.intent_distri_h[i].append(sum_h[i])
                    self.intent_distri_m[i].append(sum_m[i])
                H_intent = theta_list[idx_h]
                M_intent = theta_list[idx_m]
                self.intent_h.append(H_intent)
                self.intent_m.append(M_intent)
                self.lambda_h.append(lambda_h)
                self.lambda_m.append(lambda_m)
            else:
                p_joint_h, lambda_h = self.sim.agents[1].predicted_intent_other[-1]
                # print("-draw- p_joint_h: ", p_joint_h)
                sum_h = p_joint_h.sum(axis=0)
                sum_h = np.ndarray.tolist(sum_h)
                for i in range(len(sum_h)):
                    if not len(self.intent_distri_h) == len(sum_h):  # create 2D array
                        j = 0
                        while j in range(len(sum_h)):
                            self.intent_distri_h.append([])
                    self.intent_distri_h[i].append(sum_h[i])

                # print('sum of theta prob:', sum_h)
                idx_h = sum_h.index(max(sum_h))
                # TODO: assign list of theta and lambda somewhere in sim
                H_intent = theta_list[idx_h]
                print('probability of thetas H:', sum_h, 'H intent:', H_intent)
                self.intent_h.append(H_intent)
                self.lambda_h.append(lambda_h)

    def draw_intent(self):
        joint_infer_m = self.sim.agents[1].predicted_intent_self
        print(joint_infer_m)
        if not len(joint_infer_m) == 0:
            print("Lambdas:", self.lambda_h, self.lambda_m)
            fig2, (ax1, ax2, ax3, ax4, ax5) = pyplot.subplots(5, figsize=(5, 8))
            fig2.suptitle('Predicted intent and rationality')

            ax1.plot(self.intent_h, label='predicted H intent')
            ax1.plot(self.true_intent_h, label='true H intent', linestyle='--')
            ax1.legend()
            ax1.set_yticks(self.sim.theta_list)
            ax1.set_yticklabels(['na', 'a'])
            ax1.set(xlabel='time', ylabel='intent')

            ax2.plot(self.intent_m, label='predicted M intent')
            ax2.plot(self.true_intent_m, label='true M intent', linestyle='--')
            ax2.legend()
            ax2.set_yticks(self.sim.theta_list)
            ax2.set_yticklabels(['na', 'a'])
            ax2.set(xlabel='time', ylabel='intent')

            w = 0.15
            # TODO: generalize for more than two thetas
            x = list(range(0, len(self.intent_h)))
            x1 = [i - w for i in x]
            x2 = [i + w for i in x]
            ax3.bar(x1, self.intent_distri_h[0], width=0.15, label='theta 1')
            ax3.bar(x2, self.intent_distri_h[1], width=0.15, label='theta 2')
            ax3.legend(loc="lower right")
            ax3.set_yticks([0.25, 0.5, 0.75])
            # for i, v in enumerate(self.intent_distri_h[0]):
            #     ax3.text(v + 0.05, i + 0.2, str(v), color='blue', fontweight='bold')
            ax3.set(xlabel='time', ylabel='H distri')

            w = 0.15
            x = list(range(0, len(self.intent_m)))
            x1 = [i - w for i in x]
            x2 = [i + w for i in x]
            ax4.bar(x1, self.intent_distri_m[0], width=0.15, label='theta 1')
            ax4.bar(x2, self.intent_distri_m[1], width=0.15, label='theta 2')
            ax4.legend(loc='lower right')
            ax4.set_yticks([0.25, 0.5, 0.75])
            ax4.set(xlabel='time', ylabel='M distri')

            "plotting lambdas"
            ax5.plot(self.lambda_h, label='H ration')
            ax5.plot(self.lambda_m, label='M ration', linestyle='--')
            ax5.legend()
            ax5.set_yticks(self.sim.lambda_list)
            ax5.set(xlabel='time', ylabel='intent')

        else:
            print("predicted intent H", self.intent_h)
            print("predicted intent for H from AV:", self.sim.agents[1].predicted_intent_other)
            fig2, (ax1, ax2, ax3) = pyplot.subplots(3)
            fig2.suptitle('Predicted intent of H agent')
            ax1.plot(self.intent_h, label='predicted H intent')
            ax1.plot(self.true_intent_h, label='true H intent', linestyle='--')
            ax1.legend()
            #TODO: get actual intent from decision model/ autonomous vehicle
            ax1.set_yticks([1, 1000])
            ax1.set_yticklabels(['na', 'a'])
            ax1.set(xlabel='time', ylabel='intent')

            w = 0.15
            #TODO: generalize for more than two thetas
            x = list(range(0, len(self.intent_h)))
            x1 = [i-w for i in x]
            x2 = [i+w for i in x]
            ax2.bar(x1, self.intent_distri_h[0], width=0.15, label='theta 1')
            ax2.bar(x2, self.intent_distri_h[1], width=0.15, label='theta 2')
            ax2.legend()
            ax2.set_yticks([0.25, 0.5, 0.75])
            ax2.set(xlabel='time', ylabel='probability')

            "plotting lambda h"
            ax3.plot(self.lambda_h, label='predtd H rationality')
            ax3.legend()
            ax3.set_yticks(self.sim.lambda_list)
            ax3.set(xlabel='time', ylabel='intent')

        #TODO: plot actual distributions
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
        p_state_D, state_list = self.p_state_H[self.frame]
        #print("PLOTTING: ", state_list, "and ", p_state_D)
        "checking if predicted states are actually reached"
        if not self.frame == 0:
            # TODO: figure out how predicted state and actual state align
            past_predicted_state = self.p_state_H[self.frame-1][1][0]  # time -> state_list -> agent
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

        #pg.draw.circle()

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