# TODO: add uncertainty visualization

import pygame as pg
import pygame.gfxdraw
import numpy as np
import math
from inference_model import InferenceModel
from autonomous_vehicle import AutonomousVehicle
import time

LIGHT_GREY = (230, 230, 230)


class VisUtils:

    def __init__(self, sim):
        "for drawing state distribution"
        self.sim = sim
        self.env = sim.env
        self.p_state_H = sim.agents[1].predicted_policy_other  # get the last prediction


        if sim.decision_type == 'baseline':
            self.screen_width = 50
            self.screen_height = 50
            self.coordinate_scale = 20
            self.zoom = 0.7
            self.asset_location = "assets/"

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

            self.origin = np.array([-15.0, 15.0])
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
        frame = self.sim.frame

        # render 10 times for each step
        steps = 20

        "--dummy data--"
        black = (0, 0, 0)
        #red = (255, 0, 0)
        #r = pg.Color.r
        p_state = (0.25, [0, 0, 0, 0]) #[p_state, (sx, sy, vx, vy)]
        sx = p_state[1][0]
        sy = p_state[1][1]
        #pos1 = (sx, sy)
        pos2 = self.c2p((sx, sy))
        # def draw_circle( pos, color, radius):
        #     pg.draw.circle(self.screen, color, pos, radius) #surface,  color, (x, y),radius>=1
        #draw_circle(pos1, red, 10)
        #print('IMPORTED p state: ', self.p_state_H[-1])
        "--end of dummy data--"
        if frame == 0:
            print("do nothing")
            for i in range(self.sim.n_agents):
                pos = np.array(self.sim.agents[i].state[frame][:2])  # get 0 and 1 element (not include 2)
                "smooth out the movement between each step"
                #pos = pos_old * (1 - k * 1. / steps) + pos_new * (k * 1. / steps)
                "transform pos"
                pixel_pos_car = self.c2p(pos)
                size_car = self.car_image[i].get_size()
                self.screen.blit(self.car_image[i],
                                (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
            "drawing the map of state distribution"
            pg.draw.circle(self.screen, (255, 255, 255), pos2, 10)  # surface,  color, (x, y),radius>=1
            self.draw_prob()  # calling function to draw with data from inference

            pg.display.flip()
            pg.display.update()
        else:
            for k in range(1, steps + 1):
                self.screen.fill((255, 255, 255))
                self.draw_axes()
                # Draw Images
                for i in range(self.sim.n_agents):
                    "getting pos of agent"
                    pos_old = np.array(self.sim.agents[i].state[frame - 1][:2])
                    pos_new = np.array(self.sim.agents[i].state[frame][:2])  # get 0 and 1 element (not include 2)
                    "smooth out the movement between each step"
                    pos = pos_old * (1 - k * 1. / steps) + pos_new * (k * 1. / steps)
                    "transform pos"
                    pixel_pos_car = self.c2p(pos)
                    size_car = self.car_image[i].get_size()
                    self.screen.blit(self.car_image[i],
                                     (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
                    if self.sim.decision_type == "baseline":
                        time.sleep(0.2)
                # # Annotations
                # font = pg.font.SysFont("Arial", 30)

                "drawing the map of state distribution"
                pg.draw.circle(self.screen, (255, 255, 255), pos2, 10)  # surface,  color, (x, y),radius>=1
                self.draw_prob() #calling function to draw with data from inference

                pg.display.flip()
                pg.display.update()



    def draw_axes(self):
        # draw lanes based on environment TODO: lanes are defined as bounds of agent state spaces, need to generalize
        for a in self.env.bounds:
            bound_x, bound_y = a[0], a[1]
            if bound_x:
                b_min, b_max = bound_x[0], bound_x[1]
                _bound1 = self.c2p((b_min, 0))
                _bound2 = self.c2p((b_max, 0))
                bounds = np.array([_bound1[0], _bound2[0]])
                pg.draw.line(self.screen, LIGHT_GREY, ((bounds[1] + bounds[0])/2, 0),
                             ((bounds[1] + bounds[0])/2, self.screen_height * self.coordinate_scale,
                              ), bounds[1] - bounds[0])
            if bound_y:
                b_min, b_max = bound_y[0], bound_y[1]
                _bound1 = self.c2p((0, b_min))
                _bound2 = self.c2p((0, b_max))
                bounds = np.array([_bound1[1], _bound2[1]])
                pg.draw.line(self.screen, LIGHT_GREY, (0, (bounds[1] + bounds[0]) / 2),
                             (self.screen_width * self.coordinate_scale,
                              (bounds[1] + bounds[0]) / 2), bounds[0] - bounds[1])

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
        p_state1 = (0.25, [0, 0, 0, 0])  # [p_state, (sx, sy, vx, vy)]
        p_state_D, state_list  = self.p_state_H[-1]
        #print("PLOTTING: ", state_list, "and ", p_state_D)

        "unpacking the info"
        for k in range(len(state_list)): #time steps
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
        y = self.coordinate_scale * (-coordinates[1] + self.origin[1] + self.screen_height / 2)
        x = int(
            (x - self.screen_width * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_width * self.coordinate_scale * 0.5)
        y = int(
            (y - self.screen_height * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_height * self.coordinate_scale * 0.5)
        return np.array([x, y])
