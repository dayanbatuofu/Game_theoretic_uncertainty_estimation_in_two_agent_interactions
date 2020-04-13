# TODO: add uncertainty visualization

import pygame as pg
import pygame.gfxdraw
import numpy as np
import math


LIGHT_GREY = (230, 230, 230)


class VisUtils:

    def __init__(self, sim):
        self.screen_width = 5
        self.screen_height = 5
        self.asset_location = "assets/"
        self.fps = 24  # max framework
        self.coordinate_scale = 100
        self.zoom = 0.3

        self.sim = sim
        self.env = sim.env
        pg.init()
        self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                           self.screen_height * self.coordinate_scale))

        img_width = int(self.env.car_width * self.coordinate_scale * self.zoom)
        img_height = int(self.env.car_length * self.coordinate_scale * self.zoom)
        self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location
                                              + self.sim.agents[i].car_par["sprite"]), (img_width, img_height)),
                                              -self.sim.agents[i].car_par["orientation"])
                          for i in range(self.sim.n_agents)]

        self.origin = np.array([1.0, -1.0])
        # Draw Axis Lines
        self.screen.fill((255, 255, 255))
        self.draw_axes()
        pg.display.flip()
        pg.display.update()

    def draw_frame(self):
        # Draw the current frame
        frame = self.sim.frame

        # render 10 times for each step
        steps = 10
        for k in range(1, steps+1):
            self.screen.fill((255, 255, 255))
            self.draw_axes()
            # Draw Images
            for i in range(self.sim.n_agents):
                pos_old = np.array(self.sim.agents[i].state[frame-1][:2])
                pos_new = np.array(self.sim.agents[i].state[frame][:2])
                pos = pos_old * (1 - k * 1./steps) + pos_new * (k * 1./steps)
                pixel_pos_car = self.c2p(pos)
                size_car = self.car_image[i].get_size()
                self.screen.blit(self.car_image[i],
                                 (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))

            # # Annotations
            # font = pg.font.SysFont("Arial", 30)

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

    def c2p(self, coordinates):
        x = self.coordinate_scale * (coordinates[0] - self.origin[0] + self.screen_width / 2)
        y = self.coordinate_scale * (-coordinates[1] + self.origin[1] + self.screen_height / 2)
        x = int(
            (x - self.screen_width * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_width * self.coordinate_scale * 0.5)
        y = int((
                y - self.screen_height * self.coordinate_scale * 0.5) * self.zoom +
                self.screen_height * self.coordinate_scale * 0.5)
        return np.array([x, y])
