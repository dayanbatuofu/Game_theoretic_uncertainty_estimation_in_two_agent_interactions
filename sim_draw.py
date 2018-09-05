from constants import CONSTANTS as C
import pygame as pg
import numpy as np
import math
import os

BLACK       = (  0,  0,  0)
DARK_BLUE   = (100,100,200)
DARK_RED  = (200,100,100)
LIGHT_BLUE = (200,200,255)
LIGHT_RED = (255,200,200)
LIGHT_LIGHT_GREY  = (230,230,230)
DARK_GREY   = (100,100,100)
LIGHT_GREY  = (200,200,200)
MAGENTA     = (255,  0,255)
TEAL        = (  0,255,255)
GREEN       = (  0,255,  0)

class Sim_Draw():

    # BLACK = (0, 0, 0)
    # DARK_GREY = (0, 0, 0)
    # LIGHT_GREY = (200, 200, 200)
    MAGENTA = (255, 0, 255)
    TEAL = (0, 255, 255)
    GREEN = (0, 255, 0)

    def __init__(self, parameters, asset_loc):

        self.P = parameters

        pg.init()
        self.screen = pg.display.set_mode((self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE))

        # self.car2_image = pg.transform.rotate(pg.transform.scale(pg.image.load(asset_loc + "red_car_sized.png"),
        #                                                           (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
        #                                                            int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))), -self.P.CAR_2.ORIENTATION)
        self.car2_image = pg.transform.rotate(pg.image.load(asset_loc + "blue_car_sized.png"), -self.P.CAR_2.ORIENTATION)

        self.car1_image = pg.transform.rotate(pg.image.load(asset_loc + "red_car_sized.png"), self.P.CAR_1.ORIENTATION)
        self.coordinates_image = pg.image.load(asset_loc + "coordinates.png")
        self.origin = np.array([0, 0])

    def draw_frame(self, sim_data, car_num_display, frame):

        # Draw the current frame
        self.screen.fill((255, 255, 255))

        # self.origin = sim_data.car1_states[frame]

        # Draw Axis Lines
        self.draw_axes()

        coordinates_size = self.coordinates_image.get_size()
        self.screen.blit(self.coordinates_image, (10, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE - coordinates_size[1] - 10 / 2))

        # Draw Images
        pixel_pos_car_1 = self.c2p(sim_data.car1_states[frame])
        size_car_1 = self.car1_image.get_size()
        self.screen.blit(self.car1_image, (pixel_pos_car_1[0] - size_car_1[0] / 2, pixel_pos_car_1[1] - size_car_1[1] / 2))

        pixel_pos_car_2 = self.c2p(sim_data.car2_states[frame])
        size_car_2 = self.car2_image.get_size()
        self.screen.blit(self.car2_image, (pixel_pos_car_2[0] - size_car_2[0] / 2, pixel_pos_car_2[1] - size_car_2[1] / 2))

        # Annotations

        font = pg.font.SysFont("Arial", 15)
        label = font.render("Car 1: (%5.4f , %5.4f)" % (sim_data.car1_states[frame][0], sim_data.car1_states[frame][1]),
                            1, GREEN)
        self.screen.blit(label, (10, 10))

        label = font.render("Car 2: (%5.4f , %5.4f)" % (sim_data.car2_states[frame][0], sim_data.car2_states[frame][1]),
                            1, GREEN)
        self.screen.blit(label, (10, 30))

        label = font.render("Frame: %i" % (frame + 1), 1, GREEN)
        self.screen.blit(label, (10, 50))

        pg.display.flip()
        pg.display.update()

    def draw_axes(self):
        rel_coor_scale = C.COORDINATE_SCALE * C.ZOOM
        rel_screen_width = self.P.SCREEN_WIDTH / C.ZOOM
        rel_screen_height = self.P.SCREEN_HEIGHT / C.ZOOM

        spacing = int(C.AXES_SHOW * rel_coor_scale)
        offset_x = int(math.fmod(self.origin[1] * rel_coor_scale, spacing)) + int(math.fmod(rel_screen_width * rel_coor_scale/2,spacing))
        offset_y = int(math.fmod(self.origin[0] * rel_coor_scale, spacing)) + int(math.fmod(rel_screen_height * rel_coor_scale/2,spacing))

        distance_x = int((self.origin[1] * rel_coor_scale) / spacing)
        distance_y = int((self.origin[0] * rel_coor_scale) / spacing)

        num_vaxes = int(rel_screen_width * rel_coor_scale / spacing) + 1
        num_haxes = int(rel_screen_height * rel_coor_scale / spacing) + 1

        font = pg.font.SysFont("Arial", 15)

        # # Vertical
        # for i in range(num_vaxes):
        #     pg.draw.line(self.screen, LIGHT_LIGHT_GREY, (offset_x + i * spacing, 0),
        #                  (offset_x + i * spacing, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), 1)
        #     # label = (distance_x + 1 + i) * C.AXES_SHOW - rel_screen_width/2
        #     # text = font.render("%3.2f" % label, 1, GREY)
        #     # self.screen.blit(text, (10 + offset_x + (i * spacing), 10))
        #
        # # Horizontal
        # for i in range(num_haxes):
        #     pg.draw.line(self.screen, LIGHT_LIGHT_GREY, (0, offset_y + i * spacing),
        #                  (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, offset_y + i * spacing), 1)
        #     # label = (distance_y + 1 + i) * C.AXES_SHOW - rel_screen_height/2
        #     # text = font.render("%3.2f" % label, 1, GREY)
        #     # self.screen.blit(text, (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE - 30, 10 + offset_y + (self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE) - (i * spacing)))

        # Bounds
        if self.P.BOUND_HUMAN_X is not None:
            _bound1 = self.c2p((self.P.BOUND_HUMAN_X[0], 0))
            _bound2 = self.c2p((self.P.BOUND_HUMAN_X[1], 0))
            bounds = np.array([_bound1[1], _bound2[1]])
            # pg.draw.line(self.screen, BLACK, (0, bounds[0]), (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, bounds[0]), 2)
            pg.draw.line(self.screen, LIGHT_LIGHT_GREY, (0, (bounds[1] + bounds[0])/2),
                         (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, (bounds[1] + bounds[0])/2), bounds[0] - bounds[1])

        if self.P.BOUND_HUMAN_Y is not None:
            _bound1 = self.c2p((0, self.P.BOUND_HUMAN_Y[0]))
            _bound2 = self.c2p((0, self.P.BOUND_HUMAN_Y[1]))
            bounds = np.array([_bound1[0], _bound2[0]])
            # pg.draw.line(self.screen, BLACK, (bounds[0], 0), (bounds[0], self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), 2)
            pg.draw.line(self.screen, LIGHT_LIGHT_GREY, ((bounds[1] + bounds[0])/2, 0),
                         ((bounds[1] + bounds[0])/2, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), bounds[1] - bounds[0])

        if self.P.BOUND_MACHINE_X is not None:
            _bound1 = self.c2p((self.P.BOUND_MACHINE_X[0], 0))
            _bound2 = self.c2p((self.P.BOUND_MACHINE_X[1], 0))
            bounds = np.array([_bound1[1], _bound2[1]])
            # pg.draw.line(self.screen, BLACK, (0, bounds[0]), (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, bounds[0]), 2)
            pg.draw.line(self.screen, LIGHT_LIGHT_GREY, (0, (bounds[1] + bounds[0])/2),
                         (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, (bounds[1] + bounds[0])/2), bounds[0] - bounds[1])

        if self.P.BOUND_MACHINE_Y is not None:
            _bound1 = self.c2p((0, self.P.BOUND_MACHINE_Y[0]))
            _bound2 = self.c2p((0, self.P.BOUND_MACHINE_Y[1]))
            bounds = np.array([_bound1[0], _bound2[0]])
            # pg.draw.line(self.screen, BLACK, (bounds[0], 0), (bounds[0], self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), 2)
            pg.draw.line(self.screen, LIGHT_LIGHT_GREY, ((bounds[1] + bounds[0])/2, 0),
                         ((bounds[1] + bounds[0])/2, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), bounds[1] - bounds[0])

    def c2p(self, coordinates):
        x = C.COORDINATE_SCALE * (coordinates[1] - self.origin[1] + self.P.SCREEN_WIDTH / 2)
        y = C.COORDINATE_SCALE * (-coordinates[0] + self.origin[0] + self.P.SCREEN_HEIGHT / 2)
        x = int(
            (x - self.P.SCREEN_WIDTH * C.COORDINATE_SCALE * 0.5) * C.ZOOM + self.P.SCREEN_WIDTH * C.COORDINATE_SCALE * 0.5)
        y = int((
                y - self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE * 0.5) * C.ZOOM + self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE * 0.5)
        return np.array([x, y])