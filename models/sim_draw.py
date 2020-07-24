from constants import CONSTANTS as C
import pygame as pg
import numpy as np
import math

BLACK       = (  0,  0,  0)
DARK_CAR1   = (50,50,50)
DARK_CAR2  = (150,150,150)
LIGHT_CAR1 = (50,50,50)
LIGHT_CAR2 = (150,150,150)
LIGHT_LIGHT_GREY  = (230, 230, 230)
GREEN = (0, 0, 0)
YELLOW = (232, 145, 26)

DARK_GREY   = (100,100,100)
LIGHT_GREY  = (200,200,200)
MAGENTA     = (255,  0,255)
TEAL        = (  0,255,255)

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

        self.car2_image = pg.transform.rotate(pg.image.load(asset_loc + "red_car_sized.png"), -self.P.CAR_2.ORIENTATION)
        # self.car2_image = pg.transform.rotozoom(pg.image.load(asset_loc + "red_car_sized.png"),
        #                                         -self.P.CAR_2.ORIENTATION, 0.4)

        self.car1_image = pg.transform.rotate(pg.image.load(asset_loc + "blue_car_sized.png"), self.P.CAR_1.ORIENTATION)
        # self.car1_image = pg.transform.rotozoom(pg.image.load(asset_loc + "blue_car_sized.png"),
        #                                         -self.P.CAR_1.ORIENTATION, 0.4)
        self.coordinates_image = pg.image.load(asset_loc + "coordinates.png")
        # self.origin = np.array([-5.0, 5.0])
        self.origin = np.array([-15.0, 15.0])

    def draw_frame(self, env):

        # Draw the current frame
        self.screen.fill((255, 255, 255))

        # Draw Axis Lines
        self.draw_axes()
        ego_car_state = np.array([env.ego_car.state[0], env.ego_car.state[1]])
        ego_car_action = env.ego_car.action
        other_car_state = np.array([env.other_car.state[0], env.other_car.state[1]])
        other_car_action = env.other_car.action
        # Draw Images
        pixel_pos_car_1 = self.c2p([-ego_car_state[0], 0])
        size_car_1 = self.car1_image.get_size()
        # img_width = int(C.CAR_WIDTH * C.COORDINATE_SCALE)
        # img_height = int(C.CAR_LENGTH * C.COORDINATE_SCALE)
        img_width = int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM)
        img_height = int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM)
        self.screen.blit(pg.transform.scale(self.car1_image, (img_width, img_height)),
                         (pixel_pos_car_1[0] - img_width / 2, pixel_pos_car_1[1] - img_height / 2))

        pixel_pos_car_2 = self.c2p([0, other_car_state[0]])
        size_car_2 = self.car2_image.get_size()
        self.screen.blit(pg.transform.scale(self.car2_image, (img_height, img_width)),
                         (pixel_pos_car_2[0] - img_height / 2, pixel_pos_car_2[1] - img_width / 2))

        coordinates_size = self.coordinates_image.get_size()
        self.screen.blit(self.coordinates_image, (10, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE - coordinates_size[1] - 10 / 2))

        # Annotations
        font = pg.font.SysFont("Arial", 15)
        screen_w, screen_h = self.screen.get_size()
        label_x = screen_w - 325 # 200
        label_y = 260
        label_y_offset = 30
        label = font.render("Car 1 position and speed: (%5.4f , %5.4f)" % (ego_car_state[0], ego_car_state[1]), 1, (0, 0, 0))
        self.screen.blit(label, (label_x, label_y))
        label = font.render("Car 1 action: (%5.4f)" % ego_car_action, 1, (0, 0, 0))
        self.screen.blit(label, (label_x, label_y + label_y_offset))

        label = font.render("Car 2 position and speed: (%5.4f , %5.4f)" % (other_car_state[0], other_car_state[1]), 1, (0, 0, 0))
        self.screen.blit(label, (label_x, label_y + 2 * label_y_offset))
        label = font.render("Car 2 action: (%5.4f)" % other_car_action, 1, (0, 0, 0))
        self.screen.blit(label, (label_x, label_y + 3 * label_y_offset))

        label = font.render("isCollision: {}".format(env.collision), 1,
                            (0, 0, 0))
        self.screen.blit(label, (label_x, label_y + 4 * label_y_offset))

        # label = font.render("Lack of Courtesy: %1.4f" % (np.sum(sim_data.car1_gracefulness)/C.PARAMETERSET_2.CAR_1.ABILITY * 0.002), 1, (0, 0, 0))
        # self.screen.blit(label, (350, 10))

        label = font.render("Frame: %i" % (env.frame), 1, (0, 0, 0))
        self.screen.blit(label, (10, 10))
        import time
        time.sleep(0.25)
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

    def c2p_nozoom(self, coordinates):
        x = C.COORDINATE_SCALE * (coordinates[1] - self.origin[1] + self.P.SCREEN_WIDTH / 2)

        y = C.COORDINATE_SCALE * (-coordinates[0] + self.origin[0] + self.P.SCREEN_HEIGHT / 2)
        return np.array([int(x), int(y)])
