from constants import CONSTANTS as C
import pygame as pg
import numpy as np
import math
import os

BLACK       = (  0,  0,  0)
DARK_GREY   = (  0,  0,  0)
LIGHT_GREY  = (200,200,200)
MAGENTA     = (255,  0,255)
TEAL        = (  0,255,255)
GREEN       = (  0,255,  0)

class Sim_Draw():

    BLACK = (0, 0, 0)
    DARK_GREY = (0, 0, 0)
    LIGHT_GREY = (200, 200, 200)
    MAGENTA = (255, 0, 255)
    TEAL = (0, 255, 255)
    GREEN = (0, 255, 0)

    def __init__(self, parameters, asset_loc):

        self.P = parameters

        pg.init()
        self.screen = pg.display.set_mode((self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE))
        self.car2_image = pg.transform.rotate(pg.transform.scale(pg.image.load(asset_loc + "red_car_sized.png"),
                                                                  (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                                   int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))), -self.P.HUMAN_ORIENTATION)
        self.car1_image = pg.transform.rotate(pg.transform.scale(pg.image.load(asset_loc + "blue_car_sized.png"),
                                                                  (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                                   int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))), self.P.MACHINE_ORIENTATION)
        self.coordinates_image = pg.image.load(asset_loc + "coordinates.png")
        self.origin = np.array([0, 0])

    def draw_frame(self, sim_data, car_num_display, frame):


        # Draw the current frame
        self.screen.fill((255, 255, 255))

        self.origin = sim_data.car1_states[frame]

        # Draw Axis Lines
        self.draw_axes()

        # Draw Images
        pixel_pos_car_1 = self.c2p(sim_data.car1_states[frame])
        size_car_1 = self.machine_image.get_size()
        self.screen.blit(self.machine_image, (pixel_pos_car_1[0] - size_car_1[0] / 2, pixel_pos_car_1[1] - size_car_1[1] / 2))

        pixel_pos_car_2 = self.c2p(sim_data.car2_states[frame])
        size_car_2 = self.human_image.get_size()
        self.screen.blit(self.human_image, (pixel_pos_car_2[0] - size_car_2[0] / 2, pixel_pos_car_2[1] - size_car_2[1] / 2))

        coordinates_size = self.coordinates_image.get_size()
        self.screen.blit(self.coordinates_image, (10, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE - coordinates_size[1] - 10 / 2))

        if not car_num_display:  # If Car 1

            # Draw state
            state_range = []
            for i in range(len(sim_data.car1_actions)):
                state = sim_data.car1_states[frame] + np.sum(sim_data.car1_actions[:i + 1], axis=0)
                state_range.append(self.c2p(state))
            pg.draw.lines(self.screen, BLACK, False, state_range, 6)

            # Draw predicted state of other
            state_range = []
            for i in range(len(sim_data.car1_prediction_of_actions_of_other)):
                state = sim_data.car2_states[frame] + np.sum(sim_data.car1_prediction_of_actions_of_other[:i + 1], axis=0)
                state_range.append(self.c2p(state))
            pg.draw.lines(self.screen, DARK_GREY, False, state_range, 6)

            # Draw prediction of prediction state of self
            state_range = []
            for i in range(len(sim_data.car1_prediction_of_others_prediction_of_my_actions)):
                state = sim_data.car1_states[frame] + np.sum(sim_data.car1_prediction_of_others_prediction_of_my_actions[:i + 1], axis=0)
                state_range.append(self.c2p(state))
            pg.draw.lines(self.screen, LIGHT_GREY, False, state_range, 4)

        else:  # If Car 2

            # Draw state
            state_range = []
            for i in range(len(sim_data.car2_actions)):
                state = sim_data.car2_states[frame] + np.sum(sim_data.car2_actions[:i + 1], axis=0)
                state_range.append(self.c2p(state))
            pg.draw.lines(self.screen, BLACK, False, state_range, 6)

            # Draw predicted state of other
            state_range = []
            for i in range(len(sim_data.car2_prediction_of_actions_of_other)):
                state = sim_data.car1_states[frame] + np.sum(sim_data.car2_prediction_of_actions_of_other[:i + 1], axis=0)
                state_range.append(self.c2p(state))
            pg.draw.lines(self.screen, DARK_GREY, False, state_range, 6)

            # Draw prediction of prediction state of self
            state_range = []
            for i in range(len(sim_data.car2_prediction_of_others_prediction_of_my_actions)):
                state = sim_data.car2_states[frame] + np.sum(sim_data.car2_prediction_of_others_prediction_of_my_actions[:i + 1], axis=0)
                state_range.append(self.c2p(state))
            pg.draw.lines(self.screen, LIGHT_GREY, False, state_range, 4)


        # Annotations
        font = pg.font.SysFont("Arial", 15)
        label = font.render("Car 1: (%5.4f , %5.4f)" % (sim_data.car2_states[frame][0], sim_data.car2_states[frame][1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 10))

        label = font.render("Car 2: (%5.4f , %5.4f)" % (sim_data.car1_states[frame][0], sim_data.car1_states[frame][1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 30))

        # # label = font.render("Machine Theta: (%5.4f, %5.4f, %5.4f)" % (machine_theta[0], machine_theta[1], machine_theta[2]), 1, (0, 0, 0))
        # label = font.render("Machine Theta: (%5.4f)" % (machine_theta[0]), 1, (0, 0, 0))
        # self.screen.blit(label, (30, 60))
        # pg.draw.circle(self.screen, BLACK, (15, 70), 5)
        # pg.draw.circle(self.screen, GREEN, (15, 70), 4)
        #
        # # label = font.render("P Human Theta: (%5.4f, %5.4f, %5.4f)" % (human_predicted_theta[0], human_predicted_theta[1], human_predicted_theta[2]), 1, (0, 0, 0))
        # label = font.render("P Human Theta: (%5.4f)" % (human_predicted_theta[0]), 1, (0, 0, 0))
        # self.screen.blit(label, (30, 80))
        # pg.draw.circle(self.screen, BLACK, (15, 90), 5)
        # pg.draw.circle(self.screen, (0, 255, 255), (15, 90), 4)

        # label = font.render("PP Machine Theta: (%5.4f, %5.4f, %5.4f)" % (machine_predicted_theta[0], machine_predicted_theta[1], machine_predicted_theta[2]), 1, (0, 0, 0))
        # label = font.render("PP Machine Theta: (%5.4f)" % (machine_predicted_theta[0]), 1, (0, 0, 0))
        # self.screen.blit(label, (30, 100))
        # pg.draw.circle(self.screen, BLACK, (15, 110), 5)
        # pg.draw.circle(self.screen, MAGENTA, (15, 110), 4)

        label = font.render("Frame: %i" % (frame + 1), 1, (0, 0, 0))
        self.screen.blit(label, (10, 130))

        # label = font.render("Machine Action: (%5.4f, %5.4f)" % (machine_previous_action_set[0][0], machine_previous_action_set[0][1]), 1, (0, 0, 0))
        # self.screen.blit(label, (10, 160))
        #
        # label = font.render("P Human Action: (%5.4f, %5.4f)" % (human_previous_action_set[0][0], human_previous_action_set[0][1]), 1, (0, 0, 0))
        # self.screen.blit(label, (10, 180))
        #
        # label = font.render("PP Machine Action: (%5.4f, %5.4f)" % (machine_previous_predicted_action_set[0][0], machine_previous_predicted_action_set[0][1]), 1, (0, 0, 0))
        # self.screen.blit(label, (10, 200))

        pg.display.flip()


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

        # Vertical
        for i in range(num_vaxes):
            pg.draw.line(self.screen, LIGHT_GREY, (offset_x + i * spacing, 0),
                         (offset_x + i * spacing, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), 1)
            # label = (distance_x + 1 + i) * C.AXES_SHOW - rel_screen_width/2
            # text = font.render("%3.2f" % label, 1, GREY)
            # self.screen.blit(text, (10 + offset_x + (i * spacing), 10))

        # Horizontal
        for i in range(num_haxes):
            pg.draw.line(self.screen, LIGHT_GREY, (0, offset_y + i * spacing),
                         (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, offset_y + i * spacing), 1)
            # label = (distance_y + 1 + i) * C.AXES_SHOW - rel_screen_height/2
            # text = font.render("%3.2f" % label, 1, GREY)
            # self.screen.blit(text, (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE - 30, 10 + offset_y + (self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE) - (i * spacing)))

        # Bounds
        if self.P.BOUND_HUMAN_X is not None:
            _bound1 = self.c2p((self.P.BOUND_HUMAN_X[0], 0))
            _bound2 = self.c2p((self.P.BOUND_HUMAN_X[1], 0))
            bounds = np.array([_bound1[1], _bound2[1]])
            pg.draw.line(self.screen, BLACK, (0, bounds[0]), (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, bounds[0]), 2)
            pg.draw.line(self.screen, BLACK, (0, bounds[1]), (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, bounds[1]), 2)

        if self.P.BOUND_HUMAN_Y is not None:
            _bound1 = self.c2p((0, self.P.BOUND_HUMAN_Y[0]))
            _bound2 = self.c2p((0, self.P.BOUND_HUMAN_Y[1]))
            bounds = np.array([_bound1[0], _bound2[0]])
            pg.draw.line(self.screen, BLACK, (bounds[0], 0), (bounds[0], self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), 2)
            pg.draw.line(self.screen, BLACK, (bounds[1], 0), (bounds[1], self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), 2)

        if self.P.BOUND_MACHINE_X is not None:
            _bound1 = self.c2p((self.P.BOUND_MACHINE_X[0], 0))
            _bound2 = self.c2p((self.P.BOUND_MACHINE_X[1], 0))
            bounds = np.array([_bound1[1], _bound2[1]])
            pg.draw.line(self.screen, BLACK, (0, bounds[0]), (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, bounds[0]), 2)
            pg.draw.line(self.screen, BLACK, (0, bounds[1]), (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, bounds[1]), 2)

        if self.P.BOUND_MACHINE_Y is not None:
            _bound1 = self.c2p((0, self.P.BOUND_MACHINE_Y[0]))
            _bound2 = self.c2p((0, self.P.BOUND_MACHINE_Y[1]))
            bounds = np.array([_bound1[0], _bound2[0]])
            pg.draw.line(self.screen, BLACK, (bounds[0], 0), (bounds[0], self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), 2)
            pg.draw.line(self.screen, BLACK, (bounds[1], 0), (bounds[1], self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), 2)


    def c2p(self, coordinates):
        x = C.COORDINATE_SCALE * (coordinates[1] - self.origin[1] + self.P.SCREEN_WIDTH / 2)
        y = C.COORDINATE_SCALE * (-coordinates[0] + self.origin[0] + self.P.SCREEN_HEIGHT / 2)
        x = int(
            (x - self.P.SCREEN_WIDTH * C.COORDINATE_SCALE * 0.5) * C.ZOOM + self.P.SCREEN_WIDTH * C.COORDINATE_SCALE * 0.5)
        y = int((
                y - self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE * 0.5) * C.ZOOM + self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE * 0.5)
        return np.array([x, y])