from constants import CONSTANTS as C
import pygame as pg
import numpy as np
import math
import os

BLACK       = (  0,  0,  0)
GREY        = (200,200,200)
MAGENTA     = (255,  0,255)
TEAL        = (  0,255,255)
GREEN       = (  0,255,  0)

class Sim_Draw():

    BLACK = (0, 0, 0)
    GREY = (200, 200, 200)
    MAGENTA = (255, 0, 255)
    TEAL = (0, 255, 255)
    GREEN = (0, 255, 0)

    def __init__(self, parameters, asset_loc):

        self.P = parameters

        pg.init()
        self.screen = pg.display.set_mode((self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE))
        self.human_image = pg.transform.rotate(pg.transform.scale(pg.image.load(asset_loc + "red_car_sized.png"),
                                                                  (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                                   int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))), -self.P.HUMAN_ORIENTATION)
        self.machine_image = pg.transform.rotate(pg.transform.scale(pg.image.load(asset_loc + "blue_car_sized.png"),
                                                                  (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                                   int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))), self.P.MACHINE_ORIENTATION)
        self.coordinates_image = pg.image.load(asset_loc + "coordinates.png")
        self.origin = np.array([0, 0])

    def draw_frame(self, sim_data, frame):
        sim_data_machine, sim_data_human = sim_data

        # Define characteristics of current frame
        human_state = sim_data_human.machine_state[frame] # get this from human
        human_predicted_theta = sim_data_machine.human_predicted_theta[frame]
        human_previous_action_set = sim_data_machine.human_predicted_action_set[frame]

        machine_state = sim_data_machine.machine_state[frame]
        machine_theta = sim_data_machine.machine_theta[frame]
        machine_predicted_theta = sim_data_machine.machine_predicted_theta[frame]
        machine_previous_action_set = sim_data_machine.machine_action_set[frame]
        machine_previous_predicted_action_set = sim_data_machine.machine_predicted_action_set[frame]

        machine_predicted_theta_by_human = sim_data_human.human_predicted_theta[frame]
        machine_previous_action_set_by_human = sim_data_human.human_predicted_action_set[frame]

        human_theta = sim_data_human.machine_theta[frame]
        human_predicted_theta_by_human = sim_data_human.machine_predicted_theta[frame]
        human_previous_action_set_by_human = sim_data_human.machine_action_set[frame]
        human_previous_predicted_action_set_by_human = sim_data_human.machine_predicted_action_set[frame]


        # Draw the current frame

        self.screen.fill((255, 255, 255))

        self.origin = machine_state

        # Draw Axis Lines
        self.draw_axes()

        # Draw Images
        human_pos_pixels = self.c2p(human_state)
        human_car_size = self.human_image.get_size()
        self.screen.blit(self.human_image, (human_pos_pixels[0] - human_car_size[0] / 2, human_pos_pixels[1] - human_car_size[1] / 2))

        machine_pos_pixels = self.c2p(machine_state)
        machine_car_size = self.machine_image.get_size()
        self.screen.blit(self.machine_image, (machine_pos_pixels[0] - machine_car_size[0] / 2, machine_pos_pixels[1] - machine_car_size[1] / 2))

        coordinates_size = self.coordinates_image.get_size()
        self.screen.blit(self.coordinates_image, (10, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE - coordinates_size[1] - 10 / 2))

        # Draw machine decided state
        machine_predicted_state_pixels = []
        for i in range(len(machine_previous_action_set)):
            machine_predicted_state = machine_state + np.sum(machine_previous_action_set[:i + 1], axis=0)
            machine_predicted_state_pixels.append(self.c2p(machine_predicted_state))
        pg.draw.lines(self.screen, GREEN, False, machine_predicted_state_pixels, 6)

        # Draw human predicted state
        human_predicted_state_pixels = []
        for i in range(len(human_previous_action_set)):
            human_predicted_state = human_state + np.sum(human_previous_action_set[:i + 1], axis=0)
            human_predicted_state_pixels.append(self.c2p(human_predicted_state))
        pg.draw.lines(self.screen, TEAL, False, human_predicted_state_pixels, 6)

        # Draw machine predicted state
        machine_predicted_state_pixels = []
        for i in range(len(machine_previous_predicted_action_set)):
            machine_predicted_state = machine_state + np.sum(machine_previous_predicted_action_set[:i + 1], axis=0)
            machine_predicted_state_pixels.append(self.c2p(machine_predicted_state))
        pg.draw.lines(self.screen, MAGENTA, False, machine_predicted_state_pixels, 4)

        # Draw machine intent
        x = machine_theta[1] * np.cos(np.deg2rad(machine_theta[2]))
        y = machine_theta[1] * np.sin(np.deg2rad(machine_theta[2]))
        pos = self.c2p(np.array(machine_state) + [x, y])
        pg.draw.circle(self.screen, (0, 0, 0), pos, 7)
        pg.draw.circle(self.screen, GREEN, pos, 6)

        # Draw predicted human intent
        x = human_predicted_theta[1] * np.cos(np.deg2rad(human_predicted_theta[2]))
        y = human_predicted_theta[1] * np.sin(np.deg2rad(human_predicted_theta[2]))
        pos = self.c2p(np.array(human_state) + [x, y])
        pg.draw.circle(self.screen, (0, 0, 0), pos, 7)
        pg.draw.circle(self.screen, TEAL, pos, 6)

        # Draw predicted human's prediction of machine's intent
        x = machine_predicted_theta[1] * np.cos(np.deg2rad(machine_predicted_theta[2]))
        y = machine_predicted_theta[1] * np.sin(np.deg2rad(machine_predicted_theta[2]))
        pos = self.c2p(np.array(machine_state) + [x, y])
        pg.draw.circle(self.screen, (0, 0, 0), pos, 5)
        pg.draw.circle(self.screen, MAGENTA, pos, 4)

        # Annotations
        font = pg.font.SysFont("Arial", 15)
        label = font.render("Human State: (%5.4f , %5.4f)" % (human_state[0], human_state[1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 10))

        label = font.render("Machine State: (%5.4f , %5.4f)" % (machine_state[0], machine_state[1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 30))

        label = font.render("Machine Theta: (%5.4f, %5.4f, %5.4f)" % (machine_theta[0], machine_theta[1], machine_theta[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30, 60))
        pg.draw.circle(self.screen, BLACK, (15, 70), 5)
        pg.draw.circle(self.screen, GREEN, (15, 70), 4)

        label = font.render("P Human Theta: (%5.4f, %5.4f, %5.4f)" % (human_predicted_theta[0], human_predicted_theta[1], human_predicted_theta[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30, 80))
        pg.draw.circle(self.screen, BLACK, (15, 90), 5)
        pg.draw.circle(self.screen, (0, 255, 255), (15, 90), 4)

        label = font.render("PP Machine Theta: (%5.4f, %5.4f, %5.4f)" % (machine_predicted_theta[0], machine_predicted_theta[1], machine_predicted_theta[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30, 100))
        pg.draw.circle(self.screen, BLACK, (15, 110), 5)
        pg.draw.circle(self.screen, MAGENTA, (15, 110), 4)

        label = font.render("Frame: %i" % (frame + 1), 1, (0, 0, 0))
        self.screen.blit(label, (10, 130))

        label = font.render("Machine Action: (%5.4f, %5.4f)" % (machine_previous_action_set[0][0], machine_previous_action_set[0][1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 160))

        label = font.render("P Human Action: (%5.4f, %5.4f)" % (human_previous_action_set[0][0], human_previous_action_set[0][1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 180))

        label = font.render("PP Machine Action: (%5.4f, %5.4f)" % (machine_previous_predicted_action_set[0][0], machine_previous_predicted_action_set[0][1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 200))

        # draw from human's perspective
        gap = 400
        label = font.render("Human Theta: (%5.4f, %5.4f, %5.4f)" % (human_theta[0], human_theta[1], human_theta[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30+gap, 60))
        pg.draw.circle(self.screen, BLACK, (15+gap, 70), 5)
        pg.draw.circle(self.screen, GREEN, (15+gap, 70), 4)

        label = font.render("P Machine Theta: (%5.4f, %5.4f, %5.4f)" % (machine_predicted_theta_by_human[0], machine_predicted_theta_by_human[1],
                                                                      machine_predicted_theta_by_human[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30+gap, 80))
        pg.draw.circle(self.screen, BLACK, (15+gap, 90), 5)
        pg.draw.circle(self.screen, (0, 255, 255), (15+gap, 90), 4)

        label = font.render("PP Human Theta: (%5.4f, %5.4f, %5.4f)" % (human_predicted_theta_by_human[0], human_predicted_theta_by_human[1],
                                                                         human_predicted_theta_by_human[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30+gap, 100))
        pg.draw.circle(self.screen, BLACK, (15+gap, 110), 5)
        pg.draw.circle(self.screen, MAGENTA, (15+gap, 110), 4)

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
            pg.draw.line(self.screen, GREY, (offset_x + i * spacing, 0),
                         (offset_x + i * spacing, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), 1)
            # label = (distance_x + 1 + i) * C.AXES_SHOW - rel_screen_width/2
            # text = font.render("%3.2f" % label, 1, GREY)
            # self.screen.blit(text, (10 + offset_x + (i * spacing), 10))

        # Horizontal
        for i in range(num_haxes):
            pg.draw.line(self.screen, GREY, (0, offset_y + i * spacing),
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