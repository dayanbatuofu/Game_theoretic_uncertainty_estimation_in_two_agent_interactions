from constants import CONSTANTS as C
import pygame as pg
import pygame.gfxdraw
import numpy as np
import math
import os

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
        self.car1_image = pg.transform.rotate(pg.image.load(asset_loc + "red_car_sized.png"), self.P.CAR_1.ORIENTATION)
        self.car2_image = pg.transform.rotate(pg.image.load(asset_loc + "red_car_sized.png"), self.P.CAR_2.ORIENTATION)
        self.car3_image = pg.transform.rotate(pg.image.load(asset_loc + "blue_car_sized.png"), self.P.CAR_3.ORIENTATION)

        self.coordinates_image = pg.image.load(asset_loc + "coordinates.png")
        self.origin = np.array([-1.0, 1.0])

    def draw_frame(self, sim_data, car_num_display, frame):

        # Draw the current frame
        self.screen.fill((255, 255, 255))

        # self.origin = sim_data.car1_states[frame]

        # Draw Axis Lines
        self.draw_axes()

        # Draw Images
        pixel_pos_car_1 = self.c2p(sim_data.car1_states[frame])
        size_car_1 = self.car1_image.get_size()
        self.screen.blit(self.car1_image, (pixel_pos_car_1[0] - size_car_1[0] / 2, pixel_pos_car_1[1] - size_car_1[1] / 2))

        pixel_pos_car_2 = self.c2p(sim_data.car2_states[frame])
        size_car_2 = self.car2_image.get_size()
        self.screen.blit(self.car2_image, (pixel_pos_car_2[0] - size_car_2[0] / 2, pixel_pos_car_2[1] - size_car_2[1] / 2))

        pixel_pos_car_3 = self.c2p(sim_data.car3_states[frame])
        size_car_3 = self.car3_image.get_size()
        self.screen.blit(self.car2_image,
                         (pixel_pos_car_3[0] - size_car_3[0] / 2, pixel_pos_car_3[1] - size_car_3[1] / 2))

        coordinates_size = self.coordinates_image.get_size()
        self.screen.blit(self.coordinates_image, (10, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE - coordinates_size[1] - 10 / 2))

        if car_num_display == 1:  # If Car 1
            # Draw predicted state of other
            state_range = []
            for t in range(len(sim_data.car1_predicted_actions_other[frame])):
                state_range.append(self.c2p((sim_data.car1_predicted_actions_other[frame][t][-1]-sim_data.car1_states[frame])*0.7+sim_data.car1_states[frame]))
                # for i in range(len(sim_data.car1_predicted_actions_other[frame][t])):
                #     state = sim_data.car2_states[frame] + \
                #             np.sum(sim_data.car1_predicted_actions_other[frame][t][:i+1], axis=0)*0.7
                #     state_range.append(self.c2p(state))
            state_range_unique, index, counts = \
                np.unique(state_range, axis=0, return_index=True, return_counts=True)
            probability = np.zeros(len(state_range_unique))
            for i in range(len(state_range_unique)):
                for j in range(len(state_range)):
                    if np.array_equal(state_range[j], state_range_unique[i]):
                        probability[i] += sim_data.car1_inference_probability_proactive[frame][j]
                # pg.draw.lines(self.screen, DARK_CAR1, False, [self.c2p(sim_data.car2_states[frame]), state_range_unique[i]], 12)
                pygame.gfxdraw.filled_circle(self.screen, state_range_unique[i][0], state_range_unique[i][1],
                               int(probability[i]*36), DARK_CAR1)
                pygame.gfxdraw.filled_circle(self.screen, state_range_unique[i][0], state_range_unique[i][1],
                               max(int(probability[i]*36)-4,1), LIGHT_LIGHT_GREY)

            # Draw prediction state of self
            state_range = []
            for t in range(len(sim_data.car2_predicted_actions_other[frame])):
                # state_range.append(self.c2p(
                #     np.sum(sim_data.car2_predicted_actions_other[frame][t], axis=0)*0.7+
                #     sim_data.car1_states[frame]))
                state_range.append(self.c2p((sim_data.car2_predicted_actions_other[frame][t][-1]-sim_data.car1_states[frame])*0.7+sim_data.car1_states[frame]))
                # for i in range(len(sim_data.car2_predicted_actions_other[frame][t])):
                #     state = sim_data.car1_states[frame] + \
                #             np.sum(sim_data.car2_predicted_actions_other[frame][t][:i+1], axis=0)*0.7
                #     state_range.append(self.c2p(state))
            state_range_unique, index, counts = \
                np.unique(state_range, axis=0, return_index=True, return_counts=True)
            probability = np.zeros(len(state_range_unique))
            for i in range(len(state_range_unique)):
                for j in range(len(state_range)):
                    if np.array_equal(state_range[j], state_range_unique[i]):
                        probability[i] += sim_data.car2_inference_probability_proactive[frame][j]
                # pg.draw.lines(self.screen, DARK_CAR2, False, [self.c2p(sim_data.car1_states[frame]), state_range_unique[i]], 12)
                pygame.gfxdraw.filled_circle(self.screen, state_range_unique[i][0], state_range_unique[i][1],
                               int(probability[i]*36), DARK_CAR2)
                pygame.gfxdraw.filled_circle(self.screen, state_range_unique[i][0], state_range_unique[i][1],
                               max(int(probability[i]*36)-4,1), LIGHT_LIGHT_GREY)

            # # # Draw prediction of prediction state of other
            # # state_range = []
            # # for t in range(len(sim_data.car2_predicted_others_prediction_of_my_actions[frame])):
            # #     # state_range = []
            # #     # for i in range(len(sim_data.car2_predicted_others_prediction_of_my_actions[frame][t])):
            # #     #     state = sim_data.car2_states[frame] + \
            # #     #             np.sum(sim_data.car2_predicted_others_prediction_of_my_actions[frame][t][:i+1], axis=0)*0.7
            # #     #     state_range.append(self.c2p(state))
            # #     state_range.append(self.c2p(
            # #         np.sum(sim_data.car2_predicted_others_prediction_of_my_actions[frame][t], axis=0)*0.7+
            # #         sim_data.car2_states[frame]))
            # # state_range_unique, index, counts = \
            # #     np.unique(state_range, axis=0, return_index=True, return_counts=True)
            # # probability = np.zeros(len(state_range_unique))
            # # for i in range(len(state_range_unique)):
            # #     for j in range(len(state_range)):
            # #         if np.array_equal(state_range[j], state_range_unique[i]):
            # #             probability[i] += sim_data.car2_inference_probability[frame][j]
            # #     # pg.draw.lines(self.screen, LIGHT_CAR2, False, [self.c2p(sim_data.car2_states[frame]), state_range_unique[i]], 8)
            # #     pygame.gfxdraw.filled_circle(self.screen, state_range_unique[i][0], state_range_unique[i][1],
            # #                              int(probability[i]*28), LIGHT_CAR2)
            # #
            # # # Draw prediction of prediction state of self
            # # state_range = []
            # # for t in range(len(sim_data.car1_predicted_others_prediction_of_my_actions[frame])):
            # #     state_range.append(self.c2p(
            # #         np.sum(sim_data.car1_predicted_others_prediction_of_my_actions[frame][t], axis=0)*0.7+
            # #         sim_data.car1_states[frame]))
            # #     # for i in range(len(sim_data.car1_predicted_others_prediction_of_my_actions[frame][t])):
            # #     #     state = sim_data.car1_states[frame] + \
            # #     #             np.sum(sim_data.car1_predicted_others_prediction_of_my_actions[frame][t][:i+1], axis=0)*0.7
            # #     #     state_range.append(self.c2p(state))
            # # state_range_unique, index, counts = \
            # #     np.unique(state_range, axis=0, return_index=True, return_counts=True)
            # # probability = np.zeros(len(state_range_unique))
            # # for i in range(len(state_range_unique)):
            # #     for j in range(len(state_range)):
            # #         if np.array_equal(state_range[j], state_range_unique[i]):
            # #             probability[i] += sim_data.car1_inference_probability[frame][j]
            # #     # pg.draw.lines(self.screen, LIGHT_CAR1, False, [self.c2p(sim_data.car1_states[frame]), state_range_unique[i]], 8)
            # #     pygame.gfxdraw.filled_circle(self.screen, state_range_unique[i][0], state_range_unique[i][1],
            # #                    int(probability[i]*28), LIGHT_CAR1)
            #
            # Draw what others want me to do
            state_range = []
            for t in range(len(sim_data.car2_wanted_trajectory_other[frame])):
                # state_range.append(self.c2p(
                #     sim_data.car2_wanted_trajectory_other[frame][t]))
                state_range.append(self.c2p((sim_data.car2_wanted_states_other[frame][t][-1]-sim_data.car1_states[frame])*0.7 + sim_data.car1_states[frame]))
                # for i in range(len(sim_data.car1_predicted_others_prediction_of_my_actions[frame][t])):
                #     state = sim_data.car1_states[frame] + \
                #             np.sum(sim_data.car1_predicted_others_prediction_of_my_actions[frame][t][:i+1], axis=0)*0.7
                #     state_range.append(self.c2p(state))
            state_range_unique, index, counts = \
                np.unique(state_range, axis=0, return_index=True, return_counts=True)
            probability = np.zeros(len(state_range_unique))
            for i in range(len(state_range_unique)):
                for j in range(len(state_range)):
                    if np.array_equal(state_range[j], state_range_unique[i]):
                        probability[i] += sim_data.car2_inference_probability[frame][j]
                # pg.draw.lines(self.screen, YELLOW, False, [self.c2p(sim_data.car1_states[frame]), state_range_unique[i]], 4)
                pygame.gfxdraw.filled_circle(self.screen, state_range_unique[i][0], state_range_unique[i][1],
                               int(probability[i]*24), YELLOW)

            # Draw planned action of self
            state_range = []
            for i in range(len(sim_data.car1_planned_action_sets[frame])):
                    # state = sim_data.car1_states[frame] + \
                    #         np.sum(sim_data.car1_planned_action_sets[frame][:i+1], axis=0)*0.7
                    # state_range.append(self.c2p(state))
                state_range.append(self.c2p((sim_data.car1_planned_action_sets[frame][i]-sim_data.car1_states[frame])*0.7+sim_data.car1_states[frame]))

            pg.draw.lines(self.screen, GREEN, False, state_range, 4)

            # Draw planned action of other
            state_range = []
            for i in range(len(sim_data.car2_planned_action_sets[frame])):
                # state = sim_data.car2_states[frame] + \
                #         np.sum(sim_data.car2_planned_action_sets[frame][:i+1], axis=0)*0.7
                # state_range.append(self.c2p(state))
                state_range.append(self.c2p((sim_data.car2_planned_action_sets[frame][i]-sim_data.car2_states[frame])*0.7+sim_data.car2_states[frame]))
            pg.draw.lines(self.screen, GREEN, False, state_range, 4)

        else:  # If Car 2

            # Draw predicted state of other
            state_range = []
            for i in range(len(sim_data.car2_predicted_actions_other[frame])):
                state = sim_data.car2_predicted_actions_other[frame][i]
                state_range.append(self.c2p(state))
            pg.draw.lines(self.screen, DARK_GREY, False, state_range, 16)

            # Draw prediction of prediction state of self
            state_range = []
            for i in range(len(sim_data.car2_predicted_others_prediction_of_my_actions[frame])):
                state = sim_data.car2_states[frame] + np.sum(sim_data.car2_predicted_others_prediction_of_my_actions[frame][:i + 1], axis=0)
                state_range.append(self.c2p(state))
            pg.draw.lines(self.screen, LIGHT_GREY, False, state_range, 16)

            # Draw state
            state_range = []
            for i in range(len(sim_data.car2_planned_action_sets[frame])):
                # state = sim_data.car2_states[frame] + np.sum(sim_data.car2_planned_action_sets[frame][:i + 1], axis=0)
                # state_range.append(self.c2p(state))
                state_range.append(self.c2p((sim_data.car1_planned_action_sets[frame][i]-sim_data.car1_states[frame]) * 0.7 + sim_data.car1_states[frame]))
            pg.draw.lines(self.screen, BLACK, False, state_range, 6)

        # Annotations
        font = pg.font.SysFont("Arial", 30)

        label = font.render("Car 1 state: (%5.4f , %5.4f)" % (sim_data.car1_states[frame][0], sim_data.car1_states[frame][1]), 1, (0, 0, 0))
        self.screen.blit(label, (350, 360))
        label = font.render("Car 1 action index: (%5.4f)" % (sim_data.car1_planned_trajectory_set[frame][0]), 1, (0, 0, 0))
        self.screen.blit(label, (350, 400))

        label = font.render("Car 2 state: (%5.4f , %5.4f)" % (sim_data.car2_states[frame][0], sim_data.car2_states[frame][1]), 1, (0, 0, 0))
        self.screen.blit(label, (350, 440))
        label = font.render("Car 2 action index: (%5.4f)" % (sim_data.car2_planned_trajectory_set[frame][0]), 1, (0, 0, 0))
        self.screen.blit(label, (350, 480))


        # label = font.render("Car 1 intent by 1: %5.4f" % (np.sum(sim_data.car1_predicted_theta_self[frame])), 1, (0, 0, 0))
        # self.screen.blit(label, (410, 440))

        # label = font.render("Car 2 intent by 1: %1.2f, %1.2f, %1.2f " % (sim_data.car1_theta_probability[frame][0],
        #                                                                  sim_data.car1_theta_probability[frame][1],
        #                                                                  sim_data.car1_theta_probability[frame][2]), 1, (0, 0, 0))
        # self.screen.blit(label, (350, 480))
        #
        # label = font.render("Car 1 intent by 2: %1.2f, %1.2f, %1.2f" % (sim_data.car2_theta_probability[frame][0],
        #                                                                 sim_data.car2_theta_probability[frame][1],
        #                                                                 sim_data.car2_theta_probability[frame][2]), 1, (0, 0, 0))

        # label = font.render("Car 2 intent by 1: %1.2f, %1.2f" % (sim_data.car1_theta_probability[frame][0],
        #                                                          sim_data.car1_theta_probability[frame][1]), 1, (0, 0, 0))
        # self.screen.blit(label, (350, 480))
        #
        # label = font.render("Car 1 intent by 2: %1.2f, %1.2f" % (sim_data.car2_theta_probability[frame][0],
        #                                                          sim_data.car2_theta_probability[frame][1]), 1, (0, 0, 0))
        # self.screen.blit(label, (350, 520))

        # label = font.render("Car 2 intent by 2: %5.4f" % (np.sum(sim_data.car2_predicted_theta_self[frame])), 1, (0, 0, 0))
        # self.screen.blit(label, (410, 560))

        label = font.render("Lack of Courtesy: %1.4f" % (np.sum(sim_data.car1_gracefulness)), 1, (0, 0, 0))
        self.screen.blit(label, (350, 10))

        label = font.render("Frame: %i" % (frame + 1), 1, (0, 0, 0))
        self.screen.blit(label, (10, 10))

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

        # label = font.render("Machine Action: (%5.4f, %5.4f)" % (machine_previous_action_set[0][0], machine_previous_action_set[0][1]), 1, (0, 0, 0))
        # self.screen.blit(label, (10, 160))
        #
        # label = font.render("P Human Action: (%5.4f, %5.4f)" % (human_previous_action_set[0][0], human_previous_action_set[0][1]), 1, (0, 0, 0))
        # self.screen.blit(label, (10, 180))
        #
        # label = font.render("PP Machine Action: (%5.4f, %5.4f)" % (machine_previous_predicted_action_set[0][0], machine_previous_predicted_action_set[0][1]), 1, (0, 0, 0))
        # self.screen.blit(label, (10, 200))

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