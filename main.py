from constants import CONSTANTS as C
from human_vehicle import HumanVehicle
from machine_vehicle import MachineVehicle
from collision_box import Collision_Box
import math
import numpy as np
import pygame as pg

BLACK       = (  0,  0,  0)
GREY        = (200,200,200)
MAGENTA     = (255,  0,255)
TEAL        = (  0,255,255)
GREEN       = (  0,255,  0)

class Main():

    def __init__(self):

        self.duration = 1800

        self.P = C.PARAMETERSET_1  # Scenario parameters choice

        pg.init()
        self.screen = pg.display.set_mode((self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE))
        self.human_image = pg.transform.rotate(pg.transform.scale(pg.image.load("assets/red_car_sized.png"),
                                                                  (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                                   int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))), self.P.HUMAN_ORIENTATION)
        self.machine_image = pg.transform.rotate(pg.transform.scale(pg.image.load("assets/blue_car_sized.png"),
                                                                  (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                                   int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))), self.P.MACHINE_ORIENTATION)
        self.coordinates_image = pg.image.load("assets/coordinates.png")
        self.origin = np.array([0, 0])

        human_collision_box = Collision_Box(self.human_image.get_width() / C.COORDINATE_SCALE, self.human_image.get_height() / C.COORDINATE_SCALE)
        machine_collision_box = Collision_Box(self.machine_image.get_width() / C.COORDINATE_SCALE, self.machine_image.get_height() / C.COORDINATE_SCALE)

        # self.human_vehicle = HumanVehicle('human_state_files/intersection/human_stop.txt')
        self.human_vehicle = HumanVehicle('human_state_files/lane_change/human_change_lane.txt')
        self.machine_vehicle = MachineVehicle(self.P, human_collision_box, machine_collision_box, self.human_vehicle.get_state(0))


        # Time handling
        self.clock = pg.time.Clock()
        self.fps = C.FPS
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 1

        # Sim output
        self.sim_out = open("sim_outputs/output_maxtest.txt", "w")

        self.trial()


    def trial(self):

        while self.running:

            # Update model here
            if not self.paused and not self.end:
                self.machine_vehicle.update(self.human_vehicle.get_state(self.frame))
                self.frame += 1

            if self.frame >= self.duration:
                end = True

            # Draw frame
            if not self.end:
                self.draw_frame()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()

                    running = False

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_p:
                        self.paused = not self.paused

            # Keep fps
            self.clock.tick(self.fps)


    def draw_frame(self):

        self.screen.fill((255, 255, 255))

        self.origin = self.machine_vehicle.get_state()[0:2]

        # Draw Axis Lines
        self.draw_axes()

        # Draw Images
        human_pos = self.human_vehicle.get_state(self.frame)[0:2]
        human_pos_pixels = self.c2p(human_pos)
        human_car_size = self.human_image.get_size()
        self.screen.blit(self.human_image, (human_pos_pixels[0] - human_car_size[0] / 2, human_pos_pixels[1] - human_car_size[1] / 2))

        machine_pos = self.machine_vehicle.get_state()[0:2]
        machine_pos_pixels = self.c2p(machine_pos)
        machine_car_size = self.machine_image.get_size()
        self.screen.blit(self.machine_image, (machine_pos_pixels[0] - machine_car_size[0] / 2, machine_pos_pixels[1] - machine_car_size[1] / 2))

        coordinates_size = self.coordinates_image.get_size()
        self.screen.blit(self.coordinates_image, (10, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE - coordinates_size[1] - 10 / 2))

        # Draw machine decided state
        machine_predicted_state_pixels = []
        for i in range(len(self.machine_vehicle.machine_previous_action_set)):
            machine_predicted_state = self.machine_vehicle.machine_states[-1] + np.sum(self.machine_vehicle.machine_previous_action_set[:i+1],axis=0)
            machine_predicted_state_pixels.append(self.c2p(machine_predicted_state))
        pg.draw.lines(self.screen, GREEN, False, machine_predicted_state_pixels, 6)

        # Draw human predicted state
        human_predicted_state_pixels = []
        for i in range(len(self.machine_vehicle.human_previous_action_set)):
            human_predicted_state = self.machine_vehicle.human_states[-1] + np.sum(self.machine_vehicle.human_previous_action_set[:i+1],axis=0)
            human_predicted_state_pixels.append(self.c2p(human_predicted_state))
        pg.draw.lines(self.screen, TEAL, False, human_predicted_state_pixels, 6)

        # Draw machine predicted state
        machine_predicted_state_pixels = []
        for i in range(len(self.machine_vehicle.machine_previous_predicted_action_set)):
            machine_predicted_state = self.machine_vehicle.machine_states[-1] + np.sum(self.machine_vehicle.machine_previous_predicted_action_set[:i+1],axis=0)
            machine_predicted_state_pixels.append(self.c2p(machine_predicted_state))
        pg.draw.lines(self.screen, MAGENTA, False, machine_predicted_state_pixels, 4)

        # Draw machine intent
        pos = self.c2p(machine_pos + self.machine_vehicle.machine_theta[1:3])
        pg.draw.circle(self.screen, (0, 0, 0), pos, 7)
        pg.draw.circle(self.screen, GREEN, pos, 6)

        # Draw predicted human intent
        pos = self.c2p(human_pos + self.machine_vehicle.human_predicted_theta[1:3])
        pg.draw.circle(self.screen, (0, 0, 0), pos, 7)
        pg.draw.circle(self.screen, TEAL, pos, 6)

        # Draw predicted human's prediction of machine's intent
        pos = self.c2p(machine_pos + self.machine_vehicle.machine_predicted_theta[1:3])
        pg.draw.circle(self.screen, (0, 0, 0), pos, 5)
        pg.draw.circle(self.screen, MAGENTA, pos, 4)

        # Annotations
        font = pg.font.SysFont("Arial", 15)
        label = font.render("Human State: (%5.4f , %5.4f)" % (human_pos[0], human_pos[1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 10))

        label = font.render("Machine State: (%5.4f , %5.4f)" % (machine_pos[0], machine_pos[1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 30))

        label = font.render("Machine Theta: (%5.4f, %5.4f, %5.4f)" % (self.machine_vehicle.machine_theta[0], self.machine_vehicle.machine_theta[1], self.machine_vehicle.machine_theta[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30, 60))
        pg.draw.circle(self.screen, BLACK, (15, 70), 5)
        pg.draw.circle(self.screen, GREEN, (15, 70), 4)

        label = font.render("P Human Theta: (%5.4f, %5.4f, %5.4f)" % (self.machine_vehicle.human_predicted_theta[0], self.machine_vehicle.human_predicted_theta[1], self.machine_vehicle.human_predicted_theta[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30, 80))
        pg.draw.circle(self.screen, BLACK, (15, 90), 5)
        pg.draw.circle(self.screen, (0, 255, 255), (15, 90), 4)

        label = font.render("PP Machine Theta: (%5.4f, %5.4f, %5.4f)" % (self.machine_vehicle.machine_predicted_theta[0], self.machine_vehicle.machine_predicted_theta[1], self.machine_vehicle.machine_predicted_theta[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30, 100))
        pg.draw.circle(self.screen, BLACK, (15, 110), 5)
        pg.draw.circle(self.screen, MAGENTA, (15, 110), 4)

        label = font.render("Frame: %i" % (self.frame + 1), 1, (0, 0, 0))
        self.screen.blit(label, (10, 130))

        label = font.render("Machine Action: (%5.4f, %5.4f)" % (self.machine_vehicle.machine_previous_action_set[0][0], self.machine_vehicle.machine_previous_action_set[0][1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 160))

        label = font.render("P Human Action: (%5.4f, %5.4f)" % (self.machine_vehicle.human_previous_action_set[0][0], self.machine_vehicle.human_previous_action_set[0][1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 180))

        label = font.render("PP Machine Action: (%5.4f, %5.4f)" % (self.machine_vehicle.machine_previous_predicted_action_set[0][0], self.machine_vehicle.machine_previous_predicted_action_set[0][1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 200))

        pg.display.flip()

        if True:
            self.sim_out.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (human_pos[0],
                                                                human_pos[1],
                                                                machine_pos[0],
                                                                machine_pos[1],
                                                                human_predicted_state[0],
                                                                human_predicted_state[1],
                                                                self.machine_vehicle.human_predicted_theta[0],
                                                                self.machine_vehicle.human_predicted_theta[1],
                                                                self.machine_vehicle.human_predicted_theta[2]))

    def draw_axes(self):

        rel_coor_scale = C.COORDINATE_SCALE * C.ZOOM
        rel_screen_width = self.P.SCREEN_WIDTH / C.ZOOM
        rel_screen_height = self.P.SCREEN_HEIGHT / C.ZOOM

        spacing = int(C.AXES_SHOW * rel_coor_scale)
        offset_x = int(math.fmod(self.origin[1] * rel_coor_scale, spacing))
        offset_y = int(math.fmod(self.origin[0] * rel_coor_scale, spacing))

        distance_x = int((self.origin[1] * rel_coor_scale) / spacing)
        distance_y = int((self.origin[0] * rel_coor_scale) / spacing)

        num_vaxes = int(rel_screen_width * rel_coor_scale / spacing) + 1
        num_haxes = int(rel_screen_height * rel_coor_scale / spacing) + 1

        font = pg.font.SysFont("Arial", 15)

        # Vertical
        for i in range(num_vaxes):
            pg.draw.line(self.screen, GREY, (offset_x + i*spacing, 0), (offset_x + i*spacing, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE), 1)
            label = (distance_x + 1 + i) * C.AXES_SHOW - rel_screen_width/2
            text = font.render("%3.2f" % label, 1, GREY)
            self.screen.blit(text, (10 + offset_x + (i * spacing), 10))

        # Horizontal
        for i in range(num_haxes):
            pg.draw.line(self.screen, GREY, (0, offset_y + i*spacing), (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, offset_y + i*spacing), 1)
            label = (distance_y + 1 + i) * C.AXES_SHOW - rel_screen_height/2
            text = font.render("%3.2f" % label, 1, GREY)
            self.screen.blit(text, (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE - 30, 10 + offset_y + (self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE) - (i * spacing)))

    def c2p(self, coordinates):
        x = C.COORDINATE_SCALE * (coordinates[1] - self.origin[1] + self.P.SCREEN_WIDTH/2)
        y = C.COORDINATE_SCALE * (-coordinates[0] + self.origin[0] + self.P.SCREEN_HEIGHT/2)
        x = int((x - self.P.SCREEN_WIDTH * C.COORDINATE_SCALE * 0.5) * C.ZOOM + self.P.SCREEN_WIDTH * C.COORDINATE_SCALE * 0.5)
        y = int((y - self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE * 0.5) * C.ZOOM + self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE * 0.5)
        return np.array([x, y])


if __name__ == "__main__":
    Main()