from constants import CONSTANTS as C
from human_vehicle import HumanVehicle
from machine_vehicle import MachineVehicle
from collision_box import Collision_Box
from sim_draw import Sim_Draw
from sim_data import Sim_Data
import pickle
import numpy as np
import pygame as pg

class Main():

    def __init__(self):

        self.duration = 300

        self.P = C.PARAMETERSET_1  # Scenario parameters choice

        self.sim_draw = Sim_Draw(self.P, "assets/")

        # Time handling
        self.clock = pg.time.Clock()
        self.fps = C.FPS
        self.running = True
        self.paused = False
        self.end = False
        self.frame = -1

        # Sim output
        self.sim_data = Sim_Data()
        self.sim_out = open("sim_outputs/output_intersection.pkl", "wb")

        # Create Vehicles
        human_collision_box = Collision_Box(self.sim_draw.human_image.get_width() / C.COORDINATE_SCALE,
                                            self.sim_draw.human_image.get_height() / C.COORDINATE_SCALE)
        machine_collision_box = Collision_Box(self.sim_draw.machine_image.get_width() / C.COORDINATE_SCALE,
                                              self.sim_draw.machine_image.get_height() / C.COORDINATE_SCALE)

        # self.human_vehicle = HumanVehicle('human_state_files/intersection/human_stop_go.txt')
        self.human_vehicle = HumanVehicle('human_state_files/lane_change/human_change_lane.txt')
        self.machine_vehicle = MachineVehicle(self.P, human_collision_box, machine_collision_box,
                                              self.human_vehicle.get_state(0))

        # Go
        self.trial()


    def trial(self):

        while self.running:

            # Update model here
            if not self.paused:
                self.machine_vehicle.update(self.human_vehicle.get_state(self.frame))
                self.frame += 1

                # Update data
                self.sim_data.append(self.machine_vehicle.human_states[-1],
                                     self.machine_vehicle.human_predicted_theta,
                                     self.machine_vehicle.human_predicted_action_set,
                                     self.machine_vehicle.machine_states[-1],
                                     self.machine_vehicle.machine_theta,
                                     self.machine_vehicle.machine_predicted_theta,
                                     self.machine_vehicle.machine_action_set,
                                     self.machine_vehicle.machine_predicted_action_set)

            if self.frame >= self.duration:
                break

            # Draw frame
            self.sim_draw.draw_frame(self.sim_data, self.frame)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    self.running = False

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_p:
                        self.paused = not self.paused

                    if event.key == pg.K_q:
                        pg.quit()
                        self.running = False

            # Keep fps
            self.clock.tick(self.fps)

        pickle.dump(self.sim_data,  self.sim_out, pickle.HIGHEST_PROTOCOL)
        print('Output pickled and dumped.')


if __name__ == "__main__":
    Main()