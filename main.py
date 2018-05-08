from constants import CONSTANTS as C
from track_vehicle import TrackVehicle
from autonomous_vehicle import AutonomousVehicle
from collision_box import Collision_Box
from sim_draw import Sim_Draw
from sim_data import Sim_Data
import pickle
import numpy as np
import pygame as pg
import datetime

class Main():

    def __init__(self):

        # Setup
        self.duration = 600
        self.P = C.PARAMETERSET_2  # Scenario parameters choice


        self.sim_draw = Sim_Draw(self.P, C.ASSET_LOCATION)

        # Time handling
        self.clock = pg.time.Clock()
        self.fps = C.FPS
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 0
        self.car_num_display = 0

        # Sim output
        self.sim_data = Sim_Data()
        self.sim_out = open("sim_outputs/output_%s.pkl" % datetime.datetime.now(), "wb")

        # Vehicle Definitions ('aggressive,'reactive','passive_aggressive')
        self.car_1 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_1,
                                       car_parameters_other=self.P.CAR_2,
                                       loss_style='aggressive')
        self.car_2 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_2,
                                       car_parameters_other=self.P.CAR_1,
                                       loss_style='aggressive')

        # Go
        self.trial()

    def trial(self):

        while self.running:

            # Update model here
            if not self.paused:
                self.car_1.update(self.car_2, self.frame)
                self.car_2.update(self.car_1, self.frame)
                # self.machine_vehicle.update(self.human_vehicle, self.frame)

                # Update data
                self.sim_data.append_car1(states=self.car_1.states_s,
                                          actions=self.car_1.actions_set_s,
                                          predicted_theta_of_other=self.car_1.predicted_theta_of_other,
                                          prediction_of_actions_of_other=self.car_1.predicted_actions_of_other,
                                          prediction_of_others_prediction_of_my_actions=self.car_1.prediction_of_others_prediction_of_my_actions)

                self.sim_data.append_car2(states=self.car_2.states_s,
                                          actions=self.car_2.actions_set_s,
                                          predicted_theta_of_other=self.car_2.predicted_theta_of_other,
                                          prediction_of_actions_of_other=self.car_2.predicted_actions_of_other,
                                          prediction_of_others_prediction_of_my_actions=self.car_2.prediction_of_others_prediction_of_my_actions)

            if self.frame >= self.duration:
                break

            # Draw frame
            self.sim_draw.draw_frame(self.sim_data, self.frame, self.car_num_display)

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

                    if event.key == pg.K_d:
                        self.car_num_display = ~self.car_num_display

            # Keep fps
            self.clock.tick(self.fps)

            if not self.paused:
                self.frame += 1

        # pickle.dump([self.sim_data_machine, self.sim_data_human], self.sim_out, pickle.HIGHEST_PROTOCOL)
        # print('Output pickled and dumped.')


if __name__ == "__main__":
    Main()