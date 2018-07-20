from constants import CONSTANTS as C
from autonomous_vehicle import AutonomousVehicle
from sim_draw import Sim_Draw
from sim_data import Sim_Data
import pickle
import os
import pygame as pg
import datetime

class Main():

    def __init__(self):

        # Setup
        self.duration = 600
        self.P = C.PARAMETERSET_2  # Scenario parameters choice

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

        # self.sim_out = open("./sim_outputs/output_test.pkl", "wb")

        # Vehicle Definitions ('aggressive,'reactive','passive_aggressive')
        self.car_1 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_1,
                                       loss_style='passive_aggressive',
                                       who=1)  #M
        self.car_2 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_2,
                                       loss_style='reactive',
                                       who=0)  #H

        # Assign 'other' cars
        self.car_1.other_car = self.car_2
        self.car_2.other_car = self.car_1
        self.car_1.states_o = self.car_2.states
        self.car_2.states_o = self.car_1.states
        self.car_1.actions_set_o = self.car_2.actions_set
        self.car_2.actions_set_o = self.car_1.actions_set

        if C.DRAW:
            self.sim_draw = Sim_Draw(self.P, C.ASSET_LOCATION)
            pg.display.flip()
            self.capture = True if input("Capture video (y/n): ") else False

            output_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs("./sim_outputs/%s" % output_name)
            self.sim_out = open("./sim_outputs/%s/output.pkl" % output_name, "wb")

            if self.capture:
                self.output_dir = "./sim_outputs/%s/video/" % output_name
                os.makedirs(self.output_dir)

        # Go
        self.trial()

    def trial(self):

        while self.running:

            # Update model here
            if not self.paused:
                self.car_1.update(self.frame)
                self.car_2.update(self.frame)
                # self.machine_vehicle.update(self.human_vehicle, self.frame)

                # Update data
                self.sim_data.append_car1(states=self.car_1.states,
                                          actions=self.car_1.actions_set,
                                          action_sets=self.car_1.planned_actions_set,
                                          predicted_theta_other=self.car_1.predicted_theta_other,
                                          predicted_theta_self=self.car_1.predicted_theta_self,
                                          predicted_actions_other=self.car_1.predicted_actions_other,
                                          predicted_others_prediction_of_my_actions=
                                          self.car_1.predicted_others_prediction_of_my_actions)

                self.sim_data.append_car2(states=self.car_2.states,
                                          actions=self.car_2.actions_set,
                                          action_sets=self.car_2.planned_actions_set,
                                          predicted_theta_other=self.car_2.predicted_theta_other,
                                          predicted_theta_self=self.car_2.predicted_theta_self,
                                          predicted_actions_other=self.car_2.predicted_actions_other,
                                          predicted_others_prediction_of_my_actions=
                                          self.car_2.predicted_others_prediction_of_my_actions)

            if self.frame >= self.duration:
                break

            if C.DRAW:
                # Draw frame
                self.sim_draw.draw_frame(self.sim_data, self.car_num_display, self.frame)

                if self.capture:
                    pg.image.save(self.sim_draw.screen, "%simg%03d.jpeg" % (self.output_dir, self.frame))

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

        pg.quit()
        pickle.dump(self.sim_data, self.sim_out, pickle.HIGHEST_PROTOCOL)
        print('Output pickled and dumped.')
        if self.capture:
            # Compile to video
            os.system("ffmpeg -f image2 -framerate 5 -i %simg%%03d.jpeg %s/output_video.gif " % (self.output_dir, self.output_dir))
            # Delete images
            [os.remove(self.output_dir + file) for file in os.listdir(self.output_dir) if ".jpeg" in file]
            print("Simulation video output saved to %s." % self.output_dir)
        print("Simulation ended.")


if __name__ == "__main__":
    Main()