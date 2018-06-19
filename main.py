from constants import CONSTANTS as C
from autonomous_vehicle import AutonomousVehicle
from sim_draw import Sim_Draw
from sim_data import Sim_Data
import pickle
import os
import pygame as pg
import pygame.camera as camera
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
        output_name = "output_%s" % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.sim_data = Sim_Data()
        self.sim_out = open("./sim_outputs/%s.pkl" % output_name, "wb")

        # self.sim_out = open("./sim_outputs/output_test.pkl", "wb")

        # Vehicle Definitions ('aggressive,'reactive','passive_aggressive')
        self.car_1 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_1,
                                       car_parameters_other=self.P.CAR_2,
                                       loss_style='passive_aggressive')
        self.car_2 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_2,
                                       car_parameters_other=self.P.CAR_1,
                                       loss_style='reactive')
        self.car_1.other_car = self.car_2
        self.car_2.other_car = self.car_1

        self.sim_draw = Sim_Draw(self.P, C.ASSET_LOCATION)

        self.capture = True if input("Capture video (y/n): ") else False
        if self.capture:
            self.output_dir = "./sim_outputs/%s" % output_name
            os.makedirs(self.output_dir)
            camera.init()
            self.cam = camera.Camera("%s/%s" % (self.output_dir, output_name) , (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE))
            self.cam.start()

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
                                          action_sets=self.car_1.planned_actions_set_s,
                                          predicted_theta_of_other=self.car_1.predicted_theta_of_other,
                                          prediction_of_actions_of_other=self.car_1.predicted_actions_of_other,
                                          prediction_of_others_prediction_of_my_actions=self.car_1.prediction_of_others_prediction_of_my_actions)

                self.sim_data.append_car2(states=self.car_2.states_s,
                                          actions=self.car_2.actions_set_s,
                                          action_sets=self.car_2.planned_actions_set_s,
                                          predicted_theta_of_other=self.car_2.predicted_theta_of_other,
                                          prediction_of_actions_of_other=self.car_2.predicted_actions_of_other,
                                          prediction_of_others_prediction_of_my_actions=self.car_2.prediction_of_others_prediction_of_my_actions)

            if self.frame >= self.duration:
                break

            # Draw frame
            self.sim_draw.draw_frame(self.sim_data, self.car_num_display, self.frame)

            if self.capture:
                filename = "%s/%04d.png" % (self.output_dir, self.frame)
                image = self.cam.get_image()
                pg.image.save(image, filename)

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

        pickle.dump(self.sim_data, self.sim_out, pickle.HIGHEST_PROTOCOL)
        print('Output pickled and dumped.')
        if camera:
            os.system("avconv -r 8 -f image2 -i Snaps/%%04d.png -y -qscale 0 -s %ix%i -aspect 4:3 result.avi" % (self.P.SCREEN_WIDTH * C.COORDINATE_SCALE, self.P.SCREEN_HEIGHT * C.COORDINATE_SCALE))
            print("Simulation output saved to %s." % self.output_dir)
        input("Simulation ended.")


if __name__ == "__main__":
    Main()