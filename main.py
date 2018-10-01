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
                                       loss_style='reactive',
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

                # calculate gracefulness
                grace = []
                for wanted_trajectory_other in self.car_2.wanted_trajectory_other:
                    wanted_actions_other = self.car_2.interpolate_from_trajectory(wanted_trajectory_other)
                    grace.append((self.car_1.actions_set[-1][0] - wanted_actions_other[0][0]) ** 2)
                self.car_1.social_gracefulness.append(sum(grace*self.car_2.inference_probability))

                # Update data
                self.sim_data.append_car1(states=self.car_1.states,
                                          actions=self.car_1.actions_set,
                                          action_sets=self.car_1.planned_actions_set,
                                          predicted_theta_other=self.car_1.predicted_theta_other,
                                          predicted_theta_self=self.car_1.predicted_theta_self,
                                          predicted_actions_other=self.car_1.predicted_actions_other,
                                          predicted_others_prediction_of_my_actions=
                                          self.car_1.predicted_others_prediction_of_my_actions,
                                          wanted_trajectory_self=self.car_1.wanted_trajectory_self,
                                          wanted_trajectory_other=self.car_1.wanted_trajectory_other,
                                          inference_probability=self.car_1.inference_probability,
                                          inference_probability_proactive=self.car_1.inference_probability_proactive,
                                          theta_probability=self.car_1.theta_probability,
                                          social_gracefulness=self.car_1.social_gracefulness)

                self.sim_data.append_car2(states=self.car_2.states,
                                          actions=self.car_2.actions_set,
                                          action_sets=self.car_2.planned_actions_set,
                                          predicted_theta_other=self.car_2.predicted_theta_other,
                                          predicted_theta_self=self.car_2.predicted_theta_self,
                                          predicted_actions_other=self.car_2.predicted_actions_other,
                                          predicted_others_prediction_of_my_actions=
                                          self.car_2.predicted_others_prediction_of_my_actions,
                                          wanted_trajectory_self=self.car_2.wanted_trajectory_self,
                                          wanted_trajectory_other=self.car_2.wanted_trajectory_other,
                                          inference_probability=self.car_2.inference_probability,
                                          inference_probability_proactive=self.car_2.inference_probability_proactive,
                                          theta_probability=self.car_2.theta_probability,)

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
        # pickle.dump(self.sim_data, self.sim_out, pickle.HIGHEST_PROTOCOL)
        print('Output pickled and dumped.')
        if self.capture:
            # Compile to video
            os.system("ffmpeg -f image2 -framerate 1 -i %simg%%03d.jpeg %s/output_video.mp4 " % (self.output_dir, self.output_dir))
            # img_list = [self.output_dir+"img"+str(i).zfill(3)+".jpeg" for i in range(self.frame)]
            # import imageio
            # images = []
            # for filename in img_list:
            #     images.append(imageio.imread(filename))
            # imageio.mimsave(self.output_dir+'movie.gif', images)

            # Delete images
            [os.remove(self.output_dir + file) for file in os.listdir(self.output_dir) if ".jpeg" in file]
            print("Simulation video output saved to %s." % self.output_dir)
        print("Simulation ended.")

        import matplotlib.pyplot as plt
        import numpy as np
        car_1_theta = np.empty((0,2))
        car_2_theta = np.empty((0,2))
        for t in range(self.frame):
            car_1_theta = np.append(car_1_theta, np.expand_dims(self.sim_data.car2_theta_probability[t],axis=0),axis=0)
            car_2_theta = np.append(car_2_theta, np.expand_dims(self.sim_data.car1_theta_probability[t],axis=0),axis=0)
        plt.subplot(2, 1, 1)
        plt.plot(range(1,self.frame+1), car_1_theta[:,0], range(1,self.frame+1), car_1_theta[:,1])
        plt.subplot(2, 1, 2)
        plt.plot(range(1,self.frame+1), car_2_theta[:,0], range(1,self.frame+1), car_2_theta[:,1])
        plt.show()

if __name__ == "__main__":
    Main()