import os
import argparse
import pygame as pg
import datetime
import pickle
from constants import CONSTANTS as C
from environment import Environment
from inference_model import InferenceModel
from decision_model import DecisionModel
from autonomous_vehicle import AutonomousVehicle
from sim_draw import Sim_Draw
from sim_data import SimData


parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=str, choices=['intersection'], default='intersection')  # choose scenario
parser.add_argument('--sim_duration', type=int, default=100)  # time span for simulation
parser.add_argument('--dt', type=float, default=1.)  # time step in planning
# choose inference model: none - complete information
parser.add_argument('--inference', type=str, choices=['none', 'baseline', 'empathetic'], default='none')
# choose decision model: complete_information - complete information
parser.add_argument('--inference', type=str, choices=['none', 'baseline', 'empathetic'], default='none')

parser.add_argument('--tol', type=float, default=1e-3)  # tolerance for ode solver
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])  # method for computing gradient
parser.add_argument('--nepochs', type=int, default=100)  # number of training epochs
parser.add_argument('--lr', type=float, default=0.1)  # learning rate
parser.add_argument('--batch_size', type=int, default=20)  # batch size for training
parser.add_argument('--test_batch_size', type=int, default=20)  # batch size for validation and test
parser.add_argument('--save', type=str, default='./experiment1')  # save dir
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

class Simulation:

    def __init__(self, env, duration):

        self.duration = duration

        self.env = env
        self.agents = []

        if C.DRAW:
            self.sim_draw = Sim_Draw(self.P, C.ASSET_LOCATION)
            pg.display.flip()
            # self.capture = True if input("Capture video (y/n): ") else False
            self.capture = True
            output_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs("./sim_outputs/%s" % output_name)
            self.sim_out = open("./sim_outputs/%s/output.pkl" % output_name, "wb")

            if self.capture:
                self.output_dir = "./sim_outputs/%s/video/" % output_name
                os.makedirs(self.output_dir)

    def create_simulation(self, scenario='intersection'):

        # define scenarios
        if scenario == 'intersection':
            # intersection scenario with 2 cars

            # TODO: compile car parameters
            car_parameter = [P.CAR_1, P.CAR_2]
            inference_model = [InferenceModel("none", self), InferenceModel("none", self)]
            decision_model = [DecisionModel("complete_information", self), DecisionModel("complete_information", self)]

            # define agents
            self.agents = [AutonomousVehicle(sim=self, env=self.env, par=car_parameter[i],
                                             inference_model=inference_model[i],
                                             decision_model=decision_model[i],
                                             i=i) for i in range(len(car_parameter))]

    def run(self):
        while self.running:
            # Update model here
            if not self.paused:
                for agent in self.agents:

                    # Run simulation
                    agent.update(self)

                    # Update data
                    self.sim_data.append(agent)

                # # calculate gracefulness
                # grace = []
                # for wanted_trajectory_other in self.car_2.wanted_trajectory_other:
                #     wanted_actions_other = self.car_2.dynamic(wanted_trajectory_other)
                #     grace.append(1000*(self.car_1.states[-1][0] - wanted_actions_other[0][0]) ** 2)
                # self.car_1.social_gracefulness.append(sum(grace*self.car_2.inference_probability))

            # termination criteria
            if self.env.frame >= self.duration:
                break

            # draw stuff after each iteration
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
                # self.clock.tick(self.fps)

            if not self.paused:
                self.env.frame += 1

        pg.quit()

    def postprocess(self):
        import matplotlib.pyplot as plt
        import numpy as np
        car_1_theta = np.empty((0, 2))
        car_2_theta = np.empty((0, 2))
        for t in range(self.frame):
            car_1_theta = np.append(car_1_theta, np.expand_dims(self.sim_data.car2_theta_probability[t], axis=0), axis=0)
            car_2_theta = np.append(car_2_theta, np.expand_dims(self.sim_data.car1_theta_probability[t], axis=0), axis=0)
        plt.subplot(2, 1, 1)
        plt.plot(range(1,self.frame+1), car_1_theta[:,0], range(1,self.frame+1), car_1_theta[:,1])
        plt.subplot(2, 1, 2)
        plt.plot(range(1,self.frame+1), car_2_theta[:,0], range(1,self.frame+1), car_2_theta[:,1])
        plt.show()
        # pickle.dump(self.sim_data, self.sim_out, pickle.HIGHEST_PROTOCOL)
        print('Output pickled and dumped.')
        if self.capture:
            # Compile to video
            # os.system("ffmpeg -f image2 -framerate 1 -i %simg%%03d.jpeg %s/output_video.mp4 " % (self.output_dir, self.output_dir))
            img_list = [self.output_dir+"img"+str(i).zfill(3)+".jpeg" for i in range(self.frame)]
            import imageio
            images = []
            for filename in img_list:
                images.append(imageio.imread(filename))
            imageio.mimsave(self.output_dir+'movie.gif', images)
            #
            # # Delete images
            # [os.remove(self.output_dir + file) for file in os.listdir(self.output_dir) if ".jpeg" in file]
            # print("Simulation video output saved to %s." % self.output_dir)
        print("Simulation ended.")


if __name__ == "__main__":
    e = Environment()
    s = Simulation(e)
    s.run()
    # add analysis stuff here
    s.postprocess()