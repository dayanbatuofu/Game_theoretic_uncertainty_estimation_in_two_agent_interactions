import os
from typing import List
import pygame as pg
import datetime
import pickle
import torch as t
from inference_model import InferenceModel
from decision_model import DecisionModel
from autonomous_vehicle import AutonomousVehicle
from sim_draw import VisUtils
from models import constants as C #for terminal state check (car length)
import pdb

class Simulation:

    def __init__(self, env, duration, n_agents, inference_type, decision_type, sim_dt, sim_lr, sim_nepochs):

        self.duration = duration
        self.n_agents = n_agents
        self.dt = sim_dt
        self.running = True
        self.paused = False
        self.end = False
        self.clock = pg.time.Clock()
        self.frame = 0
        self.decision_type = decision_type
        self.decision_type_h = decision_type[0]
        self.decision_type_m = decision_type[1]
        #print("decision type:", decision_type)
        self.env = env
        self.agents = []
        self.theta_priors = None
        self.drawing_prob = True  # if function for displaying future states are enabled
        # define simulation
        car_parameter = self.env.car_par
        "theta and lambda pairs (betas):"
        self.theta_list = [1, 1000]
        self.lambda_list = [0.001, 0.005, 0.01, 0.05]
        self.action_set = [-8, -4, 0, 4, 8]

        if self.n_agents == 2:
            # simulations with 2 cars
            #Note that variable annotation is not supported in python 3.5

            inference_model: List[InferenceModel] = [InferenceModel(inference_type[i], self) for i in range(n_agents)]
            decision_model: List[DecisionModel] = [DecisionModel(decision_type[i], self) for i in range(n_agents)]

            # define agents
            self.agents = [AutonomousVehicle(sim=self, env=self.env, par=car_parameter[i],
                                             inference_model=inference_model[i],
                                             decision_model=decision_model[i],
                                             i=i) for i in range(len(car_parameter))]

        self.draw = True  # visualization during sim
        self.capture = False  # save images during visualization
        # DISPLAY
        if self.draw:
            # TODO: update visualization
            self.vis = VisUtils(self)  # initialize visualization
            # self.vis.draw_frame()
            # if self.capture:
            #     output_name = datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s")
            #     os.makedirs("./sim_outputs/%s" % output_name)
            #     self.sim_out = open("./sim_outputs/%s/output.pkl" % output_name, "wb")
            #     self.output_dir = "./sim_outputs/%s/video/" % output_name
            #     os.makedirs(self.output_dir)

    def snapshot(self):
        # take a snapshot of the current system state
        return self.agents.copy()

    def run(self):
        while self.running:
            # Update model here
            if not self.paused:
                for agent in self.agents:
                    agent.update(self)  # Run simulation

            # termination criteria

            #if self.decision_type == 'baseline':
            x_H = self.agents[0].state[-1][1] #sy_H
            x_M = self.agents[1].state[-1][0] #sx_M
            if self.frame >= 14: #TODO: modify frame limit
                break
            # if crossed the intersection, done or max time reached
            #if (x_ego >= 0.5 * C.CONSTANTS.CAR_LENGTH + 10. and x_other <= -0.5 * C.CONSTANTS.CAR_LENGTH - 10.):
            if (x_H >= 15 and x_M <= -20):
                # road width = 2.0 m
                print("terminating on vehicle passed intersection:", x_H, x_M )
                break

            if self.frame >= self.duration:
                break

            #pdb.set_trace()

            if self.decision_type == 'baseline':
                x_H = self.agents[0].state[-1][1] #sy_H ??
                x_M = self.agents[1].state[-1][0] #sx_M
                y_H = self.agents[0].state[-1][1] #sy_H
                y_M = self.agents[1].state[-1][1] #sy_M
                if self.frame >= 14: #TODO: modify frame limit
                    break
                # if crossed the intersection, done or max time reached
                #if (x_ego >= 0.5 * C.CONSTANTS.CAR_LENGTH + 10. and x_other <= -0.5 * C.CONSTANTS.CAR_LENGTH - 10.):
                if self.env.name == "trained_intersection":
                    if (x_H >= 100 and x_M <= -50):
                        # road width = 2.0 m
                        print("terminating on vehicle passed intersection:", x_H, x_M )
                        break
                if self.env.name == "merger":
                    if (y_H>= 50 and y_M>50):
                        print ("terminating on vehicle merger:")
                        break
            else:
                if self.frame >= self.duration:
                    break

            # TODO: update visualization
            # draw stuff after each iteration
            if self.draw:
                self.vis.draw_frame()  # Draw frame
                # if self.capture:
                #     pg.image.save(v.screen, "%simg%03d.jpeg" % (self.output_dir, self.frame))
                #TODO: add occupancy plot (Yi)


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
                        # if event.key == pg.K_d:
                        #     self.car_num_display = ~self.car_num_display

                # Keep fps

            if not self.paused:
                self.frame += 1
        pg.quit()
        "drawing results"
        self.vis.draw_dist()
        if self.drawing_prob:
            self.vis.draw_intent()
        print("Frames:", self.frame)
        print("states of H:", self.agents[0].state)
        print("states of H predicted by M:", self.agents[1].predicted_states_other)
        print("Action taken by H:", self.agents[0].action)
        print("Action of H predicted by M:", self.agents[1].predicted_actions_other)
        print("Action taken by M:", self.agents[1].action)

    def reset(self):
        # reset the simulation
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 0

    def postprocess(self):
        # import matplotlib.pyplot as plt
        # import numpy as np
        # car_1_theta = np.empty((0, 2))
        # car_2_theta = np.empty((0, 2))
        # for t in range(self.frame):
        #     car_1_theta = np.append(car_1_theta, np.expand_dims(self.sim_data.car2_theta_probability[t], axis=0), axis=0)
        #     car_2_theta = np.append(car_2_theta, np.expand_dims(self.sim_data.car1_theta_probability[t], axis=0), axis=0)
        # plt.subplot(2, 1, 1)
        # plt.plot(range(1,self.frame+1), car_1_theta[:,0], range(1,self.frame+1), car_1_theta[:,1])
        # plt.subplot(2, 1, 2)
        # plt.plot(range(1,self.frame+1), car_2_theta[:,0], range(1,self.frame+1), car_2_theta[:,1])
        # plt.show()
        # # pickle.dump(self.sim_data, self.sim_out, pickle.HIGHEST_PROTOCOL)
        # print('Output pickled and dumped.')
        # if self.capture:
        #     # Compile to video
        #     # os.system("ffmpeg -f image2 -framerate 1 -i %simg%%03d.jpeg %s/output_video.mp4 " % (self.output_dir, self.output_dir))
        #     img_list = [self.output_dir+"img"+str(i).zfill(3)+".jpeg" for i in range(self.frame)]
        #     import imageio
        #     images = []
        #     for filename in img_list:
        #         images.append(imageio.imread(filename))
        #     imageio.mimsave(self.output_dir+'movie.gif', images)
        #     #
        #     # # Delete images
        #     # [os.remove(self.output_dir + file) for file in os.listdir(self.output_dir) if ".jpeg" in file]
        #     # print("Simulation video output saved to %s." % self.output_dir)
        # print("Simulation ended.")
        pass



