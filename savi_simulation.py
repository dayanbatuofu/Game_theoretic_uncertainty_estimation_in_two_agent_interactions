import os
from typing import List
import pygame as pg
import datetime
import pickle
import torch as t
import numpy as np
from inference_model import InferenceModel
from decision_model import DecisionModel
from autonomous_vehicle import AutonomousVehicle
from sim_draw import VisUtils
from models import constants as C  # for terminal state check (car length)
import pdb


class Simulation:

    def __init__(self, env, duration, n_agents, inference_type, decision_type, sim_dt, sim_lr, sim_par, sim_nepochs, belief_weight):

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
        self.inference_type = inference_type
        self.env = env
        self.agents = []
        self.sharing_belief = True  # TODO: set condition for this
        self.theta_priors = None  # For test_baseline and baseline inference
        self.drawing_prob = True  # if function for displaying future states are enabled
        self.saving_png = False  # TODO: implement this in run
        if env.name == 'bvp_intersection':  # don't draw future states
            self.drawing_prob = False

        # define simulation
        car_parameter = self.env.car_par
        "theta and lambda pairs (betas):"
        self.theta_list = sim_par["theta"]
        self.lambda_list = sim_par["lambda"]
        self.action_set = sim_par["action_set"]
        # self.action_set_combo = [[-0.05,8], [0,-8], [0.05,-8], [-0.05,-4], [0,-4],
        #                          [0.05,-4], [-0.05,0], [0, 0], [0.05,0], [-0.05,4],
        #                          [0,4], [0.05,4], [-0.05, 8], [0,8], [0.05,8]]  # merging case actions
        self.action_set_combo = [[-0.05, -4], [0.05, -4], [0, 0], [0.05, 4], [-0.05, 0.4]]# CHANGE THIS LATER SUNNY

        if self.env.name == 'merger':
            self.action_set = self.action_set_combo

        'getting ground truth betas'
        self.true_params = []
        for i, par_i in enumerate(self.env.car_par):
            self.true_params.append(par_i["par"])
        'getting initial beliefs of others'
        self.belief_params = []
        for i, par_i in enumerate(self.env.car_par):
            self.belief_params.append(par_i["belief"])
        # ----------------------------------------------------------------------------------------
        # beta: [theta1, lambda1], [theta1, lambda2], ... [theta2, lambda4] (2x4 = 8 set of betas)
        # betas: [ [theta1, lambda1], [theta1, lambda2], [theta1, lambda3], [theta1, lambda4],
        #          [theta2, lambda1], [theta2, lambda2], [theta2, lambda3], [theta2, lambda4] ]
        # ----------------------------------------------------------------------------------------
        self.beta_set = []
        '1D version of beta'
        for i, theta in enumerate(self.theta_list):
            for j, _lambda in enumerate(self.lambda_list):
                self.beta_set.append([theta, _lambda])
        self.action_distri_1 = []
        self.action_distri_2 = []
        self.belief_weight = belief_weight
        self.initial_belief = self.get_initial_belief(self.env.car_par[1]['belief'][0],
                                                      self.env.car_par[0]['belief'][0],
                                                      self.env.car_par[1]['belief'][1],
                                                      self.env.car_par[0]['belief'][1],
                                                      weight=self.belief_weight)  # note: use params from the other agent's belief
        self.past_loss1 = []  # for storing loss of simulation
        self.past_loss2 = []
        if self.n_agents == 2:
            # simulations with 2 cars
            # Note that variable annotation is not supported in python 3.5

            self.current_q_values = {0: None, 1: None}  # TODO: get q values for current frame

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
            self.calc_loss()
            # termination criteria
            if self.frame >= 25:  # TODO: modify frame limit
                print('Simulation ended with duration exceeded limit')
                break
            # pdb.set_trace()
            x_H = self.agents[0].state[self.frame][0]  # sy_H ??
            x_M = self.agents[1].state[self.frame][0]  # sx_M
            y_H = self.agents[0].state[self.frame][1]  # sy_H
            y_M = self.agents[1].state[self.frame][1]  # sy_M
            if self.env.name == "merger":
                if y_H >= 50 and y_M > 50:
                    print("terminating on vehicle merger:")
                    break
            elif self.env.name == 'bvp_intersection':
                if y_H >= 40 and x_M >= 40:
                    print("terminating on vehicle passed intersection:", y_H, x_M)
                    break
            else:
                if y_H >= 5 and x_M <= -5:
                    # road width = 2.0 m
                    # if crossed the intersection, done or max time reached
                    # if (x_ego >= 0.5 * C.CONSTANTS.CAR_LENGTH + 10. and x_other <= -0.5 * C.CONSTANTS.CAR_LENGTH - 10.):
                    print("terminating on vehicle passed intersection:", x_H, x_M)
                    break

            # TODO: update visualization
            # draw stuff after each iteration
            if self.draw:
                self.vis.draw_frame()  # Draw frame
                # if self.capture:
                #     pg.image.save(v.screen, "%simg%03d.jpeg" % (self.output_dir, self.frame))

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
        self.vis.plot_loss()
        print("-------Simulation results:-------")
        print("inference types:", self.inference_type)
        print("decision types:", self.decision_type)
        print("initial intents:", self.env.car_par[0]['par'], self.env.car_par[1]['par'])
        print("Frames:", self.frame)
        print("len of action and states:", len(self.agents[0].action), len(self.agents[0].state))
        print("action distribution", self.action_distri_1)
        print("Initial belief:", self.initial_belief)
        # print("states of H:", self.agents[0].state)
        # print("states of H predicted by M:", self.agents[1].predicted_states_other)
        print("Action taken by H:", self.agents[0].action)
        print("Action of H predicted by M:", self.agents[1].predicted_actions_other)
        print("Action taken by M:", self.agents[1].action)
        print("Loss of H (p1):", self.past_loss1)
        print("Loss of M (p2):", self.past_loss1)

    # TODO: store this somewhere else
    def get_initial_belief(self, theta_h, theta_m, lambda_h, lambda_m, weight):
        """
        Obtain initial belief of the params
        :param theta_h:
        :param theta_m:
        :param lambda_h:
        :param lambda_m:
        :param weight:
        :return:
        """
        # TODO: given weights for certain param, calculate the joint distribution (p(theta_1), p(lambda_1) = 0.8, ...)
        theta_list = self.theta_list
        lambda_list = self.lambda_list
        beta_list = self.beta_set

        if self.inference_type[1] == 'empathetic' or self.inference_type[1] == 'bvp_empathetic':
            # beta_list = beta_list.flatten()
            belief = np.ones((len(beta_list), len(beta_list)))
            for i, beta_h in enumerate(beta_list):  # H: the rows
                for j, beta_m in enumerate(beta_list):  # M: the columns
                    if beta_h[0] == theta_h:  # check lambda
                        belief[i][j] *= weight
                        if beta_h[1] == lambda_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                        if beta_h[1] == lambda_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)

                    if beta_m[0] == theta_m:  # check lambda
                        belief[i][j] *= weight
                        if beta_m[1] == lambda_m:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                        if beta_m[1] == lambda_m:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)

                    # if beta_h == [lambda_h, theta_h] and beta_m == [lambda_m, theta_m]:
                    #     belief[i][j] = weight
                    # else:
                    #     belief[i][j] = 1

        # TODO: not in use! we only use the game theoretic inference
        else:  # get belief on H agent only
            belief = np.ones((len(lambda_list), len(theta_list)))
            for i, lamb in enumerate(lambda_list):
                for j, theta in enumerate(theta_list):
                    if lamb == lambda_h:  # check lambda
                        belief[i][j] *= weight
                        if theta == theta_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                        if theta == theta_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
        # THIS SHOULD NOT NEED TO BE NORMALIZED!
        # print(belief, np.sum(belief))
        assert round(np.sum(belief)) == 1
        return belief

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

    def get_current_q(self):  # TODO: get current q for both agents, for faster calculation
        pass

    def calc_loss(self):
        """
        Calculate loss function after each step
        :return:
        """
        state_h = self.agents[0].state[self.frame]
        state_m = self.agents[1].state[self.frame]
        xh = t.tensor(state_h[1], requires_grad=True, dtype=t.float32)
        xm = t.tensor(state_m[0], requires_grad=True, dtype=t.float32)
        theta1 = 1
        theta2 = 1
        R1 = 70
        R2 = 70
        W1 = 1.5
        W2 = 1.5
        L1 = 3
        L2 = 3
        beta = 10000.
        x1_in = (xh - R1 / 2 + theta2 * W2 / 2) * 10
        x1_out = -(xh - R1 / 2 - W2 / 2 - L1) * 10
        x2_in = (xm - R2 / 2 + theta1 * W1 / 2) * 10
        x2_out = -(xm - R2 / 2 - W1 / 2 - L2) * 10

        Collision_F_x = beta * t.sigmoid(x1_in) * t.sigmoid(x1_out) * \
                        t.sigmoid(x2_in) * t.sigmoid(x2_out)
        # U1 = self.agents[0].action[self.frame]
        # U2 = self.agents[1].action[self.frame]
        # L1 = U1 ** 2 + Collision_F_x.detach().numpy()
        # L2 = U2 ** 2 + Collision_F_x.detach().numpy()
        L1 = 1*Collision_F_x.detach().numpy()
        L2 = 1*Collision_F_x.detach().numpy()
        self.past_loss1.append(L1)
        self.past_loss2.append(L2)
        return
