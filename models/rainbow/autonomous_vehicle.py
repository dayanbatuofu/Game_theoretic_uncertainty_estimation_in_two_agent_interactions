import numpy as np
import pygame as pg
import sys
sys.path.append('../')
from constants import CONSTANTS as C


class AutonomousVehicle:
    def __init__(self, car_parameters, control_style, who):

        self.car_parameters = car_parameters
        self.control_style = control_style
        self.who = who
        self.image = pg.transform.rotate(pg.transform.scale(pg.image.load(C.ASSET_LOCATION + self.car_parameters.SPRITE),
                                                            (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                             int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))),
                                         self.car_parameters.ORIENTATION)

        # Initialize agent state, action
        self.state = [0, 0]
        self.state[0] = np.random.uniform(C.Intersection.CAR_1.INITIAL_STATE[0] * 0.5,
                                          C.Intersection.CAR_1.INITIAL_STATE[0] * 1.0)
        max_speed = np.sqrt((self.state[0]-1-C.CAR_LENGTH*0.5)*2.*abs(C.Intersection.MAX_DECELERATION))
        self.state[1] = np.random.uniform(max_speed * 0.1, max_speed * 0.5)
        # self.state = self.car_parameters.INITIAL_STATE
        self.isReached = False
        self.action = 0  # by default no acceleration
        self.aggressiveness = self.car_parameters.aggressiveness
        self.gracefulness = self.car_parameters.gracefulness

        # Initialize others space
        self.other_car = []

        # initialize pre-trained model if RL agent
        if self.control_style == 'pre-trained':
            pass
            #self.NN = NN(state_size, action_size, BATCH_SIZE, SIZE_HIDDEN, LEARNING_RATE, ACTIVATION)
            #self.NN.load('weights.h5')
            # delete the current graph
            # tf.reset_default_graph()

    # Loss function not penalizing deceleration
    def self_loss(self, env):
        # define instantaneous loss
        # control_loss = (self.action != 0)**2  # less control effort is better

        # loss to not go over the intersection, road width = 2.0 m
        intent_loss_self = 0

        # if not moving forward and not reaching the target, a constant loss
        if self.state[0] + C.CAR_LENGTH * 0.5 + 1. > 0. and self.state[1] <= 0.1:
            intent_loss_self = env.time_interval / env.min_time_interval  # reward when reaching the target

        if env.done and self.state[0] + C.CAR_LENGTH * 0.5 + 1. > 0.:
            intent_loss_self += 1e3

        collision_penalty = 1e3

        collision_loss = env.collision * collision_penalty

        loss_self = - (collision_loss + self.aggressiveness * intent_loss_self)

        return loss_self
        # scale the intent loss according to the aggressiveness

    # Extra Loss function penalizing deceleration
    def self_loss_acc(self, env, acc):
        # define instantaneous loss
        # control_loss = (self.action != 0)**2  # less control effort is better

        # loss to not go over the intersection, road width = 2.0 m
        intent_loss_self = 0

        # if not moving forward and not reaching the target, a constant loss
        if self.state[0] + C.CAR_LENGTH * 0.5 + 1. > 0.:
            if self.state[1] <= 0.1:
                intent_loss_self = env.time_interval / env.min_time_interval  # reward when reaching the target
            elif acc < 0 and self.state[1] > 0.1:
                intent_loss_self = 0.1

        if env.done and self.state[0] + C.CAR_LENGTH * 0.5 + 1. > 0.:
            intent_loss_self += 1e3

        collision_penalty = 1e3

        collision_loss = env.collision * collision_penalty

        loss_self = - (collision_loss + self.aggressiveness * intent_loss_self)

        return loss_self
        # scale the intent loss according to the aggressiveness

    # Loss function used by MPC policy during rollout
    def loss(self, env, state, done, collision, acc, args):
        # define instantaneous loss
        # control_loss = (self.action != 0)**2  # less control effort is better

        # loss to not go over the intersection, road width = 2.0 m
        intent_loss_self = 0

        # if not moving forward and not reaching the target, a constant loss
        if state[0] + C.CAR_LENGTH * 0.5 + 1. > 0. and state[1] <= 0.1:
            intent_loss_self = env.time_interval / env.min_time_interval  # reward when reaching the target
            # break

        if args.acc_loss:
            if state[0] + C.CAR_LENGTH * 0.5 + 1. > 0.:
                if state[1] <= 0.1:
                    intent_loss_self = env.time_interval / env.min_time_interval  # reward when reaching the target
                elif acc < 0 and state[1] > 0.1:
                    intent_loss_self = 0.1

        if done and state[0] + C.CAR_LENGTH * 0.5 + 1. > 0.:
            intent_loss_self += 1e3

        collision_penalty = 1e3

        collision_loss = collision * collision_penalty

        loss_self = - (collision_loss + self.aggressiveness * intent_loss_self)

        return loss_self
        # scale the intent loss according to the aggressiveness

    def get_collision_loss(self, env):
        collision_penalty = 1e3

        collision_loss = env.collision * collision_penalty
        return collision_loss


    def best_case_self_loss(self):
        # intent_loss = self.aggressiveness * (self.car_parameters.MAX_SPEED[0] - self.action)
        return 0

    def get_action(self):
        # reinforce_non_courteous, baseline
        if self.control_style == 'pre-trained':
            x1 = self.state[0]
            x2 = self.other_car.state[0]
            v1 = self.state[1]
            v2 = self.other_car.state[1]
            state = np.array([x1, x2, v1, v2])
            action = self.NN.best_action(state, usetarget=False)
            # compute action from action index
            if action == 0:
                a = self.agent.env.parameters.MAX_DECELERATION
            elif action == 1:
                a = self.agent.env.parameters.MAX_DECELERATION * 0.5
            elif action == 2:
                a = 0.0
            elif action == 3:
                a = self.agent.env.parameters.MAX_ACCELERATION * 0.5
            else:
                a = self.agent.env.parameters.MAX_ACCELERATION
            self.action = a
            return self.action

        elif self.control_style == 'baseline':
            #TODO implement baseline agent
            x1 = self.state[0]
            x2 = self.other_car.state[0]
            v1 = self.state[1]
            v2 = self.other_car.state[1]
            l = C.CAR_LENGTH
            w = C.CAR_WIDTH

            # if car passed the intersection, keep constant speed
            # if x1 < -0.5*l:
            #     self.action = 0
            #     return self.action

            t1_start = (x1 - 0.5 * l - 1) / (v1 + 1e-6)
            t1_end = (x1 + 0.5*l + 1)/(v1+1e-6)
            xx_start = x2 - t1_start * v2
            xx_end = x2 - t1_end*v2
            collision = True
            if (xx_start <= -0.5*l - 1 or xx_end >= 0.5*l + 1) or t1_end > 100:
                collision = False

            # if will collide, brake hard
            if collision:
                self.action = C.Intersection.MAX_DECELERATION
            # if speed too low and no collision, accelerate
            elif v1 < C.Intersection.INITIAL_SPEED and x2 < -1. - C.CAR_LENGTH * 0.5:
                self.action = C.Intersection.MAX_ACCELERATION
            # if the other car is not moving, accelerate
            elif v2 < 0.1:
                self.action = C.Intersection.MAX_ACCELERATION
            else:
                self.action = 0
            return self.action

        else:
            self.action = 0
            return self.action

