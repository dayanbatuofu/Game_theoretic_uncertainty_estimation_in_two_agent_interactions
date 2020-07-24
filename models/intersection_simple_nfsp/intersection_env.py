import random
import time

import numpy as np
import pygame as pg
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from autonomous_vehicle import AutonomousVehicle
from sim_draw import Sim_Draw
from constants import CONSTANTS as C
import logging
logging.basicConfig(level=logging.WARN)

class IntersectionEnv:
    def __init__(self, control_style_ego, control_style_other, time_interval, max_time_steps):

        # Setup
        self.duration = 100
        self.parameters = C.Intersection  # Scenario parameters choice
        # Time handling
        self.clock = pg.time.Clock()
        self.fps = C.FPS
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 0
        self.car_num_display = 0
        self.time_interval = time_interval
        self.min_time_interval = C.Intersection.MIN_TIME_INTERVAL
        self.max_time_steps = max_time_steps
        self.isCollision = False
        self.args = None
        self.state = []  # This is the state used in RL
        self.action = []  # collection of current actions from the agents
        self.collision = 0  # number of min time steps with collision

        self.ego_car = AutonomousVehicle(car_parameters=self.parameters.CAR_1,
                                         control_style=control_style_ego, who='M')  # autonomous car
        self.other_car = AutonomousVehicle(car_parameters=self.parameters.CAR_2,
                                           control_style=control_style_other, who='H')  # human car
        self.ego_car.other_car = self.other_car
        self.other_car.other_car = self.ego_car

        self.seed()
        self.renderer = Sim_Draw(self.parameters, C.ASSET_LOCATION)
        self.done = False
        self.paused = False

    def seed(self, seed=None):
        random.seed(seed)
        return [seed]

    def step(self, action):
        if action['1'] == 0:
            a = self.parameters.MAX_DECELERATION
        elif action['1'] == 1:
            a = self.parameters.MAX_DECELERATION * 0.5
        elif action['1'] == 2:
            a = 0.0
        elif action['1'] == 3:
            a = self.parameters.MAX_ACCELERATION * 0.5
        else:
            a = self.parameters.MAX_ACCELERATION
        action_self = a
        if action['2'] == 0:
            a = self.parameters.MAX_DECELERATION
        elif action['2'] == 1:
            a = self.parameters.MAX_DECELERATION * 0.5
        elif action['2'] == 2:
            a = 0.0
        elif action['2'] == 3:
            a = self.parameters.MAX_ACCELERATION * 0.5
        else:
            a = self.parameters.MAX_ACCELERATION
        action_other = a  # the other car take one step

        self.ego_car.action = action_self
        self.other_car.action = action_other

        # show what action was taken
        # self.render()

        # get current states
        x_ego = x_ego_new = self.ego_car.state[0]
        x_other = x_other_new = self.other_car.state[0]
        v_ego = v_ego_new = self.ego_car.state[1]
        v_other = v_other_new = self.other_car.state[1]

        max_speed_ego = self.ego_car.car_parameters.MAX_SPEED[0]
        min_speed_ego = self.ego_car.car_parameters.MAX_SPEED[1]
        max_speed_other = self.other_car.car_parameters.MAX_SPEED[0]
        min_speed_other = self.other_car.car_parameters.MAX_SPEED[1]

        # update state and check for collision
        l = C.CAR_LENGTH
        w = C.CAR_WIDTH
        self.collision = 0
        for t in range(int(self.time_interval / self.min_time_interval)+1):
            v_ego_new = max(min(max_speed_ego, action_self * (t) * self.min_time_interval + v_ego), min_speed_ego)
            v_other_new = max(min(max_speed_other, action_other * (t) * self.min_time_interval + v_other), min_speed_other)
            x_ego_new = x_ego - (t) * 0.5 * (v_ego_new + v_ego) * self.min_time_interval
            x_other_new = x_other - (t) * 0.5 * (v_other_new + v_other) * self.min_time_interval
            collision_box1 = [[x_ego_new - 0.5 * l, -0.5 * w],
                              [x_ego_new - 0.5 * l, 0.5 * w],
                              [x_ego_new + 0.5 * l, 0.5 * w],
                              [x_ego_new + 0.5 * l, -0.5 * w]]
            collision_box2 = [[0.5 * w, x_other_new - 0.5 * l],
                              [-0.5 * w, x_other_new - 0.5 * l],
                              [-0.5 * w, x_other_new + 0.5 * l],
                              [0.5 * w, x_other_new + 0.5 * l]]
            c = 0
            polygon = Polygon(collision_box2)
            for p in collision_box1:
                point = Point(p[0], p[1])
                c += polygon.contains(point)
                if c > 0:
                    break
            self.collision += float(c > 0)  # number of times steps of collision
        logging.debug('x_ego:{}, x_other:{}'.format(x_ego_new, x_other_new))

        if self.collision > 0:
            self.isCollision = True

        v_ego_new = max(min(max_speed_ego, action_self * self.time_interval + v_ego), min_speed_ego)
        x_ego -= 0.5 * (v_ego_new + v_ego) * self.time_interval  # start from positive distance to the center,
        # reduce to 0 when at the center
        v_ego = v_ego_new
        v_other_new = max(min(max_speed_other, action_other * self.time_interval + v_other), min_speed_other)
        x_other -= 0.5 * (v_other_new + v_other) * self.time_interval
        v_other = v_other_new
        logging.debug('x_ego:{}, x_other:{}'.format(x_ego, x_other))
        logging.debug('------')

        # flag set if cars cross intersection
        if x_ego <= -0.5 * C.CAR_LENGTH - 1.:
            self.ego_car.isReached = True

        if x_other <= -0.5 * C.CAR_LENGTH - 1.:
            self.other_car.isReached = True

        self.ego_car.state = [x_ego, v_ego]
        self.other_car.state = [x_other, v_other]
        self.state = (x_ego, v_ego, x_other, v_other)

        # if crossed the intersection, done or max time reached
        if (x_ego <= -0.5 * C.CAR_LENGTH - 1. and x_other <= -0.5 * C.CAR_LENGTH - 1.):  # road width = 2.0 m
            self.done = True

        # if max time step reached, done
        if self.frame >= self.max_time_steps:
            self.done = True

        if self.args.acc_loss:
            loss_self = self.ego_car.self_loss_acc(self, action_self)
            loss_other = self.other_car.self_loss_acc(self, action_other)

        else:
            loss_self = self.ego_car.self_loss(self)
            loss_other = self.other_car.self_loss(self)

        loss = np.array([loss_self, loss_other])
        self.frame += 1
        return np.array(self.state), loss, self.done

    def reset(self):
        self.ego_car.state[0] = np.random.uniform(C.Intersection.CAR_1.INITIAL_STATE[0] * 0.5,
                                                  C.Intersection.CAR_1.INITIAL_STATE[0] * 1.0)
        max_speed = np.sqrt((self.ego_car.state[0]-1-C.CAR_LENGTH*0.5)*2.*abs(C.Intersection.MAX_DECELERATION))
        self.ego_car.state[1] = np.random.uniform(max_speed * 0.1, max_speed * 0.5)
        self.other_car.state[0] = np.random.uniform(C.Intersection.CAR_2.INITIAL_STATE[0] * 0.5,
                                                    C.Intersection.CAR_2.INITIAL_STATE[0] * 1.0)
        max_speed = np.sqrt((self.other_car.state[0]-1-C.CAR_LENGTH*0.5) * 2. * abs(C.Intersection.MAX_DECELERATION))
        self.other_car.state[1] = np.random.uniform(max_speed * 0.1, max_speed * 0.5)
        # self.ego_car.state = C.Intersection.CAR_1.INITIAL_STATE
        # self.other_car.state = C.Intersection.CAR_2.INITIAL_STATE
        self.done = False
        self.frame = 0
        self.isCollision = False
        self.ego_car.isReached = False
        self.other_car.isReached = False
        # self.state = (self.ego_car.state[0], self.ego_car.state[1],
        #               self.other_car.state[0], self.other_car.state[1], self.frame)
        self.state = (self.ego_car.state[0], self.ego_car.state[1],
                      self.other_car.state[0], self.other_car.state[1])

        return np.array(self.state)

    def reset_state(self, state):
        self.ego_car.state[0] = state[0]
        self.ego_car.state[1] = state[1]
        self.other_car.state[0] = state[2]
        self.other_car.state[1] = state[3]

        self.done = False
        self.frame = 0
        self.isCollision = False
        self.ego_car.isReached = False
        self.other_car.isReached = False
        self.state = state[:-1]
        return np.array(self.state)


    def render(self):
        # TODO: render one step
        self.renderer.draw_frame(self)

    def render_fullcycle(self, state_1_history, state_2_history):
        self.frame = 0
        FRAME_TIME = self.time_interval
        frame = FRAME_TIME

        for state1, state2 in zip(state_1_history, state_2_history):

            if frame < FRAME_TIME:
                time.sleep(FRAME_TIME-frame)

            start = time.time()

            self.ego_car.state = [state1[0], state1[1]]
            self.other_car.state = [state1[0], state1[1]]
            self.frame += 1

            frame = time.time() - start

            self.render()

    def simulate(self):
        FRAME_TIME = self.time_interval
        frame = FRAME_TIME

        while not self.done:
            if frame < FRAME_TIME:
                time.sleep(FRAME_TIME - frame)

            start = time.time()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    self.done = True

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_p:
                        self.paused = not self.paused

                    if event.key == pg.K_q:
                        pg.quit()
                        self.done = True

                    if event.key == pg.K_d:
                        self.car_num_display = ~self.car_num_display

            if not self.paused:

                a = self.ego_car.get_action()
                self.step(a)

                self.render()

            frame = time.time() - start
