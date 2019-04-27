from constants import CONSTANTS as C
from constants import MATRICES as M
import numpy as np
import scipy
import bezier
from collision_box import Collision_Box
from loss_functions import LossFunctions
from scipy import optimize
import pygame as pg
from scipy.interpolate import spline
from scipy import stats
import time
import matplotlib.pyplot as plt


class AutonomousVehicle:
    """
    States:
            X-Position
            Y-Position
    """


    def __init__(self, scenario_parameters, car_parameters_self, who):
        self.P = scenario_parameters
        self.P_CAR = car_parameters_self
        self.who = who


        self.states = [self.P_CAR.INITIAL_POSITION]
        self.speed = [self.P_CAR.INITIAL_SPEED]
        self.ability = self.P_CAR.ABILITY
        self.orientation = [self.P_CAR.ORIENTATION]

        # Initialize others space
        self.states_o1 = []
        self.actions_set_o1 = []
        self.speed_o1 = []
        self.other_car_1 = []
        self.states_o2 = []
        self.actions_set_o2 = []
        self.speed_o2 = []
        self.other_car_2 = []
        self.collision = [0]
        self.FOV = [1]  # the vehicle sees the other dummy vehicle

    def update(self, frame):
        other1 = self.other_car_1
        other2 = self.other_car_2
        self.frame = frame

        # if len(self.states_o) > C.TRACK_BACK and len(self.states) > C.TRACK_BACK:
        self.track_back = min(C.TRACK_BACK, len(self.states))

        new_FOV = self.check_view()
        new_collision = self.check_collision()
        ########## Calculate machine actions here ###########
        planned_actions = self.get_actions(new_FOV, new_collision)


        planned_states_path, planned_speed = self.dynamic(planned_actions)
        planned_states, orientation = self.traj2path(planned_states_path,3)
        self.states.append(planned_states)
        self.speed.append(planned_speed)
        self.orientation.append(orientation)
        self.FOV.append(new_FOV)
        self.collision.append(new_collision)


    def check_view(self):

        if self.frame >= 2:
            action_car1 = (self.other_car_1.speed[-1] - self.other_car_1.speed[-2])
            if action_car1 > 0:
                new_FOV = 0
            else:
                new_FOV = 1
        else:
            new_FOV = 1

        return new_FOV


    def check_collision(self):
        car1_states = [self.other_car_1.states[-1]]
        car2_states = [self.other_car_2.states[-1]]
        car1_speed = np.asarray(self.other_car_1.speed[-1])
        car2_speed = np.asarray(self.other_car_2.speed[-1])
        car1_orientation = [self.other_car_1.orientation[-1]]
        car2_orientation = [self.other_car_2.orientation[-1]]
        distance = [np.linalg.norm(car1_states[-1]-car2_states[-1])]

        for i in range(100):
            state1 = self.path2traj(car1_states[-1],1,car1_orientation[-1])
            state2 = self.path2traj(car2_states[-1],2,car2_orientation[-1])
            new_state1,new_orientation1 = self.traj2path(state1+car1_speed, 1)
            new_state2,new_orientation2 = self.traj2path(state2+car2_speed, 2)
            car1_orientation.append(new_orientation1)
            car2_orientation.append(new_orientation2)
            car1_states.append(new_state1)
            car2_states.append(new_state2)
            distance.append(np.linalg.norm(new_state1-new_state2))
        if np.min(distance) <= 0.9:
            collision = 1
        else:
            collision = 0
        return collision



    def get_actions(self, FOV,collision):
        if collision == 1:
            if FOV == 0:
                actions = 2

            elif FOV == 1:
                actions = -2
        else:
            actions = -2
        return actions

    def dynamic(self, action):
        vel_self = np.array(self.speed[-1])
        path_state = np.array(self.states[-1])
        state = path_state[0]
        acci = np.array([action * self.ability])

        planned_speed = vel_self + acci
        planned_speed = np.clip(planned_speed, 0, C.VEHICLE_MAX_SPEED)
        predict_result_traj = state + planned_speed

        return predict_result_traj, planned_speed

    def traj2path(self, traj_state, who):
        if who == 1:
            if traj_state <= -0.8:
                path_state = np.array([traj_state[0],0.4])
                orientation = 0
            elif traj_state >= 0.6*np.pi - 0.8:
                x = 0.4
                y = -0.8 - (-0.6 * np.pi + 0.8 + traj_state)
                orientation = 90
                path_state = np.array([x, y])
            else:
                arc_lenth = traj_state[0] + 0.8
                orientation = (arc_lenth / (0.6 * np.pi)) * 90
                orientation = np.clip(orientation,0,90)
                y = np.cos(orientation * np.pi / 180)  * 1.2
                x = np.sin(orientation * np.pi / 180)  * 1.2
                path_state = np.array([x - 0.8, y - 0.8])
        elif who == 2:
            path_state = np.array([-traj_state, -1.2])
            orientation = 180

        elif who == 3:
            path_state = np.array([traj_state, -0.4])
            orientation = 180

        return path_state, orientation


    def path2traj(self, path_state, who, orientation):
        if who == 1:
            if orientation == 90:
                traj_state = (-0.8 - path_state[1]) + 0.6 * np.pi -0.8

            elif orientation == 0:
                traj_state = path_state[0]
            else:
                traj_state = -0.8 + (orientation / 90) * 0.6 * np.pi
        elif who == 2:
            traj_state = -path_state[0]
        elif who == 3:
            traj_state = path_state[0]
        return traj_state




