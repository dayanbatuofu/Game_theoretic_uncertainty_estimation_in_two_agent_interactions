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

        self.FOV = [1]  # the vehicle sees the other dummy vehicle

    def update(self, frame):
        who = self.who
        other1 = self.other_car_1
        other2 = self.other_car_2
        self.frame = frame

        # if len(self.states_o) > C.TRACK_BACK and len(self.states) > C.TRACK_BACK:
        self.track_back = min(C.TRACK_BACK, len(self.states))

        new_FOV = self.check_view()

        ########## Calculate machine actions here ###########
        planned_actions = self.get_actions(new_FOV)


        planned_states_path, planned_speed = self.dynamic(planned_actions)
        planned_states = self.path2traj(planned_states_path)
        orientation = 180
        self.states.append(planned_states)
        self.speed.append(planned_speed)
        self.orientation.append(orientation)
        self.FOV.append(new_FOV)


    def check_view(self):

        if self.frame >= 2:
            action_car2 = (self.other_car_1.speed[-2] - self.other_car_1.speed[-1])
            if action_car2 >= 0:
                new_FOV = 0
            else:
                new_FOV = 1
        else:
            new_FOV = 1

        return new_FOV



    def get_actions(self, FOV):
        if FOV == 0:
            actions = 1
        else:
            actions = -1
        return actions

    def dynamic(self, action):
        vel_self = np.array(self.speed[-1])
        path_state = np.array(self.states[-1])
        state = path_state[0]
        acci = np.array([action * self.ability])

        planned_speed = vel_self - acci
        planned_speed = np.clip(planned_speed, 0, C.VEHICLE_MAX_SPEED)
        predict_result_traj = state + planned_speed

        return predict_result_traj, planned_speed


    def path2traj(self,traj_state):
        planned_state = np.array([traj_state, -0.4])
        return planned_state



