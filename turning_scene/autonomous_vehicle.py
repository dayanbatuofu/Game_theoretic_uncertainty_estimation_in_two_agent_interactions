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

    class DummyVehicle:
        def __init__(self, scenario_parameters, car_parameters_self, who):
            self.P = scenario_parameters
            self.P_CAR = car_parameters_self
            self.who = who
            self.image = pg.transform.rotate(pg.transform.scale(pg.image.load(C.ASSET_LOCATION + self.P_CAR.SPRITE),
                                                                (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                                 int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))),
                                             self.P_CAR.ORIENTATION)

            # self.image_o = pg.transform.rotate(pg.transform.scale(pg.image.load(C.ASSET_LOCATION + self.P_CAR_O.SPRITE),
            #                                                          (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
            #                                                           int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))), self.P_CAR_O.ORIENTATION)

            self.collision_box = Collision_Box(self.image.get_width() / C.COORDINATE_SCALE / C.ZOOM,
                                               self.image.get_height() / C.COORDINATE_SCALE / C.ZOOM, self.P)

            self.states = [self.P_CAR.INITIAL_POSITION]
            self.actions_set = []
            self.trajectory = []
            self.track_back = 0
            self.speed = [self.P_CAR.INITIAL_SPEED]
            self.ability = self.P_CAR.ABILITY

            # Initialize others space
            self.states_o1 = []
            self.actions_set_o1 = []
            self.speed_o1 = []
            self.other_car_1 = []
            self.states_o2 = []
            self.actions_set_o2 = []
            self.speed_o2 = []
            self.other_car_2 = []

            self.FOV = 1  # the vehicle sees the other dummy vehicle

        def update(self, frame):
            who = self.who

            self.frame = frame

            # if len(self.states_o) > C.TRACK_BACK and len(self.states) > C.TRACK_BACK:
            self.track_back = min(C.TRACK_BACK, len(self.states))

            self.FOV = self.check_view()

            ########## Calculate machine actions here ###########
            planned_actions = self.get_actions()

            planned_actions[np.where(np.abs(planned_actions) < 1e-6)] = 0.  # remove numerical errors

            # self.states.append(np.add(self.states[-1], (planned_actions[self.track_back][0],
            #                                             planned_actions[self.track_back][1])))
            planned_states_path, planned_speed = self.dynamic(planned_actions)
            planned_states = self.path2traj(planned_path)

            self.states.append(planned_states[0])
            self.speed.append(planned_speed)


        def check_view(self):

            if self.frame >= 2:
                action_car2 = (self.speed_o2[-2] - self.speed_o2[-1])
                if action_car2 >= 0:
                    new_FOV = 0
                else:
                    new_FOV = 1
            else:
                new_FOV = 1

        return new_FOV





        def get_action(self):
            if self.FOV == 0:
                actions = 1
            else:
                actions = -1
            return actions

        def dynamic(self, action):
            vel_self = np.array(self.speed[-1])
            state_0 = np.array(self.states[-1])
            acci = np.array([action * self.ability])

            vel_0 = np.asarray(vel_self) / T  # type: ndarray # initial vel??
            A = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 2, 0],
                          [0, 0, 2, 6 * pow(T * N, 1)]])
            b1 = np.array([state_0[0], vel_0[0], acci[0], 0])
            b2 = np.array([state_0[1], vel_0[1], acci[1], 0])
            coeffx = np.linalg.solve(A, b1)
            coeffy = np.linalg.solve(A, b2)
            #     a = A.dot(coeffx)
            #     b = np.polyval(coeffx, T * N)
            velx = []
            vely = []
            for t in range(1, N + 1, 1):
                velx.append(
                    ((coeffx[3] * 3 * (pow((T * t), 2))) + (coeffx[2] * 2 * (pow((T * t), 1))) + (coeffx[1])))
            vely.append(
                ((coeffy[3] * 3 * (pow((T * t), 2))) + (coeffy[2] * 2 * (pow((T * t), 1))) + (coeffy[1])))
            if action_self[1] == 0:
                velx = np.clip(velx, 0, C.PARAMETERSET_2.VEHICLE_MAX_SPEED)
                vely = np.clip(vely, 0, 0)
            else:
                vely = np.clip(vely, -C.PARAMETERSET_2.VEHICLE_MAX_SPEED, 0)
                velx = np.clip(velx, 0, 0)
            predict_result_vel = np.column_stack((velx, vely))
            A = np.zeros([N, N])
            A[np.tril_indices(N, 0)] = 1
            predict_result_traj = np.matmul(A, predict_result_vel) + state_0
            planned_speed = vel_self - acci
            return predict_result_traj, planned_speed

        def traj2path(self, traj_state):
            return path_state

        def path2traj(self, path_state):
            return traj_state





