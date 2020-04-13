from old_files.constants import CONSTANTS as C
import numpy as np
from old_files.collision_box import Collision_Box
import pygame as pg


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
        self.speed = [self.P_CAR.INITIAL_SPEED]
        self.ability = self.P_CAR.ABILITY
        self.orientation = [self.P_CAR.ORIENTATION]


        self.FOV = [0]  # the vehicle sees the other dummy vehicle



    def update(self, frame):
        who = self.who

        self.frame = frame


        # if len(self.states_o) > C.TRACK_BACK and len(self.states) > C.TRACK_BACK:
        self.track_back = min(C.TRACK_BACK, len(self.states))

        new_FOV = self.check_view()


        ########## Calculate machine actions here ###########
        planned_actions = self.get_actions(new_FOV)


        # self.states.append(np.add(self.states[-1], (planned_actions[self.track_back][0],
        #                                             planned_actions[self.track_back][1])))

        planned_states_path, planned_speed = self.dynamic(planned_actions)

        planned_states, orientation = self.traj2path(planned_states_path)
        self.orientation.append(orientation)
        self.states.append(planned_states)
        self.speed.append(planned_speed)
        self.FOV.append(new_FOV)



    def get_actions(self, FOV):
        if self.who == 1:
            if self.orientation[-1]<90:
                if FOV == 0:
                    actions = 1
                elif FOV == 1:
                    actions  = -2
            else:
                actions = 1
        elif self.who == 2:
            actions = 0
        return actions

    def dynamic(self, action):
        vel_self = np.array(self.speed[-1])
        path_state = np.array(self.states[-1])
        state = self.path2traj(path_state)
        acci = np.array([action * self.ability])
        planned_speed = vel_self + acci
        planned_speed = np.clip(planned_speed, 0, C.VEHICLE_MAX_SPEED)

        predict_result_traj = state + planned_speed
        return predict_result_traj, planned_speed

    def traj2path(self, traj_state):
        if self.who == 1:
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
        elif self.who == 2:
            path_state = np.array([-traj_state, -1.2])
            orientation = 180

        return path_state, orientation


    def path2traj(self, path_state):
        if self.who == 1:
            if self.orientation[-1] == 90:
                traj_state = (-0.8 - path_state[1]) + 0.6 * np.pi -0.8

            elif self.orientation[-1] == 0:
                traj_state = path_state[0]
            else:
                traj_state = -0.8 + (self.orientation[-1] / 90) * 0.6 * np.pi
        elif self.who == 2:
            traj_state = -path_state[0]


        return traj_state




    def check_view(self):
        self_state = np.array(self.states[-1])
        other_state = np.array(self.other_car_1.states[-1])
        block_state = np.array(self.other_car_2.states[-1])
        x1,y1 = self_state
        x2,y2 = other_state
        x0,y0 = block_state

        xp = x0 + C.CAR_LENGTH*0.5
        yp = y0 + C.CAR_WIDTH*0.5

        xq = x0 - C.CAR_LENGTH*0.5
        yq = y0 - C.CAR_WIDTH*0.5

        a = np.divide((y2-y1), (x2-x1))
        b = np.divide((y1*x2-x1*y2), (x2-x1))

        diff1 = a*xp + b - yp
        diff2 = a*xq + b - yq


        if diff1*diff2 >= 0:
            new_FOV = 1
        else:
            new_FOV = 0



        return new_FOV



