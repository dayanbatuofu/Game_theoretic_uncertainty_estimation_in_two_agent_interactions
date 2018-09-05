import numpy as np
import pygame as pg
import bezier
from collision_box import Collision_Box
from constants import CONSTANTS as C


class ControlledVehicle:
    """
    States:
            X-Position
            Y-Position
    """

    def __init__(self, scenario_parameters, car_parameters_self, who):
        self.P = scenario_parameters
        self.P_CAR= car_parameters_self
        self.image = pg.transform.rotate(pg.transform.scale(pg.image.load(C.ASSET_LOCATION + self.P_CAR.SPRITE),
                                                              (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                               int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))),
                                           self.P_CAR.ORIENTATION)

        self.collision_box = Collision_Box(self.image.get_width() / C.COORDINATE_SCALE / C.ZOOM,
                                             self.image.get_height() / C.COORDINATE_SCALE / C.ZOOM, self.P)
        self.who = who
        # Initialize my space
        self.states = [self.P_CAR.INITIAL_POSITION]
        self.velocity = 0
        self.actions_set = [(0,0)]
        self.delta = 0.2 * C.VEHICLE_MAX_SPEED

    def update(self, Humaninput):
        Throttle = 1 - Humaninput[1]
        Brake = 1 - Humaninput[2]

        self.velocity = self.velocity + (Throttle - Brake - 0.2) * self.delta
        if self.velocity > 0:
            action = self.velocity
            if self.velocity >= C.VEHICLE_MAX_SPEED:
                self.velocity = C.VEHICLE_MAX_SPEED
        else:
            self.velocity = 0
            action = self.velocity
        self.states.append(np.add(self.states[-1], (action,0)))
        self.actions_set.append((action,0))
        # print 'action_setin controlled_veh: {}'.format(self.actions_set)
        self.planned_actions_set = (action,0)
        # print self.actions_set

    def interpolate_from_trajectory(self, trajectory):

        nodes = np.array([[0, trajectory[0]*np.cos(np.deg2rad(trajectory[1]))/2, trajectory[0]*np.cos(np.deg2rad(trajectory[1]))],
                          [0, trajectory[0]*np.sin(np.deg2rad(trajectory[1]))/2, trajectory[0]*np.sin(np.deg2rad(trajectory[1]))]])

        curve = bezier.Curve(nodes, degree=2)

        positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_NUMPOINTS + 1)))
        #TODO: skip state?
        return np.diff(positions, n=1, axis=0)