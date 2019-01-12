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



class DummyVehicle:
    def __init__(self, scenario_parameters, car_parameters_self, who):
        self.P = scenario_parameters
        self.P_CAR = car_parameters_self
        self.who = who
        self.image = pg.transform.rotate(pg.transform.scale(pg.image.load(C.ASSET_LOCATION + self.P_CAR.SPRITE),
                                                            (int(C.CAR_WIDTH * C.COORDINATE_SCALE * C.ZOOM),
                                                             int(C.CAR_LENGTH * C.COORDINATE_SCALE * C.ZOOM))),
                                         self.P_CAR.ORIENTATION)

        self.collision_box = Collision_Box(self.image.get_width() / C.COORDINATE_SCALE / C.ZOOM,
                                           self.image.get_height() / C.COORDINATE_SCALE / C.ZOOM, self.P)

        self.states = []
        self.actions_set = []
        self.FOV = 0  #wheather the vehicle sees the other dummy vehicle
        self.states_o = []
        self.actions_set_o = []
        self.other_car = []


    def update(self, frame):
        who = self.who
        other = self.other_car
        self.frame = frame
        """ Function ran on every frame of simulation"""
        ########## Update human characteristics here ########
        if who == 1:  #

        elif who == 2:













    def get_action(self, other_states):





    def dynamic(self, action):

