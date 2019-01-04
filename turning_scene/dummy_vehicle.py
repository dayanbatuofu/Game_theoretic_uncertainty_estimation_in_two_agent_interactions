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

        self.states =
        self.actions_set =
        self.intent
        self.FOV =   #wheather the vehicle sees the other dummy vehicle


