from constants import CONSTANTS as C
import pygame as pg


class Environment:

    def __init__(self):

        # default settings
        self.duration = 100
        self.parameters = C.PARAMETERSET_2  # Scenario parameters choice
        # Time handling
        self.clock = pg.time.Clock()
        self.fps = C.FPS
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 0
        self.car_num_display = 0

    def reset(self):
        self.clock = pg.time.Clock()
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 0


