from constants import CONSTANTS as C
from sim_draw import Sim_Draw
from sim_data import Sim_Data
import pickle
import pygame as pg

class Main():

    def __init__(self):

        self.duration = 300

        self.P = C.PARAMETERSET_2  # Scenario parameters choice

        self.sim_draw = Sim_Draw(self.P, "../assets/")

        # Time handling
        self.clock = pg.time.Clock()
        self.fps = C.FPS
        self.running = True
        self.paused = False
        self.end = False
        self.frame = -1

        # Sim input
        with open('output_intersection.pkl', 'rb') as input:
            self.sim_data = pickle.load(input)

        # Go
        self.trial()


    def trial(self):

        while self.running:

            if self.frame >= self.duration:
                break

            # Draw frame
            self.sim_draw.draw_frame(self.sim_data, self.frame)

            if not paused:
                self.frame += 1

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    self.running = False

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_p:
                        self.paused = not self.paused

                    if event.key == pg.K_q:
                        pg.quit()
                        self.running = False

            # Keep fps
            self.clock.tick(self.fps)


if __name__ == "__main__":
    Main()