from constants import CONSTANTS as C
from sim_draw import Sim_Draw
from sim_data import Sim_Data
import pickle
import pygame as pg
from autonomous_vehicle import AutonomousVehicle

class Main():

    def __init__(self):

        self.duration = 300

        self.P = C.PARAMETERSET_2  # Scenario parameters choice

        # Time handling
        self.clock = pg.time.Clock()
        self.fps = C.FPS
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 0
        self.car_num_display = 0

        # Sim input
        with open('./sim_outputs/output_2018-05-18-15-18-01.pkl', 'rb') as input:
            self.sim_data = pickle.load(input)

        self.sim_draw = Sim_Draw(self.P, C.ASSET_LOCATION)

        # Go
        trial_length = len(self.sim_data.car1_states)
        self.trial(trial_length)


    def trial(self, trial_length):

        while self.running and self.frame < trial_length - 1:

            if self.frame >= self.duration:
                break

            # Draw frame
            self.sim_draw.draw_frame(self.sim_data, self.car_num_display, self.frame)

            if not self.paused:
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

                    if event.key == pg.K_d:
                        self.car_num_display = ~self.car_num_display

            # Keep fps
            self.clock.tick(self.fps)

        input("Simulation playback ended. Press Enter to exit.")


if __name__ == "__main__":
    Main()