from constants import CONSTANTS as C
from human_vehicle import HumanVehicle
from machine_vehicle import MachineVehicle
import numpy as np
import pygame as pg

class Main():

    def __init__(self):

        self.duration = 1800

        self.P = C.PARAMETERSET_1  # Scenario parameters choice

        #self.human_vehicle = HumanVehicle('human_state_files/intersection/human_stop.txt')
        self.human_vehicle = HumanVehicle('human_state_files/lane_change/human_change_lane.txt')
        self.machine_vehicle = MachineVehicle(self.P, self.human_vehicle.get_state(0))

        pg.init()
        self.screen = pg.display.set_mode((self.P.SCREEN_WIDTH, self.P.SCREEN_HEIGHT))
        self.human_vehicle.image = pg.transform.rotate(pg.image.load("assets/red_car_sized.png"), self.P.HUMAN_ORIENTATION)
        self.machine_vehicle.image = pg.transform.rotate(pg.image.load("assets/blue_car_sized.png"), self.P.MACHINE_ORIENTATION)
        self.coordinates_image = pg.image.load("assets/coordinates.png")

        # Time handling
        self.clock = pg.time.Clock()
        self.fps = C.FPS
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 1

        # Sim output
        self.sim_out = open("sim_outputs/output_maxtest.txt", "w")

        self.trial()


    def trial(self):

        while self.running:

            # Update model here
            if not self.paused and not self.end:
                self.machine_vehicle.update(self.human_vehicle.get_state(self.frame))
                self.frame += 1

            if self.frame >= self.duration:
                end = True

            # Draw frame
            if not self.end:
                self.draw_frame()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()

                    running = False

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_p:
                        self.paused = not self.paused

            # Keep fps
            self.clock.tick(self.fps)


    def draw_frame(self):

        self.screen.fill((255, 255, 255))

        # Draw Images

        human_pos = self.human_vehicle.get_state(self.frame)[0:2]
        human_pos_pixels = self.c2p(human_pos)
        human_car_size = self.human_vehicle.image.get_size()
        self.screen.blit(self.human_vehicle.image, (human_pos_pixels[0] - human_car_size[0] / 2, human_pos_pixels[1] - human_car_size[1] / 2))

        machine_pos = self.machine_vehicle.get_state()[0:2]
        machine_pos_pixels = self.c2p(machine_pos)
        machine_car_size = self.machine_vehicle.image.get_size()
        self.screen.blit(self.machine_vehicle.image, (machine_pos_pixels[0] - machine_car_size[0] / 2, machine_pos_pixels[1] - machine_car_size[1] / 2))

        coordinates_size = self.coordinates_image.get_size()
        self.screen.blit(self.coordinates_image, (10, self.P.SCREEN_HEIGHT - coordinates_size[1] - 10 / 2))

        # Draw human predicted state
        for i in range(len(self.machine_vehicle.human_previous_action_set)):
            human_predicted_state = self.machine_vehicle.human_states[-1] + np.sum(self.machine_vehicle.human_previous_action_set[:i+1],axis=0)
            human_predicted_state_pixels = self.c2p(human_predicted_state)
            pg.draw.circle(self.screen, (0, 255 - i*10, 255 - i*10), human_predicted_state_pixels, 6)

        # Draw machine decided state
        for i in range(len(self.machine_vehicle.machine_previous_action_set)):
            machine_predicted_state = self.machine_vehicle.machine_states[-1] + np.sum(self.machine_vehicle.machine_previous_action_set[:i+1],axis=0)
            machine_predicted_state_pixels = self.c2p(machine_predicted_state)
            pg.draw.circle(self.screen, (0, 255 - i*10, 0), machine_predicted_state_pixels, 6)

        # Draw machine predicted state
        for i in range(len(self.machine_vehicle.machine_previous_predicted_action_set)):
            machine_predicted_state = self.machine_vehicle.machine_states[-1] + np.sum(self.machine_vehicle.machine_previous_predicted_action_set[:i+1],axis=0)
            machine_predicted_state_pixels = self.c2p(machine_predicted_state)
            pg.draw.circle(self.screen, (255 - i*10, 0, 255 - i*10), machine_predicted_state_pixels, 4)

        # Draw predicted human intent
        pos = self.c2p(human_pos + self.machine_vehicle.human_predicted_theta[1:3] * 0.5)
        pg.draw.circle(self.screen, (0, 0, 0), pos, 7)
        pg.draw.circle(self.screen, (0, 255, 255), pos, 6)

        # Draw machine intent
        pos = self.c2p(machine_pos + self.machine_vehicle.machine_theta[1:3])
        pg.draw.circle(self.screen, (0, 0, 0), pos, 7)
        pg.draw.circle(self.screen, (0, 255, 0), pos, 6)

        # Draw predicted human's prediction of machine's intent
        pos = self.c2p(machine_pos + self.machine_vehicle.machine_predicted_theta[1:3])
        pg.draw.circle(self.screen, (0, 0, 0), pos, 5)
        pg.draw.circle(self.screen, (255, 0, 255), pos, 4)

        # Annotations

        font = pg.font.SysFont("Arial", 15)
        label = font.render("Human State: (%5.2f , %5.2f)" % (human_pos[0], human_pos[1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 10))

        label = font.render("Machine State: (%5.2f , %5.2f)" % (machine_pos[0], machine_pos[1]), 1, (0, 0, 0))
        self.screen.blit(label, (10, 30))

        label = font.render("Machine Theta: (%5.2f, %5.2f, %5.2f)" % (self.machine_vehicle.machine_theta[0], self.machine_vehicle.machine_theta[1], self.machine_vehicle.machine_theta[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30, 60))
        pg.draw.circle(self.screen, (0, 0, 0), (15, 65), 5)
        pg.draw.circle(self.screen, (0, 255, 0), (15, 65), 4)

        label = font.render("P Human Theta: (%5.2f, %5.2f, %5.2f)" % (self.machine_vehicle.human_predicted_theta[0], self.machine_vehicle.human_predicted_theta[1], self.machine_vehicle.human_predicted_theta[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30, 80))
        pg.draw.circle(self.screen, (0, 0, 0), (15, 85), 5)
        pg.draw.circle(self.screen, (0, 255, 255), (15, 85), 4)

        label = font.render("PP Machine Theta: (%5.2f, %5.2f, %5.2f)" % (self.machine_vehicle.machine_predicted_theta[0], self.machine_vehicle.machine_predicted_theta[1], self.machine_vehicle.machine_predicted_theta[2]), 1, (0, 0, 0))
        self.screen.blit(label, (30, 100))
        pg.draw.circle(self.screen, (0, 0, 0), (15, 105), 5)
        pg.draw.circle(self.screen, (255, 0, 255), (15, 105), 4)

        # label = font.render("Effort: %f" % (machine_vehicle.human_predicted_theta[3]), 1, (0, 0, 0))
        # screen.blit(label, (10, 70))
        label = font.render("Frame: %i" % (self.frame + 1), 1, (0, 0, 0))
        self.screen.blit(label, (10, 130))

        pg.display.flip()

        if True:
            self.sim_out.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (human_pos[0],
                                                                human_pos[1],
                                                                machine_pos[0],
                                                                machine_pos[1],
                                                                human_predicted_state[0],
                                                                human_predicted_state[1],
                                                                self.machine_vehicle.human_predicted_theta[0],
                                                                self.machine_vehicle.human_predicted_theta[1],
                                                                self.machine_vehicle.human_predicted_theta[2]))

    def c2p(self, coordinates):
        x = int(self.P.COORDINATE_SCALE * coordinates[1] + self.P.ORIGIN[0])
        y = int(self.P.COORDINATE_SCALE * -coordinates[0] + self.P.ORIGIN[1])
        return np.array([x, y])


if __name__ == "__main__":
    Main()