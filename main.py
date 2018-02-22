from constants import CONSTANTS as C
from human_vehicle import HumanVehicle
from machine_vehicle import MachineVehicle
import pygame as pg


def main():

    duration = 1800
    trial(duration)


def trial(duration):

    human_vehicle = HumanVehicle()
    machine_vehicle = MachineVehicle(machine_initial_state=C.MACHINE_INITIAL_POSITION,
                                     human_initial_state=human_vehicle.get_state(0))

    pg.init()
    screen = pg.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
    human_vehicle.image = pg.image.load("assets/red_car_sized.png")
    machine_vehicle.image = pg.image.load("assets/blue_car_sized.png")

    fps = C.FPS

    clock = pg.time.Clock()
    running = True
    paused = False
    end = False
    frame = 0

    while running:

        # Update model here
        if not paused and not end:
            machine_vehicle.update(human_vehicle.get_state(frame))
            frame += 1

        if frame >= duration:
            end = True

        # Draw frame
        if not end:
            draw_frame(screen, frame, human_vehicle, machine_vehicle)


        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    paused = not paused

        # Keep fps
        clock.tick(fps)


def draw_frame(screen, frame, human_vehicle, machine_vehicle):
    screen.fill((255, 255, 255))

    human_pos = human_vehicle.get_state(frame)[0:2]
    human_pos_pixels = c2p(human_pos)
    screen.blit(human_vehicle.image, (human_pos_pixels[0] - C.CAR_WIDTH / 2, human_pos_pixels[1] - C.CAR_LENGTH / 2))

    machine_pos = machine_vehicle.get_state()[0:2]
    machine_pos_pixels = c2p(machine_pos)
    screen.blit(machine_vehicle.image, (machine_pos_pixels[0] - C.CAR_WIDTH / 2, machine_pos_pixels[1] - C.CAR_LENGTH / 2))


    pg.draw.circle(screen, (0, 255, 0), c2p(machine_vehicle.human_predicted_state), 10)

    #pg.draw.circle(screen, (255, 0, 255), c2p(machine_vehicle.human_predicted_s_desired), 5)

    font = pg.font.SysFont("Arial", 15)
    label = font.render("Human State: (%f , %f)" % (human_pos[0], human_pos[1]), 1, (0, 0, 0))
    screen.blit(label, (10, 10))
    label = font.render("Machine State: (%f , %f)" % (machine_pos[0], machine_pos[1]), 1, (0, 0, 0))
    screen.blit(label, (10, 30))
    label = font.render("Frame: %i" % (frame+1), 1, (0, 0, 0))
    screen.blit(label, (10, 50))
    label = font.render("Debug 1: %f" % machine_vehicle.debug_1, 1, (0, 0, 0))
    screen.blit(label, (10, 90))
    label = font.render("Debug 2: %f" % machine_vehicle.debug_2, 1, (0, 0, 0))
    screen.blit(label, (10, 110))
    label = font.render("Debug 3: %f" % machine_vehicle.debug_3, 1, (0, 0, 0))
    screen.blit(label, (10, 130))

    pg.display.flip()


def c2p(coordinates):
    x = int(C.LANE_WIDTH * (coordinates[1] + 0.5))
    y = int(C.LANE_WIDTH * -coordinates[0] + C.SCREEN_HEIGHT/2)
    return [x, y]


if __name__ == "__main__":
    main()