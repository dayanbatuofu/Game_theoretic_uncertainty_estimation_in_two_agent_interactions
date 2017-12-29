from constants import CONSTANTS as C
from human_vehicle import HumanVehicle
from machine_vehicle import MachineVehicle
import pygame as pg


def main():

    duration = 1800
    trial(duration)

def trial(duration):

    human_vehicle = HumanVehicle()
    machine_vehicle = MachineVehicle(machine_initial_state=(0, 0, 0, 0),
                                     human_initial_state=human_vehicle.get_state(0))

    pg.init()
    screen = pg.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
    human_vehicle.image = pg.image.load("assets/red_car_sized.png")
    machine_vehicle.image = pg.image.load("assets/blue_car_sized.png")

    fps = 60

    clock = pg.time.Clock()
    running = True
    for frame in range(duration):

        # Update model here
        machine_vehicle.update(human_vehicle.get_state(frame))

        # Draw frame
        draw_frame(screen, frame, human_vehicle, machine_vehicle)


        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                running = False
        if not running:
            break

        # Keep fps
        clock.tick(fps)


def draw_frame(screen, frame, human_vehicle, machine_vehicle):
    screen.fill((255,255,255))

    LANE_WIDTH = C.SCREEN_WIDTH/2

    human_pos = human_vehicle.get_state(frame)[0:2]
    human_pos_adjusted_x = LANE_WIDTH * (human_pos[1] + 0.5)  # flip axis and add offset
    human_pos_adjusted_y = (C.SCREEN_HEIGHT / 2) * (human_pos[0] + 1)
    screen.blit(human_vehicle.image, (human_pos_adjusted_x - C.CAR_WIDTH / 2, human_pos_adjusted_y - C.CAR_LENGTH / 2))

    machine_pos = machine_vehicle.get_state()[0:2]
    machine_pos_adjusted_x = LANE_WIDTH * (machine_pos[1] + 0.5)  # flip axis and add offset
    machine_pos_adjusted_y = (C.SCREEN_HEIGHT / 2) * (machine_pos[0] + 1)
    screen.blit(machine_vehicle.image, (machine_pos_adjusted_x - C.CAR_WIDTH / 2, machine_pos_adjusted_y - C.CAR_LENGTH / 2))

    pg.display.flip()


if __name__ == "__main__":
    main()

# frame_start = time.time()
# frame_time = time.time()
# if frame_time < 1 / fps:
#     time.sleep(frame_time - 1 / fps)