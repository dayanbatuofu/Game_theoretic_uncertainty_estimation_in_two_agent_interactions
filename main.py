
from human_vehicle import HumanVehicle
from machine_vehicle import MachineVehicle
import time
import pygame as pg


def main():

    duration = 1800
    trial(duration)

def trial(duration):

    human_vehicle = HumanVehicle()
    machine_vehicle = MachineVehicle(initial_state=(0, 0, 0, 0))

    pg.init()
    car_width = 100
    car_height = 200
    screen_width = 400
    screen_height = 800
    screen = pg.display.set_mode((screen_width, screen_height))
    human_vehicle.image = pg.image.load("assets/red_car_sized.png")
    machine_vehicle.image = pg.image.load("assets/blue_car_sized.png")

    fps = 60

    clock = pg.time.Clock()
    running = True
    for frame in range(duration):
        frame_start = time.time()

        # Update model here
        machine_vehicle.update()

        # Draw frame
        draw_frame(screen, screen_height, screen_width, frame, human_vehicle, machine_vehicle, car_width, car_height)


        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                running = False
        if not running:
            break

        # Keep fps
        clock.tick(fps)


def draw_frame(screen, screen_height, screen_width, frame, human_vehicle, machine_vehicle, car_width, car_height):
    screen.fill((255,255,255))

    human_pos = human_vehicle.get_position(frame)
    human_pos_adjusted_x = (screen_width / 3) * (human_pos[1] + 1)  # flip axis
    human_pos_adjusted_y = (screen_height / 2) * (human_pos[0] + 1)
    screen.blit(human_vehicle.image, (human_pos_adjusted_x - car_width/2, human_pos_adjusted_y - car_height/2))

    machine_pos = machine_vehicle.get_position()
    machine_pos_adjusted_x = (screen_width / 3) * (machine_pos[1] + 1)  # flip axis
    machine_pos_adjusted_y = (screen_height / 2) * (machine_pos[0] + 1)
    screen.blit(machine_vehicle.image, (machine_pos_adjusted_x - car_width/2, machine_pos_adjusted_y - car_height/2))


    pg.display.flip()


if __name__ == "__main__":
    main()

# frame_start = time.time()
# frame_time = time.time()
# if frame_time < 1 / fps:
#     time.sleep(frame_time - 1 / fps)