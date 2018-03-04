from constants import CONSTANTS as C
import numpy as np
import pygame as pg
import csv


def main():
    duration = 1800
    trial(duration)


def trial(duration):


    pg.init()
    screen = pg.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
    human_vehicle_image = pg.image.load("../assets/red_car_sized.png")
    machine_vehicle_image = pg.image.load("../assets/blue_car_sized.png")

    fps = C.FPS

    clock = pg.time.Clock()
    running = True
    paused = False
    end = False
    frame = 0

    # Read file
    with open('output_weffort.txt') as f:
        reader = csv.reader(f, delimiter="\t")
        data = np.array(list(reader)).astype('float32')

    while running:

        # Update model here
        if not paused and not end:
            frame += 1

        if frame >= duration:
            end = True

        # Draw frame
        if not end:
            draw_frame(screen, data, frame, human_vehicle_image, machine_vehicle_image)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                cv2.destroyAllWindows()

                running = False

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    paused = not paused

        # Keep fps
        clock.tick(fps)


def draw_frame(screen, data, frame, human_vehicle_image, machine_vehicle_image):

    human_state                 = (data[frame][0], data[frame][1])
    machine_state               = (data[frame][2], data[frame][3])
    human_predicted_state       = (data[frame][4], data[frame][5])
    human_predicted_theta       = (data[frame][6], data[frame][7], data[frame][8], data[frame][9])

    screen.fill((255, 255, 255))

    human_pos = human_state
    human_pos_pixels = c2p(human_pos)
    screen.blit(human_vehicle_image, (human_pos_pixels[0] - C.CAR_WIDTH / 2, human_pos_pixels[1] - C.CAR_LENGTH / 2))

    machine_pos = machine_state
    machine_pos_pixels = c2p(machine_pos)
    screen.blit(machine_vehicle_image,
                (machine_pos_pixels[0] - C.CAR_WIDTH / 2, machine_pos_pixels[1] - C.CAR_LENGTH / 2))

    pg.draw.circle(screen, (0, 255, 0), c2p(human_predicted_state), 10)

    font = pg.font.SysFont("Arial", 15)
    label = font.render("Human State: (%f , %f)" % (human_pos[0], human_pos[1]), 1, (0, 0, 0))
    screen.blit(label, (10, 10))
    label = font.render("Machine State: (%f , %f)" % (machine_pos[0], machine_pos[1]), 1, (0, 0, 0))
    screen.blit(label, (10, 30))
    label = font.render("P Human Theta: (%f, %f, %f)" % (human_predicted_theta[0], human_predicted_theta[1], human_predicted_theta[2]), 1, (0, 0, 0))
    screen.blit(label, (10, 50))
    label = font.render("Effort: %f" % (human_predicted_theta[3]), 1, (0, 0, 0))
    screen.blit(label, (10, 70))
    label = font.render("Frame: %i" % (frame + 1), 1, (0, 0, 0))
    screen.blit(label, (10, 110))

    pg.display.flip()


def c2p(coordinates):
    x = int(C.LANE_WIDTH * (coordinates[1] + 0.5))
    y = int(C.LANE_WIDTH * -coordinates[0] + C.SCREEN_HEIGHT / 2)
    return [x, y]


if __name__ == "__main__":
    main()