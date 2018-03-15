from constants import CONSTANTS as C
import numpy as np

file_name = 'human_stop.txt'
file = open(file_name, 'w')


duration = 1800
start_y = 3

intersection_y = 1.5

x = 0
y = start_y
x_vel = 0
y_vel = 0

for step in range(duration):

    if y > intersection_y:
        y += -C.VEHICLE_MOVEMENT_SPEED

    file.write("%f %f\n" % (x, y))