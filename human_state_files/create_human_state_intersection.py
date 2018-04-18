from constants import CONSTANTS as C
import numpy as np

file_name = 'human_stop_immediately.txt'
file = open(file_name, 'w')


duration = 200
start_y = 1.5

intersection_y = 1.5

x = 0
y = start_y
x_vel = 0
y_vel = 0

for step in range(duration):

    # if y > intersection_y:
    y += 0

    file.write("%f %f\n" % (x, y))