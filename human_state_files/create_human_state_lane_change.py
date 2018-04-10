from constants import CONSTANTS as C
import numpy as np

file_name = 'human_change_lane_immediately.txt'
file = open(file_name, 'w')


duration = 1800
start_lane = 1
end_lane = 0

laneChange_distance = end_lane-start_lane
laneChange_direction = np.sign(end_lane-start_lane)
laneChange_start = 10

laneChange_speed = C.PARAMETERSET_1.VEHICLE_MAX_SPEED * 0.1

x = 0
y = start_lane
x_vel = 0
y_vel = 0

for step in range(duration):
    laneChange = (laneChange_start - step <= 0) and (laneChange_start - step - laneChange_distance/laneChange_speed >= 0)

    x += 0.075  # Make the human's speed slower than the machine's max speed in order for it to pass

    if laneChange:
        y += -0.01

    file.write("%f %f\n" % (x, y))