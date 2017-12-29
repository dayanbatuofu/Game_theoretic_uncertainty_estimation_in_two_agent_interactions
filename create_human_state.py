from constants import CONSTANTS as C
import numpy as np

file_name = 'human_no_change_lane.txt'
file = open(file_name, 'w')


duration = 1800
start_lane = 1
end_lane = 0

laneChange_distance = end_lane-start_lane
laneChange_direction = np.sign(end_lane-start_lane)
laneChange_start = 1800

x = 0
y = start_lane
x_vel = 0
y_vel = 0

for step in range(duration):
    laneChange = (laneChange_start - step <= 0) and (laneChange_start - step - laneChange_distance/C.VEHICLE_LATERAL_MOVEMENT_SPEED >= 0)

    if laneChange:
        y += C.VEHICLE_LATERAL_MOVEMENT_SPEED * laneChange_direction

    file.write("%f %f\n" % (x, y))