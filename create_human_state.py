
file_name = 'human_change_lane_immediately.txt'
file = open(file_name, 'w')

start_lane = 1
end_lane = 0

duration = 1800
laneChange_start = 0
laneChange_duration = 900


laneChange_speed = (end_lane-start_lane)/laneChange_duration

x = 0
y = start_lane
x_vel = 0
y_vel = 0

for step in range(duration):
    laneChange = (laneChange_start - step <= 0) and (laneChange_start + laneChange_duration - 1 - step >= 0)

    if laneChange:
        y += laneChange_speed
        y_vel = laneChange_speed
    else:
        y_vel = 0

    file.write("%f %f %f %f\n" % (x, y, x_vel, y_vel))