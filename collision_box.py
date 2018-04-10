import math
from constants import CONSTANTS as C
import numpy as np

class Collision_Box():

    def __init__(self, width, height):

            self.width = width
            self.height = height

    def get_minimum_distance(self, my_pos, other_pos, other_box):

        ## Rectangular boxes

        # for i in range(len(my_pos)):
        #
        #     if other_pos[i,0] - other_box.height/2 > my_pos[i,0] + self.height/2:  # Other is on top
        #
        #         if other_pos[i,1] - other_box.width/2 > my_pos[i,1] + self.width/2:  # Other is on top-right
        #
        #             distance.append(math.hypot((other_pos[i,0] - other_box.height / 2) - (my_pos[i,0] + self.height / 2),
        #                               (other_pos[i,1] - other_box.width / 2) - (my_pos[i,1] + self.width / 2)))
        #
        #         elif other_pos[i,1] + other_box.width/2 < my_pos[i,1] - self.width/2: # Other is on top-left
        #
        #             distance.append(math.hypot((other_pos[i,0] - other_box.height / 2) - (my_pos[i,0] + self.height / 2),
        #                               (other_pos[i,1] + other_box.width / 2) - (my_pos[i,1] - self.width / 2)))
        #
        #         else: # Other is on top
        #
        #             distance.append(np.abs((other_pos[i, 0] - other_box.height / 2) - (my_pos[i, 0] + self.height / 2)))
        #
        #     elif other_pos[i,0] + other_box.height/2 < my_pos[i,0] - self.height/2:  # Other is on bottom
        #
        #         if other_pos[i,1] - other_box.width / 2 > my_pos[i,1] + self.width / 2:  # Other is on bottom-right
        #
        #             distance.append(math.hypot((other_pos[i,0] + other_box.height / 2) - (my_pos[i,0] - self.height / 2),
        #                               (other_pos[i,1] - other_box.width / 2) - (my_pos[i,1] + self.width / 2)))
        #
        #         elif other_pos[i,1] + other_box.width / 2 < my_pos[i,1] - self.width / 2:  # Other is on bottom-left
        #
        #             distance.append(math.hypot((other_pos[i,0] + other_box.height / 2) - (my_pos[i,0] - self.height / 2),
        #                               (other_pos[i,1] + other_box.width / 2) - (my_pos[i,1] - self.width / 2)))
        #
        #         else: # Other is on bottom
        #
        #             distance.append(np.abs((other_pos[i, 0] + other_box.height / 2) - (my_pos[i, 0] - self.height / 2)))
        #
        #     else: # Other is to the left or right
        #
        #         if other_pos[i, 1] - other_box.width / 2 > my_pos[i, 1] + self.width / 2: # Other is on left
        #
        #             distance.append(np.abs((other_pos[i, 1] + other_box.width / 2) - (my_pos[i, 1] - self.width / 2)))
        #
        #         else: # Other is on right
        #
        #             distance.append(np.abs((other_pos[i, 1] - other_box.width / 2) - (my_pos[i, 1] + self.width / 2)))
        #
        # return np.array(distance)


        ## Ellipse boxes

        # distance = []
        #
        # for i in range(len(my_pos)):
        #
        #     angle = np.arctan2(other_pos[i, 1] - my_pos[i, 1], other_pos[i, 0] - other_pos[i, 0])
        #
        #     my_rad = self.radius_at_angle(angle, self.width, self.height)
        #     other_rad = self.radius_at_angle(angle, other_box.width, other_box.height)
        #
        #     center_distance = math.hypot(other_pos[i, 0] - my_pos[i, 0], other_pos[i, 1] - my_pos[i, 1])
        #
        #     distance.append(center_distance - my_rad - other_rad)


        ## Max implementation
        distance = np.sum((my_pos - other_pos)**2, axis=1)

        # return np.array(1 / (1 + np.exp(-distance + C.CAR_LENGTH*2))) # try sigmoid
        return distance

    def radius_at_angle(self, angle, a, b):

        return a * b / np.sqrt((a**2 * np.sin(angle)**2) + (b**2 * np.cos(angle)**2))

