import math
import numpy as np

class Collision_Box():

    def __init__(self, width, height):

            self.width = width
            self.height = height

    def get_minimum_distance(self, my_pos, other_pos, other_box):

        distance = []

        for i in range(len(my_pos)):

            if other_pos[i,0] - other_box.height/2 > my_pos[i,0] + self.height/2:  # Other is on top

                if other_pos[i,1] - other_box.width/2 > my_pos[i,1] + self.width/2:  # Other is on top-right

                    distance.append(math.hypot((other_pos[i,0] - other_box.height / 2) - (my_pos[i,0] + self.height / 2),
                                      (other_pos[i,1] - other_box.width / 2) - (my_pos[i,1] + self.width / 2)))

                else: # Other is on top-left

                    distance.append(math.hypot((other_pos[i,0] - other_box.height / 2) - (my_pos + self.height / 2),
                                      (other_pos[i,1] + other_box.width / 2) - (my_pos - self.width / 2)))

            else:  # Other is on bottom

                if other_pos[i,1] - other_box.width / 2 > my_pos[i,1] + self.width / 2:  # Other is on bottom-right

                    distance.append(math.hypot((other_pos[i,0] + other_box.height / 2) - (my_pos[i,0] - self.height / 2),
                                      (other_pos[i,1] - other_box.width / 2) - (my_pos[i,1] + self.width / 2)))

                else:  # Other is on bottom-left

                    distance.append(math.hypot((other_pos[i,0] + other_box.height / 2) - (my_pos[i,0] - self.height / 2),
                                      (other_pos[i,1] + other_box.width / 2) - (my_pos[i,1] - self.width / 2)))

        return np.array(distance)
