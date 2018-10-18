import math
from constants import CONSTANTS as C
import numpy as np

class Collision_Box():

    def __init__(self, width, height, P):

            self.P = P
            self.width = width
            self.height = height

    def get_collision_loss(self, my_pos, other_pos, other_box):
        """

        :type other_pos: object
        """
        # collision_loss = []
        #
        # for i in range(len(my_pos)):
        #
        #     in_collision_box = False
        #
        #     # Loop through collision boxes
        #     for j in range(len(self.P.COLLISION_BOXES)): #TODO: what should be the size of the collision box?
        #
        #         # Check if other in box
        #         if (self.P.COLLISION_BOXES[j, 1] > other_pos[i, 0] > self.P.COLLISION_BOXES[j, 0]) and \
        #                 (self.P.COLLISION_BOXES[j, 3] > other_pos[i, 1] > self.P.COLLISION_BOXES[j, 2]):
        #             other_in = True
        #         else:
        #             other_in = False
        #
        #         # Check if self in box
        #         if (self.P.COLLISION_BOXES[j, 1] > my_pos[i, 0] > self.P.COLLISION_BOXES[j, 0]) and \
        #                 (self.P.COLLISION_BOXES[j, 3] > my_pos[i, 1] > self.P.COLLISION_BOXES[j, 2]):
        #             self_in = True
        #         else:
        #             self_in = False
        #
        #         if other_in and self_in:
        #             in_collision_box = True
        #             break
        #
        #     if not in_collision_box:
        #
        #         collision_loss.append(0)
        #
        #     else:
        #
        #         distance = np.sum((my_pos - other_pos) ** 2, axis=1)
        #         collision_loss.append(np.sum(np.exp(C.EXPCOLLISION * (-distance + C.CAR_LENGTH ** 2 * 1.5))))
        other_pos[np.where(np.abs(other_pos)<1e-12)] = 0
        distance = np.sum((my_pos - other_pos)**2, axis=1)
        # collision_loss = np.sum(np.exp(C.EXPCOLLISION * (-distance + C.CAR_LENGTH ** 2 * 1.5)))
        return np.array(distance)
        # return np.array(collision_loss)


