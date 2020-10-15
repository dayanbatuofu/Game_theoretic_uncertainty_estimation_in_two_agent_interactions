# TODO: add uncertainty visualization

import pygame as pg
import numpy as np
import time
import scipy.io
import os
from examples.choose_problem import system, problem, config

LIGHT_GREY = (230, 230, 230)

class VisUtils:

    def __init__(self):

        self.screen_width = 10  # 50
        self.screen_height = 10  # 50
        self.coordinate_scale = 80
        self.zoom = 0.1
        self.asset_location = 'assets/'
        self.fps = 24  # max framework

        self.car_width = 1.5
        self.car_length = 3

        load_path = 'examples/vehicle/data_train.mat'
        self.train_data = scipy.io.loadmat(load_path)

        self.new_data = self.generate(self.train_data)

        self.T = self.new_data['t']

        self.car_par = [{'sprite': 'grey_car_sized.png',
                         'state': self.new_data['X'][:2, :],  # pos_x, vel_x
                         'orientation': 0.},
                        {'sprite': 'white_car_sized.png',
                         'state': self.new_data['X'][2:, :],  # pos_x, vel_x
                         'orientation': -90.}
                        ]

        img_width = int(self.car_width * self.coordinate_scale * self.zoom)
        img_height = int(self.car_length * self.coordinate_scale * self.zoom)

        "initialize pygame"
        pg.init()
        self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                           self.screen_height * self.coordinate_scale))

        self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location
                                                             + self.car_par[i]['sprite']),
                                               (img_width, img_height)),
                            - self.car_par[i]['orientation']) for i in range(len(self.car_par))]

        self.origin = np.array([25, 25])

        "Draw Axis Lines"

        self.screen.fill((255, 255, 255))
        self.draw_axes() #calling draw axis function
        pg.display.flip()
        pg.display.update()

    def draw_frame(self):
        '''state[t] = [s_x, s_y, v_x, v_y]_t'''
        '''state = [state_t, state_t+1, ...]'''
        # Draw the current frame
        '''frame is counting which solution step'''

        steps = self.T.shape[0]  # 10/0.1 + 1 = 101

        for k in range(steps - 1):
            self.screen.fill((255, 255, 255))
            self.draw_axes()
            # Draw Images
            n_agents = 2
            for i in range(n_agents):
                '''getting pos of agent: (x, y)'''
                pos_old = np.array(self.car_par[i]['state'][0][k])  # car position
                pos_new = np.array(self.car_par[i]['state'][0][k+1])  # get 0 and 1 element (not include 2) : (x, y)
                '''smooth out the movement between each step'''
                pos = pos_old * (1 - k * 1. / steps) + pos_new * (k * 1. / steps)
                if i == 0:
                    pos = (26.5, pos)  # car position
                if i == 1:
                    pos = (pos, 26.5)  # car position
                '''transform pos'''
                pixel_pos_car = self.c2p(pos)
                size_car = self.car_image[i].get_size()
                self.screen.blit(self.car_image[i],
                                 (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
                time.sleep(0.05)
                # if self.sim.decision_type == "baseline":
                #     time.sleep(0.05)
            # Annotations
            font = pg.font.SysFont("Arial", 15)
            screen_w, screen_h = self.screen.get_size()
            label_x = screen_w - 555
            label_y = 260
            label_y_offset = 30
            pos_h, speed_h = self.car_par[0]['state'][0][k+1], self.car_par[0]['state'][1][k+1]  #y axis
            label = font.render("Car 1 position and speed: (%5.4f , %5.4f)" % (pos_h, speed_h), 1,
                                (0, 0, 0))
            self.screen.blit(label, (label_x, label_y))
            pos_m, speed_m = self.car_par[1]['state'][0][k+1], self.car_par[1]['state'][1][k+1] #x axis
            label = font.render("Car 2 position and speed: (%5.4f , %5.4f)" % (pos_m, speed_m), 1,
                                (0, 0, 0))
            self.screen.blit(label, (label_x, label_y + label_y_offset))

            label = font.render("Frame: %i" % steps, 1, (0, 0, 0))
            self.screen.blit(label, (10, 10))

            recording_path = 'image_recording/'
            pg.image.save(self.screen, "%simg%03d.png" % (recording_path, k))

            "drawing the map of state distribution"
            # pg.draw.circle(self.screen, (255, 255, 255), self.c2p(self.origin), 10)  # surface,  color, (x, y),radius>=1

            # time.sleep(1)

            pg.display.flip()
            pg.display.update()

    def draw_axes(self):
        # draw lanes based on environment
        pg.draw.line(self.screen, LIGHT_GREY, self.c2p((26.5, -50)), self.c2p((26.5, 100)), 25)
        pg.draw.line(self.screen, LIGHT_GREY, self.c2p((100, 26.5)), self.c2p((-50, 26.5)), 25)

        # bound1 = [[-1, 1], None] #[xbound, ybound], xbound = [x_min, x_max]  what is the meaning of bound?
        # bound2 = [None, [-1, 1]]
        # bound_set = [[[-12.5, 12.5], None], [None, [-1, 1]]]
        # for a in bound_set:
        #     bound_x, bound_y = a[0], a[1] #vehicle width
        #     if bound_x:
        #         b_min, b_max = bound_x[0], bound_x[1]
        #         _bound1 = self.c2p((b_min, 25))
        #         _bound2 = self.c2p((b_max, 25))
        #         bounds = np.array([_bound1[0], _bound2[0]])
        #         pg.draw.line(self.screen, LIGHT_GREY, ((bounds[1] + bounds[0])/2, -50),
        #                      ((bounds[1] + bounds[0])/2, self.screen_height * self.coordinate_scale,
        #                       ), bounds[1] - bounds[0])
        #
        #     if bound_y:
        #         b_min, b_max = bound_y[0], bound_y[1]
        #         _bound1 = self.c2p((25, b_min))
        #         _bound2 = self.c2p((25, b_max))
        #         bounds = np.array([_bound1[1], _bound2[1]])
        #         # pg.draw.line(self.screen, LIGHT_GREY, (0, (self.screen_width * self.coordinate_scale,
        #         #                 (bounds[1] + bounds[0]) / 2),
        #         #                 (bounds[1] + bounds[0]) / 2), bounds[0] - bounds[1])
        #
        #         pg.draw.line(self.screen, LIGHT_GREY, (self.screen_width * self.coordinate_scale,
        #                         (bounds[1] + bounds[0]) / 2), (-50, (bounds[1] + bounds[0]) / 2),
        #                         bounds[0] - bounds[1])

    def c2p(self, coordinates):
        '''coordinates = x, y position in your environment(vehicle position)'''
        x = self.coordinate_scale * (- coordinates[0] + self.origin[0] + self.screen_width / 2)
        y = self.coordinate_scale * (- coordinates[1] + self.origin[1] + self.screen_height / 2)
        x = int(
            (x - self.screen_width * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_width * self.coordinate_scale * 0.5)
        y = int(
            (y - self.screen_height * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_height * self.coordinate_scale * 0.5)
        '''returns x, y for the pygame window'''

        return np.array([x, y])

    def generate(self, data):
        t_bar = np.arange(0, 10, 0.1)  # t_bar = 0: 0.1: 10
        X_bar = np.zeros((4, t_bar.shape[0]))
        i = 0
        j = 0
        time = 0
        t = data['t']  # time is from train_data
        X = data['X']
        while time <= 9.9:
            while t[0][i] <= time:
                i += 1
            X_bar[0][j] = (time - t[0][i-1]) * (X[0][i] - X[0][i-1])/(t[0][i] - t[0][i-1]) + X[0][i-1]
            X_bar[1][j] = (time - t[0][i - 1]) * (X[1][i] - X[1][i - 1]) / (t[0][i] - t[0][i - 1]) + X[1][i - 1]
            X_bar[2][j] = (time - t[0][i - 1]) * (X[2][i] - X[2][i - 1]) / (t[0][i] - t[0][i - 1]) + X[2][i - 1]
            X_bar[3][j] = (time - t[0][i - 1]) * (X[3][i] - X[3][i - 1]) / (t[0][i] - t[0][i - 1]) + X[3][i - 1]
            time = time + 0.1
            j += 1

        new_data = dict()
        new_data.update({'t': t_bar,
                         'X': X_bar})

        result = new_data

        return new_data

if __name__ == '__main__':
    vis = VisUtils()
    vis.draw_frame()

    path = 'image_recording/'
    import glob
    image = glob.glob(path+"*.png")
    # print(image)
    # episode_step_count = len(image)
    img_list = image # [path + "img" + str(i).zfill(3) + ".png" for i in range(episode_step_count)]

    import imageio
    images = []
    for filename in img_list:
        images.append(imageio.imread(filename))
    tag = 'theta1' + '=' + str(problem.theta1) + '_' + 'theta2' + '=' + str(problem.theta2) +  '_' +'time horizon' + '=' + str(config.t1)
    imageio.mimsave(path + 'movie_' + tag + '.gif', images, 'GIF', duration=0.2)
    # Delete images
    [os.remove(path + file) for file in os.listdir(path) if ".png" in file]




