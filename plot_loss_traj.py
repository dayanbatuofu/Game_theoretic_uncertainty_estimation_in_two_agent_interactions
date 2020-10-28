"""
for plotting multiple loss trajectory from a cvs file
"""
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import csv
from matplotlib.collections import LineCollection

directory = '/experiment'
path = "experiment/*.csv"
na_na_path = "experiment/55/*.csv"
a_a_path = "experiment/new_11_2/*.csv"
a_na_path = "experiment/new_15/*.csv"
na_a_path = "experiment/51/*.csv"

def read_csv_loss():
    # for root,dirs,files in os.walk(directory):
    #     for file in files:
    #         if file.endswith('.csv'):
    #             f = open(file, 'r')
    #             f.close()
    time = []
    loss_1 = []
    loss_2 = []
    x1 = []
    x2 = []
    "choose path here"
    for filename in glob.glob(path):
        print(filename)
        with open(filename, 'r') as csv_file:
            # creating a csv reader object
            csv_reader = csv.reader(csv_file)

            # extracting each data row one by one
            rows = []
            for row in csv_reader:
                rows.append(row)
            time.append(rows[0])
            loss_1.append(rows[2])
            loss_2.append(rows[4])
            x1.append(rows[6])
            x2.append(rows[8])

    return loss_1, x1, x2


def plot_loss():
    loss_s, x1_s, x2_s = read_csv_loss()
    loss = []
    x1 = []
    x2 = []
    for i in range(len(loss_s)):
        loss.append([float(j) for j in loss_s[i]])
        x1.append([float(j) for j in x1_s[i]])
        x2.append([float(j) for j in x2_s[i]])
    # print(loss)
    # plt.show()
    n = len(loss)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n):
        print(len(x1[i]), len(x2[i]), len(loss[i]))
        ax.scatter(x1[i], x2[i], loss[i])
    # ax = fig.add_subplot(111, projection='3d')
    # print(len(x1[0]), len(x2[0]), len(loss[0]))
    # print(x1[0][0])
    # ax.scatter(x1[0], x2[0], loss[0])
    ax.invert_xaxis()
    ax.set_xlabel('P1 location')
    ax.set_ylabel('P2 location')
    ax.set_zlabel('Loss')
    ax.set_xticks([15, 20, 25, 30, 35, 40, 45])
    ax.set_yticks([15, 20, 25, 30, 35, 40, 45])
    # ax.xlim([15, 40])
    # ax.ylim([15, 40])
    # ax.axis('equal')
    plt.show()


def plot_loss_color():
    loss_s, x1_s, x2_s = read_csv_loss()
    loss = []
    x1 = []
    x2 = []
    max_loss = 0
    min_loss = 0
    for i in range(len(loss_s)):
        loss.append([float(j) for j in loss_s[i]])
        x1.append([float(j) for j in x1_s[i]])
        x2.append([float(j) for j in x2_s[i]])
        if max(loss[i]) > max_loss:
            max_loss = max(loss[i])
        if min(loss[i])<min_loss:
            min_loss = min(loss[i])
    # print(loss)
    # plt.show()
    n = len(loss)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(min_loss, max_loss)
    for i in range(n):
        # print(len(x1[i]), len(x2[i]), len(loss[i]))
        # x = np.zeros((len(x1[i]), 2))
        # for j in range(len(x1[i])):
        #     x[j][0] = x1[i][j]
        #     x[j][1] = x2[i][j]
        x = np.column_stack((x1[i], x2[i]))
        print(x)
        lc = LineCollection(x, cmap='viridis', norm=norm)
        lc.set_array(loss[i])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        # ax.scatter(x1[i], x2[i], loss[i])
    # ax = fig.add_subplot(111, projection='3d')
    # print(len(x1[0]), len(x2[0]), len(loss[0]))
    # print(x1[0][0])
    # ax.scatter(x1[0], x2[0], loss[0])
    ax.invert_xaxis()
    ax.set_xlabel('P1 location')
    ax.set_ylabel('P2 location')
    # ax.set_zlabel('Loss')
    fig.colorbar(line, ax=ax[0])
    ax.set_xticks([15, 20, 25, 30, 35, 40, 45])
    ax.set_yticks([15, 20, 25, 30, 35, 40, 45])
    # ax.xlim([15, 40])
    # ax.ylim([15, 40])
    # ax.axis('equal')
    plt.show()


if __name__ == '__main__':
    plot_loss()
