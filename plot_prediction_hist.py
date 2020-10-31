"""
Plot histogram of the count when each parameter is chosen
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import csv


def plot_hist():
    x = [0.5, 1.5, 2.5, 3.5]  # NA_NA, NA_A, A_NA, A_A
    x1 = [0.4, 1.4, 2.4, 3.4]
    x2 = [0.6, 1.6, 2.6, 3.6]
    w = 0.1
    x_ticks_label = ['NA_NA', 'NA_A', 'A_NA', 'A_A']
    " count for the parameters "
    y_na = [[0, 333, 480, 153], [133, 338, 244, 325]]  # A_A belief, [[NE], [E]]
    y_a = [[14, 382, 540, 0], [14, 372, 539, 71]]  # NA_NA belief, [[NE], [E]]
    fig2, (ax1, ax2) = plt.subplots(2)
    fig2.suptitle('Predicted Parameters Count')

    'plotting for NA_NA case'
    ax1.set_title('NA_NA agents with A_A belief')
    ax1.bar(x1, y_na[0], label='Non-empathetic', width=w)
    ax1.bar(x2, y_na[1], label='Empathetic', width=w)
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_ticks_label)
    ax1.set(xlabel='Parameters', ylabel='Count')

    'plotting for A_A case'
    ax2.set_title('A_A agents with NA_NA belief')
    ax2.bar(x1, y_a[0], label='Non-empathetic', width=w)
    ax2.bar(x2, y_a[1], label='Empathetic', width=w)
    ax2.legend()
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_ticks_label)
    ax2.set(xlabel='Parameters', ylabel='Count')

    plt.show()
    return


if __name__ == '__main__':
    plot_hist()
