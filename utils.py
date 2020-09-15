# all utility classes here

import torch
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import logging
from torch.utils.data import DataLoader
from torch import nn

matplotlib.use("TkAgg")
# class Visualization:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.compare = len(dataset) == 2
#
#     # compare two datasets
#     def compare_data(self):
#         assert self.compare
#         dataset1 = self.dataset[0]
#         dataset2 = self.dataset[1]
#         trj1 = dataset1.X
#         phase1 = dataset1.phase
#         trj2 = dataset2  # TODO: right now Neural ODE model does not output phase info, dataset2 is just an array
#
#         fig = plt.figure()
#         ax = plt.axes(xlim=(dataset1.domain[0][0], dataset1.domain[0][1]),
#                       ylim=(dataset1.domain[1][0], dataset1.domain[1][1]))
#
#         # placeholders for the first dataset
#         lines = [ax.plot([], [], '-k')[0] for i in range(trj1.shape[0])]
#         lobj_marker1 = ax.plot([], [], 'or')[0]
#         lobj_marker2 = ax.plot([], [], 'ob')[0]
#         lines.append(lobj_marker1)
#         lines.append(lobj_marker2)
#
#         # placeholders for the second dataset
#         lines2 = [ax.plot([], [], '-.k')[0] for i in range(trj2.shape[0])]
#         # lobj_marker1 = ax.plot([], [], 'pr')[0]
#         # lobj_marker2 = ax.plot([], [], 'pb')[0]
#         # lines2.append(lobj_marker1)
#         # lines2.append(lobj_marker2)
#
#         # combine
#         lines.extend(lines2)
#
#         def init():
#             for line in lines:
#                 line.set_data([], [])
#             return lines
#
#         def animate(i):
#             xdata1 = trj1[:, 0, :i]
#             ydata1 = trj1[:, 1, :i]
#             xdata2 = trj2[:, 0, :i]
#             ydata2 = trj2[:, 1, :i]
#
#             for j in range(trj1.shape[0]):
#                 lines[j].set_data(xdata1[j], ydata1[j])  # set data for each line separately.
#             lines[trj1.shape[0]].set_data(trj1[phase1[:,i-1]==0, 0, i - 1], trj1[phase1[:,i-1]==0, 1, i - 1])
#             lines[trj1.shape[0]+1].set_data(trj1[phase1[:,i-1]==1, 0, i - 1], trj1[phase1[:,i-1]==1, 1, i - 1])
#
#             for j in range(trj2.shape[0]):
#                 lines[trj1.shape[0]+2+j].set_data(xdata2[j], ydata2[j])  # set data for each line separately.
#             # lines[-2].set_data(trj2[phase2[:,i-1]==0, 0, i - 1], trj2[phase2[:,i-1]==0, 1, i - 1])
#             # lines[-1].set_data(trj2[phase2[:,i-1]==1, 0, i - 1], trj2[phase2[:,i-1]==1, 1, i - 1])
#
#             return lines
#
#         interval = 10
#         anim = FuncAnimation(fig, animate, init_func=init,
#                              frames=trj1.shape[2], interval=interval, blit=True)
#         plt.show()
#
#     # plot simulation data from dataset
#     def plot_data(self):
#         if self.compare:
#             dataset = self.dataset[0]
#         trj = dataset.X
#         phase = dataset.phase
#         fig = plt.figure()
#         ax = plt.axes(xlim=(dataset.domain[0][0], dataset.domain[0][1]),
#                       ylim=(dataset.domain[1][0], dataset.domain[1][1]))
#         lines = [ax.plot([], [], '-k')[0] for i in range(trj.shape[0])]
#         lobj_marker1 = ax.plot([], [], 'or')[0]
#         lobj_marker2 = ax.plot([], [], 'ob')[0]
#         lines.append(lobj_marker1)
#         lines.append(lobj_marker2)
#
#         def init():
#             for line in lines:
#                 line.set_data([], [])
#             return lines
#
#         def animate(i):
#             xdata = trj[:, 0, :i]
#             ydata = trj[:, 1, :i]
#             for j in range(trj.shape[0]):
#                 lines[j].set_data(xdata[j], ydata[j])  # set data for each line separately.
#             lines[-2].set_data(trj[phase[:,i-1]==0, 0, i - 1], trj[phase[:,i-1]==0, 1, i - 1])
#             lines[-1].set_data(trj[phase[:,i-1]==1, 0, i - 1], trj[phase[:,i-1]==1, 1, i - 1])
#
#             return lines
#
#         interval = 10
#         anim = FuncAnimation(fig, animate, init_func=init,
#                              frames=trj.shape[2], interval=interval, blit=True)
#         plt.show()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


# use data loaders to stream data for batch training
def get_data_loaders(data, batch_size=128, test_batch_size=128):

    train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

    train_eval_loader = DataLoader(dataset=data, batch_size=test_batch_size, shuffle=False)

    test_loader = DataLoader(dataset=data, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, train_eval_loader


# count the number of parameters used
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(lr, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = lr

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


# create a moving window observer
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# define the accuracy, this is the goodness measure
# TODO: goodness defined on physically meaningful features?
def accuracy(model, dt, criterion, dataset_loader, device):
    mse = 0
    for x in dataset_loader:
        target = x[:,:,1:].to(device).float()
        x0 = x[:,:,0].to(device)
        predicted = model(x0.float(), dt)
        mse += criterion(predicted, target).cpu().detach().numpy()
    return mse / len(dataset_loader.dataset)