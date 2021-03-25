import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

font = {'family' : 'normal','weight' : 'normal','size'   : 16}

plt.rc('font', **font)

# Road length
R = 70
R = torch.tensor(R, dtype=torch.float32)
# Vehicle width
W = 1.5
W = torch.tensor(W, dtype=torch.float32)
# Vehicle length
L = 3
L = torch.tensor(L, dtype=torch.float32)

# weight for sigmoid function
beta = 10000

def sigmoid(x1,x2, theta1, theta2):
    x1 = torch.tensor(x1, dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    return beta * (torch.sigmoid((x1 - R / 2 + theta1 * W / 2) * 10) * torch.sigmoid(-(x1 - R / 2 - W / 2 - L) * 10) *
                torch.sigmoid((x2 - R / 2 + theta2 * W / 2) * 10) * torch.sigmoid(-(x2 - R / 2 - W / 2 - L) * 10)).detach().numpy()

x1_axis = np.arange(20,50, 0.1)
x2_axis = np.arange(20,50, 0.1)
x1, x2 = np.meshgrid(x1_axis, x2_axis)

x3_axis = np.arange(20,50, 0.1)
x4_axis = np.arange(20,50, 0.1)
x3, x4 = np.meshgrid(x3_axis, x4_axis)

f1 = sigmoid(x1, x2, 5, 5)
f2 = sigmoid(x3, x4, 1, 1)

fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection="3d")
ax.plot_surface(x1, x2, f1, color='red', alpha=0.3)
ax.plot_surface(x3, x4, f2, color='blue', alpha=0.3)
ax.plot_wireframe(x1, x2, f1, colors=['red'], alpha=0.2)
ax.plot_wireframe(x3, x4, f2, colors=['blue'], alpha=0.2)
ax.set_xlabel('d1',labelpad=20)
ax.set_xlim(20,50)
ax.set_ylabel('d2',labelpad=20)
ax.set_ylim(20,50)
ax.set_zlabel('f$^{(c)}$',labelpad=20)
ax.view_init(25, 240)
ax.text2D(0.35, 0.95, "Loss Function", transform=ax.transAxes)

plt.show()

