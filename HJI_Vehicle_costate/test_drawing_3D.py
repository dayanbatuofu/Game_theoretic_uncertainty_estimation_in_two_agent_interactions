import numpy as np
import matplotlib.pyplot as plt
import torch

# Road length
R = 50
R = torch.tensor(R, dtype=torch.float32)
# Vehicle width
W = 1.5
W = torch.tensor(W, dtype=torch.float32)
# Vehicle length
L = 3
L = torch.tensor(L, dtype=torch.float32)

# weight for sigmoid function
theta1 = 5
theta2 = 5
beta = 1

def sigmoid(x1,x2):
    x1 = torch.tensor(x1, dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    return beta - beta * (torch.sigmoid((x1 - R / 2 + theta2 * W / 2) * 10) * torch.sigmoid(-(x1 - R / 2 - theta2 * W / 2 - L) * 10) *
                torch.sigmoid((x2 - R / 2 + theta1 * W / 2) * 10) * torch.sigmoid(-(x2 - R / 2 - theta1 * W / 2 - L) * 10)).detach().numpy()

# def sigmoid(x1,x2):
#     x1 = torch.tensor(x1, dtype=torch.float32)
#     x2 = torch.tensor(x2, dtype=torch.float32)
#     return beta * (torch.sigmoid((x1 - R / 2 - theta2 * W / 2) * 10) + torch.sigmoid(-(x1 - R / 2 + theta2 * W / 2) * 10) +
#                 torch.sigmoid((x2 - R / 2 - theta1 * W / 2) * 10) + torch.sigmoid(-(x2 - R / 2 + theta1 * W / 2) * 10)).detach().numpy()

# def sigmoid(x1,x2):
#     x1 = torch.tensor(x1, dtype=torch.float32)
#     x2 = torch.tensor(x2, dtype=torch.float32)
#     return torch.sigmoid(-10 * (torch.sqrt((R / 2 - x1 + L /2)**2 + (R / 2 - x2 + L /2)**2) -
#                           2 * torch.sqrt((W/2)**2 + (L/2)**2))).detach().numpy()

# def sigmoid(x1,x2):
#     x1 = torch.tensor(x1, dtype=torch.float32)
#     x2 = torch.tensor(x2, dtype=torch.float32)
#     return torch.sigmoid(-10 * ((torch.sqrt((x1 - x2)**2) - 2 * W/2))).detach().numpy()

x1_axis = np.arange(0, 50, 0.1)
x2_axis = np.arange(0, 50, 0.1)
x1, x2 = np.meshgrid(x1_axis, x2_axis)

f = sigmoid(x1, x2)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.contour3D(x1, x2, f, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

# def f(x, y):
#     return 1/2*x**2*(-1/2)*y**2
#
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
#
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()
