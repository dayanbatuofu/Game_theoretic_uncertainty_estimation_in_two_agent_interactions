import numpy as np
import matplotlib.pyplot as plt
import torch

# Road length
R = 50
# Vehicle width
W = 1.5
# Vehicle length
L = 3

# A = 5

# weight for sigmoid function
theta = 5

x = np.arange(0, 51, 0.1)
x = torch.tensor(x, dtype=torch.float32)
x_in = (x - R / 2 + theta * W / 2) * 10
x_out = -(x - R / 2 - W / 2 - L) * 10

# x_in = (x - R / 2 - A) * 10
# x_out = -(x - R / 2 + A) * 10

def sigmoid_test_1(x_in):
    return torch.sigmoid(x_in).detach().numpy()

def sigmoid_test_2(x_out):
    return torch.sigmoid(x_out).detach().numpy()

def sigmoid(x_in, x_out):
    return (torch.sigmoid(x_in) * torch.sigmoid(x_out)).detach().numpy()  # use '+' for lane merge

X_axis = np.arange(0, 51, 0.1)
sigmoid_outputs_test1 = sigmoid_test_1(x_in)
sigmoid_outputs_test2 = sigmoid_test_2(x_out)
sigmoid_outputs = sigmoid(x_in, x_out)

plt.plot(X_axis, sigmoid_outputs_test1)
plt.xlabel("Trajectory")
plt.ylabel("Collision Outputs after Sigmoid")
plt.show()

plt.plot(X_axis, sigmoid_outputs_test2)
plt.xlabel("Trajectory")
plt.ylabel("Collision Outputs after Sigmoid")
plt.show()

plt.plot(X_axis, sigmoid_outputs)
plt.xlabel("Trajectory")
plt.ylabel("Collision function" + ' ' + '(theta = ' + str(theta) + ')')
my_x_ticks = np.arange(0, 51, 1)
my_y_ticks = np.arange(0, 1.2, 0.1)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.show()
