import numpy as np
import torch as t
import sys
import random
from scipy.special import logsumexp
sys.path.append('/models/rainbow/')
from models.rainbow.arguments import get_args
import models.rainbow.arguments
from models.rainbow.set_nfsp_models import get_models
from HJI_Vehicle.NN_output import get_Q_value
# from HJI_Vehicle.NN_output_costate import get_Hamilton as get_Q_value
import dynamics
from scipy.optimize import Bounds, minimize, minimize_scalar
import savi_simulation
from scipy.integrate import quad


class TestOptimize:
    def __init__(self, sim):
        self.x0 = []
        self.x1_nn = np.array([[24], [19], [19], [18]])  # x1 v1 x2 v2
        self.x2_nn = np.array([[20], [20], [15], [18]])
        self.time = np.array([[1]])
        self.dt = 0.05
        self.theta1 = 5
        self.theta2 = 5
        self.last_action = 0

    def bvp_optimize(self, _bounds):
        q_fun = self.q_function_helper1
        res = minimize_scalar(self.q_function_helper1, bounds=(-5, 10), method='bounded')
        return res

    def q_function_helper1(self, action):  # for agent 1
        print(self.x1_nn)
        q1, q2 = get_Q_value(self.x1_nn, self.time, np.array([[action], [self.last_action]]),
                             (self.theta1, self.theta2), self.dt)
        q1 = -q1[0][0]
        print(action, q1)

        return q1

    def optimize_example(self):
        # bounds = Bounds([-5, 10])  # action bounds

        return

    def calc(self, x, a, b):
        return a * x ** 2 + b

    def integrate(self, a, b):
        return quad(self.calc, 0, 1, args=(a, b))[0]

    def q_func_integrate_helper1(self, X, T, Ui, theta, deltaT):
        q1, q2 = get_Q_value(X, T, np.array([[Ui], [self.last_action]]), theta)

        return q1[0][0]

    def integrate_action(self, X, T, theta, deltaT):
        return quad(self.q_func_integrate_helper1, -5, 10, args=(X, T, theta, deltaT))[0]


def calc(x, a, b):
    return a*x**2 + b


def integrate(a, b):
    print("res: ", calc(1, 1, 1))
    return quad(calc, 0, 1, args=(a, b))[0]


def q_func_integrate_helper1(Ui, X, last_action, T, theta, deltaT):
    q1, q2 = get_Q_value(X, T, np.array([[Ui], [last_action]]), theta)

    exp_q1 = np.exp(0.1 * q1[0][0])
    assert exp_q1 != 0

    return exp_q1


def integrate_action(X, last_action,T, theta, deltaT):
    return quad(q_func_integrate_helper1, -5, 10, args=(X, last_action,T, theta, deltaT))[0]


if __name__ == "__main__":
    test = TestOptimize(savi_simulation)
    bounds = [-5, 10]
    x1 = np.array([[24], [19], [23], [18]])
    dt = 0.05
    theta1 = 5
    theta2 = 5
    last_action = 0
    # res1 = test.bvp_optimize(bounds)
    # print("optimized action:", res1.x)
    # res2 = test.integrate_action(x1, np.array([[0.5]]), (theta1, theta2), dt)
    res2 = integrate_action(x1, 0, np.array([[0.5]]), (theta1, theta2), dt)
    # res2 = np.exp(res2)
    print("integrated:", res2)

