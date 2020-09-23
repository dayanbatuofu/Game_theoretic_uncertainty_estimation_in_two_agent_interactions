"""
Defines dynamics for universal usage
"""
import numpy as np


def dynamics_1d(x, u, dt, min_speed, max_speed):
    """
    In this case where dynamics is 1D
    :param state:
    :param action_set:
    :param dt:
    :return: resulting state from given current state and action
    """
    sx, sy, vx, vy = x[0], x[1], x[2], x[3]
    "Deceleration only leads to small/min velocity!"
    if sx == 0 and vx == 0:  # y axis movement
        # print("Y axis movement detected")
        vx_new = vx  # + u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
        vy_new = vy + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
        if vy_new < min_speed:
            vy_new = min_speed
        else:
            vy_new = max(min(vy_new, max_speed), min_speed)
        sx_new = sx  # + (vx + vx_new) * dt * 0.5
        sy_new = sy + (vy + vy_new) * dt * 0.5
    elif sy == 0 and vy == 0:  # x axis movement
        vx_new = abs(vx) + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
        vy_new = vy
        if vx_new < min_speed:
            # print("vehicle M is exceeding min speed", vx_new, u)
            vx_new = min_speed
        else:
            vx_new = max(min(vx_new, max_speed), min_speed)
        vx_new = -vx_new
        sx_new = sx + (vx + vx_new) * dt * 0.5
        sy_new = sy  # + (vy + vy_new) * dt * 0.5
    else:  # TODO: in the case of more than 1D movement??
        print("Y axis movement detected (else)")
        vx_new = vx  # + u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
        vy_new = vy + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
        if vy_new < 0:
            vy_new = 0
        sx_new = sx  # + (vx + vx_new) * dt * 0.5
        sy_new = sy + (vy + vy_new) * dt * 0.5

    # TODO: add a cieling for how fast they can go
    return sx_new, sy_new, vx_new, vy_new


def dynamics_2d(x, u, dt, min_speed, max_speed):
    """
    In the case steering action is available
    :param state:
    :param action_set:
    :param dt:
    :return: resulting state from given current state and action
    """

    sx, sy, theta, delta, vy = x[0], x[1], x[2], x[3], x[4] # x, y, heading, velocity steering, velocity
    L = 3 # length of the vehicle 

    vy_new = vy + u[1] * dt #* vy / (np.linalg.norm([vx, vy]) + 1e-12)
    delta_new = delta + u[0] * dt
    if vy_new < min_speed:
        vy_new = min_speed
    else:
        vy_new = max(min(vy_new, max_speed), min_speed)
    sx_new = sx + (vy_new) * dt *np.sin(theta)
    sy_new = sy + (vy_new) * dt *np.cos(theta)
    theta_new = theta + vy_new/L*np.tan(delta_new) *dt

    #print("ID:", self.id, "action:", u[0],"," ,u[1], "old vel:", vy, "new vel:", vy_new, "angle", theta_new)
    return sx_new, sy_new, theta_new, delta_new, vy_new



"backup from inference model"
# def calc_state(x, u, dt):
#     sx, sy, vx, vy = x[0], x[1], x[2], x[3]
#     "Deceleration only leads to zero velocity!"
#     if sx == 0 and vx == 0:  # y axis movement
#         vx_new = vx  # + u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
#         vy_new = vy + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
#         if vy_new < 0:
#             vy_new = 0
#         sx_new = sx  # + (vx + vx_new) * dt * 0.5
#         sy_new = sy + (vy + vy_new) * dt * 0.5
#     elif sy == 0 and vy == 0:  # x axis movement
#         vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
#         vy_new = vy  # + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
#         if vx_new < 0:
#             vx_new = 0
#         sx_new = sx + (vx + vx_new) * dt * 0.5
#         sy_new = sy  # + (vy + vy_new) * dt * 0.5
#     else:  # TODO: assume x axis movement for single agent case!!
#         vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
#         vy_new = vy  # + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
#         if vx_new < 0:
#             vx_new = 0
#         sx_new = sx + (vx + vx_new) * dt * 0.5
#         sy_new = sy  # + (vy + vy_new) * dt * 0.5
#
#     # TODO: add a cieling for how fast they can go
#     return sx_new, sy_new, vx_new, vy_new

"backup for autonomous_vehicle dynamics"
# def f(x, u, dt):
#     sx, sy, vx, vy = x[0], x[1], x[2], x[3]
#     if self.id == 0:
#         vx_new = vx
#         vy_new = vy + u * dt #* vy / (np.linalg.norm([vx, vy]) + 1e-12)
#         if vy_new < self.min_speed:
#             vy_new = self.min_speed
#         else:
#             vy_new = max(min(vy_new, self.max_speed), self.min_speed)
#         sx_new = sx
#         sy_new = sy + (vy + vy_new) * dt * 0.5
#
#     elif self.id == 1:  # white vehicle (M) (agent[1]), x axis, moving towards negative
#         #u = -u
#         vx_new = abs(vx) + u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
#         vy_new = vy
#         if vx_new < self.min_speed:
#             # print("vehicle M is exceeding min speed", vx_new, u)
#             vx_new = self.min_speed
#         else:
#             vx_new = max(min(vx_new, self.max_speed), self.min_speed)
#         vx_new = -vx_new
#         sx_new = sx + (vx + vx_new) * dt * 0.5
#         sy_new = sy
#
#     else:
#         vx_new = vx + u * dt * vx #/ (np.linalg.norm([vx, vy]) + 1e-12)
#         vy_new = vy + u * dt * vy #/ (np.linalg.norm([vx, vy]) + 1e-12)
#         sx_new = sx + (vx + vx_new) * dt * 0.5
#         sy_new = sy + (vy + vy_new) * dt * 0.5
#     # print("ID:", self.id, "action:", u, "old vel:", vx, vy, "new vel:", vx_new, vy_new)
#     return sx_new, sy_new, vx_new, vy_new