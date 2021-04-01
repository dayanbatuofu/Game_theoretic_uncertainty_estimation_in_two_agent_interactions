"""
python 3.6 and up is required
records agent info for i = 0, 1
"""
import numpy as np
import scipy
from sim_data import DataUtil
import pygame as pg
import dynamics
import logging
logging.basicConfig(level=logging.INFO)

class AutonomousVehicle:
    """
    States:
            X-Position
            Y-Position
    """
    def __init__(self, sim, env, par, inference_model, decision_model, i):
        self.sim = sim
        self.env = env  # environment parameters
        self.car_par = par  # car parameters
        self.inference_model = inference_model
        self.decision_model = decision_model
        self.id = i

        # Initialize variables
        self.state = self.car_par["initial_state"]  # state is cumulative
        self.intent = self.car_par["par"]
        self.action = self.car_par["initial_action"]  # action is cumulative
        self.trajectory = []
        self.planned_actions_set = []
        self.planned_trajectory_set = []

        # Initialize prediction variables
        self.predicted_intent_all = []
        self.predicted_intent_other = []
        self.predicted_intent_self = []
        self.predicted_policy_other = []
        self.predicted_policy_self = []
        "for recording predicted state from inference"
        self.predicted_actions_other = [0]  # assume initial action of other agent = 0
        self.predicted_actions_self = [0]
        self.predicted_states_self = []
        self.predicted_states_other = []
        self.belief_count = []
        self.policy_choice = []
        self.min_speed = self.env.MIN_SPEED
        self.max_speed = self.env.MAX_SPEED

    def update(self, sim):
        other = sim.agents[:self.id]+sim.agents[self.id+1:]  # get all other agents
        frame = sim.frame

        # take a snapshot of the state at beginning of the frame before agents update their states
        snapshot = sim.snapshot()  # snapshot = agent.copy() => snapshot taken before updating

        # perform inference
        # if not self.sim.frame == 0:
        inference = self.inference_model.infer(snapshot, sim)
        DataUtil.update(self, inference)

        # planning
        plan = self.decision_model.plan()

        # update state
        action = plan["action"]

        if self.sim.decision_type[self.id] == 'constant_speed':
            pass
        else:
            action = action[self.id]
            plan = {"action": action}
        DataUtil.update(self, plan)
        logging.debug("chosen action", action)
        self.dynamics(action)

    def dynamics(self, action):  # Dynamic of cubic polynomial on velocity
        # TODO: add steering
        # define the discrete time dynamical model

        def f_environment(x, u, dt): # x, y, theta, velocity
            sx, sy, vx, vy = x[0], x[1], x[2], x[3]
            if self.id == 0 or self.id == 1:
                vx_new = vx
                vy_new = vy + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                if vy_new < self.min_speed:
                    vy_new = self.min_speed
                else:
                    vy_new = max(min(vy_new, self.max_speed), self.min_speed)
                sx_new = sx
                sy_new = sy + (vy + vy_new) * dt * 0.5
            else:
                vx_new = vx + u * dt * vx #/ (np.linalg.norm([vx, vy]) + 1e-12)
                vy_new = vy + u * dt * vy #/ (np.linalg.norm([vx, vy]) + 1e-12)
                sx_new = sx + (vx + vx_new) * dt * 0.5
                sy_new = sy + (vy + vy_new) * dt * 0.5
            logging.debug("ID:", self.id, "action:", u, "old vel:", vx, vy, "new vel:", vx_new, vy_new)
            return sx_new, sy_new, vx_new, vy_new

        # if self.env.name == "merger":
        #     self.state.append(f_environment(self.state[-1], action, self.sim.dt))

        # else:  # using dynamics defined in dynamics.py, for easier access
        #     self.state.append(dynamics.dynamics_1d(self.state[-1], action, self.sim.dt, self.min_speed, self.max_speed))
        # def f_environment(x, u, dt): # x, y, theta, velocity
        #     sx, sy, theta, vy = x[0], x[1], x[2], x[3]
        #     if self.id == 0 or self.id == 1:
        #         vy_new = vy + u[1] * dt #* vy / (np.linalg.norm([vx, vy]) + 1e-12)
        #         if vy_new < self.min_speed:
        #             vy_new = self.min_speed
        #         else:
        #             vy_new = max(min(vy_new, self.max_speed), self.min_speed)
        #         sx_new = sx + (vy + vy_new) * dt *np.cos(theta)
        #         sy_new = sy + (vy + vy_new) * dt *np.sin(theta)
        #         theta_new = theta + u[0]
        #     else:
        #         vx_new = vx + u * dt * vx #/ (np.linalg.norm([vx, vy]) + 1e-12)
        #         vy_new = vy + u * dt * vy #/ (np.linalg.norm([vx, vy]) + 1e-12)
        #         sx_new = sx + (vx + vx_new) * dt * 0.5
        #         sy_new = sy + (vy + vy_new) * dt * 0.5
        #         theta_new = theta + u[0]
        #     logging.debug("ID:", self.id, "action:", u[0],"," ,u[1], "old vel:", vy, "new vel:", vy_new, "angle", theta_new)
        #     return sx_new, sy_new, theta_new, vy_new
        # if self.env.name == "merger":
        #     self.state.append(f_environment(self.state[-1], action, self.sim.dt))
        # else:
            # self.state.append(f(self.state[-1], action, self.sim.dt))

        def f_environment_sc(x, u, dt): # x, y, heading, velocity steering, velocity 
            sx, sy, theta, delta, vy = x[0], x[1], x[2], x[3], x[4]
            L = 3 # length of the vehicle 
            if self.id == 0 or self.id == 1:
                vy_new = vy + u[1] * dt #* vy / (np.linalg.norm([vx, vy]) + 1e-12)
                delta_new = delta + u[0] * dt
                if vy_new < self.min_speed:
                    vy_new = self.min_speed
                else:
                    vy_new = max(min(vy_new, self.max_speed), self.min_speed)
                sx_new = sx + (vy_new) * dt *np.sin(theta)
                sy_new = sy + (vy_new) * dt *np.cos(theta)
                theta_new = theta + vy_new/L*np.tan(delta_new) *dt
            else:
                pass
                # vx_new = vx + u * dt * vx #/ (np.linalg.norm([vx, vy]) + 1e-12)
                # vy_new = vy + u * dt * vy #/ (np.linalg.norm([vx, vy]) + 1e-12)
                # sx_new = sx + (vx + vx_new) * dt * 0.5
                # sy_new = sy + (vy + vy_new) * dt * 0.5
                # theta_new = theta + u[0]
            logging.debug("ID:", self.id, "action:", u[0], "," , u[1], "old vel:", vy, "new vel:", vy_new, "angle", theta_new)
            return sx_new, sy_new, theta_new, delta_new, vy_new
        if self.env.name == "merger":
            self.state.append(f_environment_sc(self.state[-1], action, self.sim.dt))
        elif self.env.name == 'bvp_intersection':
            self.state.append(dynamics.bvp_dynamics_1d(self.state[self.sim.frame], action, self.sim.dt))
        else:  # for nfsp intersection, or other intersection case
            # self.state.append(f(self.state[-1], action, self.sim.dt))
            self.state.append(dynamics.dynamics_1d(self.state[-1], action, self.sim.dt, self.min_speed, self.max_speed))

        return


# dummy class
class dummy(object):
    pass


if __name__ == '__main__':
    Car1 = {"initial_state": [[200, 200, 0, 0, 0]], "par": 1, "initial_action":1}
    from environment import *
    env = Environment("merger")
    sim = dummy()
    sim.dt = 1
    # def __init__(self, sim, env, par, inference_model, decision_model, i):
    test_car = AutonomousVehicle(sim, env, Car1, "baseline", "baseline", 0)
    i = 1
    while i < 20:
        test_car.dynamics([1, 1])
        i += 1


