"""
python 3.6 and up is required
records agent info
"""
import numpy as np
import scipy
from sim_data import DataUtil
import pygame as pg
import dynamics


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
        self.initial_belief = self.get_initial_belief(self.env.car_par[1]['belief'][0], self.env.car_par[0]['belief'][0],
                                                      self.env.car_par[1]['belief'][1], self.env.car_par[0]['belief'][1],
                                                      weight=0.8)  # note: use params from the other agent's belief
        # Initialize prediction variables
        self.predicted_intent_all = []
        self.predicted_intent_other = []
        self.predicted_intent_self = []
        self.predicted_policy_other = []
        self.predicted_policy_self = []
        "for recording predicted state from inference"
        # TODO: check this predicted action: to match the time steps we put 0 initially, but fixes?
        self.predicted_actions_other = [0]  # assume initial action of other agent = 0
        self.predicted_actions_self = [0]
        self.predicted_states_self = []
        self.predicted_states_other = []
        self.min_speed = 0.1
        self.max_speed = 30

    def update(self, sim):
        other = sim.agents[:self.id]+sim.agents[self.id+1:]  # get all other agents
        frame = sim.frame

        # take a snapshot of the state at beginning of the frame before agents update their states
        snapshot = sim.snapshot()  # snapshot = agent.copy() => snapshot taken before updating

        # perform inference
        inference = self.inference_model.infer(snapshot, sim)
        DataUtil.update(self, inference)

        # planning
        plan = self.decision_model.plan()

        # update state
        action = plan["action"]
        if self.sim.decision_type[self.id] == "baseline" \
                or self.sim.decision_type[self.id] == "baseline2" \
                or self.sim.decision_type[self.id] == "non-empathetic"\
                or self.sim.decision_type[self.id] == "empathetic":  #TODO: need to be able to check indivisual type
            action = action[self.id]
            plan = {"action": action}
        DataUtil.update(self, plan)
        print("chosen action", action)
        self.dynamics(action)

    def dynamics(self, action):  # Dynamic of cubic polynomial on velocity
        # TODO: add steering
        # define the discrete time dynamical model

        def f_environment(x, u, dt): # x, y, theta, velocity
            sx, sy, vx, vy = x[0], x[1], x[2], x[3]
            if self.id == 0 or self.id == 1:
                vx_new = vx
                vy_new = vy + u * dt #* vy / (np.linalg.norm([vx, vy]) + 1e-12)
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
            print("ID:", self.id, "action:", u, "old vel:", vx, vy, "new vel:", vx_new, vy_new)
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
        #     print("ID:", self.id, "action:", u[0],"," ,u[1], "old vel:", vy, "new vel:", vy_new, "angle", theta_new)
        #     return sx_new, sy_new, theta_new, vy_new
        # if self.env.name == "merger":
        #     self.state.append(f_environment(self.state[-1], action, self.sim.dt))
        # else:
            #self.state.append(f(self.state[-1], action, self.sim.dt))

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
                vx_new = vx + u * dt * vx #/ (np.linalg.norm([vx, vy]) + 1e-12)
                vy_new = vy + u * dt * vy #/ (np.linalg.norm([vx, vy]) + 1e-12)
                sx_new = sx + (vx + vx_new) * dt * 0.5
                sy_new = sy + (vy + vy_new) * dt * 0.5
                theta_new = theta + u[0]
            print("ID:", self.id, "action:", u[0],"," ,u[1], "old vel:", vy, "new vel:", vy_new, "angle", theta_new)
            return sx_new, sy_new, theta_new, delta_new, vy_new
        if self.env.name == "merger":
            self.state.append(f_environment_sc(self.state[-1], action, self.sim.dt))
        else:
            # self.state.append(f(self.state[-1], action, self.sim.dt))
            self.state.append(dynamics.dynamics_1d(self.state[-1], action, self.sim.dt, self.min_speed, self.max_speed))

        return

    def get_initial_belief(self, theta_h, theta_m, lambda_h, lambda_m, weight):
        """
        Obtain initial belief of the params
        :param theta_h:
        :param theta_m:
        :param lambda_h:
        :param lambda_m:
        :param weight:
        :return:
        """
        # TODO: given weights for certain param, calculate the joint distribution (p(theta_1), p(lambda_1) = 0.8, ...)
        theta_list = self.sim.theta_list
        lambda_list = self.sim.lambda_list
        beta_list = self.sim.beta_set

        if self.sim.inference_type[1] == 'empathetic':
            # beta_list = beta_list.flatten()
            belief = np.ones((len(beta_list), len(beta_list)))
            for i, beta_h in enumerate(beta_list):  # H: the rows
                for j, beta_m in enumerate(beta_list):  # M: the columns
                    if beta_h[0] == theta_h:  # check lambda
                        belief[i][j] *= weight
                        if beta_h[1] == lambda_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                        if beta_h[1] == lambda_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)

                    if beta_m[0] == theta_m:  # check lambda
                        belief[i][j] *= weight
                        if beta_m[1] == lambda_m:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                        if beta_m[1] == lambda_m:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)

                    # if beta_h == [lambda_h, theta_h] and beta_m == [lambda_m, theta_m]:
                    #     belief[i][j] = weight
                    # else:
                    #     belief[i][j] = 1

        # TODO: not in use! we only use the game theoretic inference
        else:  # get belief on H agent only
            belief = np.ones((len(lambda_list), len(theta_list)))
            for i, lamb in enumerate(lambda_list):
                for j, theta in enumerate(theta_list):
                    if lamb == lambda_h:  # check lambda
                        belief[i][j] *= weight
                        if theta == theta_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                        if theta == theta_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
        # THIS SHOULD NOT NEED TO BE NORMALIZED!
        # print(belief, np.sum(belief))
        assert round(np.sum(belief)) == 1
        return belief


# dummy class
class dummy(object):
    pass


if __name__ == '__main__':
    Car1 = {"initial_state": [[200, 200, 0, 0, 0]], "par": 1, "initial_action":1}
    from environment import *
    env = Environment("merger")
    sim = dummy()
    sim.dt = 1
    #def __init__(self, sim, env, par, inference_model, decision_model, i):
    test_car = AutonomousVehicle(sim, env, Car1, "baseline", "baseline", 0)
    i = 1
    while i<20:
        test_car.dynamics([1, 1])
        i+= 1

    
    # def multi_search(self, guess_set):
    #     s = self
    #     o = s.other_car
    #     theta_s = s.intent
    #     box_o = o.collision_box
    #     box_s = s.collision_box
    #     orientation_o = o.car_par.ORIENTATION
    #     orientation_s = s.car_par.ORIENTATION
    #
    #     """ run multiple searches with different initial guesses """
    #     trajectory_set = np.empty((0, 2))  # TODO: need to generalize
    #     trajectory_other_set = []
    #     loss_value_set = []
    #     inference_probability_set = []
    #
    #     for guess in guess_set:
    #         fun, trajectory_other, inference_probability = self.loss.loss(guess, self, [])
    #         trajectory_set = np.append(trajectory_set, [guess], axis=0)
    #         trajectory_other_set.append(trajectory_other)
    #         inference_probability_set.append(inference_probability)
    #         loss_value_set = np.append(loss_value_set, fun)
    #
    #     candidates = np.where(loss_value_set == np.min(loss_value_set))[0][0]
    #     self.predicted_trajectory_other = trajectory_other_set[candidates]
    #     self.predicted_actions_other = [self.dynamic(self.predicted_trajectory_other[i])
    #                                     for i in range(len(self.predicted_trajectory_other))]
    #     self.inference_probability_proactive = inference_probability_set[candidates]
    #
    #     trajectory = trajectory_set[candidates]
    #
    #     return trajectory
    #
    # def multi_search_courteous(self, guess_set):
    #     s = self
    #     o = s.other_car
    #     theta_s = s.intent
    #     box_o = o.collision_box
    #     box_s = s.collision_box
    #     orientation_o = o.car_par.ORIENTATION
    #     orientation_s = s.car_par.ORIENTATION
    #
    #     """ run multiple searches with different initial guesses """
    #     trajectory_set = np.empty((0, 2))  # TODO: need to generalize
    #     trajectory_other_set = []
    #     loss_value_set = []
    #     inference_probability_set = []
    #
    #     # first find courteous baseline payoff of the other agent
    #     baseline_loss_all = []
    #     for wanted_trajectory_self, theta_other in zip(s.wanted_trajectory_self, s.predicted_theta_other):
    #         baseline_loss_all.append(s.loss.courteous_baseline_loss(
    #             agent=s, action=wanted_trajectory_self, other_agent_intent=theta_other))
    #
    #     for guess in guess_set:
    #         fun = self.loss.berkeley_courtesy_loss(
    #                 agent=s, action=guess, baseline=baseline_loss_all, beta=C.COURTESY_CONSTANT)
    #         trajectory_set = np.append(trajectory_set, [guess], axis=0)
    #         loss_value_set = np.append(loss_value_set, fun)
    #     candidates = np.where(loss_value_set == np.min(loss_value_set))[0][0]
    #     trajectory = guess_set[candidates]
    #
    #     # get responsive actions from the other agent when ego agent takes "trajectory"
    #     trajectory_other_set = []
    #     for other_intent, other_intent_p in zip(s.predicted_theta_other, s.inference_probability):
    #         # predict how others perceive your action
    #         trajectory_other_all, trajectory_probability = \
    #             self.loss.other_agent_response(agent=s, action=trajectory, other_agent_intent=other_intent)
    #         trajectory_other_set.append([other_intent,
    #                                      other_intent_p,
    #                                      trajectory_other_all,
    #                                      trajectory_probability,
    #                                      1./len(trajectory_other_all)])
    #
    #     self.inference_probability_proactive = []
    #     self.predicted_trajectory_other = []
    #     self.predicted_actions_other = []
    #     for i in range(len(trajectory_other_set)):
    #         for trajectory_other in trajectory_other_set[i][2]:
    #             self.inference_probability_proactive.append(s.inference_probability[i]*trajectory_other_set[i][4])
    #             self.predicted_trajectory_other.append(trajectory_other)
    #             self.predicted_actions_other.append(self.loss.dynamic(trajectory_other, o)[0])
    #
    #     self.inference_probability_proactive = \
    #         np.array(self.inference_probability_proactive)/sum(self.inference_probability_proactive)
    #
    #     return trajectory
    #
    # def multi_search_berkeley_courteous(self, guess_set):
    #     s = self
    #     o = s.other_car
    #     theta_s = s.intent
    #     box_o = o.collision_box
    #     box_s = s.collision_box
    #     orientation_o = o.car_par.ORIENTATION
    #     orientation_s = s.car_par.ORIENTATION
    #
    #     """ run multiple searches with different initial guesses """
    #     trajectory_set = np.empty((0, 2))  # TODO: need to generalize
    #     trajectory_other_set = []
    #     loss_value_set = []
    #     inference_probability_set = []
    #
    #     # first find courteous baseline payoff of the other agent
    #     baseline_loss_all = []
    #     for theta_other in s.predicted_theta_other:
    #         baseline_temp = []
    #         for guess in guess_set:
    #             baseline_temp.append(s.loss.courteous_baseline_loss(
    #                 agent=s, action=guess, other_agent_intent=theta_other))
    #         baseline_loss_all.append(min(baseline_temp))
    #
    #     for guess in guess_set:
    #         fun = self.loss.berkeley_courtesy_loss(
    #                 agent=s, action=guess, baseline=baseline_loss_all, beta=C.COURTESY_CONSTANT)
    #         trajectory_set = np.append(trajectory_set, [guess], axis=0)
    #         loss_value_set = np.append(loss_value_set, fun)
    #     candidates = np.where(loss_value_set == np.min(loss_value_set))[0][0]
    #     trajectory = guess_set[candidates]
    #
    #     # get responsive actions from the other agent when ego agent takes "trajectory"
    #     trajectory_other_set = []
    #     for other_intent, other_intent_p in zip(s.predicted_theta_other, s.inference_probability):
    #         # predict how others perceive your action
    #         trajectory_other_all, trajectory_probability = \
    #             self.loss.other_agent_response(agent=s, action=trajectory, other_agent_intent=other_intent)
    #         trajectory_other_set.append([other_intent,
    #                                      other_intent_p,
    #                                      trajectory_other_all,
    #                                      trajectory_probability,
    #                                      1./len(trajectory_other_all)])
    #
    #     self.inference_probability_proactive = []
    #     self.predicted_trajectory_other = []
    #     self.predicted_actions_other = []
    #     for i in range(len(trajectory_other_set)):
    #         for trajectory_other in trajectory_other_set[i][2]:
    #             self.inference_probability_proactive.append(s.inference_probability[i]*trajectory_other_set[i][4])
    #             self.predicted_trajectory_other.append(trajectory_other)
    #             self.predicted_actions_other.append(self.loss.dynamic(trajectory_other, o)[0])
    #
    #     self.inference_probability_proactive = \
    #         np.array(self.inference_probability_proactive)/sum(self.inference_probability_proactive)
    #
    #     return trajectory
    #
    #
    # def get_predicted_intent_of_other(self):
    #     """ predict the aggressiveness of the agent and what the agent expect me to do """
    #     who = self.who
    #     cons = []
    #     if who == 1:  # machine looking at human
    #         if self.env_par.BOUND_HUMAN_X is not None:  # intersection
    #             intent_bounds = [(0.1, None),  # alpha
    #                              (0 * self.env_par.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.env_par.VEHICLE_MAX_SPEED),
    #                              # radius
    #                              (-C.ACTION_TURNANGLE + self.car_par.ORIENTATION,
    #                               C.ACTION_TURNANGLE + self.car_par.ORIENTATION)]  # angle, to accommodate crazy behavior
    #             cons.append({'type': 'ineq',
    #                          'fun': lambda x: self.states[-1][1] + (x[0] * scipy.sin(np.deg2rad(x[1]))) - (0.4 - 0.33)})
    #             cons.append({'type': 'ineq',
    #                          'fun': lambda x: - self.states[-1][1] - (x[0] * scipy.sin(np.deg2rad(x[1]))) + (
    #                          -0.4 + 0.33)})
    #
    #         else:  # TODO: update this part
    #             intent_bounds = [(0.1, None),  # alpha
    #                              (-C.ACTION_TIMESTEPS * self.env_par.VEHICLE_MAX_SPEED,
    #                               C.ACTION_TIMESTEPS * self.env_par.VEHICLE_MAX_SPEED),  # radius
    #                              (-90, 90)]  # angle, to accommodate crazy behavior
    #     else:  # human looking at machine
    #         if self.env_par.BOUND_HUMAN_X is not None:  # intersection
    #             intent_bounds = [(0.1, None),  # alpha
    #                              (0 * self.env_par.VEHICLE_MAX_SPEED, C.ACTION_TIMESTEPS * self.env_par.VEHICLE_MAX_SPEED),
    #                              # radius
    #                              (-C.ACTION_TURNANGLE + self.car_par.ORIENTATION,
    #                               C.ACTION_TURNANGLE + self.car_par.ORIENTATION)]  # angle, to accommodate crazy behavior
    #             cons.append({'type': 'ineq',
    #                          'fun': lambda x: self.states[-1][0] + (x[0] * scipy.cos(np.deg2rad(x[1]))) - (0.4 - 0.33)})
    #             cons.append({'type': 'ineq',
    #                          'fun': lambda x: - self.states[-1][0] - (x[0] * scipy.cos(np.deg2rad(x[1]))) + (
    #                          -0.4 + 0.33)})
    #
    #         else:  # TODO: update this part
    #             intent_bounds = [(0.1, None),  # alpha
    #                              (-C.ACTION_TIMESTEPS * self.env_par.VEHICLE_MAX_SPEED,
    #                               C.ACTION_TIMESTEPS * self.env_par.VEHICLE_MAX_SPEED),  # radius
    #                              (-90, 90)]  # angle, to accommodate crazy behavior
    #
    #     # TODO: damn...I don't know why the solver is not working, insert a valid solution, output nan...
    #     # trials_trajectory_other = np.arange(5,-0.1,-0.1)
    #     # trials_trajectory_self = np.arange(5,-0.1,-0.1)
    #
    #     # guess_set = np.hstack((np.ones((trials.size,1)) * self.human_predicted_theta[0], np.expand_dims(trials, axis=1),
    #     #                        np.ones((trials.size,1)) * self.machine_orientation))
    #
    #     # if self.loss.characterization is 'reactive':
    #     intent_optimization_results = self.multi_search_intent()  # believes none is aggressive
    #     # elif self.loss.characterization is 'aggressive':
    #     #     intent_optimization_results = self.multi_search_intent_aggressive()  # believes both are aggressive
    #     # elif self.loss.characterization is 'passive_aggressive':
    #     #     intent_optimization_results = self.multi_search_intent_passive_aggressive()  # believes both are aggressive
    #
    #     # theta_other, theta_self, trajectory_other, trajectory_self = intent_optimization_results
    #
    #     return intent_optimization_results
    #
    # def multi_search_intent(self):
    #     """ run multiple searches with different initial guesses """
    #     s = self
    #     who = self.who
    #     trials_theta_other = C.THETA_SET
    #     # if who == 1:
    #     #    trials_theta = [1.]
    #     # else:
    #     #    trials_theta = C.THETA_SET
    #     trials_theta = C.THETA_SET
    #     inference_set = []  # T0poODO: need to generalize
    #     loss_value_set = []
    #     for theta_self in trials_theta:
    #     # for theta_self in [s.intent]:
    #         if s.who == 1:
    #             theta_self = s.intent
    #         for theta_other in trials_theta_other:
    #             # for theta_other in [1.]:
    #             trajectory_self, trajectory_other, my_loss_all, other_loss_all = self.equilibrium(theta_self,
    #                                                                                               theta_other, s,
    #                                                                                               s.other_car)
    #
    #             # my_trajectory = [trajectory_self[i] for i in
    #             #                  np.where(other_loss_all == np.min(other_loss_all))[0]]  # I believe others move fast
    #             # other_trajectory = [trajectory_other[i] for i in
    #             #                     np.where(my_loss_all == np.min(my_loss_all))[0]]  # others believe I move fast
    #             my_trajectory = trajectory_self
    #             other_trajectory = trajectory_other
    #             # other_trajectory_conservative = \
    #             #     [trajectory_other[i] for i in
    #             #      np.where(other_loss_all == np.min(other_loss_all))[0]]  # others move fast
    #
    #             trajectory_self_wanted_other = []
    #             other_trajectory_wanted = []
    #
    #             if trajectory_self is not []:
    #                 action_self = [self.dynamic(my_trajectory[i])
    #                                for i in range(len(my_trajectory))]
    #                 action_other = [self.dynamic(other_trajectory[i])
    #                                 for i in range(len(other_trajectory))]
    #                 # print '*********'
    #                 # print action_other
    #                 # print action_self
    #                 # print "&&&&&&&&&"
    #                 if self.frame == 0:
    #                     if self.who == 1:
    #                         fun_self = [np.linalg.norm(np.sum((action_self[i][0] - s.states[-1]) - s.actions_set[-1]) - self.car_par.ABILITY * my_trajectory[i][0])
    #                                     for i in range(len(my_trajectory))]
    #                         fun_other = [np.linalg.norm(np.sum((action_other[i][0] - s.states_o[-1]) - s.actions_set_o[-1]) + self.car_par.ABILITY_O * other_trajectory[i][0])
    #                                      for i in range(len(other_trajectory))]
    #                     else:
    #                         fun_self = [np.linalg.norm(np.sum((action_self[i][0] - s.states[-1]) - s.actions_set[-1]) + self.car_par.ABILITY * my_trajectory[i][0])
    #                                     for i in range(len(my_trajectory))]
    #                         fun_other = [np.linalg.norm(np.sum((action_other[i][0] - s.states_o[-1]) - s.actions_set_o[-1]) - self.car_par.ABILITY_O * other_trajectory[i][0])
    #                                      for i in range(len(other_trajectory))]
    #                 else:
    #                     if self.who == 1:
    #                         fun_self = [np.linalg.norm(np.sum(s.actions_set[-1] - s.actions_set[-2]) - self.car_par.ABILITY * my_trajectory[i][0])
    #                                     for i in range(len(my_trajectory))]
    #                         fun_other = [np.linalg.norm(np.sum(s.actions_set_o[-1] - s.actions_set_o[-2]) + self.car_par.ABILITY_O * other_trajectory[i][0])
    #                                      for i in range(len(other_trajectory))]
    #                     else:
    #                         fun_self = [np.linalg.norm(np.sum(s.actions_set[-1] - s.actions_set[-2]) + self.car_par.ABILITY * my_trajectory[i][0])
    #                                     for i in range(len(my_trajectory))]
    #                         fun_other = [np.linalg.norm(np.sum(s.actions_set_o[-1] - s.actions_set_o[-2]) - self.car_par.ABILITY_O * other_trajectory[i][0])
    #                                      for i in range(len(other_trajectory))]
    #                 # print fun_self
    #                 # print fun_other
    #                 if len(fun_other) != 0:
    #                     fun = min(fun_other)
    #                 else:
    #                     break
    #
    #                 # what I think other want me to do if he wants to take the benefit
    #                 trajectory_self_wanted_other = \
    #                     [trajectory_self[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]
    #
    #                 # what I want other to do
    #                 other_trajectory_wanted = \
    #                     [trajectory_other[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
    #
    #                 # what I think others expect me to do
    #                 trajectory_self = np.atleast_2d(
    #                     [my_trajectory[i] for i in np.where(fun_self == np.min(fun_self))[0]])
    #
    #                 # what I think others will do
    #                 trajectory_other = np.atleast_2d(
    #                     [other_trajectory[i] for i in np.where(fun_other == fun)[0]])
    #                 # else:
    #                 #     fun = 0
    #                 #
    #                 #     # what I think other want me to do if he wants to take the benefit
    #                 #     trajectory_self_wanted_other = \
    #                 #         [trajectory_self[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]
    #                 #
    #                 #     # what I want other to do
    #                 #     other_trajectory_wanted = \
    #                 #         [trajectory_other[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
    #                 #
    #                 #     # what I think others expect me to do
    #                 #     trajectory_self = np.atleast_2d(
    #                 #         [my_trajectory[i] for i in np.where(fun_self == np.min(fun_self))[0]])
    #                 #
    #                 #     # what I think others will do
    #                 #     trajectory_other = np.atleast_2d(
    #                 #         [other_trajectory[i] for i in np.where(fun_other == fun)[0]])
    #             else:
    #                 fun = 1e32
    #
    #             inference_set.append([theta_self,
    #                                   theta_other,
    #                                   trajectory_other,
    #                                   trajectory_self,
    #                                   trajectory_self_wanted_other,
    #                                   other_trajectory_wanted,
    #                                   1./len(trajectory_other)*len(trajectory_self_wanted_other)*len(other_trajectory_wanted)])
    #             loss_value_set.append(round(fun*1e12)/1e12)
    #
    #     candidate = np.where(loss_value_set == np.min(loss_value_set))[0]
    #     theta_self_out = []
    #     theta_other_out = []
    #     trajectory_self_out = []
    #     trajectory_other_out = []
    #     trajectory_self_wanted_other_out = []
    #     other_trajectory_wanted_out = []
    #     inference_probability_out = []
    #     theta_probability = []
    #
    #     for i in range(len(candidate)):
    #         for j in range(len(inference_set[candidate[i]][2])):
    #             for k in range(len(inference_set[candidate[i]][3])):
    #                 for l in range(len(inference_set[candidate[i]][4])):
    #                     for p in range(len(inference_set[candidate[i]][5])):
    #                         theta_self_out.append(inference_set[candidate[i]][0])
    #                         theta_other_out.append(inference_set[candidate[i]][1])
    #                         trajectory_other_out.append(inference_set[candidate[i]][2][j])
    #                         trajectory_self_out.append(inference_set[candidate[i]][3][k])
    #                         trajectory_self_wanted_other_out.append(inference_set[candidate[i]][4][l])
    #                         other_trajectory_wanted_out.append(inference_set[candidate[i]][5][p])
    #                         inference_probability_out.append(1./len(candidate)*inference_set[candidate[i]][6])
    #
    #     inference_probability_out = np.array(inference_probability_out)
    #     # update theta probability
    #     for theta_other in trials_theta_other:
    #         theta_probability.append(sum(inference_probability_out[np.where(theta_other_out==theta_other)[0]]))
    #     #theta_probability = (self.theta_probability * self.frame + theta_probability) / (self.frame + 1)
    #     theta_probability = self.theta_probability * theta_probability
    #     if sum(theta_probability) > 0:
    #         theta_probability = theta_probability/sum(theta_probability)
    #     else:
    #         theta_probability = np.ones(C.THETA_SET.shape)/C.THETA_SET.size
    #
    #     # update inference probability accordingly
    #     for i in range(len(trials_theta_other)):
    #         id = np.where(theta_other_out == trials_theta_other[i])[0]
    #         inference_probability_out[id] = inference_probability_out[id]/\
    #                                          sum(inference_probability_out[id]) * theta_probability[i]
    #     inference_probability_out = inference_probability_out/sum(inference_probability_out)
    #
    #     return theta_other_out, theta_self_out, trajectory_other_out, trajectory_self_out, \
    #            trajectory_self_wanted_other_out, other_trajectory_wanted_out, inference_probability_out, \
    #            theta_probability
    #
    # def multi_search_intent_aggressive(self):
    #     """ run multiple searches with different initial guesses """
    #     s = self
    #     who = self.who
    #     trials_theta = C.THETA_SET
    #     inference_set = []  # TODO: need to generalize
    #     loss_value_set = []
    #
    #     for theta_self in trials_theta:
    #         for theta_other in trials_theta:
    #             # for theta_other in [1.]:
    #             trajectory_self, trajectory_other, my_loss_all, other_loss_all = self.equilibrium(theta_self,
    #                                                                                               theta_other, s,
    #                                                                                               s.other_car)
    #
    #             my_trajectory = [trajectory_self[i] for i in
    #                              np.where(other_loss_all == np.min(other_loss_all))[0]]  # I believe others move fast
    #             other_trajectory = [trajectory_other[i] for i in
    #                                 np.where(my_loss_all == np.min(my_loss_all))[0]]  # I believe others move fast
    #             other_trajectory_conservative = [trajectory_other[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]  # others move slow
    #
    #             if trajectory_self is not []:
    #                 action_self = [self.interpolate_from_trajectory(my_trajectory[i])
    #                                for i in range(len(my_trajectory))]
    #                 action_other = [self.interpolate_from_trajectory(other_trajectory[i])
    #                                 for i in range(len(other_trajectory))]
    #
    #                 fun_self = [np.linalg.norm(action_self[i][:s.track_back] - s.actions_set[-s.track_back:])
    #                             for i in range(len(action_self))]
    #                 fun_other = [np.linalg.norm(action_other[i][:s.track_back] - s.actions_set_o[-s.track_back:])
    #                              for i in range(len(action_other))]
    #
    #                 fun = min(fun_other)
    #
    #                 # what I think other want me to do if he wants to take the benefit
    #                 trajectory_self_wanted_other = \
    #                     [trajectory_self[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]
    #                 trajectory_self_wanted_other = \
    #                     [trajectory_self_wanted_other[i] for i in np.where(fun_other == fun)[0]]
    #
    #                 trajectory_self = np.atleast_2d(
    #                     [my_trajectory[i] for i in np.where(fun_self == np.min(fun_self))[0]])
    #                 trajectory_other = np.atleast_2d(other_trajectory_conservative)
    #                 # my_loss_all = [my_loss_all[i] for i in np.where(fun_self == np.min(fun_self))[0]]
    #                 #
    #                 # trajectory_self = [trajectory_self[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
    #                 # trajectory_other = [trajectory_other[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
    #             else:
    #                 fun = 1e32
    #
    #             inference_set.append([theta_self,
    #                                   theta_other,
    #                                   trajectory_other,
    #                                   trajectory_self,
    #                                   trajectory_self_wanted_other])
    #             loss_value_set.append(fun)
    #
    #     candidate = np.where(loss_value_set == np.min(loss_value_set))[0]
    #     # inference = inference_set[candidate[np.random.randint(len(candidate))]]
    #     theta_self_out = []
    #     theta_other_out = []
    #     trajectory_self_out = []
    #     trajectory_other_out = []
    #     trajectory_self_wanted_other_out = []
    #     for i in range(len(candidate)):
    #         for j in range(len(inference_set[candidate[i]][2])):
    #             for k in range(len(inference_set[candidate[i]][3])):
    #                 for q in range(len(inference_set[candidate[i]][4])):
    #                     theta_self_out.append(inference_set[candidate[i]][0])
    #                     theta_other_out.append(inference_set[candidate[i]][1])
    #                     trajectory_other_out.append(inference_set[candidate[i]][2][k])
    #                     trajectory_self_out.append(inference_set[candidate[i]][3][j])
    #                     trajectory_self_wanted_other_out.append(inference_set[candidate[i]][4][q])
    #
    #
    #     return theta_other_out, theta_self_out, trajectory_other_out, trajectory_self_out, \
    #            trajectory_self_wanted_other_out
    #
    # def multi_search_intent_passive_aggressive(self):
    #     """ run multiple searches with different initial guesses """
    #     s = self
    #     who = self.who
    #     trials_theta = C.THETA_SET
    #     inference_set = []  # TODO: need to generalize
    #     loss_value_set = []
    #
    #     for theta_self in trials_theta:
    #         for theta_other in trials_theta:
    #             trajectory_self, trajectory_other, my_loss_all, other_loss_all = self.equilibrium(theta_self,
    #                                                                                               theta_other, s,
    #                                                                                               s.other_car)
    #
    #             my_trajectory = [trajectory_self[i] for i in
    #                              np.where(other_loss_all == np.min(other_loss_all))[0]]  # I believe others move fast
    #             other_trajectory = [trajectory_other[i] for i in
    #                                 np.where(my_loss_all == np.min(my_loss_all))[0]]  # I believe others move fast
    #             other_trajectory_conservative = \
    #                 [trajectory_other[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]  # others move slow
    #
    #             if trajectory_self is not []:
    #                 action_self = [self.interpolate_from_trajectory(my_trajectory[i])
    #                                for i in range(len(my_trajectory))]
    #                 action_other = [self.interpolate_from_trajectory(other_trajectory[i])
    #                                 for i in range(len(other_trajectory))]
    #
    #                 fun_self = [np.linalg.norm(action_self[i][:s.track_back] - s.actions_set[-s.track_back:])
    #                             for i in range(len(action_self))]
    #                 fun_other = [np.linalg.norm(action_other[i][:s.track_back] - s.actions_set_o[-s.track_back:])
    #                              for i in range(len(action_other))]
    #                 if fun_other is not []:
    #                     fun = min(fun_other)
    #
    #                     # what I think other want me to do if he wants to take the benefit
    #                     trajectory_self_wanted_other = \
    #                         [trajectory_self[i] for i in np.where(other_loss_all == np.min(other_loss_all))[0]]
    #                     trajectory_self_wanted_other = \
    #                         [trajectory_self_wanted_other[i] for i in np.where(fun_other == fun)[0]]
    #
    #                     trajectory_self = np.atleast_2d(
    #                         [my_trajectory[i] for i in np.where(fun_self == np.min(fun_self))[0]])
    #                     trajectory_other = np.atleast_2d(other_trajectory_conservative)
    #                     # my_loss_all = [my_loss_all[i] for i in np.where(fun_self == np.min(fun_self))[0]]
    #                     #
    #                     # trajectory_self = [trajectory_self[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
    #                     # trajectory_other = [trajectory_other[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
    #                 else:
    #                     fun = 1e32
    #             else:
    #                 fun = 1e32
    #
    #             inference_set.append([theta_self,
    #                                   theta_other,
    #                                   trajectory_other,
    #                                   trajectory_self,
    #                                   trajectory_self_wanted_other])
    #             loss_value_set.append(fun)
    #
    #     candidate = np.where(loss_value_set == np.min(loss_value_set))[0]
    #     # inference = inference_set[candidate[np.random.randint(len(candidate))]]
    #     theta_self_out = []
    #     theta_other_out = []
    #     trajectory_self_out = []
    #     trajectory_other_out = []
    #     trajectory_self_wanted_other_out = []
    #     for i in range(len(candidate)):
    #         for j in range(len(inference_set[candidate[i]][2])):
    #             for k in range(len(inference_set[candidate[i]][3])):
    #                 for q in range(len(inference_set[candidate[i]][4])):
    #                     theta_self_out.append(inference_set[candidate[i]][0])
    #                     theta_other_out.append(inference_set[candidate[i]][1])
    #                     trajectory_other_out.append(inference_set[candidate[i]][2][k])
    #                     trajectory_self_out.append(inference_set[candidate[i]][3][j])
    #                     trajectory_self_wanted_other_out.append(inference_set[candidate[i]][4][q])
    #     return theta_other_out, theta_self_out, trajectory_other_out, trajectory_self_out, \
    #            trajectory_self_wanted_other_out
    #
    # def equilibrium(self, theta_self, theta_other, s, o):
    #     action_guess = C.TRAJECTORY_SET
    #     trials_trajectory_self = np.hstack((np.expand_dims(action_guess, axis=1),
    #                                         np.ones((action_guess.size, 1)) * s.car_par.ORIENTATION))
    #     trials_trajectory_other = np.hstack((np.expand_dims(action_guess, axis=1),
    #                                          np.ones((action_guess.size, 1)) * o.car_par.ORIENTATION))
    #     loss_matrix = np.zeros((trials_trajectory_self.shape[0], trials_trajectory_other.shape[0], 2))
    #     for i in range(trials_trajectory_self.shape[0]):
    #         for j in range(trials_trajectory_other.shape[0]):
    #             loss_matrix[i, j, :] = self.simulate_game([trials_trajectory_self[i]], [trials_trajectory_other[j]],
    #                                                       theta_self, theta_other, s, o)
    #
    #     # find equilibrium
    #     my_loss_all = []
    #     other_loss_all = []
    #     eq_all = []
    #     for j in range(trials_trajectory_other.shape[0]):
    #         id_s = np.atleast_1d(np.argmin(loss_matrix[:, j, 0]))
    #         for i in range(id_s.size):
    #             id_o = np.atleast_1d(np.argmin(loss_matrix[id_s[i], :, 1]))
    #             # print sum(np.isin(id_o, j))
    #             if sum(np.isin(id_o, j)) > 0:
    #                 eq_all.append([id_s[i], j])
    #                 my_loss_all.append(loss_matrix[id_s[i], j, 0])
    #                 other_loss_all.append(loss_matrix[id_s[i], j, 1])  # put self in the other's shoes
    #
    #     # eq = [eq_all[i] for i in np.where(my_loss_all == np.min(my_loss_all))[0]]
    #     # print eq_all # skip when no pure equilibrium.
    #     # if len(eq_all) != 0:
    #     trajectory_self = [trials_trajectory_self[eq_all[i][0]] for i in range(len(eq_all))]
    #     trajectory_other = [trials_trajectory_other[eq_all[i][1]] for i in range(len(eq_all))]
    #
    #     # else:
    #     #     # trajectory_self = []
    #     #     # trajectory_other = []
    #     #     eq_all = np.array([[5,0],[0,5]])
    #     #     trajectory_self = [trials_trajectory_self[eq_all[i][0]] for i in range(len(eq_all))]
    #     #     trajectory_other = [trials_trajectory_other[eq_all[i][1]] for i in range(len(eq_all))]
    #     #     my_loss_all = [loss_matrix[5, 0, 0],loss_matrix[0,5,0]]
    #     #     other_loss_all = [loss_matrix[5, 0, 1],loss_matrix[0,5,1]]  # put self in the other's shoes
    #     #     eq_all = []
    #     # print trajectory_self
    #     # print trajectory_other
    #     return trajectory_self, trajectory_other, my_loss_all, other_loss_all
    #
    # def simulate_game(self, trajectory_self, trajectory_other, theta_self, theta_other, s, o):
    #     # if len(s.states) == 1:
    #     loss_s = self.loss.reactive_loss(theta_self, trajectory_self, trajectory_other, [1],
    #                                      s.states[-s.track_back],
    #                                      [0,0], [0,0],
    #                                      s.states_o[-s.track_back],
    #                                      [0,0], [0,0],
    #                                      s)
    #     loss_o = self.loss.reactive_loss(theta_other, trajectory_other, trajectory_self, [1],
    #                                      s.states_o[-s.track_back],
    #                                      [0,0], [0,0],
    #                                      s.states[-s.track_back],
    #                                      [0,0], [0,0],
    #                                      s)
    #     # elif len(s.states)== 2:
    #     #
    #     #     loss_s = self.loss.reactive_loss(theta_self, trajectory_self, trajectory_other, [1],
    #     #                                      s.states[-s.track_back],
    #     #                                      s.states[-s.track_back] - s.states[-s.track_back - 1],s.states[-s.track_back] - s.states[-s.track_back - 1],
    #     #                                      s.states_o[-s.track_back],
    #     #                                      s.states_o[-s.track_back] - s.states_o[-s.track_back - 1],s.states_o[-s.track_back] - s.states_o[-s.track_back - 1],
    #     #                                      s)
    #     #     loss_o = self.loss.reactive_loss(theta_other, trajectory_other, trajectory_self, [1],
    #     #                                      s.states_o[-s.track_back],
    #     #                                      s.states_o[-s.track_back] - s.states_o[-s.track_back - 1],s.states_o[-s.track_back] - s.states_o[-s.track_back - 1],
    #     #                                      s.states[-s.track_back],
    #     #                                      s.states[-s.track_back] - s.states[-s.track_back - 1],s.states[-s.track_back] - s.states[-s.track_back - 1] ,
    #     #                                      o)
    #     #
    #     # else:
    #     #
    #     #     loss_s = self.loss.reactive_loss(theta_self, trajectory_self, trajectory_other, [1],
    #     #                                      s.states[-s.track_back],
    #     #                                      s.states[-s.track_back]-s.states[-s.track_back-1],(s.states[-s.track_back]-s.states[-s.track_back-1])-(s.states[-s.track_back-1]-s.states[-s.track_back-2]) ,
    #     #                                      s.states_o[-s.track_back],
    #     #                                      s.states_o[-s.track_back]-s.states_o[-s.track_back-1],(s.states_o[-s.track_back]-s.states_o[-s.track_back-1])-(s.states_o[-s.track_back-1]-s.states_o[-s.track_back-2]),
    #     #                                      s)
    #     #     loss_o = self.loss.reactive_loss(theta_other, trajectory_other, trajectory_self, [1],
    #     #                                      s.states_o[-s.track_back],
    #     #                                      s.states_o[-s.track_back]-s.states_o[-s.track_back-1],(s.states_o[-s.track_back]-s.states_o[-s.track_back-1])-(s.states_o[-s.track_back-1]-s.states_o[-s.track_back-2]),
    #     #                                      s.states[-s.track_back],
    #     #                                      s.states[-s.track_back]-s.states[-s.track_back-1],(s.states[-s.track_back]-s.states[-s.track_back-1])-(s.states[-s.track_back-1]-s.states[-s.track_back-2]) ,
    #     #                                      o)
    #
    #     return loss_s, loss_o
    #
    # def intent_loss_func(self, intent):
    #     orientation_self = self.P_CAR_S.ORIENTATION
    #     state_self = self.states_s[-C.TRACK_BACK]
    #     state_other = self.states_o[-C.TRACK_BACK]
    #     action_other = self.actions_set_o[-C.TRACK_BACK]
    #     who = 1 - (self.P_CAR_S.BOUND_X is None)
    #
    #     # alpha = intent[0] #aggressiveness of the agent
    #     trajectory = intent  # what I was expected to do
    #
    #     # what I could have done and been
    #     s_other = np.array(state_self)
    #     nodes = np.array([[s_other[0], s_other[0] + trajectory[0] * np.cos(np.deg2rad(self.P_CAR_S.ORIENTATION)) / 2,
    #                        s_other[0] + trajectory[0] * np.cos(np.deg2rad(trajectory[1]))],
    #                       [s_other[1], s_other[1] + trajectory[0] * np.sin(np.deg2rad(orientation_self)) / 2,
    #                        s_other[1] + trajectory[0] * np.sin(np.deg2rad(trajectory[1]))]])
    #     curve = bezier.Curve(nodes, degree=2)
    #     positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_TIMESTEPS + 1)))
    #     a_other = np.diff(positions, n=1, axis=0)
    #     s_other_traj = np.array(s_other + np.matmul(M.LOWER_TRIANGULAR_MATRIX, a_other))
    #
    #     # actions and states of the agent
    #     s_self = np.array(state_other)  # self.human_states_set[-1]
    #     a_self = np.array(C.ACTION_TIMESTEPS * [action_other])  # project current agent actions to future
    #     s_self_traj = np.array(s_self + np.matmul(M.LOWER_TRIANGULAR_MATRIX, a_self))
    #
    #     # I expect the agent to be this much aggressive
    #     # theta_self = self.human_predicted_theta
    #
    #     # calculate the gradient of the control objective
    #     A = M.LOWER_TRIANGULAR_MATRIX
    #     D = np.sum((s_other_traj - s_self_traj) ** 2.,
    #                axis=1) + 1e-12  # should be t_steps by 1, add small number for numerical stability
    #     # need to check if states are in the collision box
    #     gap = 1.05
    #     for i in range(s_self_traj.shape[0]):
    #         if who == 1:
    #             if s_self_traj[i, 0] <= -gap + 1e-12 or s_self_traj[i, 0] >= gap - 1e-12 or s_other_traj[
    #                 i, 1] >= gap - 1e-12 or s_other_traj[i, 1] <= -gap + 1e-12:
    #                 D[i] = np.inf
    #         elif who == 0:
    #             if s_self_traj[i, 1] <= -gap + 1e-12 or s_self_traj[i, 1] >= gap - 1e-12 or s_other_traj[
    #                 i, 0] >= gap - 1e-12 or s_other_traj[i, 0] <= -gap + 1e-12:
    #                 D[i] = np.inf
    #
    #     # dD/da
    #     # dDda_self = - np.dot(np.expand_dims(np.dot(A.transpose(), sigD**(-2)*dsigD),axis=1), np.expand_dims(ds, axis=0)) \
    #     #        - np.dot(np.dot(A.transpose(), np.diag(sigD**(-2)*dsigD)), np.dot(A, a_self - a_other))
    #     dDda_self = - 2 * C.EXPCOLLISION * np.dot(A.transpose(), (s_self_traj - s_other_traj) *
    #                                               np.expand_dims(
    #                                                   np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5)),
    #                                                   axis=1))
    #
    #     # update theta_hat_H
    #     w = - dDda_self  # negative gradient direction
    #
    #     if who == 0:
    #         if self.env_par.BOUND_HUMAN_X is not None:  # intersection
    #             w[np.all([s_self_traj[:, 0] <= 1e-12, w[:, 0] <= 1e-12],
    #                      axis=0), 0] = 0  # if against wall and push towards the wall, get a reaction force
    #             w[np.all([s_self_traj[:, 0] >= -1e-12, w[:, 0] >= -1e-12],
    #                      axis=0), 0] = 0  # TODO: these two lines are hard coded for intersection, need to check the interval
    #             # print(w)
    #         else:  # lane changing
    #             w[np.all([s_self_traj[:, 1] <= 1e-12, w[:, 1] <= 1e-12],
    #                      axis=0), 1] = 0  # if against wall and push towards the wall, get a reaction force
    #             w[np.all([s_self_traj[:, 1] >= 1 - 1e-12, w[:, 1] >= -1e-12],
    #                      axis=0), 1] = 0  # TODO: these two lines are hard coded for lane changing
    #     else:
    #         if self.env_par.BOUND_HUMAN_X is not None:  # intersection
    #             w[np.all([s_self_traj[:, 1] <= 1e-12, w[:, 1] <= 1e-12],
    #                      axis=0), 1] = 0  # if against wall and push towards the wall, get a reaction force
    #             w[np.all([s_self_traj[:, 1] >= -1e-12, w[:, 1] >= -1e-12],
    #                      axis=0), 1] = 0  # TODO: these two lines are hard coded for intersection, need to check the interval
    #         else:  # lane changing
    #             w[np.all([s_self_traj[:, 0] <= 1e-12, w[:, 0] <= 1e-12],
    #                      axis=0), 0] = 0  # if against wall and push towards the wall, get a reaction force
    #             w[np.all([s_self_traj[:, 0] >= -1e-12, w[:, 0] >= -1e-12],
    #                      axis=0), 0] = 0  # TODO: these two lines are hard coded for lane changing
    #     w = -w
    #
    #     # calculate best alpha for the enumeration of trajectory
    #
    #     if who == 1:
    #         l = np.array([- C.EXPTHETA * np.exp(C.EXPTHETA * (-s_self_traj[-1][0] + 0.4)), 0.])
    #         # don't take into account the time steps where one car has already passed
    #         decay = (((s_self_traj - s_other_traj)[:, 0] < gap) + 0.0) * (
    #         (s_self_traj - s_other_traj)[:, 1] < gap + 0.0)
    #     else:
    #         l = np.array([0., C.EXPTHETA * np.exp(C.EXPTHETA * (s_self_traj[-1][1] + 0.4))])
    #         decay = (((s_other_traj - s_self_traj)[:, 0] < gap) + 0.0) * (
    #         (s_other_traj - s_self_traj)[:, 1] < gap + 0.0)
    #     decay = decay * np.exp(np.linspace(0, -10, C.ACTION_TIMESTEPS))
    #     w = w * np.expand_dims(decay, axis=1)
    #     l = l * np.expand_dims(decay, axis=1)
    #     alpha = np.max((- np.trace(np.dot(np.transpose(w), l)) / (np.sum(l ** 2) + 1e-12), 0.1))
    #
    #     # if who == 0:
    #     # alpha = 1.
    #
    #     x = w + alpha * l
    #     L = np.sum(x ** 2)
    #     return L, alpha
    #
    # def interpolate_from_trajectory(self, trajectory):
    #
    #     nodes = np.array([[0, trajectory[0] * np.cos(np.deg2rad(trajectory[1])) / 2,
    #                        trajectory[0] * np.cos(np.deg2rad(trajectory[1]))],
    #                       [0, trajectory[0] * np.sin(np.deg2rad(trajectory[1])) / 2,
    #                        trajectory[0] * np.sin(np.deg2rad(trajectory[1]))]])
    #     # print nodes
    #     curve = bezier.Curve(nodes, degree=2)
    #
    #     positions = np.transpose(curve.evaluate_multi(np.linspace(0, 1, C.ACTION_NUMPOINTS + 1)))
    #     # TODO: skip state?
    #     return np.diff(positions, n=1, axis=0)

