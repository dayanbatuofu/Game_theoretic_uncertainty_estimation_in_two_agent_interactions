import torch
import numpy as np
import pygame as pg
from shapely.geometry import Polygon, Point
from constants import CONSTANTS as C

# Input is initial state and trajectory
class DynamicsMain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, init_state, trajectory, env, args, q_set):
        # Q_g_vals = torch.zeros(init_state.size(0), device=args.device, dtype=torch.float)
        # action_vals = torch.tensor([-8, -4, 0, 4, 8], device=args.device, dtype=torch.float)
        p1_t_cost = torch.zeros(init_state.size(0), device=args.device, dtype=torch.float)
        p2_t_cost = torch.zeros(init_state.size(0), device=args.device, dtype=torch.float)

        # Send all the data for batch processing on GPU
        dynamics = DynamicsBatch.apply

        ns_b, ns_other = init_state, init_state

        for i in range(trajectory.size(1)):
            ns_b, ns_other, p1_c, p2_c = dynamics(ns_b, trajectory.narrow(1, i, 1), env, args)
            p1_t_cost = p1_t_cost.add(p1_c)
            p2_t_cost = p2_t_cost.add(p2_c)

        Q_i_t = p1_t_cost
        Q_j_t = p2_t_cost

        Q_i_vals = q_set[env.t1_idx].forward(ns_b).to(args.device, dtype=torch.float)
        Q_j_vals = q_set[env.t2_idx].forward(ns_other).to(args.device, dtype=torch.float)

        Q_i_t_T_max, Q_i_t_T_index = torch.max(Q_i_vals, dim=1)
        Q_j_t_T_max, Q_j_t_T_index = torch.max(Q_j_vals, dim=1)

        Q_i_t += Q_i_t_T_max
        Q_j_t += Q_j_t_T_max
        # print(Q_i_t, Q_j_t)
        # print(env.ego_car.gracefulness)
        if env.ego_car.gracefulness < 0:
            Q_g_vals = env.ego_car.gracefulness * Q_j_t
        else:
            Q_g_vals = env.ego_car.gracefulness * Q_j_t + (1 - env.ego_car.gracefulness) * Q_i_t

        return torch.max(Q_g_vals), torch.argmax(Q_g_vals)
        # return trajectory[torch.argmax(Q_g_vals)][0]


    @staticmethod
    def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            return grad_input



class DynamicsBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u, env, args):
        ctx.save_for_backward(x, u)
        states = x
        actions = u
        next_states = torch.randn(x.size(), device=args.device, dtype=torch.float, requires_grad=True)
        next_states_other = torch.randn(x.size(), device=args.device, dtype=torch.float, requires_grad=True)
        loss_self_batch = torch.randn(x.size(0), device=args.device, dtype=torch.float, requires_grad=True)
        loss_other_batch = torch.randn(x.size(0), device=args.device, dtype=torch.float, requires_grad=True)

        i = 0
        # can also use memoization; recomputing some values anyway
        transition_dict = {}
        cost1_dict = {}
        cost2_dict = {}
        found_flag = False
        for state, action in zip(states, actions):
            s_list = state.tolist()
            a_list = action.tolist()
            # print(s_list, a_list)
            key = (s_list[0], s_list[1], s_list[2], s_list[3], a_list[0])

            if key not in transition_dict.keys():
                found_flag = False
            else:
                found_flag = True

            if not found_flag:
                next_state = torch.randn(x.size(1), device=args.device, dtype=torch.float, requires_grad=True)
                next_state_other = torch.randn(x.size(1), device=args.device, dtype=torch.float, requires_grad=True)
                done = False
                # print(state, action)
                # print(action)
                if action == 0:
                    a = env.parameters.MAX_DECELERATION
                elif action == 1:
                    a = env.parameters.MAX_DECELERATION * 0.5
                elif action == 2:
                    a = 0.0
                elif action == 3:
                    a = env.parameters.MAX_ACCELERATION * 0.5
                else:
                    a = env.parameters.MAX_ACCELERATION
                action_self = torch.tensor(a, dtype=torch.float, device=args.device, requires_grad=True)

                p2_state = [state[2], state[3], state[0], state[1]]
                # print(p2_state)

                dist = env.nfsp_models[env.inf_idx].act_dist(torch.FloatTensor(p2_state).to(args.device))

                dist_a = dist.tolist()[0]
                action_p2 = np.argmax(dist_a)
                if action_p2 == 0:
                    a = env.parameters.MAX_DECELERATION
                elif action_p2 == 1:
                    a = env.parameters.MAX_DECELERATION * 0.5
                elif action_p2 == 2:
                    a = 0.0
                elif action_p2 == 3:
                    a = env.parameters.MAX_ACCELERATION * 0.5
                else:
                    a = env.parameters.MAX_ACCELERATION
                action_other = torch.tensor(a, dtype=torch.float, device=args.device, requires_grad=True)

                # get current states
                x_ego = x_ego_new = state[0]
                x_other = x_other_new = state[2]
                v_ego = v_ego_new = state[1]
                v_other = v_other_new = state[3]

                max_speed_ego = torch.tensor(env.ego_car.car_parameters.MAX_SPEED[0], dtype=torch.float,
                                             device=args.device)  # , requires_grad=True)
                min_speed_ego = torch.tensor(env.ego_car.car_parameters.MAX_SPEED[1], dtype=torch.float,
                                             device=args.device)  # , requires_grad=True)
                max_speed_other = torch.tensor(env.other_car.car_parameters.MAX_SPEED[0], dtype=torch.float,
                                               device=args.device)  # , requires_grad=True)
                min_speed_other = torch.tensor(env.other_car.car_parameters.MAX_SPEED[1], dtype=torch.float,
                                               device=args.device)  # , requires_grad=True)

                # update state and check for collision
                l = C.CAR_LENGTH
                w = C.CAR_WIDTH
                collision = 0
                for t in range(int(env.time_interval / env.min_time_interval + 1)):
                    v_ego_new = max(min(max_speed_ego, action_self * t * env.min_time_interval + v_ego), min_speed_ego)
                    v_other_new = max(min(max_speed_other, action_other * t * env.min_time_interval + v_other),
                                      min_speed_other)
                    x_ego_new = x_ego - t * 0.5 * (v_ego_new + v_ego) * env.min_time_interval
                    x_other_new = x_other - t * 0.5 * (v_other_new + v_other) * env.min_time_interval
                    collision_box1 = [[x_ego_new - 0.5 * l, -0.5 * w],
                                      [x_ego_new - 0.5 * l, 0.5 * w],
                                      [x_ego_new + 0.5 * l, 0.5 * w],
                                      [x_ego_new + 0.5 * l, -0.5 * w]]
                    collision_box2 = [[0.5 * w, x_other_new - 0.5 * l],
                                      [-0.5 * w, x_other_new - 0.5 * l],
                                      [-0.5 * w, x_other_new + 0.5 * l],
                                      [0.5 * w, x_other_new + 0.5 * l]]
                    c = 0
                    polygon = Polygon(collision_box2)
                    for p in collision_box1:
                        point = Point(p[0], p[1])
                        c += polygon.contains(point)
                        if c > 0:
                            break
                    collision += float(c > 0)  # number of times steps of collision

                v_ego_new = max(min(max_speed_ego, action_self * env.time_interval + v_ego), min_speed_ego)
                x_ego_new -= 0.5 * (v_ego_new + v_ego) * env.time_interval  # start from positive distance to the center,
                # reduce to 0 when at the center
                v_ego = v_ego_new
                x_ego = x_ego_new

                v_other_new = max(min(max_speed_other, action_other * env.time_interval + v_other), min_speed_other)
                x_other_new -= 0.5 * (v_other_new + v_other) * env.time_interval
                v_other = v_other_new
                x_other = x_other_new

                # next_state = [x_ego, v_ego, x_other, v_other]
                next_state[0] = x_ego
                next_state[1] = v_ego
                next_state[2] = x_other
                next_state[3] = v_other

                next_state_other[0] = x_other
                next_state_other[1] = v_other
                next_state_other[2] = x_ego
                next_state_other[3] = v_ego

                if (x_ego <= -0.5 * C.CAR_LENGTH - 1. and x_other <= -0.5 * C.CAR_LENGTH - 1.) or env.frame >= env.max_time_steps:  # road width = 2.0 m
                    done = True

                loss_self = env.ego_car.loss(env, [x_ego, v_ego], done, collision, action_self, args)
                loss_other = env.other_car.loss(env, [x_other, v_other], done, collision, action_other, args)

                # print(loss_self, loss_other)

                loss_self = torch.tensor(int(loss_self), dtype=torch.float, device=args.device)  # , requires_grad=True)
                loss_other = torch.tensor(int(loss_other), dtype=torch.float, device=args.device)  # , requires_grad=True)

                loss_self_batch[i] = loss_self
                loss_other_batch[i] = loss_other
                next_states[i] = next_state
                next_states_other[i] = next_state_other
                transition_dict[key] = next_state
                cost1_dict[key] = loss_self
                cost2_dict[key] = loss_other
            else:
                next_states[i] = transition_dict[key]

                next_state = next_states[i]
                next_state_other = torch.tensor([next_state[2], next_state[3], next_state[0], next_state[1]], device=args.device, dtype=torch.float)
                next_states_other[i] = next_state_other

                # need to add costs also
                loss_self_batch[i] = cost1_dict[key]
                loss_other_batch[i] = cost2_dict[key]

            i += 1

        return next_states, next_states_other, loss_self_batch, loss_other_batch

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2):
        # print('Dynamics backward()')
        x, u, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_y = grad_output1.clone()
        return grad_x, grad_y, None, None

class Dynamics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u, env, args):
        ctx.save_for_backward(x, u)
        state = x  # x.detach()
        action = u
        next_state = torch.randn(4, device=args.device, dtype=torch.float, requires_grad=True)

        done = False
        if action == 0:
            a = env.parameters.MAX_DECELERATION
        elif action == 1:
            a = env.parameters.MAX_DECELERATION * 0.5
        elif action == 2:
            a = 0.0
        elif action == 3:
            a = env.parameters.MAX_ACCELERATION * 0.5
        else:
            a = env.parameters.MAX_ACCELERATION
        action_self = torch.tensor(a, dtype=torch.float, device=args.device, requires_grad=True)

        p2_state = [state[2], state[3], state[0], state[1]]
        # print(p2_state)

        dist = env.nfsp_models[env.inf_idx].act_dist(torch.FloatTensor(p2_state).to(args.device))

        dist_a = dist.tolist()[0]
        action_p2 = np.argmax(dist_a)
        if action_p2 == 0:
            a = env.parameters.MAX_DECELERATION
        elif action_p2 == 1:
            a = env.parameters.MAX_DECELERATION * 0.5
        elif action_p2 == 2:
            a = 0.0
        elif action_p2 == 3:
            a = env.parameters.MAX_ACCELERATION * 0.5
        else:
            a = env.parameters.MAX_ACCELERATION
        action_other = torch.tensor(a, dtype=torch.float, device=args.device, requires_grad=True)

        # get current states
        x_ego = x_ego_new = state[0]
        x_other = x_other_new = state[2]
        v_ego = v_ego_new = state[1]
        v_other = v_other_new = state[3]

        max_speed_ego = torch.tensor(env.ego_car.car_parameters.MAX_SPEED[0], dtype=torch.float,
                                     device=args.device)  # , requires_grad=True)
        min_speed_ego = torch.tensor(env.ego_car.car_parameters.MAX_SPEED[1], dtype=torch.float,
                                     device=args.device)  # , requires_grad=True)
        max_speed_other = torch.tensor(env.other_car.car_parameters.MAX_SPEED[0], dtype=torch.float,
                                       device=args.device)  # , requires_grad=True)
        min_speed_other = torch.tensor(env.other_car.car_parameters.MAX_SPEED[1], dtype=torch.float,
                                       device=args.device)  # , requires_grad=True)

        # update state and check for collision
        l = C.CAR_LENGTH
        w = C.CAR_WIDTH
        collision = 0
        for t in range(int(env.time_interval / env.min_time_interval + 1)):
            v_ego_new = max(min(max_speed_ego, action_self * t * env.min_time_interval + v_ego), min_speed_ego)
            v_other_new = max(min(max_speed_other, action_other * t * env.min_time_interval + v_other),
                              min_speed_other)
            x_ego_new = x_ego - t * 0.5 * (v_ego_new + v_ego) * env.min_time_interval
            x_other_new = x_other - t * 0.5 * (v_other_new + v_other) * env.min_time_interval
            collision_box1 = [[x_ego_new - 0.5 * l, -0.5 * w],
                              [x_ego_new - 0.5 * l, 0.5 * w],
                              [x_ego_new + 0.5 * l, 0.5 * w],
                              [x_ego_new + 0.5 * l, -0.5 * w]]
            collision_box2 = [[0.5 * w, x_other_new - 0.5 * l],
                              [-0.5 * w, x_other_new - 0.5 * l],
                              [-0.5 * w, x_other_new + 0.5 * l],
                              [0.5 * w, x_other_new + 0.5 * l]]
            c = 0
            polygon = Polygon(collision_box2)
            for p in collision_box1:
                point = Point(p[0], p[1])
                c += polygon.contains(point)
                if c > 0:
                    break
            collision += float(c > 0)  # number of times steps of collision

        v_ego_new = max(min(max_speed_ego, action_self * env.time_interval + v_ego), min_speed_ego)
        x_ego_new -= 0.5 * (v_ego_new + v_ego) * env.time_interval  # start from positive distance to the center,
        # reduce to 0 when at the center
        v_ego = v_ego_new
        x_ego = x_ego_new

        v_other_new = max(min(max_speed_other, action_other * env.time_interval + v_other), min_speed_other)
        x_other_new -= 0.5 * (v_other_new + v_other) * env.time_interval
        v_other = v_other_new
        x_other = x_other_new

        # next_state = [x_ego, v_ego, x_other, v_other]
        next_state[0] = x_ego
        next_state[1] = v_ego
        next_state[2] = x_other
        next_state[3] = v_other

        if (
                x_ego <= -0.5 * C.CAR_LENGTH - 1. and x_other <= -0.5 * C.CAR_LENGTH - 1.) or env.frame >= env.max_time_steps:  # road width = 2.0 m
            done = True

        loss_self = env.ego_car.loss(env, [x_ego, v_ego], done, collision, action_self, args)
        loss_other = env.other_car.loss(env, [x_other, v_other], done, collision, action_other, args)

        loss_self = torch.tensor(loss_self, dtype=torch.float, device=args.device)  # , requires_grad=True)
        loss_other = torch.tensor(loss_other, dtype=torch.float, device=args.device)  # , requires_grad=True)

        # print(next_state)
        # with torch.no_grad():
        # next_state = torch.stack((x_ego, v_ego, x_other, v_other), 0)
        # n_s = torch.tensor(next_state, dtype=float, device=args.device, requires_grad=True)
        # print(n_s)

        return next_state, loss_self, loss_other

        # return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2):
        # print('Dynamics backward()')
        x, u, = ctx.saved_tensors
        # print(x)
        # print(u)
        # print('go:{}'.format(grad_output))
        grad_x = grad_output.clone()
        grad_y = grad_output1.clone()
        # grad_u = grad_output1.clone()
        # print('dyn_grad_x:{}'.format(grad_x))
        # print('dyn_grad_y:{}'.format(grad_y))
        # grad_x[x < 0] = 0
        return grad_x, grad_y, None, None

class TotalCost(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, state, costs, args):
        # print('here')
        # print(state)
        # ctx.save_for_backward(state, costs) #, costs)
        temp = torch.tensor([costs, costs, costs, costs], device=args.device, dtype=torch.float)
        # ctx.mark_non_differentiable(costs)
        return temp

    @staticmethod
    def backward(ctx, grad_output):
        # (state, costs) = ctx.saved_tensors
        # print('go_size: {}'.format(grad_output.size()))
        # print(state.size())
        # print('state_Grad: {}'.format(state.grad))
        # print('state_is_leaf: {}'.format(state.is_leaf))
        grad_x0 = grad_output.clone()
        # grad_x0 = state.mul(grad_output.clone())
        # print('total_cost_grad_x:{}'.format(grad_x0))
        # grad_x[x < 0] = 0
        return grad_x0, None, None  # , grad_x1, grad_x2