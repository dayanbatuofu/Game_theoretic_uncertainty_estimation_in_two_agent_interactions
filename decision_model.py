import numpy as np
# TODO pytorch version
import torch as t

#TODO: organize imports
import sys
sys.path.append('/models/rainbow/')
from models.rainbow.arguments import get_args
import models.rainbow.arguments
from models.rainbow.set_nfsp_models import get_models
#sys.path.append('/models/intersection_simple_nfsp/')
#from arguments import get_args
from models.rainbow.common.utils import epsilon_scheduler, beta_scheduler, update_target, print_log
#from set_nfsp_models import get_models

class DecisionModel:
    def __init__(self, model, sim):
        self.sim = sim
        if model == 'constant_speed':
            self.plan = self.constant_speed
        elif model == 'complete_information':
            self.plan = self.complete_information
        elif model == 'baseline':
            self.plan = self.baseline
        elif model == 'baseline2':
            self.plan = self.baseline2
        elif model == 'reactive_point':
            self.plan = self.reactive_point
        elif model == 'reactive_uncertainty':
            self.plan = self.reactive_uncertainty
        else:
            # placeholder for future development
            pass

    @staticmethod
    def constant_speed():
        return {'action': 0}  # just keep the speed

    def complete_information(self, *args):
        # TODO: generalize for n agents

        # find nash equilibrial policies when intents (t1, t2) are known
        states, actions, intents = args  # state of agent 1, 2; latest actions 1, 2; intent of agent 1, 2

        loss = [self.create_long_term_loss(states, actions, intents[i]) for i in range(len(intents))]
        # iterate to find nash equilibrium
        # TODO: check convergence
        learning_rate = self.sim.par.learning_rate_planning
        optimizers = [t.optim.Adam(actions[i], lr=learning_rate) for i in range(len(intents))]
        for j in range(self.sim.par.max_iter_planning):
            for i in range(len(intents)):
                optimizers[i].zero_grad()
                loss[i].backward()
                optimizers[i].step()

    def baseline(self):
        # randomly pick one of the nash equilibrial policy

        #TODO: import args, env here
        (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2), \
        (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()

        "sorting states to obtain action from pre-trained model"
        #y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[-1]
        p2_state = self.sim.agents[1].state[-1]

        p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))#s_ego, v_ego, s_other, v_other
        p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))

        args = get_args()
        "action for H"
        action1 = policy_a_na.act(t.FloatTensor(p1_state).to(args.device))

        "action for M"
        action2 = policy_na_a.act(t.FloatTensor(p2_state).to(args.device))

        action_set = [-8, -4, 0, 4, 8]
        # if self.sim.agents[0]:
        #     action = action_set[action1]
        # else:
        #     action = action_set[action2]
        action1 = action_set[action1]
        action2 = action_set[action2]
        actions = [action1, action2]
        print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)
        #actions = {"1": action1, "2": action2}
        #actions = [action1, action2]
        return {'action': actions}
    def baseline2(self):
        # randomly pick one of the nash equilibrial policy

        # TODO: import args, env here
        (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2), \
        (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()

        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[-1]
        p2_state = self.sim.agents[1].state[-1]

        p1_state = (-p1_state[1], -p1_state[3], p2_state[0], p2_state[2])  # s_ego, v_ego, s_other, v_other
        p2_state = (p2_state[0], p2_state[2], -p1_state[1], -p1_state[3])

        args = get_args()
        "action for H"
        action1 = policy_a_na.act(t.FloatTensor(p1_state).to(args.device))

        "action for M"
        action2 = policy_na_a.act(t.FloatTensor(p2_state).to(args.device))

        action_set = [-8, -4, 0, 4, 8]
        # if self.sim.agents[0]:
        #     action = action_set[action1]
        # else:
        #     action = action_set[action2]
        action1 = action_set[action1]
        action2 = action_set[action2]
        actions = [action1, action2]
        print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)
        # actions = {"1": action1, "2": action2}
        # actions = [action1, action2]
        return {'action': actions}

    def reactive_point(self):
        # implement reactive planning based on point estimates of future trajectories
        pass

    def reactive_uncertainty(self):
        # implement reactive planning based on inference of future trajectories
        pass

    # create long term loss as a pytorch object
    def create_long_term_loss(self, states, action1, action2, intent):
        # define instantaneous loss
        def loss(x, u1, u2, i):

            pass

        steps = self.sim.duration - self.sim.env.frame  # time left
        l = t.tensor(0)
        for i in range(steps):
            # compute instantaneous loss
            l = l + loss(states, action1[i], action2[i], intent)

            # update state
            states = self.sim.update(states, [action1[i], action2[i]])

        return l
