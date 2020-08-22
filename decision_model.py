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

        self.policy_or_Q = 'policy'

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

        "sorting states to obtain action from pre-trained model"
        #y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[-1]
        p2_state = self.sim.agents[1].state[-1]
        action_set = [-8, -4, 0, 4, 8]

        args = get_args()

        if self.policy_or_Q == 'policy':
            (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2), \
            (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()

            p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
            p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))
            "action for H"
            action1 = policy_a_na.act(t.FloatTensor(p1_state).to(args.device))

            "action for M"
            action2 = policy_na_a.act(t.FloatTensor(p2_state).to(args.device))
            action1 = action_set[action1]
            action2 = action_set[action2]
            actions = [action1, action2]
        else:
            def trained_q_function(state_h, state_m):
                """
                Import Q function from nfsp given states
                :param state_h:
                :param state_m:
                :return:
                """
                q_set = get_models()[0]  # 0: q func, 1: policy
                # Q = q_set[0]  # use na_na for now

                "Q values for given state over a set of actions:"
                # Q_vals = Q.forward(torch.FloatTensor(state).to(torch.device("cpu")))
                return q_set

            def q_values_pair(state_h, state_m, intent):
                q_set = trained_q_function(state_h, state_m)
                # Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a,
                # TODO: consider when we have more than 1 Q pair!
                if intent == "na_na":
                    Q_h = q_set[0]
                    Q_m = q_set[1]
                else:  # use a_na
                    Q_h = q_set[3]
                    Q_m = q_set[2]

                "Need state for agent H: xH, vH, xM, vM"
                state_h = [-state_h[1], abs(state_h[3]), state_m[0], abs(state_m[2])]
                state_m = [state_m[0], abs(state_m[2]), -state_h[1], abs(state_h[3])]
                #p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
                #p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))
                "Q values for each action"
                Q_vals_h = Q_h.forward(t.FloatTensor(state_h).to(t.device("cpu")))
                Q_vals_m = Q_m.forward(t.FloatTensor(state_m).to(t.device("cpu")))
                return [Q_vals_h, Q_vals_m]

            def action_prob(state_h, state_m, _lambda, intent):
                """
                calculate action prob for both agents
                :param state_h:
                :param state_m:
                :param _lambda:
                :param theta:
                :return:
                """
                # TODO: do we need beta_m???

                'intent has to be na_na or a_na'
                # q_vals = q_values(state_h, state_m, intent=intent)
                q_vals_pair = q_values_pair(state_h, state_m, intent)
                q_vals_h = q_vals_pair[0]
                q_vals_m = q_vals_pair[1]

                "Q*lambda"
                # np.multiply(Q,_lambda,out = Q)
                q_vals_h = q_vals_h.detach().numpy()  # detaching tensor
                q_vals_m = q_vals_m.detach().numpy()
                q_vals_pair = [q_vals_h, q_vals_m]
                # print("q values: ",q_vals)
                exp_Q_pair = []

                for q_vals in q_vals_pair:
                    exp_Q = []
                    Q = [q * _lambda for q in q_vals]
                    # print("Q*lambda:", Q)
                    "Q*lambda/(sum(Q*lambda))"
                    # np.exp(Q, out=Q)

                    for q in Q:
                        # print(exp_Q)
                        exp_Q.append(np.exp(q))
                    # print("EXP_Q:", exp_Q)

                    "normalizing"
                    # normalize(exp_Q, norm = 'l1', copy = False)
                    exp_Q /= sum(exp_Q)
                    exp_Q_pair.append(exp_Q)

                return exp_Q_pair  # [exp_Q_h, exp_Q_m]

            "calling function for Boltzmann model"
            intent_list = ['na_na', 'a_na']
            #TODO: does it work for both agents????
            p_actions = action_prob(p1_state, p2_state, _lambda=1, intent='na_na')
            actions = []
            for p_a in p_actions:
                p_a = np.array(p_a).tolist()
                id = p_a.index(max(p_a))
                actions.append(action_set[id])

        # if self.sim.agents[0]:
        #     action = action_set[action1]
        # else:
        #     action = action_set[action2]


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
