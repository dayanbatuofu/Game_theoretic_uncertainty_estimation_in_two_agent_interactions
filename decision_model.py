"""
for obtaining action for each agent

"""
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
        # TODO: check the info imported from inference is the right frame!
        # assert self.sim.frame == len(self.sim.agents[1].predicted_intent_other) - 1
        if model == 'constant_speed':
            self.plan = self.constant_speed
        elif model == 'complete_information':
            self.plan = self.complete_information
        elif model == 'baseline':  # use with trained models
            self.plan = self.baseline
        elif model == 'baseline2':  # doesn't do anything different yet!
            self.plan = self.baseline2
        elif model == 'reactive_point':  # non-game, using NFSP, import estimated params to choose action
            self.plan = self.reactive_point
            # import estimated values; use estimation of other's param to get an action for self
            # self.H_intent = self.sim.agents[1].predicted_intent_other
            # self.H_action = self.sim.agents[1].predicted_actions_other
        elif model == 'reactive_uncertainty':  # game, using NFSP, import inferred params for both agents
            self.plan = self.reactive_uncertainty
            # import estimated values; use estimation of other and self params to get an action for both
            # self.H_intent = self.sim.agents[1].predicted_intent_other
            # self.H_action = self.sim.agents[1].predicted_actions_other
            # self.predicted_action_m = self.sim.agents[1].predicted_actions_self
        else:
            # placeholder for future development
            pass

        self.policy_or_Q = 'Q'

        self.true_intents = []
        for i, par_i in enumerate(self.sim.env.car_par):
            self.true_intents.append(par_i["par"])

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
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        action_set = [-8, -4, 0, 4, 8]

        args = get_args()

        if self.policy_or_Q == 'policy':
            (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2), \
            (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()

            "re-organizing for NFSP definition"
            _p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
            _p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))
            "action for H"
            action1 = policy_a_na.act(t.FloatTensor(_p1_state).to(args.device))

            "action for M"
            action2 = policy_na_a.act(t.FloatTensor(_p2_state).to(args.device))
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
                # Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2,
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
            p_actions = action_prob(p1_state, p2_state, _lambda=self.sim.lambda_list[-1], intent='a_na')
            # TODO: DRAW action instead of pulling highest mass
            assert (not pa == 0 for pa in p_actions)
            actions = []
            for p_a in p_actions:
                p_a = np.array(p_a).tolist()
                id = p_a.index(max(p_a))
                actions.append(action_set[id])


        print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)
        #actions = {"1": action1, "2": action2}
        #actions = [action1, action2]
        return {'action': actions}

    def baseline2(self):
        """
        This is for H to act according to the models (Policy)
        :return:
        """
        # randomly pick one of the nash equilibrial policy

        # TODO: import args, env here
        (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2), \
        (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()

        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[-1]
        p2_state = self.sim.agents[1].state[-1]

        p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
        p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))
        # state_h = [-state_h[1], abs(state_h[3]), state_m[0], abs(state_m[2])]
        # state_m = [state_m[0], abs(state_m[2]), -state_h[1], abs(state_h[3])]
        args = get_args()

        def action_prob(q_vals, _lambda):
            """
            Equation 1
            Noisy-rational model
            calculates probability distribution of action given hardmax Q values
            Uses:
            1. Softmax algorithm
            2. Q-value given state and theta(intent)
            3. lambda: "rationality coefficient"
            => P(uH|xH;beta,theta) = exp(beta*QH(xH,uH;theta))/sum_u_tilde[exp(beta*QH(xH,u_tilde;theta))]
            :return: Normalized probability distributions of available actions at a given state and lambda
            """
            # q_vals = q_values(state_h, state_m, intent=intent)
            exp_Q = []
            "Q*lambda"
            q_vals = q_vals.detach().numpy()  # detaching tensor
            Q = [q * _lambda for q in q_vals]
            "Q*lambda/(sum(Q*lambda))"

            for q in Q:
                exp_Q.append(np.exp(q))

            "normalizing"
            exp_Q /= sum(exp_Q)
            # print("exp_Q normalized:", exp_Q)
            return exp_Q

        "action for H"
        q_h = Q_a_na  # TODO: GROUND TRUTH
        q_vals_h = q_h.forward(t.FloatTensor(p1_state).to(t.device("cpu")))
        lambda_h = self.sim.lambda_list[-1]  # the most rational coefficient
        p_action_h = action_prob(q_vals_h, lambda_h)
        action1 = np.argmax(p_action_h)
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
        """
        Get appropriate action based on predicted intent of the other agent (H)
        :return:
        """
        # implement reactive planning based on point estimates of future trajectories
        # TODO: import HJI BVP model
        "----------This is placeholder until we have BVP result-------------"
        (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2)= get_models()[0]

        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
        p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))

        args = get_args()

        def action_prob(q_vals, _lambda):
            """
            Equation 1
            Noisy-rational model
            calculates probability distribution of action given hardmax Q values
            Uses:
            1. Softmax algorithm
            2. Q-value given state and theta(intent)
            3. lambda: "rationality coefficient"
            => P(uH|xH;beta,theta) = exp(beta*QH(xH,uH;theta))/sum_u_tilde[exp(beta*QH(xH,u_tilde;theta))]
            :return: Normalized probability distributions of available actions at a given state and lambda
            """
            #q_vals = q_values(state_h, state_m, intent=intent)
            exp_Q = []
            "Q*lambda"
            q_vals = q_vals.detach().numpy() #detaching tensor
            Q = [q * _lambda for q in q_vals]
            "Q*lambda/(sum(Q*lambda))"

            for q in Q:
                exp_Q.append(np.exp(q))

            "normalizing"
            exp_Q /= sum(exp_Q)
            #print("exp_Q normalized:", exp_Q)
            assert len(exp_Q) == len(self.sim.action_set)
            return exp_Q

        "action for H"
        q_h = Q_a_na  # TODO: GROUND TRUTH
        q_vals_h = q_h.forward(t.FloatTensor(p1_state).to(t.device("cpu")))
        lambda_h = self.sim.lambda_list[-1]  # the most rational coefficient
        p_action_h = action_prob(q_vals_h, lambda_h)
        action1 = np.argmax(p_action_h)
        # action1 = policy_a_na.act(t.FloatTensor(p1_state).to(args.device))

        "action for M: we know our intent, get best response to H's intent"
        theta_list = self.sim.theta_list
        lambda_m = self.sim.lambda_list[-1]  # the most rational coefficient
        p_joint_h = self.sim.agents[1].predicted_intent_other[-1][0]
        p_theta = np.zeros(len(theta_list))
        for i, p_t in enumerate(p_joint_h.transpose()):  # get marginal prob of theta: p(theta) from joint prob p(lambda, theta)
            p_theta[i] = sum(p_t)
        h_intent = theta_list[np.argmax(p_theta)]
        if h_intent == theta_list[0]:  # NA
            if self.true_intents[1] == theta_list[0]:
                q_m = Q_na_na_2  # TODO: check which Q_na_na
            else:
                q_m = Q_a_na
        else:  # A
            if self.true_intents[1] == theta_list[0]:
                q_m = Q_na_a
            else:
                q_m = Q_a_a_2  # TODO: check which Q_a_a
        q_vals_m = q_m.forward(t.FloatTensor(p2_state).to(t.device("cpu")))
        p_action_m = action_prob(q_vals_m, lambda_m)
        # TODO: DRAW action instead of pulling highest mass
        action2 = np.argmax(p_action_m)

        action_set = self.sim.action_set
        # if self.sim.agents[0]:
        #     action = action_set[action1]
        # else:
        #     action = action_set[action2]
        action1 = action_set[action1]
        action2 = action_set[action2]
        actions = [action1, action2]
        # print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)
        # actions = {"1": action1, "2": action2}
        # actions = [action1, action2]
        return {'action': actions}

    def reactive_uncertainty(self):
        # implement reactive planning based on inference of future trajectories
        # TODO: import HJI BVP model
        "----------This is placeholder until we have BVP result-------------"
        (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2), \
        (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()

        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[-1]
        p2_state = self.sim.agents[1].state[-1]

        p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
        p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))

        args = get_args()

        def action_prob(q_vals, _lambda):
            """
            Equation 1
            Noisy-rational model
            calculates probability distribution of action given hardmax Q values
            Uses:
            1. Softmax algorithm
            2. Q-value given state and theta(intent)
            3. lambda: "rationality coefficient"
            => P(uH|xH;beta,theta) = exp(beta*QH(xH,uH;theta))/sum_u_tilde[exp(beta*QH(xH,u_tilde;theta))]
            :return: Normalized probability distributions of available actions at a given state and lambda
            """
            # q_vals = q_values(state_h, state_m, intent=intent)
            exp_Q = []
            "Q*lambda"
            q_vals = q_vals.detach().numpy()  # detaching tensor
            Q = [q * _lambda for q in q_vals]
            "Q*lambda/(sum(Q*lambda))"

            for q in Q:
                exp_Q.append(np.exp(q))

            "normalizing"
            exp_Q /= sum(exp_Q)
            print("exp_Q normalized:", exp_Q)
            return exp_Q

        # TODO: get action from NE instead of separate marginal p_beta_i
        "action for H"
        q_h = Q_a_na  # TODO: GROUND TRUTH
        q_vals_h = q_h.forward(t.FloatTensor(p1_state).to(t.device("cpu")))
        lambda_h = self.sim.lambda_list[-1]  # the most rational coefficient
        p_action_h = action_prob(q_vals_h, lambda_h)
        # TODO: DRAW action instead of pulling highest mass
        action1 = np.argmax(p_action_h)
        # action1 = policy_a_na.act(t.FloatTensor(p1_state).to(args.device))

        "action for M: choose action based on the equilibrium intent set"
        # TODO: DRAW action instead of pulling highest mass
        action2 = self.sim.agents[1].predicted_actions_self[-1]

        action_set = [-8, -4, 0, 4, 8]
        # if self.sim.agents[0]:
        #     action = action_set[action1]
        # else:
        #     action = action_set[action2]
        action1 = action_set[action1]
        #action2 = action_set[action2]
        actions = [action1, action2]
        # print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)
        # actions = {"1": action1, "2": action2}
        # actions = [action1, action2]
        return {'action': actions}

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
