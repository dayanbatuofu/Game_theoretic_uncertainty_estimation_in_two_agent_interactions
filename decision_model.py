"""
for obtaining action for each agent

"""
import numpy as np
# TODO pytorch version
import torch as t
import sys
import random
# TODO: organize this
sys.path.append('/models/rainbow/')
from models.rainbow.arguments import get_args
import models.rainbow.arguments
from models.rainbow.set_nfsp_models import get_models
from HJI_Vehicle.NN_output import get_Q_value
#sys.path.append('/models/intersection_simple_nfsp/')
#from arguments import get_args
from models.rainbow.common.utils import epsilon_scheduler, beta_scheduler, update_target, print_log
#from set_nfsp_models import get_models


class DecisionModel:
    def __init__(self, model, sim):
        self.sim = sim
        self.frame = self.sim.frame
        if model == 'constant_speed':
            self.plan = self.constant_speed
        elif model == 'complete_information':
            self.plan = self.complete_information
        elif model == 'nfsp_baseline':  # use with trained models
            self.plan = self.nfsp_baseline
        elif model == 'bvp_baseline':
            self.plan = self.bvp_baseline
        elif model == 'non-empathetic':  # use estimated params of other and known param of self to choose action
            self.plan = self.non_empathetic
        elif model == 'empathetic':  # game, using NFSP, import inferred params for both agents
            self.plan = self.empathetic
        elif model == 'bvp_non-empathetic':
            self.plan = self.bvp_non_empathetic
        elif model == 'bvp_empathetic':
            self.plan = self.bvp_empathetic
            # import estimated values; use estimation of other and other's of self to get an action for both
        else:
            # placeholder for future development
            print("WARNING!!! NO DECISION MODEL DETECTED")
            pass

        self.policy_or_Q = 'Q'
        self.noisy = False  # if baseline is randomly picking action based on distribution

        self.true_params = self.sim.true_params
        self.belief_params = self.sim.initial_belief
        self.action_set = self.sim.action_set
        self.action_set_combo = self.sim.action_set_combo
        self.theta_list = self.sim.theta_list
        self.lambda_list = self.sim.lambda_list
        self.beta_set = self.sim.beta_set

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

    def nfsp_baseline(self):
        """
        Choose action according to given intent, using the NFSP trained model
        :return:
        """
        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        p1_state_nn = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
        p2_state_nn = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))
        action_set = self.action_set
        lambda_list = self.lambda_list
        theta_list = self.theta_list
        args = get_args()
        (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2), \
        (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()
        if self.policy_or_Q == 'policy':
            "action for H"
            action1 = policy_a_na.act(t.FloatTensor(p1_state_nn).to(args.device))
            "action for M"
            action2 = policy_na_a.act(t.FloatTensor(p2_state_nn).to(args.device))
            action1 = action_set[action1]
            action2 = action_set[action2]
            actions = [action1, action2]
        else:
            "calling function for Boltzmann model"
            beta_h, beta_m = self.true_params
            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m
            betas = [beta_h, beta_m]
            lamb_id = []
            theta_id = []
            for b in betas:
                lamb_id.append(lambda_list.index(b[1]))
                theta_id.append(theta_list.index(b[0]))
            " the following assumes 2 thetas"
            # TODO: create a function for this
            if theta_id[0] == 0:
                if theta_id[1] == 0:
                    q_1 = Q_na_na_2
                    q_2 = Q_na_na
                elif theta_id[1] == 1:  # 1: na, 2:a
                    q_1 = Q_na_a
                    q_2 = Q_a_na
                else:
                    print("ERROR: THETA DOES NOT EXIST")
            elif theta_id[0] == 1:  # 1: A
                if theta_id[1] == 0:
                    q_1 = Q_a_na
                    q_2 = Q_na_a
                elif theta_id[1] == 1:
                    q_1 = Q_a_a_2
                    q_2 = Q_a_a
                else:
                    print("ERROR: THETA DOES NOT EXIST")
            else:
                print("ERROR: THETA DOES NOT EXIST")
            q1_vals = q_1.forward(t.FloatTensor(p1_state_nn).to(t.device("cpu")))
            q2_vals = q_2.forward(t.FloatTensor(p2_state_nn).to(t.device("cpu")))
            p_action1 = self.nfsp_action_prob(q1_vals, beta_h[1])
            p_action2 = self.nfsp_action_prob(q2_vals, beta_m[1])
            actions = []
            for p_a in (p_action1, p_action2):
                a_i = np.argmax(p_a)
                "drawing action from action set using the distribution"
                if self.noisy:
                    p_a = np.array(p_a).tolist()
                    action = random.choices(action_set, weights=p_a, k=1)
                else: # choose highest prob
                    action = action_set[a_i]
                actions.append(action[0])  # TODO: check why it's list

        # print("action taken for baseline:", actions, "current state (y is reversed):", p1_state, p2_state)
        return {'action': actions}

    def bvp_baseline(self):
        """
        Choose action according to given intent, using the BVP value approximated model
        :return:
        """
        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        p1_state_nn = (p1_state[1], p1_state[3], p2_state[0], p2_state[2])  # s_ego, v_ego, s_other, v_other
        p2_state_nn = (p2_state[0], p2_state[2], p1_state[1], p1_state[3])

        lambda_list = self.lambda_list
        theta_list = self.theta_list
        # beta_h = self.true_params[0]  # includes theta and lambda
        # beta_m = self.true_params[1]
        # beta_h_b = self.belief_params[0]
        # beta_m_b = self.belief_params[1]

        action_set = self.action_set

        "Using true param of self and other"
        true_beta_h, true_beta_m = self.true_params
        p_action1, p_action2_n = self.bvp_action_prob(p1_state, p2_state, true_beta_h, true_beta_m)
        p_action1_n, p_action2 = self.bvp_action_prob(p1_state, p2_state, true_beta_h, true_beta_m)

        actions = []
        for i, p_a in enumerate([p_action1, p_action2]):

            if self.noisy:
                # TODO: flatten p_a -> draw action id -> get action from set
                p_a = np.array(p_a).tolist()
                "drawing action from action set using the distribution"
                # TODO: need to obtain the mixed strategy array
                # ===================================
                p_a_s = []
                for pa in p_a:  # summing over rows
                    p_a_s.append(sum(pa))
                assert len(p_a_s) == len(action_set)
                # ===================================
                action = random.choices(action_set, weights=p_a_s, k=1)  # p_a needs 1D array
                actions.append(action[0])
            else:
                # TODO: get u_k-1 for the other agent
                ui = self.sim.agents[i-1].action[self.frame - 1]  # other agent's last action
                ui_i = action_set.index(ui)
                if i == 0:
                    p_a_t = np.transpose(p_a)
                    p_a_self = p_a_t[ui_i]
                elif i == 1:
                    p_a_self = p_a[ui_i]
                else:
                    print("WARNING! AGENT EXCEEDS 2 IS NOT SUPPORTED")
                action_id = np.argmax(p_a_self)
                # action_id = np.unravel_index(p_a.argmax(), p_a.shape)
                actions.append(action_set[action_id])

        # print("action taken for baseline:", actions, "current state (y is reversed):", p1_state, p2_state)
        return {'action': actions}

    def non_empathetic(self):
        """
        Get appropriate action based on predicted intent of the other agent and self intent
        :return: appropriate action for both agents
        """
        # implement reactive planning based on point estimates of future trajectories
        # TODO: import HJI BVP model
        "----------This is placeholder until we have BVP result-------------"
        (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2) = get_models()[0]

        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        p1_state_nn = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
        p2_state_nn = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))

        args = get_args()

        lambda_list = self.lambda_list
        theta_list = self.theta_list
        if self.sim.frame == 0:
            p_beta = self.sim.initial_belief
            beta_pair_id = np.unravel_index(p_beta.argmax(), p_beta.shape)
            beta_h = self.beta_set[beta_pair_id[0]]
            beta_m = self.beta_set[beta_pair_id[1]]
            assert beta_h[0], beta_m[0] == self.true_params

        else:
            p_beta, [beta_h, beta_m] = self.sim.agents[1].predicted_intent_all[-1]

        action_set = self.action_set

        betas = [beta_h, beta_m]
        lamb_id = []
        theta_id = []
        for b in betas:
            lamb_id.append(lambda_list.index(b[1]))
            theta_id.append(theta_list.index(b[0]))
        "getting true intent id"
        true_intent_id = []
        for _beta in self.true_params:
            true_intent_id.append(self.theta_list.index(_beta[0]))
        assert theta_list[true_intent_id[1]] == self.true_params[1][0]

        "the following assumes 2 thetas"
        # TODO: create a function for this
        "for agent 1 (H): using true self intent and guess of other's intent"
        if true_intent_id[0] == 0:
            if theta_id[1] == 0:
                q_1 = Q_na_na_2
            elif theta_id[1] == 1:  # 1: na, 2:a
                q_1 = Q_na_a
            else:
                print("ERROR: THETA DOES NOT EXIST")
        elif true_intent_id[0] == 1:  # 1: A
            if theta_id[1] == 0:
                q_1 = Q_a_na
            elif theta_id[1] == 1:
                q_1 = Q_a_a_2
            else:
                print("ERROR: THETA DOES NOT EXIST")
        else:
            print("ERROR: THETA DOES NOT EXIST")
        "for agent 2 (M)"
        if true_intent_id[1] == 0:
            if theta_id[0] == 0:
                q_2 = Q_na_na
            elif theta_id[0] == 1:  # 2: na, 1:a
                q_2 = Q_a_na
            else:
                print("ERROR: THETA DOES NOT EXIST")
        elif true_intent_id[1] == 1:  # 2: A
            if theta_id[0] == 0:
                q_2 = Q_na_a
            elif theta_id[0] == 1:
                q_2 = Q_a_a
            else:
                print("ERROR: THETA DOES NOT EXIST")
        else:
            print("ERROR: THETA DOES NOT EXIST")
        q1_vals = q_1.forward(t.FloatTensor(p1_state_nn).to(t.device("cpu")))
        q2_vals = q_2.forward(t.FloatTensor(p2_state_nn).to(t.device("cpu")))
        # TODO: what to use for lambda?? (use true beta for self
        p_action1 = self.nfsp_action_prob(q1_vals, _lambda=beta_h[1])
        p_action2 = self.nfsp_action_prob(q2_vals, _lambda=beta_m[1])
        actions = []
        for p_a in (p_action1, p_action2):
            p_a = np.array(p_a).tolist()
            "drawing action from action set using the distribution"
            action = random.choices(action_set, weights=p_a, k=1)
            actions.append(action[0])  # TODO: check why it's list
        self.sim.action_distri_1.append(p_action1)
        self.sim.action_distri_2.append(p_action2)
        # print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)
        # actions = [action1, action2]
        return {'action': actions}

    def empathetic(self):
        """
        Choose action from Nash Equilibrium, according to the inference model
        :return: actions for both agents
        """
        # implement reactive planning based on inference of future trajectories
        # TODO: import HJI BVP model
        "----------This is placeholder until we have BVP result-------------"
        (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2), \
        (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()

        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]

        p1_state_nn = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
        p2_state_nn = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))

        lambda_list = self.lambda_list
        theta_list = self.theta_list
        action_set = self.action_set
        if self.sim.frame == 0:
            p_beta = self.sim.initial_belief
            beta_pair_id = np.unravel_index(p_beta.argmax(), p_beta.shape)
            beta_h = self.beta_set[beta_pair_id[0]]
            beta_m = self.beta_set[beta_pair_id[1]]

        else:
            beta_h, beta_m  = self.sim.agents[1].predicted_intent_all[-1][1]


        args = get_args()
        betas = [beta_h, beta_m]
        lamb_id =[]
        theta_id = []
        for b in betas:
            lamb_id.append(lambda_list.index(b[1]))
            theta_id.append(theta_list.index(b[0]))
        " the following assumes 2 thetas"
        # TODO: create a function for this
        if theta_id[0] == 0:
            if theta_id[1] == 0:
                q_1 = Q_na_na_2
                q_2 = Q_na_na
            elif theta_id[1] == 1:  # 1: na, 2:a
                q_1 = Q_na_a
                q_2 = Q_a_na
            else:
                print("ERROR: THETA DOES NOT EXIST")

        elif theta_id[0] == 1:  # 1: A
            if theta_id[1] == 0:
                q_1 = Q_a_na
                q_2 = Q_na_a
            elif theta_id[1] == 1:
                q_1 = Q_a_a_2
                q_2 = Q_a_a
            else:
                print("ERROR: THETA DOES NOT EXIST")
        else:
            print("ERROR: THETA DOES NOT EXIST")
        q1_vals = q_1.forward(t.FloatTensor(p1_state_nn).to(t.device("cpu")))
        q2_vals = q_2.forward(t.FloatTensor(p2_state_nn).to(t.device("cpu")))
        p_action1 = self.nfsp_action_prob(q1_vals, beta_h[1])
        p_action2 = self.nfsp_action_prob(q2_vals, beta_m[1])
        actions = []
        for p_a in (p_action1, p_action2):
            p_a = np.array(p_a).tolist()
            "drawing action from action set using the distribution"
            action = random.choices(action_set, weights=p_a, k=1)
            actions.append(action[0])  # TODO: check why it's list
        self.sim.action_distri_1.append(p_action1)
        self.sim.action_distri_2.append(p_action2)
        # print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)

        return {'action': actions}

    def bvp_non_empathetic(self):
        """
        Get appropriate action based on predicted intent of the other agent and self intent
        :return: appropriate action for both agents
        """
        # implement reactive planning based on point estimates of future trajectories
        # TODO: import HJI BVP model

        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        p1_state_nn = (p1_state[1], p1_state[3], p2_state[0], p2_state[2])  # s_ego, v_ego, s_other, v_other
        p2_state_nn = (p2_state[0], p2_state[2], p1_state[1], p1_state[3])

        lambda_list = self.lambda_list
        theta_list = self.theta_list
        if self.sim.frame == 0:
            p_beta = self.sim.initial_belief
            beta_pair_id = np.unravel_index(p_beta.argmax(), p_beta.shape)
            beta_h = self.beta_set[beta_pair_id[0]]
            beta_m = self.beta_set[beta_pair_id[1]]
            assert beta_h[0], beta_m[0] == self.true_params
        else:
            p_beta, [beta_h, beta_m] = self.sim.agents[1].predicted_intent_all[-1]

        action_set = self.action_set

        # TODO: this is placeholder; probably not right to do this
        "this is where non_empathetic is different: using true param of self"
        true_beta_h, true_beta_m = self.true_params
        p_action1, p_action2_n = self.bvp_action_prob(p1_state, p2_state, true_beta_h, beta_m)
        p_action1_n, p_action2 = self.bvp_action_prob(p1_state, p2_state, beta_h, true_beta_m)

        actions = []
        for p_a in (p_action1, p_action2):
            # TODO: flatten p_a -> draw action id -> get action from set
            p_a = np.array(p_a).tolist()
            "drawing action from action set using the distribution"
            # TODO: need to obtain the mixed strategy array
            # ===================================
            p_a_s = []
            for pa in p_a:
                p_a_s.append(sum(pa))
            assert len(p_a_s) == len(action_set)
            # ===================================
            action = random.choices(action_set, weights=p_a_s, k=1)  # p_a needs 1D array
            actions.append(action[0])  # TODO: check why it's list
        self.sim.action_distri_1.append(p_action1)
        self.sim.action_distri_2.append(p_action2)
        # print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)
        # actions = [action1, action2]
        return {'action': actions}

    def bvp_empathetic(self):
        """
        Choose action from Nash Equilibrium, according to the inference model
        :return: actions for both agents
        """
        # implement reactive planning based on inference of future trajectories
        # TODO: import HJI BVP model

        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        p1_state_nn = (p1_state[1], p1_state[3], p2_state[0], p2_state[2])  # s_ego, v_ego, s_other, v_other
        p2_state_nn = ([p2_state[0]], [p2_state[2]], [p1_state[1]], [p1_state[3]])

        lambda_list = self.lambda_list
        theta_list = self.theta_list
        if self.sim.frame == 0:
            p_beta = self.sim.initial_belief
            beta_pair_id = np.unravel_index(p_beta.argmax(), p_beta.shape)
            beta_h = self.beta_set[beta_pair_id[0]]
            beta_m = self.beta_set[beta_pair_id[1]]
            assert beta_h[0], beta_m[0] == self.true_params
        else:
            p_beta, [beta_h, beta_m] = self.sim.agents[1].predicted_intent_all[-1]

        action_set = self.action_set

        # TODO: this is placeholder; probably not right to do this
        "this is where empathetic is different: using predicted param of self"
        true_beta_h, true_beta_m = self.true_params
        p_action1, p_action2 = self.bvp_action_prob(p1_state, p2_state, beta_h, beta_m)

        actions = []
        for p_a in (p_action1, p_action2):
            # TODO: flatten p_a -> draw action id -> get action from set
            p_a = np.array(p_a).tolist()
            "drawing action from action set using the distribution"
            # TODO: need to obtain the mixed strategy array
            # ===================================
            p_a_s = []
            for pa in p_a:
                p_a_s.append(sum(pa))
            assert len(p_a_s) == len(action_set)
            # ===================================
            action = random.choices(action_set, weights=p_a_s, k=1)
            actions.append(action[0])  # TODO: check why it's list
        self.sim.action_distri_1.append(p_action1)
        self.sim.action_distri_2.append(p_action2)
        # print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)
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

    "-------------Utilities:---------------"
    def nfsp_action_prob(self, q_vals, _lambda):
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
            exp_q = np.exp(q)
            assert exp_q != 0
            exp_Q.append(np.exp(q))

        "normalizing"

        exp_Q /= sum(exp_Q)
        # TODO: do the assertion work???
        assert (not pa == 0 for pa in exp_Q)
        assert (np.isnan(pa) for pa in exp_Q)
        print("exp_Q:", exp_Q)
        return exp_Q

    def bvp_action_prob(self, state_h, state_m, beta_h, beta_m):
        """
        Equation 1
        calculate action prob for both agents
        :param state_h:
        :param state_m:
        :param _lambda:
        :param theta:
        :return: [p_action_H, p_action_M], where p_action = [p_a1, ..., p_a5]
        """

        theta_h, lambda_h = beta_h
        theta_m, lambda_m = beta_m
        action_set = self.action_set

        _lambda = [lambda_h, lambda_m]

        "Need state for agent H: xH, vH, xM, vM"  # TODO: check if this is right
        p1_state_nn = np.array([[state_h[1]], [state_h[3]], [state_m[0]], [state_m[2]]])
        p2_state_nn = np.array([[state_m[0]], [state_m[2]], [state_h[1]], [state_h[3]]])

        # TODO: math needs to be checked
        _p_action_1 = np.zeros((len(action_set), len(action_set)))
        _p_action_2 = np.zeros((len(action_set), len(action_set)))
        for i, p_a_h in enumerate(_p_action_1):
            for j, p_a_m in enumerate(_p_action_1[i]):
                q1, q2 = get_Q_value(p1_state_nn, np.array([[action_set[i]], [action_set[j]]]), (1, 1))  # TODO: theta is not considered yet! all will get the same theta
                lamb_Q1 = q1 * lambda_h
                _p_action_1[i][j] = np.exp(lamb_Q1)
                lamb_Q2 = q2 * lambda_m
                _p_action_2[i][j] = np.exp(lamb_Q2)

        "normalizing"  # TODO: check if this works
        _p_action_1 /= np.sum(_p_action_1)
        _p_action_2 /= np.sum(_p_action_2)
        print('p1 state:', p1_state_nn)
        print("action prob 1 from bvp:", _p_action_1)
        print("action prob 2 from bvp:", _p_action_2)
        assert round(np.sum(_p_action_1)) == 1
        assert round(np.sum(_p_action_2)) == 1


        return [_p_action_1, _p_action_2]  # [exp_Q_h, exp_Q_m]

    # TODO: get q vals given beta set
    def get_q_vals(self):
        return



