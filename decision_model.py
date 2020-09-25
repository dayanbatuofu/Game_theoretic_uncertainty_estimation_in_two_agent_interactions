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
        elif model == 'non-empathetic':  # use estimated params of other and known param of self to choose action
            self.plan = self.non_empathetic
        elif model == 'empathetic':  # game, using NFSP, import inferred params for both agents
            self.plan = self.empathetic
            # import estimated values; use estimation of other and other's of self to get an action for both
        else:
            # placeholder for future development
            pass

        self.policy_or_Q = 'Q'

        self.true_intents = self.sim.true_params
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

    def baseline(self):
        """
        Choose action according to given intent, using the NFSP trained model
        :return:
        """
        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        action_set = self.action_set

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
                """
                return get_models()[0]  # 0: q func, 1: policy

            def q_values_pair(state_h, state_m, theta_h, theta_m):
                q_set = trained_q_function(state_h, state_m)
                # Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2,
                # TODO: consider when we have more than 1 Q pair!
                id_h = self.sim.theta_list.index(theta_h)
                id_m = self.sim.theta_list.index(theta_m)
                if id_h == 0:
                    if id_m == 0:  # TODO: IMPORTANT: CHECK WHICH ONE IS NA2 IN DECISION
                        Q_h = q_set[0]
                        Q_m = q_set[1]
                    elif id_m == 1:  # M is aggressive
                        Q_h = q_set[2]
                        Q_m = q_set[3]
                    else:
                        print("ID FOR THETA DOES NOT EXIST")
                elif id_h == 1:
                    if id_m == 0:
                        Q_h = q_set[3]
                        Q_m = q_set[2]
                    elif id_m == 1:  # TODO: IMPORTANT: CHECK WHICH ONE IS A2 IN DECISION
                        Q_h = q_set[4]
                        Q_m = q_set[5]
                    else:
                        print("ID FOR THETA DOES NOT EXIST")

                "Need state for agent H: xH, vH, xM, vM"
                state_h = [-state_h[1], abs(state_h[3]), state_m[0], abs(state_m[2])]
                state_m = [state_m[0], abs(state_m[2]), -state_h[1], abs(state_h[3])]
                #p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
                #p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))
                "Q values for each action"
                Q_vals_h = Q_h.forward(t.FloatTensor(state_h).to(t.device("cpu")))
                Q_vals_m = Q_m.forward(t.FloatTensor(state_m).to(t.device("cpu")))
                return [Q_vals_h, Q_vals_m]

            def action_prob(state_h, state_m, _lambda, theta_h, theta_m):
                """
                calculate action prob for both agents
                """
                # TODO: do we need beta_m???

                'intent has to be na_na or a_na'
                # q_vals = q_values(state_h, state_m, intent=intent)
                q_vals_pair = q_values_pair(state_h, state_m, theta_h, theta_m)
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
            beta_h, beta_m = self.true_intents
            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m
            p_actions = action_prob(p1_state, p2_state, self.lambda_list[-1], theta_h, theta_m)
            assert (not pa == 0 for pa in p_actions)
            actions = []
            for p_a in p_actions:
                p_a = np.array(p_a).tolist()
                "drawing action from action set using the distribution"
                action = random.choices(action_set, weights=p_a, k=1)
                actions.append(action[0])  # TODO: check why it's list from random??
                # id = p_a.index(max(p_a))
                # actions.append(action_set[id])

        print("action taken for baseline:", actions, "current state (y is reversed):", p1_state, p2_state)
        # actions = {"1": action1, "2": action2}
        # actions = [action1, action2]
        return {'action': actions}

    def baseline2(self):
        # TODO: not in use
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
        p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
        p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))

        args = get_args()

        lambda_list = self.lambda_list
        theta_list = self.theta_list
        if self.sim.frame == 0:
            p_beta = self.sim.agents[0].initial_belief
            beta_pair_id = np.unravel_index(p_beta.argmax(), p_beta.shape)
            beta_h = self.beta_set[beta_pair_id[0]]
            beta_m = self.beta_set[beta_pair_id[1]]
            assert beta_h[0], beta_m[0] == self.true_intents

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
        for _beta in self.true_intents:
            true_intent_id.append(self.theta_list.index(_beta[0]))
        assert theta_list[true_intent_id[1]] == self.true_intents[1][0]

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
        q1_vals = q_1.forward(t.FloatTensor(p1_state).to(t.device("cpu")))
        q2_vals = q_2.forward(t.FloatTensor(p2_state).to(t.device("cpu")))
        # TODO: what to use for lambda?? (use true beta for self
        p_action1 = self.action_prob(q1_vals, _lambda=beta_h[1])
        p_action2 = self.action_prob(q2_vals, _lambda=beta_m[1])
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

        p1_state = (-p1_state[1], abs(p1_state[3]), p2_state[0], abs(p2_state[2]))  # s_ego, v_ego, s_other, v_other
        p2_state = (p2_state[0], abs(p2_state[2]), -p1_state[1], abs(p1_state[3]))

        lambda_list = self.lambda_list
        theta_list = self.theta_list
        action_set = self.action_set
        if self.sim.frame == 0:
            p_beta = self.sim.agents[0].initial_belief
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
        q1_vals = q_1.forward(t.FloatTensor(p1_state).to(t.device("cpu")))
        q2_vals = q_2.forward(t.FloatTensor(p2_state).to(t.device("cpu")))
        p_action1 = self.action_prob(q1_vals, beta_h[1])
        p_action2 = self.action_prob(q2_vals, beta_m[1])
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
    def action_prob(self, q_vals, _lambda):
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

    # TODO: get q vals given beta set
    def get_q_vals(self):
        return

