"""
Perform inference on other agent
"""

import numpy as np
#from sklearn.processing import normalize
# TODO pytorch version
from autonomous_vehicle import AutonomousVehicle
import discrete_sets as sets

#class Lambdas(FloatEnums):

class InferenceModel:
    def __init__(self, model, sim): #model = inference type, sim = simulation class
        if model == 'none':
            self.infer = self.no_inference
        elif model == 'baseline':
            self.infer = self.baseline_inference
        elif model == 'empathetic':
            self.infer = self.empathetic_inference
        else:
            # placeholder for future development
            pass

        "---simulation info: only the static variables!---"
        self.sim = sim
        self.agents = sim.agents
        self.n_agents = sim.n_agents
        self.frame = sim.frame
        self.T = 1  # one step look ahead/ Time Horizon
        self.dt = sim.dt #default is 1s: assigned in main.py
        self.car_par = sim.env.car_par

        self.past_scores = {} #score for each lambda #TODO: record past scores for time/frame k-1
        # for i in range(self.n_agents):
        #     if i == 1:
        #         self.h_car_par = sim.env.car_par[i]
        #     if i == 2:
        #         self.m_car_par = sim.env.car_par[i]
        self.theta_priors = None
        self.action_set = [-1, -0.2, 0.0, 0.1, 0.5]  # accelerations (m/s^2) #TODO: give appropriate actions
        #safe deceleration: 15ft/s
        
        "---Agent info---"
        # "importing initial agents information"
        # self.initial_state = [self.car_par[0]["initial_state"], self.car_par[1]["initial_state"]]
        # if self.frame == 0: #no agent info yet
        #     self.actions = [self.car_par[0]["initial_action"],self.car_par[1]["initial_action"]]
        #     self.curr_state = [self.car_par[0]["initial_state"], self.car_par[1]["initial_state"]]
        #     self.traj_h = [self.curr_state[0], self.actions[0]]
        #     self.traj_m = [self.curr_state[1], self.actions[1]]
        #     self.traj = [[self.curr_state[0], self.curr_state[1]],[self.actions[0], self.actions[1]]]

        # else:
        #     "importing agents information from Autonomous Vehicle (sim.agents)"
        #     for i in range(self.n_agents):
        #         if i == 0:
        #             self.curr_state_h = self.sim.agents[i].state[-1]
        #         if i == 1:
        #             self.curr_state_m = self.sim.agents[i].state[-1]
        #     self.curr_state = self.sim.agents.state[-1]
        #

        #     self.actions = [-2, -0.5, 0, 0.5, 2]  # accelerations (m/s^2)
        #
        #     "trajectory: need to process and combine past states and actions together"
        #     self.states_h = self.sim.agents[0].state
        #     self.states_m = self.sim.agents[1].state
        #     self.past_actions_h = sim.agents[0].action
        #     self.past_actions_m = sim.agents[1].action
        #     self.traj = []
        #     self.traj_h = []
        #     for i in range(self.frame):
        #         self.traj.append([[self.states_h[i],self.states_m[i]],
        #                           [self.past_actions_h[i], self.past_actions_m[i]]])
        #         self.traj_h.append([self.states_h[i], self.past_actions_h[i]])
        # # self.traj = AutonomousVehicle.planned_trajectory_set
        "---end of agent info---"

        "---goal states---"
        self.goal = [self.car_par[0]["desired_state"], self.car_par[1]["desired_state"]]

        "---parameters(theta and lambda)---"
        self.lambdas = [0.01, 0.1, 1, 10]  # range?
        self.thetas = [1, 1000]  # range?

        #"--------------------------------------------------------"
        "Some reference variables from pedestrian prediction"
        #self.q_cache = {}
        #"defining reward for (s, a) pair"
        #self.default_reward = -1
        #self.rewards = np.zeros[self.agents.s, self.agents.a] #Needs fixing
        #self.rewards.fill(self.default_reward)
        #"--------------------------------------------------------"

    #@staticmethod
    def no_inference(self, agents, sim):
        #pass
        print("frame {}".format(sim.frame))
        return

    #@staticmethod
    def baseline_inference(self, agents, sim):
        # TODO: implement Fridovich-Keil et al. "Confidence-aware motion prediction for real-time collision avoidance"
        """
        for each agent, estimate its par (agent.par) based on its last action (agent.action[-1]),
        system state (agent.state[-1] for all agent), and prior dist. of par
        :return: #TODO: check what to return

        Important equations implemented here:
        - Equation 1 (action_probabilities):
        P(u|x,theta,lambda) = exp(Q*lambda)/sum(exp(Q*lambda)), Q size = action space at state x
        
        - Equation 2 (belief_update):
         #Pseudo code for intent inference to obtain P(lambda, theta) based on past action sequence D(k-1):
        #P(lambda, theta|D(k)) = P(u| x;lambda, theta)P(lambda, theta | D(k-1)) / sum[ P(u|x;lambda', theta') P(lambda', theta' | D(k-1))]
        #equivalent: P = action_prob * P(k-1)/{sum_(lambda,theta)[action_prob * P(k-1)]}
        
        - Equation 3 (belief_resample):
        #resampled_prior = (1 - epsilon)*prior + epsilon * initial_belief
       
        """

        "importing agents information from Autonomous Vehicle (sim.agents)"
        curr_state_h = sim.agents[0].state[-1]

        "trajectory: need to process and combine past states and actions together"
        states_h = sim.agents[0].state
        past_actions_h = sim.agents[0].action
        traj_h = []

        for i in range(sim.frame):
            traj_h.append([states_h[i], past_actions_h[i]])
        if sim.frame == 0:
            traj_h.append([states_h[0], past_actions_h[0]])
        "---end of agent info---"
        self.frame = sim.frame #updating curr frame


        def q_function(current_s, action, goal_s, dt):
            """
            Calculates Q value for a state action pair (s, a)

            Current Implementation:
                Q is negatively proportional to the time it takes to reach the goal

            Params:
                current_s [tuple?] -- Current state containing x-state, y-state,
                    x-velocity, y-velocity
                action [IntEnum] -- potential action taken by agent H
                goal_s [tuple?] -- Goal state, with the format (sx, sy)
                dt[int / float] -- discrete time interval
            Returns:
                Q: an value correspond to action a in state s
            """
            # Q = -(s_goal - s_current)/v_next #estimates time required to reach goal with current state and chosen action
            u = action
            delta = 1  # to prevent denominator from becoming zero
            sx, sy, vx, vy = current_s[0], current_s[1], current_s[2], current_s[3]
            # TODO: implement a variable for filtering which agent we are looking at (agent = H or M) to choose goal!
            if sx == 0 and vx == 0:
                # Q = FUNCTION MATH HERE USING SY, VY
                print("Y direction movement detected")
                goal_s = goal_s[0]
                next_v = vy + action * dt
                "Deceleration only leads to 0 velocity!"
                if next_v < 0.0:
                    # let v_next = 0
                    Q = -(goal_s - sy) / delta
                else:
                    Q = -(goal_s - sy) / (vy + action * dt + delta)
            elif sy == 0 and vy == 0:
                # Q = FUNCTION MATH HERE USING SX, VX
                print("X direction movement detected")
                goal_s = goal_s[1]
                next_v = vx + action * dt
                "Deceleration only leads to 0 velocity!"
                if next_v < 0.0:
                    # let v_next = 0
                    Q = -(goal_s - sx) / delta
                else:
                    Q = -(goal_s - sx) / (vx + action * dt + delta)
            else:
                # Q = FUNCTION FOR 2D MODELS
                goal_x = goal_s[0]
                goal_y = goal_s[1]
                next_vx = vx + action * dt
                next_vy = vy + action * dt

                "Deceleration only leads to 0 velocity!"
                if next_vx < 0:
                    if next_vy < 0:  # both are negative
                        Q = -((goal_y - sy) / delta + (goal_x - sx) / delta)
                    else:  # only vx is negative
                        Q = -((goal_y - sy) / (vy + action * dt + delta) + (goal_x - sx) / delta)
                elif next_vy < 0:  # only vy is negative
                    Q = -((goal_y - sy) / delta + (goal_x - sx) / (vx + action * dt + delta))
                else:  # both are non negative
                    Q = -((goal_y - sy) / (vy + action * dt + delta) + (goal_x - sx) / (vx + action * dt + delta))
                # TODO: add a cieling for how fast they can go
            return Q

        def q_values(state, goal):
            # TODO documentation for function
            """
            Calls q_function and return a list of q values corresponding to an action set at a given state
            :param state:
            :param goal:
            return:
                A 1D list of values for a given state s with action set A


            """
            print("q_values function is being called,{0}, {1}".format(state, goal))
            # current_s = states[-1]
            # Q = {} #dict type
            Q = []  # list type
            actions = self.action_set  # TODO: check that actions are imported in init
            for a in actions:  # sets: file for defining sets
                # Q[a] = q_function(state, a, goal, self.dt)  #dict type
                Q.append(q_function(state, a, goal, self.dt))  # list type

            return Q

            ##from PP code::   CHANGE PARAMETER NAMES
            # Inferring agent's current param based on its last action and state in last time step
            # """
            # s_last = self.agents.state[-1]
            # a_last = self.agents.action[-1]
            # x_last = s_last[x]
            # v_last = s_last[v]
            # Q = -v_last*dt - np.abs(x_last + v_last*dt - goal) #2D version of Q value from confidence aware paper
            # """

        def get_state_list(state, T):
            """
            2D case: calculate an array of state (T x S at depth T)
            1D case: calculate a list of state (1 X (1 + Action_set^T))
            :param
                state: any state
                T: time horizon / look ahead
            :return:
                list of resulting states from taking each action at a given state
            """
            # TODO: Check this modification so that action probability is calculated for states within a time horizon
            # ---Code: append all states reachable within time T----
            # states = self.states  # TODO: import a state to start from
            actions = self.action_set
            # T = self.T  # this should be the time horizon/look ahead: not using predefined T to generalize for usage
            dt = self.dt  # TODO: confirm where dt is defined

            def get_s_prime(_states, _actions):
                _s_prime = []

                def calc_state(x, u, dt):
                    sx, sy, vx, vy = x[0], x[1], x[2], x[3]
                    "Deceleration only leads to zero velocity!"
                    if sx == 0 and vx == 0:  # y axis movement
                        vx_new = vx  # + u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vy_new < 0:
                            vy_new = 0
                        sx_new = sx  # + (vx + vx_new) * dt * 0.5
                        sy_new = sy + (vy + vy_new) * dt * 0.5
                    elif sy == 0 and vy == 0:  # x axis movement
                        vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy  # + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vx_new < 0:
                            vx_new = 0
                        sx_new = sx + (vx + vx_new) * dt * 0.5
                        sy_new = sy  # + (vy + vy_new) * dt * 0.5
                    else:  # TODO: assume x axis movement for single agent case!!
                        vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy  # + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vx_new < 0:
                            vx_new = 0
                        sx_new = sx + (vx + vx_new) * dt * 0.5
                        sy_new = sy  # + (vy + vy_new) * dt * 0.5

                    # TODO: add a cieling for how fast they can go
                    return sx_new, sy_new, vx_new, vy_new

                # Cases where only one state is in states:
                # if len(_states) == 1:
                for a in _actions:
                    _s_prime.append(calc_state(_states, a, dt))
                # else:
                #     for s in _states:
                #         for a in _actions:
                #             _s_prime.append(calc_state(s, a, dt))
                print("s prime:", _s_prime)
                return _s_prime

            # TODO: Check the state!
            # states = self.initial_state  # or use current state

            i = 0  # row counter
            # L = len(actions)*T
            # state_list = np.zeros([T, L])
            state_list = {}  # use dict to keep track of time step
            # state_list = []
            # state_list.append(state) #ADDING the current state!
            for t in range(0, T):
                s_prime = get_s_prime(state, actions)  # separate pos and speed!
                state_list[i] = s_prime
                # state_list.append(s_prime)
                state = s_prime  # get s prime for the new states
                i += 1  # move onto next row
            return state_list
            # -------end of code--------

        def action_probabilities(state, _lambda):  # equation 1
            """
            refer to mdp.py
            Noisy-rational model
            calculates probability distribution of action given hardmax Q values
            Uses:
            1. Softmax algorithm
            2. Q-value given state action pair (s, a)
            3. beta: "rationality coefficient"
            => P(uH|xH;beta,theta) = exp(beta*QH(xH,uH;theta))/sum_u_tilde[exp(beta*QH(xH,u_tilde;theta))]
            :return:
                Normalized probability distributions of available actions at a given state and lambda
            """
            # Need to add some filtering for states with no legal action: q = -inf
            # exp_Q_list = np.zeros(shape=state_list) #create an array of exp_Q recording for each state
            # for i, s in enumerate(state_list):
            Q = q_values(state, self.goal[0])  # TODO: maybe import goal through args?
            #print("Q values array:", Q)
            # exp_Q = np.empty(len(Q))
            exp_Q = []

            "Q*lambda"
            # np.multiply(Q,_lambda,out = Q)
            Q = [q * _lambda for q in Q]
            #print("Q*lambda:", Q)

            "Q*lambda/(sum(Q*lambda))"
            # np.exp(Q, out=Q)
            for q in Q:
                exp_Q.append(np.exp(q))
            #print("exp_Q:", exp_Q)

            "normalizing"
            # normalize(exp_Q, norm = 'l1', copy = False)
            exp_Q /= sum(exp_Q)
            print("exp_Q normalized:", exp_Q)
            return exp_Q
            # exp_Q_list[i] = exp_Q

            # return exp_Q_list #array of exp_Q for an array of states


        def traj_probabilities(state, _lambda, prior = None):
            # TODO: What does summarizing over x(k) and u(k) do?
            """
            refer to mdp.py
            Calculates probability of being in a set of states at time k+1: P(x(k+1)| lambda, theta)
            :params:
                state: current state at time k
                _lambda: given lambda/rational coefficient
                prior: probability of agent being at "state" at time k (default is 1)
            :return:
                possible resulting states at k+1 with probabilities for being at each one of them
            """

            "for now we consider prior = 1 for observed state at time k"
            if prior == None:
                p_traj = 1  # initialize
            T = self.T  # TODO: predefined time horizon
            state_list = get_state_list(state, T)  # get list of state given curr_state/init_state from self._init_

            # p_states = np.zeros(shape=state_list)
            p_states = []

            "This is for the case where we have 2D array of state list"
            # for row in p_states:
            #     if row == 0: #first row has the initial state so prob is 1
            #         p_states[0, 0] = p_traj
            #     else:
            #         #TODO: generalize for more than 1 time step!
            #         for i in row: #calculate prob for subsequent states
            #             p_action = action_probabilities(state_list[0, 0], _lambda)
            #             #p_action = action_probabilities(state, _lambda)
            #             p_states[row, i] = p_traj * p_action[i]

            "for the case where state list is 1D"
            for i in range(len(state_list)):
                p_states.append(action_probabilities(state, _lambda))  # 1 step look ahead only depends on action prob
                # transition is deterministic -> 1, prob state(k) = 1

            #TODO: make a dictionary to store future states at time k+1?
            return state_list, p_states

        def belief_resample(priors, epsilon):
            """
            Equation 3
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            initial_belief = np.ones(len(priors)) / len(priors)
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def theta_joint_update(thetas, lambdas, traj, goal, epsilon=0.05, theta_priors=None):
            """
            refer to destination.py
            Calculates joint probability of lambda and theta, P(lambda, theta|D(k))
            :params:
                thetas: list of theta/intent/aggressiveness measure
                lambdas: list of lambda/rational coefficient
                traj: array of past state action pair
                goal: array describing the goal state
                epsilon: resample coefficient/weight
                theta_priors: prior of joint distribution of lambda and theta, P(lambda, theta|D(k-1))
            :return:
                posterior probabilities of each theta and corresponding lambda maximizing the probability
            """
            # TODO: simplify the code and reduce the redundant calculation
            # theta_init = np.ones(len(thetas))/len(thetas) #initial belief of theta

            if theta_priors is None:
                theta_priors = np.ones(len(thetas)) / len(thetas)
            print("-----theta priors: {}".format(theta_priors))
            print("traj: {}".format(traj))
            suited_lambdas = np.empty(len(thetas))
            L = len(lambdas)

            "processing traj data, in the case that datas of 2 agents are stored together in tupples:"

            # traj_state = []
            # traj_action = []
            # for i, traj_t in enumerate(traj):
            #     traj_state.append(traj_t[0])
            #     traj_action.append(traj_t[1])
            # h_states = []
            # h_actions = []
            # for j, s_t in enumerate(traj_state):
            #     h_states.append(traj_state[0])
            # for k, a_t in enumerate(traj_action):
            #     h_actions.append(traj_action[0])

            # scores = np.empty(len(lambdas))
            # TODO: should we record scores for past traj?
            def compute_score(traj, _lambda, L):
                # scores = np.empty(L)
                scores = []
                for i, (s, a) in enumerate(traj):  # pp score calculation method
                    #print("--i, (s, a), lambda:", i, (s, a), _lambda)
                    p_a = action_probabilities(s, _lambda)  # get probability of action in each state given a lambda
                    # scores[i] = p_a[s, a]
                    #print("-p_a[a]:", p_a[a_i])
                    # scores[i] = p_a[a]
                    a_i = self.action_set.index(a)
                    scores.append(p_a[a_i])
                print("scores at frame {}:".format(self.frame), scores)
                log_scores = np.log(scores)
                return np.sum(log_scores)

            def get_last_score(traj, _lambda, L): #add score to existing list of score
                p_a = action_probabilities(traj[-1][0], _lambda)
                a = traj[-1][1]
                a_i = self.action_set.index(a)
                if _lambda in self.past_scores: #add to existing list
                    scores = self.past_scores[_lambda]
                    scores.append(p_a[a_i])
                else:
                    self.past_scores[_lambda] = [p_a[a_i]]
                    scores = self.past_scores[_lambda]
                log_scores = np.log(scores)
                return np.sum(log_scores)

            "executing compute_score/get_last_score to get the best suited lambdas"
            for i, theta in enumerate(thetas):  # get a best suited lambda for each theta
                score_list = []
                for j, lamb in enumerate(lambdas):
                    #score_list.append(compute_score(traj, lamb, L))
                    score_list.append(get_last_score(traj, lamb, L))
                    #l = str(lamb)
                    # if lamb in self.past_scores: #add to current list
                    #     self.past_scores[lamb].append(get_last_score(traj, lamb, L))
                    # else: #create a list
                    #     self.past_scores[lamb] = [get_last_score(traj, lamb, L)]
                max_lambda_j = np.argmax(score_list)
                #max_lambda_j = np.argmax(self.past_scores)
                suited_lambdas[i] = lambdas[max_lambda_j]  # recording the best suited lambda for corresponding theta[i]
                #print("theta being analyzed:", theta, "best lambda:", lambdas[max_lambda_j])

            # Instead of iterating we calculate action prob right before calculating p_theta_prime!

            p_theta = np.copy(theta_priors)
            "re-sampling from initial distribution (shouldn't matter if p_theta = prior?)"
            p_theta = belief_resample(p_theta, epsilon == 0.05)  # resample from uniform belief
            p_theta_prime = np.empty(len(thetas))

            "joint inference update for (lambda, theta)"
            #TODO: optimize calculation to increase speed
            for t, (s, a) in enumerate(traj):  # enumerate through past traj#TODO: CHECK PAST TRAJ FROM CLASS AV
                if t == 0:  # initially there's only one state and not past
                    for theta_t in range(len(thetas)):  # cycle through list of thetas
                        p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
                        a_i = self.action_set.index(a)
                        p_theta_prime[theta_t] = p_action[a_i] * p_theta[theta_t]
                else:  # for state action pair that is not at time zero
                    for theta_t in range(len(thetas)):  # cycle through theta at time t or K
                        p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
                        for theta_past in range(len(thetas)):  # cycle through theta probability from past time K-1
                            a_i = self.action_set.index(a)
                            p_theta_prime[theta_t] += p_action[a_i] * p_theta[theta_past]
            p_theta_prime /= sum(p_theta_prime)  # normalize

            assert np.sum(p_theta_prime) == 1  # check if it is properly normalized
            print("-----p_thetas at frame {0}: {1}".format(self.frame, p_theta_prime))
            return {'predicted_intent_other': [p_theta_prime, suited_lambdas]}

        def marginal_prob(p_state_beta, p_theta):
            """
            Calculates the marginal probability P(x(k+1) | D(k))
            General procedure:
            1. P(lambda, theta) is calculated first and best lambda is obtained
            2. P(x(k+1) | lambda, theta) is calculated with the best lambda from previous step
            3. Calculate P(x(k+1) | D(k)) by multiplying the results from first 2 steps together, with
               the same lambda #TODO: need confirmation!
            :param
                p_state_beta: P(x(k+1)|lambda, theta)
                p_theta: P(lambda, theta), the joint probability of theta and corresponding lambda
            :return:
                p_state_D: P(x(k+1) | D(k)), the marginal probability of agent being in state x(k+1)
                           given observation D(k)
            """

            #TODO: calculate marginal prob
            "get look up table for P(lambda, theta)"
            #TODO: calculate prob for beta (a table instead of just for 2 thetas???)

            "calculate marginal"
            p_state_D = np.zeros(len(p_state_beta))
            for i in range(p_state_beta):
                for j in range(p_theta): #TODO: multiply by the p_theta with the corresponding lambda????
                    if j == 0:
                        p_state_D[i] = p_state_beta[i] * p_theta[j]
                    else:
                        p_state_D[i] += p_state_beta[i] * p_theta[j]

            return p_state_D

        "------------------------------"
        "executing the above functions!"
        "------------------------------"

        "#calling functions for baseline inference"
        joint_probability = theta_joint_update(thetas=self.thetas, theta_priors=self.theta_priors,
                                               lambdas=self.lambdas, traj=traj_h, goal=self.goal, epsilon=0.05)

        "#take a snapshot of the theta prob for next time step"
        self.theta_priors = joint_probability['predicted_intent_other'][0]

        "calculate the marginal state distribution / prediction" #TODO: implement function once completed

        return joint_probability #TODO: CHECK WHAT TO RETURN


        # def lambda_update( self, lambdas, traj, priors, goals, k):
        #     #This function is not in use! But it is a good reference for update algorithm
        #     """
        #     refer to beta.py
        #     Simple lambda updates without theta joint inference
        #     Update belief over set of beta with Baysian update
        #     params:
        #     traj: for obtaining trajectory up to time step k
        #     k: current time step
        #     trajectory probabilities: calculates probability of action taken at given state and params
        #     :return: Posterior belief over betas
        #     """
        #
        #     if priors is None:
        #         priors = np.ones(len(lambdas)) #assume uniform priors
        #         priors /= len(lambdas) #normalize
        #
        #     resampled_prior = self.belief_resample(priors, epsilon = 0.05) #0.05 is what they used
        #
        #     if k is not None:
        #         traj = traj[-k:]
        #
        #     #calculating posteriors
        #     post_lambda = np.copy(priors)
        #     for i,beta in enumerate(priors):
        #         #multiply by action probability given state and params
        #         post_lambda[i] *= self.trajectory_probabilities(goals, traj=traj, beta=beta)
        #
        #     np.divide(post_lambda, np.sum(post_lambda), out=post_lambda) #normalize
        #
        #     return post_lambda
        #
        #     pass


        "functions for references, not in use:"
            # def state_probabilities_infer(self, traj, goal, state_priors, thetas, theta_priors, lambdas, T):
        #     #TODO: maybbe we dont need this function? as our transition is deterministic and have only one destination
        #     #Not in use
        #     """
        #     refer to state.py and occupancy.py
        #     Infer the state probabilities before observation of lambda and theta.
        #     Equation 4: P(x(k+1);lambda, theta) = P(u(k)|x(k);lambda,theta) * P(x(k),lambda, theta)
        #                                         = action_probabilities * p_state(t-1)
        #     params:
        #     T: time horizon for state probabilities inference
        #     :return:
        #     probability of theta
        #     probability the agent is at each state given correct theta
        #     corresponding lambda
        #     """
        #     #TODO theta_priors could be None!! Design the function around it!
        #     p_theta, lambdas = self.theta_joint_update(thetas, theta_priors, traj,goal, epsilon = 0.05)
        #     # TODO: modify the following sample code:
        #
        #
        #     def infer_simple(self, T):
        #         p_state = np.zeros([T+1, self.states])
        #         p_state[0][self.initial_state] = 1 #start with prob of 1 at starting point
        #         p_action = self.baseline_inference.action_probabilities()
        #         for t in range(1, T + 1):
        #             p = p_state[t - 1]
        #             p_prime = p_state[t]
        #             p_prime *= p_action #TODO: check this calculation! Maybe need p_action for each time step
        #         return p_state
        #
        #     # Take the weighted sum of the occupancy given each individual destination.
        #     # iterate through multiple thetas
        #     for theta, lamb, p_action, p_theta in zip(thetas, lambdas, p_actions, p_theta):
        #         p_state = infer_simple(T)
        #         np.multiply(p_state, p_theta, out=p_state)
        #         #TODO: I am confused... need to check back for how to calculate p_state in our case
        #
        #
        #     pass

        # def value_iter():
        #     #Not in use
        #     """
        #     refer to hardmax.py
        #     Calculate V where ith entry is the value of starting at state s till i
        #     :return:
        #     """
        #     pass


    #@staticmethod
    def empathetic_inference(self, agent, sim):
        """
        When QH also depends on xM,uM
        :return:P(beta_h, beta_m_hat | D(k))
        """
        #TODO: documentation
        """
        Equation 6
        Equation 7
        Equation 8
        Equation 9
        Equation 10
        Equation 11
        """
        #----------------------------#
        # TODO: implement proposed:
        # variables: predicted_intent_other: BH hat, predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat, predicted_policy_self: QM tilde
        #----------------------------#
        #TODO: get necessary agent info:
        "---agent info---"

        "---end of agent info---"
        #TODO: import Q pairs from Nash Equilibrium computed with RL
        #TODO: how will betas arrays (lambda1, theta1), ... be constructed?
        #TODO: THREE total priors needed to store: p(betas|D(k-1)) and p(Q pairs|D(k-1)) and P(x(k)|Qh,Qm)
        #NOTE: action prob is considering only one Nash Equilibrium (Qh, Qm) instead of a set of them!!!
        #TODO: for a set of NE iterate through them using the below functions!
        def h_action_prob(self, state, _lambda_h, q_values_h):
            # TODO: documentation
            """
            Calculate H's action probability based on the Q values from Nash Equilibrium (QH,QM)eq
            :param self:
            :return: H's action probabilities correspond to each action available at state s
            """
            # TODO: consider how q value from nash equilibrium is imported
            #Pseudo code, need confirmation:
            q = self.q_h_eq #get q values for H from nash equilibrium

            #reusing q variable to save storage
            "p_action_h = exp(q*lambda)/sum(exp(q*lambda))"
            np.multiply(q, _lambda_h, out = q)
            np.exp(q, out = q)
            normalize(q, norm = 'l1', copy = False)

            # TODO: CONSIDER if there are more than 1 set of Qs: maybe just implement them in outer scope
            pass
            return q

        def m_action_prob(self, state, _lambda_m, q_values_m):
            # TODO: documentation
            """
            Calculate M's action probability based on the Q values obtained from Nash Equilibrium (QH,QM)eq
            :param self:
            :return: M's action probabilities correspond to each action available at state s
            """
            # Pseudo code, need confirmation:
            #TODO: consider how q value from nash equilibrium is imported

            q = self.q_m_eq  # get q values for H from nash equilibrium
            # reusing q variable to save storage
            "p_action_h = exp(q*lambda)/sum(exp(q*lambda))"
            np.multiply(q, _lambda_m, out=q)
            np.exp(q, out=q)
            normalize(q, norm='l1', copy=False)

            pass
            return q

        def belief_resample(self, priors, epsilon):
            """
            Equation 3
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            initial_belief = np.ones(len(priors)) / len(priors)
            resampled_priors = (1 - epsilon)*priors + epsilon * initial_belief
            return resampled_priors

        def q_pair_prob(self, prior ,q_pairs, state, lambda_h):
            #TODO: documentation
            """
            Equation 6
            Calculates Q function pairs probabilities for use in beta pair probabilities calculation,
            since each beta pair may map to multiple Q function/value pair.

            :requires: p_action_h, p_pair_prob(k-1) or prior, q_pairs
            :param:
            self:
            q_pairs: all Q function pairs (QH, QM)
            p_action_h
            p_q2

            :return:
            """
            #TODO: code is still in work
            if prior is None:
                prior = np.ones(q_pairs)/len(q_pairs) #TODO:: assuming uniform prior?

            "resample from initial/uniform distribution"
            prior = self.belief_resample(prior, epsilon=0.05)

            p_action_h = h_action_prob(state, lambda_h)
            p_q2 = np.empty(q_pairs)

            #TODO:: I'm assuming a 2D array of q pairs is imported, needs confirmation!!!
            #TODO: also I need to confirm if this calculation is correct
            "Calculating probability of each Q pair: Equation 6"
            for i,q_h in enumerate(q_pairs): #rows
                for j,q_m in enumerate(q_h): #cols
                    p_q2[i, j] = p_action_h[i] *prior[i, j] #action prob corresponds to H's Q so it's "i"

            p_q2 /= sum(p_q2) #normalize
            assert sum(p_q2) == 1 #check if properly normalized
            pass
            return p_q2
        def prob_beta_given_q(self, beta_D_prior, beta_H, beta_M):
            #TODO: this is a placeholder, needs to be implemented
            """
            Equation 8: using Bayes rule
            Calculates probability of beta pair (Bh, BM_hat) given Q pair (QH, QM): P(Bh, BM_hat | QH, QM),
            for beta_pair_prob formula.
            :param self:
            :return:
            """
            # TODO: code is still in work
            #TODO: check betas
            if beta_D_prior is None:
                prior = np.ones([beta_H, beta_M]) / len([beta_H, beta_M])  # TODO:: assuming uniform prior?
            "import prob of beta pair given D(k-1) from Equation 7"
            p_betas_prev = beta_pair_prob()
            "calculate prob of beta pair given Q pair"

            pass


        def beta_pair_prob(self, beta_H, beta_M, q_pairs):
            """
            Equation 7
            Calculates probability of beta pair (BH, BM_hat) given past observation D(k).
            :param self:
            :return:
            """
            #TODO: This code is still in work
            "importing prob of Q pair given observation D(k)"
            p_q2 = q_pair_prob() #p_q2_d
            "importing prob of beta pair given Q pair"
            #TODO: this is a placeholder!
            p_betas_q = prob_beta_given_q() #does not return anything yet
            "Calculate prob of beta pair given D(k) by summing over Q pair"

            # TODO: resample from initial belief! HOW??? (no prior is used!)
            "resample from initial/uniform distribution"

            p_betas_d = np.zeros(p_betas_q)
            r = len(q_pairs)
            c = len(q_pairs[0])#for iterating through 2D array p_q_pairs
            for p in range(len(p_betas_d)): #iterating through 2D array p_betas
                for q in range(len(p_betas_d)[0]):
                    p_beta_q = p_betas_q[p, q]
                    for i in range(r): #iterate through 2D array p_q2 or p_q2_d
                        for j in range(c):
                            #if p_betas[p,q] == p_beta:
                            if (i, j) == (0,0): #first element
                                p_betas_d[p, q] = p_beta_q * p_q2[0, 0]
                            else:
                                p_betas_d[p, q] += p_beta_q * p_q2[i,j]
            pass
            return p_betas_d

        def get_state_list_h(self, T):
            """
            calculate an array of state (T x S at depth T)
            shape: t0[x(k)  ]
                   t1[x(k+1)], for 1 step look ahead
            :param self:
            :return:
            """
            # TODO: Check this modification so that action probability is calculated for states within a time horizon
            # ---Code: append all states reachable within time T----
            states = self.states  # TODO: import a state to start from
            actions = self.action_set
            #T = self.T  # this should be the time horizon/look ahead: not using predefined T to generalize for usage
            dt = self.sim.dt

            # TODO: MODIFY for calculating state list for agent M
            def get_s_prime(_states, _actions):
                _s_prime = []

                def calc_state(x, u, dt):
                    sx, sy, vx, vy = x[0], x[1], x[2], x[3]
                    vx_new = vx + u * dt * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                    vy_new = vy + u * dt * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                    sx_new = sx + (vx + vx_new) * dt * 0.5
                    sy_new = sy + (vy + vy_new) * dt * 0.5

                    return sx_new, sy_new, vx_new, vy_new

                for s in _states:
                    for a in _actions:
                        _s_prime.append(calc_state(s, a, dt))
                return _s_prime

            # TODO: Check the state!
            states = self.initial_state  # or use current state
            i = 0  # row counter
            state_list = np.zeros([T, actions * T])
            for t in range(0, T):
                s_prime = get_s_prime(states, actions)  # separate pos and speed!
                state_list[i] = s_prime
                state = s_prime  # get s prime for the new states
                i += 1  # move onto next row
            return state_list
            # -------end of code--------
        def get_state_list_m(self, T):
            """
            calculate an array of state (T x S at depth T)
            shape: t0[x(k)  ]
                   t1[x(k+1)], for 1 step look ahead
            :param self:
            :return:
            """
            # TODO: Check this modification so that action probability is calculated for states within a time horizon
            # ---Code: append all states reachable within time T----
            states = self.states  # TODO: import a state to start from
            actions = self.action_set
            #T = self.T  # this should be the time horizon/look ahead: not using predefined T to generalize for usage
            dt = self.sim.dt
            #TODO: MODIFY for calculating state list for agent M
            def get_s_prime(_states, _actions):
                _s_prime = []

                def calc_state(x, u, dt):
                    sx, sy, vx, vy = x[0], x[1], x[2], x[3]
                    vx_new = vx + u * dt * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                    vy_new = vy + u * dt * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                    sx_new = sx + (vx + vx_new) * dt * 0.5
                    sy_new = sy + (vy + vy_new) * dt * 0.5

                    return sx_new, sy_new, vx_new, vy_new

                for s in _states:
                    for a in _actions:
                        _s_prime.append(calc_state(s, a, dt))
                return _s_prime

            #TODO: Check the state! As well as the state info for M (state[:,1]?? or simply self.h_state?)!
            states = self.initial_state  # or use current state
            i = 0  # row counter
            state_list = np.zeros([T, actions * T])
            for t in range(0, T):
                s_prime = get_s_prime(states, actions)  # separate pos and speed!
                state_list[i] = s_prime
                state = s_prime  # get s prime for the new states
                i += 1  # move onto next row
            return state_list
            # -------end of code--------

        def get_state_list(self):
            """

            :return:
            """
            h_states = self.get_state_list_h(self.T)
            h_states = h_states[1,:] #extract states from t = k+1, which is in the 2nd row
            m_states = self.get_state_list_m(self.T)
            m_states = m_states[1,:] #extract states from t = k+1, which is in the 2nd row
            states = np.empty([len(h_states),len(m_states)])
            for i in range(len(h_states)):
                for j in range(len(m_states)):
                    states[i,j] = (h_states[i],m_states[j]) #TODO: Check states data type in sim
            return states #h_state by m_state matrix

        def state_prob(self, curr_state,Qh,Qm):
            """
            Equation 11 (includes 9 and 10)
            Calculates state probabilities given past observation D(k)
            :param self:
            :return:
            """
            state_list = get_state_list(self.T) #CHECK Universal Time Horizon!!!!
            def p_action_pair(state, _lambda):
                """
                This implementation only considers current given state! Implement outside for iterating through states
                Equation 10: p(uh,um|x(k);Qh,Qm)
                Computes joint probabilities of H and M's action prob
                :return:
                """
                #TODO: consider if the state contains 2 agents information!
                p_action_h = self.h_action_prob(state, _lambda,q_values_h=q_value_h) #1D arrays
                p_action_m = self.m_action_prob(state, _lambda,q_values_m=q_value_m)
                p_a2 = np.zeros([p_action_h,p_action_m])
                for ah in p_action_h:
                    for am in p_action_m:
                        p_a2[ah,am] = p_action_h[ah]*p_action_m[am]
                return p_a2
                #pass
            def state_prob_q(prior = None):
                """
                Equation 9
                :return:
                """
                "import p(uh,um|x(k);Qh,Qm)"
                p_a2 = p_action_pair()
                "import prior p(x(k)|Qh,Qm)"
                #TODO: do we need to check if prior is available?
                #p_state_prev = state_prob_q() #previous time step
                "In the case where transition is deterministic, and prior P(x(k)) is 1 as it is already observed:"
                "And time horizon T = 1"
                return p_a2 #P(x(k+1)|Q2) is solely determined by p(uh,um|x(k),Q2)
                #pass
            #p_actions_list =
            "Import Q pair prob from Equation 6"
            p_q2 = q_pair_prob()
            "Call the function: state prob q"
            p_state_q = state_prob_q()
            "Equation 11: calculate state probabilities given past observation D(k)"
            #p_s = np.multiply(p_q2,p_state_q) <- incorrect, we are calculating the marginal

            #TODO: check the size of state_list and p_q2
            p_s = np.zeros(state_list) #p_s should have the same size as s_list
            for p in range(len(state_list)): #assuming state_list is 2D
                for q in range(len(state_list[0])):
                    p_state_q_i = p_state_q[p, q] #extract an element for multiplication
                    for i in range(len(p_q2)): #assuming p_q2 is a 2D array
                        for j in range(len(p_q2[0])):
                            if (i, j) == (0,0): #for the first element
                                p_s[p, q] = p_state_q_i * p_q2[0,0]
                            else:
                                p_s[p, q] += p_state_q_i * p_q2[i, j]


            return p_s #size of uH x uM

        "#execution of functions and return for empathetic inference"
        pass #end of empathetic inference

    @staticmethod
    def less_inference():
        # implement Bobu et al. "LESS is More:
        # Rethinking Probabilistic Models of Human Behavior"
        pass
