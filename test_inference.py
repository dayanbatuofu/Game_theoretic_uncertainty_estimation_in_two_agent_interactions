import numpy as np
#from sklearn.processing import normalize
from autonomous_vehicle import AutonomousVehicle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TestInference:

    def __init__(self,model,sim):

        # importing agents information
        self.sim = sim
        self.agents = AutonomousVehicle
        #self.curr_state = AutonomousVehicle.state  # cumulative #TODO: import this!
        #self.goal = sim.goal  # CHECK THIS
        #self.traj = AutonomousVehicle.planned_trajectory_set  # TODO: check if this is right!
        self.T = 1  # one step look ahead/ Time Horizon
        self.dt = 1

        "dummy data"
        self.curr_state = [(-20, 0, 0, 0), (0, -20, 0, 0)] #sx, sy, vx, vy
        self.goal = [(0, 20), (20, 0)] #goal for both H and M
        self.actions = [-2, -0.5, 0, 0.5, 2] #accelerations (m/s^2)
        self.lambdas = [0.01, 0.1, 1, 10] #range?
        self.thetas = [1, 1000] #range?
        self.traj = [(((-20, 0, 0, 0), (0, -20, 0, 0)), (2, 2))] #recordings of past states
        self.h_traj = [((-20, 0, 0, 0), 0)] #traj of H agent only
    def baseline_inference(self):

        def q_function( current_s, action, goal_s, dt):
            """Calculate Q value

            Current Implementation:
                Q is negatively proportional to the time it takes to reach the goal

            Params:
                current_s [tuple?] -- Current state containing x-state, y-state,
                    x-velocity, y-velocity
                action [IntEnum] -- potential action taken by car
                goal_s [tuple?] -- Goal state, same format as current_s
                dt[int / float] -- discrete time interval
            """
            #Q = -(s_goal - s_current)/v_next #estimates time required to reach goal with current state and chosen action
            u = action
            delta = 1 #to prevent denominator from becoming zero
            sx, sy, vx, vy = current_s[0], current_s[1], current_s[2], current_s[3]
            #TODO: implement a variable for filtering which agent we are looking at (agent = H or M) to choose goal!
            if sx == 0 and vx == 0:
                #Q = FUNCTION MATH HERE USING SY, VY
                print("Y direction movement detected")
                goal_s = goal_s[0]
                next_v = vy + action * dt
                "Deceleration only leads to 0 velocity!"
                if next_v < 0.0:
                    #let v_next = 0
                    Q = -(goal_s - sy) / delta
                else:
                    Q = -(goal_s - sy) / (vy + action * dt + delta)
            elif sy == 0 and vy == 0:
                #Q = FUNCTION MATH HERE USING SX, VX
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
                #Q = FUNCTION FOR 2D MODELS
                goal_x = goal_s[0]
                goal_y = goal_s[1]
                next_vx = vx + action * dt
                next_vy = vy + action * dt

                "Deceleration only leads to 0 velocity!"
                if next_vx < 0:
                    if next_vy < 0: #both are negative
                        Q = -((goal_y - sy) / delta + (goal_x - sx) / delta)
                    else: #only vx is negative
                        Q = -((goal_y - sy) / (vy + action * dt + delta) + (goal_x - sx)/delta)
                elif next_vy < 0:#only vy is negative
                    Q = -((goal_y - sy) / delta + (goal_x - sx) / (vx + action * dt + delta))
                else: #both are non negative
                    Q = -((goal_y - sy) / (vy + action * dt + delta) + (goal_x - sx) / (vx + action * dt + delta))
                #TODO: add a cieling for how fast they can go
            return Q

        def q_values(state, goal):
            #TODO documentation for function
            """

            :param self:
            :param states:
            :return:
            """
            print("q_values function is being called,{0}, {1}".format(state,goal))
            #current_s = states[-1]
            #Q = {} #dict type
            Q = [] #list type
            actions = self.actions #TODO: check that actions are imported in init
            for a in actions: #sets: file for defining sets
                #Q[a] = q_function(state, a, goal, self.dt)  #dict type
                Q.append(q_function(state, a, goal, self.dt)) #list type

            return Q

            ##from PP code::   CHANGE PARAMETER NAMES
            #Inferring agent's current param based on its last action and state in last time step
            #"""
            #s_last = self.agents.state[-1]
            #a_last = self.agents.action[-1]
            #x_last = s_last[x]
            #v_last = s_last[v]
            #Q = -v_last*dt - np.abs(x_last + v_last*dt - goal) #2D version of Q value from confidence aware paper
            #"""

        def get_state_list(state, T):
            """
            calculate an array of state (T x S at depth T)
            :param self:
            :return:
            """
            # TODO: Check this modification so that action probability is calculated for states within a time horizon
            # ---Code: append all states reachable within time T----
            #states = self.states  # TODO: import a state to start from
            actions = self.actions
            #T = self.T  # this should be the time horizon/look ahead: not using predefined T to generalize for usage
            dt = self.dt #TODO: confirm where dt is defined

            def get_s_prime(_states, _actions):
                _s_prime = []

                def calc_state(x, u, dt):
                    sx, sy, vx, vy = x[0], x[1], x[2], x[3]
                    "Deceleration only leads to zero velocity!"
                    if sx == 0 and vx == 0: #y axis movement
                        vx_new = vx #+ u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy + u * dt #* vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vy_new < 0:
                            vy_new = 0
                        sx_new = sx #+ (vx + vx_new) * dt * 0.5
                        sy_new = sy + (vy + vy_new) * dt * 0.5
                    elif sy == 0 and vy == 0: #x axis movement
                        vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy #+ u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vx_new < 0:
                            vx_new = 0
                        sx_new = sx + (vx + vx_new) * dt * 0.5
                        sy_new = sy #+ (vy + vy_new) * dt * 0.5
                    else: #TODO: assume x axis movement for single agent case!!
                        vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy #+ u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vx_new < 0:
                            vx_new = 0
                        sx_new = sx + (vx + vx_new) * dt * 0.5
                        sy_new = sy #+ (vy + vy_new) * dt * 0.5

                    # TODO: add a cieling for how fast they can go
                    return sx_new, sy_new, vx_new, vy_new

                #Cases where only one state is in states:
                #if len(_states) == 1:
                for a in _actions:
                    _s_prime.append(calc_state(_states, a, dt))
                # else:
                #     for s in _states:
                #         for a in _actions:
                #             _s_prime.append(calc_state(s, a, dt))
                print("s prime:", _s_prime)
                return _s_prime

            # TODO: Check the state!
            #states = self.initial_state  # or use current state

            i = 0  # row counter
            #L = len(actions)*T
            #state_list = np.zeros([T, L])
            state_list = {} #use dict to keep track of time step
            #state_list = []
            #state_list.append(state) #ADDING the current state!
            for t in range(0, T):
                s_prime = get_s_prime(state, actions)  # separate pos and speed!
                state_list[i] = s_prime
                #state_list.append(s_prime)
                state = s_prime  # get s prime for the new states
                i += 1  # move onto next row
            return state_list
            # -------end of code--------

        def action_probabilities(state, _lambda):  #equation 1
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
            """
            #Need to add some filtering for states with no legal action: q = -inf
            #exp_Q_list = np.zeros(shape=state_list) #create an array of exp_Q recording for each state
            #for i, s in enumerate(state_list):
            Q = q_values(state, self.goal[0]) #TODO: maybe import goal through args?
            print("Q values array:", Q)
            #exp_Q = np.empty(len(Q))
            exp_Q = []

            "Q*lambda"
            #np.multiply(Q,_lambda,out = Q)
            Q = [q*_lambda for q in Q]
            print("Q*lambda:",Q)

            "Q*lambda/(sum(Q*lambda))"
            #np.exp(Q, out=Q)
            for q in Q:
                exp_Q.append(np.exp(q))
            print("exp_Q:",exp_Q)

            "normalizing"
            #normalize(exp_Q, norm = 'l1', copy = False)
            exp_Q /= sum(exp_Q)

            return exp_Q
            #exp_Q_list[i] = exp_Q

            #return exp_Q_list #array of exp_Q for an array of states
            #TODO: check data type! make sure the data can be easily accessed(2D array with 2 for loops?)

        def traj_probabilities(state, _lambda):
            #TODO: think about how trajectory is generated
            #TODO: Modify this so that state distribution is calculated for future 1 time step
            #TODO: What does summarizing over x(k) and u(k) do?
            """
            refer to mdp.py
            multiply over action probabilities to obtain trajectory probabilities given (s, a)
            params:
            traj: composed of sequence of (s, a) pair
            p_action: probability of action taken at each state
            p_traj: probability of action taken accumulated over states, i.e. p_traj = p(a|s1)*p(a|s2)*p(a|s3)*...
            :return:
            """
            #Pseudo code

            #p_action = action_probabilities(_lambda)
            p_traj = 1 #initialize
            T = self.T #TODO: predefined time horizon
            state_list = get_state_list(state, T) #get list of state given curr_state/init_state from self._init_
            #p_states = np.zeros(shape=state_list)
            p_states = []

            # for row in p_states:
            #     if row == 0: #first row has the initial state so prob is 1
            #         p_states[0, 0] = p_traj
            #     else:
            #         #TODO: generalize for more than 1 time step!
            #         for i in row: #calculate prob for subsequent states
            #             p_action = action_probabilities(state_list[0, 0], _lambda)
            #             #p_action = action_probabilities(state, _lambda)
            #             p_states[row, i] = p_traj * p_action[i]
            for i in range(len(state_list)):
                p_states.append(action_probabilities(state,_lambda)) #1 step look ahead only depends on action prob
                #transition is deterministic -> 1, prob state(k) = 1

            return state_list, p_states

        def belief_resample(priors, epsilon):
            """
            Equation 3
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            initial_belief = np.ones(len(priors)) / len(priors)
            resampled_priors = (1 - epsilon)*priors + epsilon * initial_belief
            return resampled_priors

        def theta_joint_update(thetas,  lambdas, traj, goal, epsilon=0.05,theta_priors=None):
            """
            refer to destination.py
            :return:posterior probabilities of each theta and corresponding lambda maximizing the probability
            """
            #TODO: simplify the code and reduce the redundant calculation
            #theta_init = np.ones(len(thetas))/len(thetas) #initial belief of theta

            if theta_priors is None:
                theta_priors = np.ones(len(thetas))/len(thetas)

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

            #scores = np.empty(len(lambdas))
            #TODO: how to get recorded traj for evaluation(Maybe AutonomousVehicle.state?)?
            def compute_score(traj, _lambda, L):
                #scores = np.empty(L)
                scores = []
                for i, (s, a) in enumerate(traj): #pp score calculation method
                    print("--i, (s, a), lambda:", i, (s, a), _lambda)
                    p_a = action_probabilities(s, _lambda)  # get probability of action in each state given a lambda
                    #scores[i] = p_a[s, a]
                    print("-p_a[a]:", p_a[a])
                    #scores[i] = p_a[a]
                    scores.append(p_a[a])
                print("scores:", scores)
                log_scores = np.log(scores)
                return np.sum(log_scores)

            "executing compute_score to get the best suited lambdas"
            for i, theta in enumerate(thetas):#get a best suited lambda for each theta
                #lambdas[i] = self.binary_search(traj, gradient)
                score_list = []
                for j, lamb in enumerate(lambdas):
                    score_list.append(compute_score(traj, lamb, L))
                max_lambda_j = np.argmax(score_list)
                suited_lambdas[i] = lambdas[max_lambda_j]  #recording the best suited lambda for corresponding theta[i]
                print("theta being analyzed:", theta, "best lambda:", lambdas[max_lambda_j])
            #Instead of iterating we calculate action prob right before calculating p_theta_prime!
            #p_action = np.empty([thetas, traj, self.agents.a]) #storing 3D info of [thetas, states, actions]
            #for i, (lamb, theta) in enumerate(zip(lambdas, thetas)):
            #    for s, a in traj:
            #        p_action[i] = action_probabilities(s, lamb)

            p_theta = np.copy(theta_priors)
            "re-sampling from initial distribution (shouldn't matter if p_theta = prior?)"
            p_theta = belief_resample(p_theta, epsilon == 0.05) #resample from uniform belief
            p_theta_prime = np.empty(len(thetas))

            "joint inference update for (lambda, theta)"
            for t,(s, a) in enumerate(traj): #enumerate through past traj#TODO: CHECK PAST TRAJ FROM CLASS AV
                if t == 0: #initially there's only one state and not past
                    for theta_t in range(len(thetas)): #cycle through list of thetas
                        p_action = action_probabilities(s, suited_lambdas[theta_t]) #1D array
                        p_theta_prime[theta_t] = p_action[a]  * p_theta[theta_t]
                else: #for state action pair that is not at time zero
                    for theta_t in range(len(thetas)): #cycle through theta at time t or K
                        p_action = action_probabilities(s, suited_lambdas[theta_t]) #1D array
                        for theta_past in range(len(thetas)): #cycle through theta probability from past time K-1
                            p_theta_prime[theta_t] += p_action[a]  * p_theta[theta_past]
            p_theta_prime /= sum(p_theta_prime) #normalize
            assert np.sum(p_theta_prime) == 1 #check if it is properly normalized

            return p_theta_prime, suited_lambdas
        #pass
        "testing the functions: return the following when baseline is called"
        #return q_values(self.curr_state[0], self.goal[0])
        #return get_state_list(self.curr_state[0], self.T)
        #return action_probabilities(self.curr_state[0], self.lambdas[1])
        #return traj_probabilities(self.curr_state[0], self.lambdas[0])
        #return theta_joint_update(self.thetas, lambdas=self.lambdas, traj=self.h_traj, goal=self.goal[0], epsilon=0.05, theta_priors=None)

        return traj_probabilities(self.curr_state[0], self.lambdas[0])

    def test_plot(self, states, p_states):
        """

        :param states: takes in list of states (sx, sy, vx, vy)i
        :param p_states: takes in list of probabilities correspond to the list of states
        :return: contour plot
        """
        """
        x = self.states[pos]
        y = self.states[speed]
        z = self.p_state
        fig,(ax1, ax2, ax3, ...) = plt.subplots(nrows = 2) #plot separately for different thetas

        #------
        #plot 1: distribution with the pair (lambda1, theta1)
        #------
        ax1.contour(x, y, z, levels = 10, linewidths = 1,colors = 'k' )
        #TODO: modify the params
        cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
        fig.colorbar(cntr1, ax=ax1)
        ax1.plot(x, y, 'ko', ms=3)
        ax1.set(xlim=(-2, 2), ylim=(-2, 2))
        ax1.set_title('state probability distribution with theta 1')
        """
        x = []
        y = []
        z = []

        "testing purposes"
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for s in states:
            x.append(s[0])
            y.append(s[1])
        z = p_states
        #plt.scatter(x, y)
        ax.scatter(x, y, z, zdir='z')
        plt.xlim(-20, 20)
        plt.show()

        "Plotting contour"
        # for s in states:
        #     x.append((s[0], s[1]))
        #     y.append((s[1], s[2]))
        # z = p_states

        # fig, ax1 = plt.subplot(nrows = 2)
        # ax1.contour(x, y, z, levels=10)
        # #cntr1 = ax1.contourf()
        # ax1.plot(x, y, 'ko', ms=3)
        # ax1.set(xlim=(-20, 20), ylim=(0, 20))
        # ax1.set_title('state probability distribution with theta 1')


test = TestInference(1,1)
print("prinnting baseline:", test.baseline_inference()[0][0], test.baseline_inference()[1])
test.test_plot(test.baseline_inference()[0][0], test.baseline_inference()[1])
