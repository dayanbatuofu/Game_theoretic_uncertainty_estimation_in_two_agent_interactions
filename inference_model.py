import numpy as np
from sklearn.processing import normalize
# TODO pytorch version
from autonomous_vehicle import AutonomousVehicle
import discrete_sets as sets

class Lambdas(FloatEnums)
class InferenceModel:
    def __init__(self, model, sim):
        if model == 'none':
            self.infer = self.no_inference
        elif model == 'baseline':
            self.infer = self.baseline_inference
        elif model == 'empathetic':
            self.infer = self.empathetic_inference
        else:
            # placeholder for future development
            pass
        self.sim = sim

        # TODO: obtain state and action space information
        # importing agents information
        self.agents = AutonomousVehicle
        self.initial_state = AutonomousVehicle.initial_state #TODO: import this!
        #"--------------------------------------------------------"
        #"imported variables from pedestrian prediction"
        self.q_cache = {}
        #"defining reward for (s, a) pair"
        self.default_reward = -1
        self.rewards = np.zeros[self.agents.s, self.agents.a] #Needs fixing
        self.rewards.fill(self.default_reward)
        #"--------------------------------------------------------"

    @staticmethod
    def no_inference(agents, sim):
        pass

    @staticmethod
    def baseline_inference(self,agents, sim):
        # TODO: implement Fridovich-Keil et al. "Confidence-aware motion prediction for real-time collision avoidance"
        """
        for each agent, estimate its par (agent.par) based on its last action (agent.action[-1]),
        system state (agent.state[-1] for all agent), and prior dist. of par
        :return:
        """
        """
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


        def q_function(self, current_s, action, goal_s, dt):
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
            if sx == 0 and vx == 0:
                #Q = FUNCTION MATH HERE USING SY, VY
                Q = -(goal_s - sx)/(vx + action * dt + delta)
            elif sy == 0 and vy == 0:
                #Q = FUNCTION MATH HERE USING SX, VX
                Q = -(goal_s - sy) / (vy + action * dt + delta)
            else:
                #Q = FUNCTION FOR 2D MODELS
                Q = -((goal_s - sy) / (vy + action * dt + delta) + (goal_s - sx)/(vx + action * dt + delta))
            return Q

        def q_values(self, state):
            #TODO documentation for function
            """

            :param self:
            :param states:
            :return:
            """
            #current_s = states[-1]
            Q = {}
            #TODO: IMPORT GOAL.  goal = whatever
            #TODO: IMPORT DT
            for a in sets.ActionsActions:
                Q[a] = self.q_function(current_s, sets.getActionVal(a), goal, dt)

            return Q

            #Estimating Q value at state s with time to goal from s, assuming agent moves along the y axis:
            """
            #Pseudo code
            s_current = self.agents.state[s_y]
            s_goal = goal.state[s_y]
            v_current = self.agents.state[v_y]
            Q = np.empty([agent.Action]) #size of action available at state s
            v_next = v_current + agent.Action * dt
            Q = -(s_goal - s_current)/v_next #estimates time required to reach goal with current state and chosen action
            return Q
            """
            #Inferring agent's current param based on its last action and state in last time step
            """
            s_last = self.agents.state[-1]
            a_last = self.agents.action[-1]
            x_last = s_last[x]
            v_last = s_last[v]
            Q = -v_last*dt - np.abs(x_last + v_last*dt - goal) #2D version of Q value from confidence aware paper
            """
            #from berkeley code::   CHANGE PARAMETER NAMES

        def action_probabilities(self,_lambda):  #equation 1
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
            #TODO: Check this modification so that action probability is calculated for states within a time horizon
            #-----pseudo code: append all states reachable within time T----
            states = self.states #TODO: import a state to start from
            actions = self.actions
            state_list = []
            dt = self.sim.dt
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
                        _s_prime.append(calc_state(s, a, dt)) #maybe use AutonomousVehicle.dynamics(action)?
                return _s_prime
            states = self.initial_state
            for t in range(1, T):
                s_prime = get_s_prime(states, actions) #separate pos and speed!
                state_list.append(s_prime)
                state = s_prime  # get s prime for the new states
            #-----end of pseudo code------


            #Need to add some filtering for states with no legal action: q = -inf
            exp_Q_list = np.empty(len(state_list)) #create an array of exp_Q recording for each state
            for i, s in enumerate(state_list):
                Q = self.q_values(s)
                exp_Q = np.empty([Q])

                "Q*lambda"
                np.multiply(Q,_lambda,out = Q)

                "Q*lambda/(sum(Q*lambda))"
                np.exp(Q, out=exp_Q)
                normalize(exp_Q, norm = 'l1', copy = False)
                exp_Q_list[i] = exp_Q
            return exp_Q_list #array of exp_Q for an array of states
            #TODO: check data type! make sure the data can be easily accessed(2D array with 2 for loops?)
            #pass

        def traj_probabilities(self, traj, _lambda):
            #TODO: think about how trajectory is generated
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

            p_action = self.action_probabilities(_lambda)
            p_traj = 1 #initialize
            p_states = np.empty(len(traj))
            for i, (s, a) in enumerate(traj):
                p_traj *= p_action
                p_states[i] = p_traj #5/28 update: add probability at each state to a list
            return p_states

            #pass

        def lambda_update( self, lambdas, traj, priors, goals, k):
            #This function is not in use!
            """
            refer to beta.py
            Simple lambda updates without theta joint inference
            Update belief over set of beta with Baysian update
            params:
            traj: for obtaining trajectory up to time step k
            k: current time step
            trajectory probabilities: calculates probability of action taken at given state and params
            :return: Posterior belief over betas
            """
            #Psuedo code
            #TODO: use the resampled prior from function belief_resample

            if priors is None:
                priors = np.ones(len(lambdas)) #assume uniform priors
                priors /= len(lambdas) #normalize

            #TODO: choose epsilon
            resampled_prior = self.belief_resample(priors, epsilon = 0.05) #0.05 is what they used

            if k is not None:
                traj = traj[-k:]
            
            #calculating posteriors
            post_lambda = np.copy(priors)
            for i,beta in enumerate(priors):
                #multiply by action probability given state and params
                post_lambda[i] *= self.trajectory_probabilities(goals, traj=traj, beta=beta)
                
            np.divide(post_lambda, np.sum(post_lambda), out=post_lambda) #normalize

            return post_lambda
            

            pass
        def belief_resample(self, priors, epsilon):
            """
            Equation 3
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            initial_belief = np.ones(len(priors)) / len(priors)
            resampled_priors = (1 - epsilon)*priors + epsilon * initial_belief
            return resampled_priors

        def theta_joint_update(self, thetas, theta_priors, lambdas, traj,goal, epsilon = 0.05):
            """
            refer to destination.py
            :return:posterior probabilities of each theta and corresponding lambda maximizing the probability
            """

            if theta_priors is None:
                theta_priors = np.ones(len(thetas))/len(thetas)

            #TODO: enumerate through all the lambdas instead of searching!
            suited_lambdas = np.empty(len(thetas))
            L = len(lambdas)
            #scores = np.empty(len(lambdas))
            def compute_score(self, traj, _lambda, L):
                scores = np.empty(L)
                p_a = self.action_probabilities(_lambda)
                for i, (s, a) in enumerate(traj): #pp score calculation method
                    scores[i] = p_a[s, a]
                log_scores = np.log(scores)
                return np.sum(log_scores)

            for i, theta in enumerate(thetas):#get a best suited lambda for each theta
                #lambdas[i] = self.binary_search(traj, gradient)
                score_list = []
                for j, lamb in enumerate(lambdas):
                    score_list.append(compute_score(traj, lamb, L))
                max_lambda_j = np.argmax(score_list)
                suited_lambdas[i] = lambdas[max_lambda_j]  #recording the best suited lambda for corresponding theta[i]

            #TODO: check state and action for one agent case
            p_action = np.empty([self.agents.s,self.agents.a])
            for i, (lamb, theta) in enumerate(zip(lambdas, thetas)):
                p_action[i] = action_probabilities(lamb)

            p_theta = np.copy(theta_priors)
            p_theta_prime = np.empty(len(thetas))

            "joint inference update for (lambda, theta)"
            for t,(s, a) in enumerate(traj):
                if t ==0:
                    for theta_t in range(len(thetas)):
                        p_theta_prime[theta_t] = p_action[theta_t,s,a]  * p_theta[theta_t]
                else:
                    for theta_t in range(len(thetas)):
                        for theta_past in range(len(thetas)):
                            p_theta_prime[theta_t] += p_action[theta_t,s,a]  * p_theta[theta_past]
            p_theta_prime /= sum(p_theta_prime) #normalize
            assert np.sum(p_theta_prime) == 1 #check if it is properly normalized
            #TODO: use equation 3 to resample from initial belief

            pass
            return p_theta_prime, lambdas
        def state_probabilities_infer(self, traj, goal, state_priors, thetas, theta_priors, lambdas, T):
            #TODO: maybbe we dont need this function? as our transition is deterministic and have only one destination
            """
            refer to state.py and occupancy.py
            Infer the state probabilities before observation of lambda and theta.
            Equation 4: P(x(k+1);lambda, theta) = P(u(k)|x(k);lambda,theta) * P(x(k),lambda, theta)
                                                = action_probabilities * p_state(t-1)
            params:
            T: time horizon for state probabilities inference
            :return:
            probability of theta
            probability the agent is at each state given correct theta
            corresponding lambda
            """
            #TODO theta_priors could be None!! Design the function around it!
            p_theta, lambdas = self.theta_joint_update(thetas, theta_priors, traj,goal, epsilon = 0.05)
            # TODO: modify the following sample code:


            def infer_simple(self, T):
                p_state = np.zeros([T+1, self.states])
                p_state[0][self.initial_state] = 1 #start with prob of 1 at starting point
                p_action = self.baseline_inference.action_probabilities()
                for t in range(1, T + 1):
                    p = p_state[t - 1]
                    p_prime = p_state[t]
                    p_prime *= p_action #TODO: check this calculation! Maybe need p_action for each time step
                return p_state

            # Take the weighted sum of the occupancy given each individual destination.
            # iterate through multiple thetas
            for theta, lamb, p_action, p_theta in zip(thetas, lambdas, p_actions, p_theta):
                p_state = infer_simple(T)
                np.multiply(p_state, p_theta, out=p_state)
                #TODO: I am confused... need to check back for how to calculate p_state in our case


            pass

        def value_iter():
            #Not in use
            """
            refer to hardmax.py
            Calculate V where ith entry is the value of starting at state s till i
            :return:
            """
            pass

    @staticmethod
    def empathetic_inference():
        """
        When QH also depends on xM,uM
        :return:
        """
        # TODO: implement proposed

        # predicted_intent_other, predicted_intent_self, predicted_policy_other, predicted_policy_self
        pass

    @staticmethod
    def less_inference():
        # implement Bobu et al. "LESS is More:
        # Rethinking Probabilistic Models of Human Behavior"
        pass
