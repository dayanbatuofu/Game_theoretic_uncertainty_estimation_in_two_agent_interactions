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
            sx, sy, vx, vy = current_s[0], current_s[1], current_s[2], current_s[3]
            if sx == 0 and vx == 0:
                #Q = FUNCTION MATH HERE USING SY, VY
                Q = -(goal_s - sx)/(vx + action * dt)
            elif sy == 0 and vy == 0:
                #Q = FUNCTION MATH HERE USING SX, VX
                Q = -(goal_s - sy) / (vy + action * dt)
            else:
                #Q = FUNCTION FOR 2D MODELS
                Q = -((goal_s - sy) / (vy + action * dt) + (goal_s - sx)/(vx + action * dt))
            return Q

        def q_values(self, states):
            #TODO documentation for function

            current_s = states[-1]
            Q = {}
            #TODO: IMPORT GOAL.  goal = whatever
            #TODO: IMPORT DT
            for a in sets.ActionsActions:
                Q[a] = self.q_function(current_s, sets.getActionVal(a), goal, dt)

            return Q


            """
            refer to classic.py, car.py
            Calculates hardmax Q-value given state-action pair.
            Q(s,a) = R(s,a)  +V(s')   #All f
            Q(s, a) = (s_goal - s_current)/v_next
            """
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

            #Q values using reward and value iteration:
            """
            V = value_iter(goal)
            Q = np.empty([agents.S, agents.A])
            Q.fill(-np.inf)

            for s in range(agents.S):
                if s == goal and goal_stuck:
                    Q[s, agents.A] = 0
                    continue
                for a in range(agents.A):
                    if s == goal and a = Actions.ABSORB:
                        Q[s, a] = 0
                    else:
                        Q[s, a] = self.rewards[s, a] + V[transition(s,a)]
            return np.copy(Q)
            
            """
            pass



        def transition_probabilities(self,beta):
            """
            Refer to mdp.py
            No need since our car transition is deterministic?
            :return:
            """
            """
            #Pseudo code
     
            """
            pass
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



            #TODO: Yi's notes 5/17
            #Need to add some filtering for states with no legal action: q = -inf
            Q = self.q_values()
            exp_Q = np.empty([Q])

            "Q*lambda"
            np.multiply(Q,_lambda,out = Q)

            "Q*lambda/(sum(Q*lambda))"
            np.exp(Q, out=exp_Q)
            normalize(exp_Q, norm = 'l1', copy = False)

            return exp_Q
            #pass

        def traj_probabilities(self, traj):
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

            p_action = self.action_probabilities()
            p_traj = 1 #initialize
            
            for s, a in traj:
                p_traj *= p_action
            return p_traj

            #pass
        def binary_search(self, traj,  gradient):
            """
            refer to gradient_descent_shared.py
            For finding most suited beta corresponding to theta
            :return:
            """
            #Pseudo code
            #TODO modify for our uses
            """
            lo, hi = min_beta, max_beta
            
            if guess is None:
                mid = (lo + hi) / 2
            else:
                mid = guess

            if len(traj) == 0:
                return guess

            for i in xrange(max_iters):
                assert lo <= mid <= hi
                grad = compute_grad(g, traj, goal, mid)
                if verbose:
                    print u"i={}\t mid={}\t grad={}".format(i, mid, grad)

                if i >= min_iters and abs(grad) < grad_threshold:
                    break

                if grad > 0:
                   lo = mid
                else:
                    hi = mid
                if i >= min_iters and hi - lo < beta_threshold:
                    break

                mid = (lo + hi)/2

            if verbose:
                print u"final answer: beta=", mid
            return mid
            """
            pass
        def gradient_ascent():
            """
            refer to gradient_descent_shared.py
            For obtaining local maximum
            :return:
            """
            """
            
            """
            pass
        def lambda_update( self, lambdas, traj, priors, goals, k):
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

            #TODO: choose epsilonn
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
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            initial_belief = np.ones(len(priors)) / len(priors)
            resampled_priors = (1 - epsilon)*priors + epsilon * initial_belief
            return resampled_priors

        def theta_joint_update(self, thetas, theta_priors, traj,goal, epsilon = 0.05):
            """
            refer to destination.py
            :return:posterior probabilities of each theta and corresponding lambda maximizing the probability
            """
            #TODO: organize all algorithms for joint inference
            if theta_priors is None:
                theta_priors = np.ones(len(thetas))/len(thetas)

            lambdas = np.empty(len(thetas))
            for i, theta in enumerate(thetas):
                lambdas[i] = self.binary_search(traj, gradient)

            #TODO: check state and action for one agent case
            p_action = np.empty([self.agents.s,self.agents.a])
            for i, (lamb, theta) in enumerate(zip(lambdas, thetas)):
                p_action[i] = action_probabilities(lamb)

            p_theta = np.copy(theta_priors)
            p_theta_prime = np.empty(len(thetas))

            "probability of theta"
            for t,(s, a) in enumerate(traj):
                if t ==0:
                    for theta_t in range(len(thetas)):
                        p_theta_prime[theta_t] = p_action[theta_t,s,a]  * p_theta[theta_t]
                else:
                    for theta_t in range(len(thetas)):
                        for theta_past in range(len(thetas)):
                            p_theta_prime[theta_t] += p_action[theta_t,s,a]  * p_theta[theta_past]
            p_theta_prime /= sum(p_theta_prime) #normalize

            #TODO: joint inference with theta and lambda
            pass
            return p_theta_prime, lambdas
        def state_probabilities_infer(self, traj, goal, thetas,theta_priors, p_theta,lambdas, T):
            """
            refer to state.py and occupancy.py
            Infer the state probabilities before observation of lambda and theta.
            params:
            T: time horizon for state probabilities inference
            :return:
            probability of theta
            probability the agent is at each state given correct theta
            corresponding lambda
            """
            #TODO theta_priors could be None!! Design the function around it!
            prob_theta, lambdas = self.theta_joint_update(thetas, theta_priors, traj,goal, epsilon = 0.05)
            # TODO: modify the following sample code:
            """
            # Take the weighted sum of the occupancy given each individual destination.
            D = np.zeros(g.S)
            D_dests = []  # Only for verbose_return
            for dest, beta, act_prob, dest_prob in zip(
                    dests, betas, act_probs, dest_probs):
                D_dest = infer_simple(g, init_state, dest, T, beta=beta,
                        action_prob=act_prob)
                if verbose_return:
                    D_dests.append(np.copy(D_dest))
                np.multiply(D_dest, dest_prob, out=D_dest)
                np.add(D, D_dest, out=D)

            if not verbose_return:
                return D
            else:
                return D, D_dests, dest_probs, betas
            """
            pass

        def value_iter():
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
