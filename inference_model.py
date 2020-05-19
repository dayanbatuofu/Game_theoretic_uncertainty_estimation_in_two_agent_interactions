import numpy as np
#from sklearn.processing import normalize   UNCOMMENT WHEN IMPLEMENT
# TODO pytorch version
from autonomous_vehicle import AutonomousVehicle

class Actions(IntEnums):
    #CHANGE THESE NAMES
    BIG_ACC = 5
    LIL_ACC = 3
    NO_ACC = 0
    LIL_BRAKE = -3
    BIG_BRAKE = -5
    
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

        #"Yi 5/14-------------------------------------------------"
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
        #Pseudo code for intent inference to obtain P(lambda, theta) based on past action sequence D(k-1):
        #P(lambda, theta|D(k)) = P(u| x;lambda, theta)P(lambda, theta | D(k-1)) / sum[ P(u|x;lambda', theta') P(lambda', theta' | D(k-1))]
        #equivalent: P = action_prob * P(k-1)/sum_(lambda,theta)[action_prob * P(k-1)]
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
            for a in Actions:
                Q[a] = self.q_function(current_s, a, goal)

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
        def action_probabilities():  #equation 1
            """
            refer to mdp.py
            calculates probability distribution of action given hardmax Q values
            Uses:
            1. Softmax algorithm
            2. Q-value given state action pair (s, a)
            3. beta: "rationality coefficient"
            => P(uH|xH;beta,theta) = exp(beta*QH(xH,uH;theta))/sum_u_tilde[exp(beta*QH(xH,u_tilde;theta))]
            :return:
            """

            #Pseudocode 
            """
                Iterate over Q values where x in u|x = current state{
                    calculate lambda*Q(x,u,theta) --> store in T
                }

                //where T is lambda*Q(x,u,theta) for all u|x
                np.exp(T, out=T)                        //import numpy as np
                normalize(T, norm='l1', copy=False)     //from sklearn.preprocessing import normalize
                
            """

            """
            Yi's notes 5/17
            #Need to add some filtering for states with no legal action: q = -inf
            Q = self.q_values()
            np.exp(Q,out=exp_Q)
            normalize(exp_Q, norm = 'l1', copy = False)
            """
            pass
        def traj_probabilities():
            """
            refer to mdp.py
            multiply over action probabilities to obtain trajectory probabilities given (s, a)
            :return:
            """
            pass
        def value_iter():
            """
            refer to hardmax.py
            Calculate V where ith entry is the value of starting at state s till i
            :return:
            """
            pass
        pass
        def transition_helper(g, s, a, alert_illegal = False):
            """
            refer to classic.py, car.py
            defines the transition when action a is taken in state s
            :return:
            """
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
