import numpy as np
#from sklearn.processing import normalize
from autonomous_vehicle import AutonomousVehicle
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
        self.goal = [0, 20]
        self.actions = [-2, -0.5, 0, 2] #accelerations (m/s^2)
        self.lambdas = [0.01, 0.1, 1, 10] #range?
        self.thetas = [1, 1000] #range?
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
            #TODO: how to determine which direction or axis the agent is moving on!
            if sx == 0 and vx == 0:
                #Q = FUNCTION MATH HERE USING SY, VY
                print("Y direction movement detected")
                goal_s = goal_s[0]
                Q = -(goal_s - sy) / (vy + action * dt + delta)
            elif sy == 0 and vy == 0:
                #Q = FUNCTION MATH HERE USING SX, VX
                print("Y direction movement detected")
                goal_s = goal_s[1]
                Q = -(goal_s - sx) / (vx + action * dt + delta)
            else:
                #Q = FUNCTION FOR 2D MODELS
                goal_x = goal_s[0]
                goal_y = goal_s[1]
                Q = -((goal_y - sy) / (vy + action * dt + delta) + (goal_x - sx)/(vx + action * dt + delta))
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
                    #TODO: what if we started out having zero velocity?
                    if sx == 0 and vx == 0: #y axis movement
                        vx_new = vx #+ u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy + u * dt #* vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        sx_new = sx #+ (vx + vx_new) * dt * 0.5
                        sy_new = sy + (vy + vy_new) * dt * 0.5
                    elif sy == 0 and vy == 0: #x axis movement
                        vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy #+ u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        sx_new = sx + (vx + vx_new) * dt * 0.5
                        sy_new = sy #+ (vy + vy_new) * dt * 0.5
                    else: #TODO: assume x axis movement for single agent case!!
                        vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy #+ u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        sx_new = sx + (vx + vx_new) * dt * 0.5
                        sy_new = sy #+ (vy + vy_new) * dt * 0.5
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
            state_list = {}
            #state_list.append(state) #ADDING the current state!
            for t in range(0, T):
                s_prime = get_s_prime(state, actions)  # separate pos and speed!
                state_list[i] = s_prime
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
            Q = q_values(state, self.goal) #TODO: check function var
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


        #pass
        #return q_values(self.curr_state[0], self.goal)
        #return get_state_list(self.curr_state[0], self.T)
        return action_probabilities(self.curr_state[0], self.lambdas[0])
test = TestInference(1,1)
print(test.baseline_inference())