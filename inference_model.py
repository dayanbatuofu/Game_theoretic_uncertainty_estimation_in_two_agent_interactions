import numpy as np
#from sklearn.processing import normalize
# TODO pytorch version


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

    @staticmethod
    def no_inference(agents, sim):
        pass

    @staticmethod
    def baseline_inference(agents, sim):
        # TODO: implement Fridovich-Keil et al. "Confidence-aware motion prediction for real-time collision avoidance"
        """
        for each agent, estimate its par (agent.par) based on its last action (agent.action[-1]),
        system state (agent.state[-1] for all agent), and prior dist. of par
        :return:
        """
        def q_values():
            """
            refer to classic.py, car.py
            Calculates hardmax Q-value given state-action pair.
            Q(s,a) = R(s,a)  +V(s')
            """
            pass
        def transition_probabilities():
            """
            Refer to mdp.py
            :return:
            """
            pass
        def action_probabilities():
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
