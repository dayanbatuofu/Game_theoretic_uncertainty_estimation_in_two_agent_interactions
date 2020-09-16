"""
parse variables to be updated in autonomous_vehicle
"""
import pickle


class DataUtil:

    def __init__(self, n):
        pass

    @staticmethod
    def update(agent, data):
        # list of all variables of interest. Not all studies requires all of them.
        key = ['state',  # actual states
               'action',  # actual actions (converted from trajectory)
               'par',  # parameters of the agent's objective
               'planned_action_sets',  # planned future actions
               'planned_trajectory_set',  # future state trajectories based on the planned actions
               'predicted_par_other',  # other's intent, estimated
               'predicted_par_self',  # other's belief of self intent, estimated
               'predicted_actions_other',  # other's future actions, estimated
               'predicted_actions_self',  # other's belief of self future actions, estimated
               'predicted_states_other',  # prediction of other's future states
               'predicted_states_self',  # prediction of other's future states
               'predicted_intent_all',  # calculated joint probabilities of intent and rationality for all agents
               'predicted_intent_other',  # calculated joint probabilities of intent and rationality
               'predicted_intent_self',  # calculated joint probabilities of intent and rationality
               'wanted_action_self',  # most favorable self future state traj by others, estimated
               'wanted_action_self',  # most favorable other's future state traj by self
               'inference_probability',  # inference probabilities
               'gracefulness',  # TODO: check definition
               'predicted_policy_other'
               ]

        if data:
            for k in list(data.keys()):
                if k in key:
                    target = getattr(agent, k)
                    source = data[k]
                    target.append(source)
