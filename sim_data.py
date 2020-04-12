import pickle


class SimData:

    def __init__(self, n):

        # list of all variables of interest. Not all studies requires all of them.
        self.key = ['states',  # actual states
                    'actions',  # actual actions (converted from trajectory)
                    'intent',
                    'planned_action_sets',  # planned future actions
                    'planned_trajectory_set',  # future state trajectories based on the planned actions
                    'predicted_intent_other',  # other's intent, estimated
                    'predicted_intent_self',  # other's belief of self intent, estimated
                    'predicted_actions_other',  # other's future actions, estimated
                    'predicted_actions_self',  # other's belief of self future actions, estimated
                    'wanted_action_self',  # most favorable self future state traj by others, estimated
                    'wanted_action_self',  # most favorable other's future state traj by self
                    'inference_probability',  # inference probabilities
                    'gracefulness',  # TODO: check definition
                    ]

        self.raw = dict.fromkeys(self.key, [])
