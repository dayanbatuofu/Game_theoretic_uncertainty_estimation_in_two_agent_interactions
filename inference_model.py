import numpy as np
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

    def no_inference(self):
        pass

    @staticmethod
    def baseline_inference():
        # implement Fridovich-Keil et al. "Confidence-aware motion prediction for real-time collision avoidance"

        pass

    @staticmethod
    def empathetic_inference():
        # implement proposed

        # predicted_intent_other, predicted_intent_self, predicted_policy_other, predicted_policy_self
        pass

    @staticmethod
    def less_inference():
        # implement Bobu et al. "LESS is More:
        # Rethinking Probabilistic Models of Human Behavior"
        pass
