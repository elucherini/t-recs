from .base_component import BaseComponent, FromNdArray
import numpy as np

class PredictedScores(FromNdArray, BaseComponent):
    def __init__(self, predicted_scores=None, verbose=False):
        # general input checks
        if predicted_scores is not None:
            if not isinstance(predicted_scores, (list, np.ndarray)):
                raise TypeError("predicted_scores must be a list or numpy.ndarray")
        self.predicted_scores = predicted_scores
        # Initialize component state
        BaseComponent.__init__(self, verbose=verbose, init_value=self.predicted_scores)

    def store_state(self):
        self.component_data.append(np.copy(self.predicted_scores))
