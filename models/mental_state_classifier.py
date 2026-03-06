import numpy as np


class MentalStateClassifier:

    def __init__(self):

        # weights (initially random or heuristic)
        self.w = np.array([
            -1.2,   # valence
            0.8,    # arousal
            1.5,    # distress
            0.02,   # eye openness
            0.03    # mouth openness
        ])

        self.b = 0.1


    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))


    def predict_probability(self, feature_vector):

        z = np.dot(self.w, feature_vector) + self.b

        probability = self.sigmoid(z)

        return probability


    def predict_label(self, feature_vector):

        p = self.predict_probability(feature_vector)

        if p > 0.65:
            return "distressed"
        else:
            return "normal"