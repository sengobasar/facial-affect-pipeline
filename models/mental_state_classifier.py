import numpy as np


class MentalStateClassifier:

    def __init__(self):

        # Weight vector for 12 features
        # [Vf, Af, Df, eye, mouth, Vv, Av, Dv, pitch, jitter, shimmer, energy]

        self.w = np.array([

            -1.2,   # face valence
             0.8,   # face arousal
             1.5,   # face distress

             0.02,  # eye openness
             0.03,  # mouth openness

            -1.0,   # voice valence
             0.7,   # voice arousal
             1.3,   # voice distress

             0.01,  # pitch
             0.5,   # jitter
             0.5,   # shimmer
             0.01   # energy
        ])

        # bias
        self.b = 0.1


    # --------------------------
    # Sigmoid function
    # --------------------------

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))


    # --------------------------
    # Predict probability
    # --------------------------

    def predict_probability(self, feature_vector):

        feature_vector = np.array(feature_vector)

        # safety check
        if feature_vector.shape[0] != self.w.shape[0]:
            raise ValueError(
                f"Feature vector size {feature_vector.shape[0]} "
                f"does not match weight size {self.w.shape[0]}"
            )

        z = np.dot(self.w, feature_vector) + self.b

        probability = self.sigmoid(z)

        return float(probability)


    # --------------------------
    # Predict label
    # --------------------------

    def predict_label(self, feature_vector):

        p = self.predict_probability(feature_vector)

        if p > 0.65:
            return "distressed"

        elif p > 0.45:
            return "elevated"

        else:
            return "normal"