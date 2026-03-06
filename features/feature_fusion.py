import numpy as np


class FeatureFusion:

    def __init__(self):
        pass

    def fuse(self, valence, arousal, distress, behavior_features):

        eye_open = behavior_features["eye_openness"]
        mouth_open = behavior_features["mouth_openness"]

        feature_vector = np.array([
            valence,
            arousal,
            distress,
            eye_open,
            mouth_open
        ])

        return feature_vector