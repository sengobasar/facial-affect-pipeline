import numpy as np


class FacialBehaviorExtractor:

    def __init__(self):
        pass

    def distance(self, p1, p2):

        return np.sqrt(
            (p1[0] - p2[0])**2 +
            (p1[1] - p2[1])**2
        )

    def compute_eye_openness(self, landmarks):

        # example landmark indices (approximate)
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]

        eye_open = self.distance(
            left_eye_top,
            left_eye_bottom
        )

        return eye_open

    def compute_mouth_openness(self, landmarks):

        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]

        mouth_open = self.distance(
            mouth_top,
            mouth_bottom
        )

        return mouth_open

    def compute_features(self, landmarks):

        eye_open = self.compute_eye_openness(landmarks)

        mouth_open = self.compute_mouth_openness(landmarks)

        return {
            "eye_openness": eye_open,
            "mouth_openness": mouth_open
        }