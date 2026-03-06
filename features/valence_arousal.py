import numpy as np


class ValenceArousalCalculator:

    def __init__(self):

        # Valence weights
        self.valence_weights = {
            "happy": 1.0,
            "surprise": 0.4,
            "neutral": 0.0,
            "sad": -0.7,
            "angry": -0.6,
            "fear": -0.8,
            "disgust": -0.7
        }

        # Arousal weights
        self.arousal_weights = {
            "happy": 0.6,
            "surprise": 0.8,
            "neutral": 0.0,
            "sad": -0.3,
            "angry": 0.7,
            "fear": 0.9,
            "disgust": 0.4
        }


    def compute(self, emotion_probs):

        # Normalize probabilities (DeepFace gives 0–100)
        total = sum(emotion_probs.values())

        if total == 0:
            return 0.0, 0.0

        normalized = {
            emotion: prob / total
            for emotion, prob in emotion_probs.items()
        }

        v = 0.0
        a = 0.0

        for emotion, prob in normalized.items():

            v += prob * self.valence_weights.get(emotion, 0)
            a += prob * self.arousal_weights.get(emotion, 0)

        return v, a