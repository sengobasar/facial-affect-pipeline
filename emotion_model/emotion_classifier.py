import cv2
import numpy as np
from deepface import DeepFace


class EmotionClassifier:

    def __init__(self):

        # emotion labels
        self.emotions = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral"
        ]

    def predict_emotion(self, face):

        if face.size == 0:
            return None

        # resize face for CNN
        face = cv2.resize(face, (224, 224))

        # DeepFace emotion prediction
        result = DeepFace.analyze(
            face,
            actions=['emotion'],
            enforce_detection=False
        )

        probabilities = result[0]["emotion"]

        return probabilities