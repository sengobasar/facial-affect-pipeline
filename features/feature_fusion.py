import numpy as np


class FeatureFusion:

    def __init__(self):
        pass


    def fuse(
        self,
        face_valence,
        face_arousal,
        face_distress,
        behavior_features,
        voice_emotion=None,
        voice_features=None
    ):

        # ----------------------
        # Behavioral features
        # ----------------------

        eye_open = behavior_features.get("eye_openness", 0.0)
        mouth_open = behavior_features.get("mouth_openness", 0.0)

        # ----------------------
        # Face features
        # ----------------------

        features = [
            face_valence,
            face_arousal,
            face_distress,
            eye_open,
            mouth_open
        ]

        # ----------------------
        # Voice features (optional)
        # ----------------------

        if voice_emotion and voice_features:

            v_valence = voice_emotion.get("valence", 0.0)
            v_arousal = voice_emotion.get("arousal", 0.0)
            v_distress = voice_emotion.get("distress", 0.0)

            pitch = voice_features.get("pitch", 0.0)
            jitter = voice_features.get("jitter", 0.0)
            shimmer = voice_features.get("shimmer", 0.0)
            energy = voice_features.get("energy", 0.0)

            voice_vector = [
                v_valence,
                v_arousal,
                v_distress,
                pitch,
                jitter,
                shimmer,
                energy
            ]

            features.extend(voice_vector)

        return np.array(features, dtype=float)