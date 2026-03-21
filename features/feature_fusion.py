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
        voice_features=None,
        temporal_stats=None
    ):

        # ----------------------
        # Behavioral features
        # ----------------------

        eye_open = behavior_features.get("eye_openness", 0.0)
        mouth_open = behavior_features.get("mouth_openness", 0.0)

        # ----------------------
        # Face + behavior vector
        # ----------------------

        features = [
            face_valence,   # Vf
            face_arousal,   # Af
            face_distress,  # Df
            eye_open,       # EAR
            mouth_open      # MAR
        ]

        # ----------------------
        # Voice features
        # ----------------------

        if voice_emotion is None or voice_features is None:

            voice_vector = [0.0] * 7

        else:

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

        # ----------------------
        # Temporal features
        # ----------------------

        if temporal_stats:
            temporal_vector = [
                temporal_stats.get("valence_var", 0.0),
                temporal_stats.get("distress_var", 0.0),
                temporal_stats.get("high_distress_ratio", 0.0),
                temporal_stats.get("positive_valence_ratio", 0.0)
            ]
        else:
            temporal_vector = [0.0] * 4

        # ----------------------
        # Final fusion vector
        # ----------------------

        features.extend(voice_vector)
        features.extend(temporal_vector)

        return np.array(features, dtype=float)