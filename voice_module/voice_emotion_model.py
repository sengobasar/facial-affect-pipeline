import numpy as np


class VoiceEmotionModel:

    def __init__(self):

        # weights for valence
        self.w_pitch = 0.4
        self.w_energy = 0.3
        self.w_mfcc = 0.3

        # weights for arousal
        self.w_pitch_var = 0.5
        self.w_jitter = 0.25
        self.w_shimmer = 0.25


    def compute(self, mfcc_features, pitch, jitter, shimmer, energy):

        """
        mfcc_features : MFCC coefficient array
        pitch : fundamental frequency (Hz)
        jitter : pitch instability
        shimmer : amplitude instability
        energy : signal energy
        """

        # ---------------------------
        # Feature normalization
        # ---------------------------

        pitch_norm = pitch / 300.0
        energy_norm = energy / 1000.0

        mfcc_mean = np.mean(mfcc_features)

        # ---------------------------
        # Valence Calculation
        # ---------------------------

        valence = (
            self.w_pitch * pitch_norm +
            self.w_energy * energy_norm +
            self.w_mfcc * mfcc_mean
        )

        valence = np.tanh(valence)


        # ---------------------------
        # Arousal Calculation
        # ---------------------------

        pitch_variance = pitch_norm

        arousal = (
            self.w_pitch_var * pitch_variance +
            self.w_jitter * jitter +
            self.w_shimmer * shimmer
        )

        arousal = np.tanh(arousal)


        # ---------------------------
        # Distress Score
        # ---------------------------

        distress = (1 - valence) * abs(arousal)


        return {
            "valence": float(valence),
            "arousal": float(arousal),
            "distress": float(distress)
        }