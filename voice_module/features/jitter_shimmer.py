import numpy as np


class JitterShimmerExtractor:

    def compute(self, pitch_series, amplitude_series):

        pitch_series = np.array(pitch_series)
        amplitude_series = np.array(amplitude_series)

        # --------------------
        # Jitter
        # --------------------
        pitch_diff = np.abs(np.diff(pitch_series))
        jitter = np.mean(pitch_diff)

        # --------------------
        # Shimmer
        # --------------------
        amp_diff = np.abs(np.diff(amplitude_series))
        shimmer = np.mean(amp_diff)

        if np.isnan(jitter):
            jitter = 0.0

        if np.isnan(shimmer):
            shimmer = 0.0

        return jitter, shimmer