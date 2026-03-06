from collections import deque
import numpy as np


class TemporalEmotionTracker:

    def __init__(self, window_size=30):

        # store last N emotional states
        self.window_size = window_size

        self.valence_history = deque(maxlen=window_size)
        self.arousal_history = deque(maxlen=window_size)
        self.distress_history = deque(maxlen=window_size)

    def update(self, valence, arousal, distress):

        self.valence_history.append(valence)
        self.arousal_history.append(arousal)
        self.distress_history.append(distress)

    def get_temporal_state(self):

        if len(self.valence_history) == 0:
            return None

        v_mean = np.mean(self.valence_history)
        a_mean = np.mean(self.arousal_history)
        d_mean = np.mean(self.distress_history)

        return {
            "valence_avg": v_mean,
            "arousal_avg": a_mean,
            "distress_avg": d_mean
        }