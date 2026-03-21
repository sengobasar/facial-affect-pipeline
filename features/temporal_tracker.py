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

        # Means
        v_mean = float(np.mean(self.valence_history))
        a_mean = float(np.mean(self.arousal_history))
        d_mean = float(np.mean(self.distress_history))

        # Variances (measure of emotional volatility)
        v_var = float(np.var(self.valence_history)) if len(self.valence_history) > 1 else 0.0
        d_var = float(np.var(self.distress_history)) if len(self.distress_history) > 1 else 0.0

        # Durations (percentage of time in high distress or positive valence)
        # Assuming sample rate is consistent
        high_distress_count = sum(1 for d in self.distress_history if d > 0.6)
        positive_valence_count = sum(1 for v in self.valence_history if v > 0.6)
        
        total = len(self.distress_history)
        d_duration = high_distress_count / total if total > 0 else 0.0
        v_duration = positive_valence_count / total if total > 0 else 0.0

        return {
            "valence_avg": v_mean,
            "arousal_avg": a_mean,
            "distress_avg": d_mean,
            "valence_var": v_var,
            "distress_var": d_var,
            "high_distress_ratio": d_duration,
            "positive_valence_ratio": v_duration
        }