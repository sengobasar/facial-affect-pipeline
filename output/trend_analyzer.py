from collections import deque
import numpy as np


class TrendAnalyzer:

    def __init__(self, window_size=50):

        self.window_size = window_size

        self.history = deque(maxlen=window_size)


    def update(self, distress_probability):

        self.history.append(distress_probability)


    def compute_average(self):

        if len(self.history) == 0:
            return 0

        return np.mean(self.history)


    def compute_trend(self):

        if len(self.history) < 2:
            return "stable"

        slope = self.history[-1] - self.history[0]

        if slope > 0.1:
            return "increasing"

        elif slope < -0.1:
            return "decreasing"

        else:
            return "stable"


    def analyze(self):

        avg = self.compute_average()

        trend = self.compute_trend()

        return {
            "average_distress": avg,
            "trend": trend
        }