class RiskAnalyzer:

    def __init__(self):
        pass


    def compute_risk_level(self, probability):

        if probability < 0.3:
            return "Low"

        elif probability < 0.7:
            return "Moderate"

        else:
            return "High"


    def compute_confidence(self, probability):

        confidence = abs(probability - 0.5) * 2

        return confidence


    def analyze(self, probability):

        risk = self.compute_risk_level(probability)

        confidence = self.compute_confidence(probability)

        return {
            "distress_probability": probability,
            "risk_level": risk,
            "confidence": confidence
        }