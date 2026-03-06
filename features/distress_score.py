class DistressScoreCalculator:

    def __init__(self):
        pass

    def compute(self, valence, arousal):

        distress = (1 - valence) * abs(arousal)

        return distress