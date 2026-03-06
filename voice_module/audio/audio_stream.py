import sounddevice as sd
import numpy as np


class AudioStream:

    def __init__(self, sample_rate=16000, duration=3):

        self.sample_rate = sample_rate
        self.duration = duration

    def record(self):

        print("Recording voice...")

        audio = sd.rec(
            int(self.sample_rate * self.duration),
            samplerate=self.sample_rate,
            channels=1
        )

        sd.wait()

        audio = audio.flatten()

        # ------------------------
        # Energy Calculation
        # ------------------------

        energy = np.sum(audio ** 2)

        return audio, energy