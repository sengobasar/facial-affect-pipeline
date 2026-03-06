import librosa
import numpy as np


class PitchExtractor:

    def __init__(self, sr=16000):
        self.sr = sr

    def extract(self, audio_signal):

        pitches = librosa.yin(
            audio_signal,
            fmin=50,
            fmax=300,
            sr=self.sr
        )

        pitch = np.nanmean(pitches)

        if np.isnan(pitch):
            pitch = 0.0

        return pitch