import librosa
import numpy as np


class MFCCExtractor:

    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def extract(self, audio_signal):

        """
        audio_signal : numpy array
        """

        mfcc = librosa.feature.mfcc(
            y=audio_signal,
            sr=self.sr,
            n_mfcc=self.n_mfcc
        )

        # transpose so frames × coefficients
        mfcc = mfcc.T

        return mfcc