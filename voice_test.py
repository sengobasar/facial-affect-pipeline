from voice_module.audio.audio_stream import AudioStream
from voice_module.features.mfcc_extractor import MFCCExtractor
from voice_module.features.pitch_extractor import PitchExtractor
from voice_module.features.jitter_shimmer import JitterShimmerExtractor
from voice_module.voice_emotion_model import VoiceEmotionModel

import numpy as np


audio_stream = AudioStream()
mfcc_extractor = MFCCExtractor()
pitch_extractor = PitchExtractor()
js_extractor = JitterShimmerExtractor()
emotion_model = VoiceEmotionModel()


audio_signal, energy = audio_stream.record()

mfcc = mfcc_extractor.extract(audio_signal)

pitch = pitch_extractor.extract(audio_signal)

pitch_series = [pitch] * 20
amp_series = np.abs(audio_signal[:20])

jitter, shimmer = js_extractor.compute(pitch_series, amp_series)

voice_emotion = emotion_model.compute(
    mfcc,
    pitch,
    jitter,
    shimmer,
    energy
)

print("Voice Emotion:", voice_emotion)