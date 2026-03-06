import cv2
import time
import numpy as np

from camera.camera_stream import CameraStream
from detection.face_detector import FaceDetector
from alignment.face_landmarks import FaceLandmarks

from emotion_model.emotion_classifier import EmotionClassifier

from features.valence_arousal import ValenceArousalCalculator
from features.distress_score import DistressScoreCalculator
from features.temporal_tracker import TemporalEmotionTracker
from features.facial_behavior import FacialBehaviorExtractor
from features.feature_fusion import FeatureFusion

from models.mental_state_classifier import MentalStateClassifier

from output.risk_analyzer import RiskAnalyzer
from output.trend_analyzer import TrendAnalyzer


# -----------------------------
# VOICE MODULE IMPORTS
# -----------------------------

from voice_module.audio.audio_stream import AudioStream
from voice_module.features.mfcc_extractor import MFCCExtractor
from voice_module.features.pitch_extractor import PitchExtractor
from voice_module.features.jitter_shimmer import JitterShimmerExtractor
from voice_module.voice_emotion_model import VoiceEmotionModel


def main():

    # -----------------------------
    # FACE MODULE INITIALIZATION
    # -----------------------------

    camera = CameraStream()
    detector = FaceDetector()
    landmarks_detector = FaceLandmarks()

    classifier = EmotionClassifier()

    va_calculator = ValenceArousalCalculator()
    distress_calculator = DistressScoreCalculator()

    behavior_extractor = FacialBehaviorExtractor()
    fusion = FeatureFusion()

    mental_classifier = MentalStateClassifier()

    risk_analyzer = RiskAnalyzer()
    trend_analyzer = TrendAnalyzer()

    tracker = TemporalEmotionTracker(window_size=30)

    # -----------------------------
    # VOICE MODULE INITIALIZATION
    # -----------------------------

    audio_stream = AudioStream()
    mfcc_extractor = MFCCExtractor()
    pitch_extractor = PitchExtractor()
    js_extractor = JitterShimmerExtractor()
    voice_model = VoiceEmotionModel()

    last_voice_time = 0
    voice_interval = 8   # seconds

    voice_emotion = None
    voice_features = None

    # -----------------------------
    # MAIN LOOP
    # -----------------------------

    while True:

        frame = camera.get_frame()
        faces = detector.detect_faces(frame)

        # --------------------------------
        # VOICE RECORDING (every few seconds)
        # --------------------------------

        current_time = time.time()

        if current_time - last_voice_time > voice_interval:

            try:

                audio_signal, energy = audio_stream.record()

                mfcc = mfcc_extractor.extract(audio_signal)

                pitch = pitch_extractor.extract(audio_signal)

                pitch_series = [pitch] * 20
                amp_series = np.abs(audio_signal[:20])

                jitter, shimmer = js_extractor.compute(
                    pitch_series,
                    amp_series
                )

                voice_features = {
                    "pitch": pitch,
                    "jitter": jitter,
                    "shimmer": shimmer,
                    "energy": energy
                }

                voice_emotion = voice_model.compute(
                    mfcc,
                    pitch,
                    jitter,
                    shimmer,
                    energy
                )

                last_voice_time = current_time

                print("Voice Emotion:", voice_emotion)

            except Exception as e:
                print("Voice processing error:", e)

        # --------------------------------
        # FACE PROCESSING
        # --------------------------------

        for (x, y, w, h, face) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            if face is None or face.size == 0:
                continue

            try:

                # --------------------
                # Emotion recognition
                # --------------------

                emotion_probs = classifier.predict_emotion(face)

                if emotion_probs is None:
                    continue

                emotion = max(emotion_probs, key=emotion_probs.get)

                # --------------------
                # Valence-Arousal
                # --------------------

                v, a = va_calculator.compute(emotion_probs)

                # --------------------
                # Distress score
                # --------------------

                d = distress_calculator.compute(v, a)

                tracker.update(v, a, d)

                temporal_state = tracker.get_temporal_state()

                # --------------------
                # Face landmarks
                # --------------------

                landmarks = landmarks_detector.extract_landmarks(face)

                if landmarks is None:
                    continue

                # --------------------
                # Behavioral features
                # --------------------

                behavior_features = behavior_extractor.compute_features(landmarks)

                # --------------------
                # FEATURE FUSION
                # --------------------

                feature_vector = fusion.fuse(
                    v,
                    a,
                    d,
                    behavior_features,
                    voice_emotion,
                    voice_features
                )

                # --------------------
                # Mental state prediction
                # --------------------

                probability = mental_classifier.predict_probability(feature_vector)

                label = mental_classifier.predict_label(feature_vector)

                # --------------------
                # Risk analysis
                # --------------------

                risk_info = risk_analyzer.analyze(probability)

                trend_analyzer.update(probability)

                trend_info = trend_analyzer.analyze()

                # --------------------
                # DISPLAY RESULTS
                # --------------------

                cv2.putText(
                    frame,
                    emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"V:{v:.2f} A:{a:.2f}",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"D:{d:.2f}",
                    (x, y + h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                if temporal_state is not None:

                    cv2.putText(
                        frame,
                        f"AvgD:{temporal_state['distress_avg']:.2f}",
                        (x, y + h + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                cv2.putText(
                    frame,
                    f"State:{label}",
                    (x, y + h + 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Risk:{risk_info['risk_level']}",
                    (x, y + h + 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Trend:{trend_info['trend']}",
                    (x, y + h + 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

            except Exception as e:
                print("Emotion detection error:", e)

        cv2.imshow("Mental Health Detection - Multimodal", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()