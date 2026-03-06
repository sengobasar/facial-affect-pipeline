import cv2

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


def main():

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

    while True:

        frame = camera.get_frame()

        faces = detector.detect_faces(frame)

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
                # Feature fusion
                # --------------------

                feature_vector = fusion.fuse(
                    v,
                    a,
                    d,
                    behavior_features
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
                # Display results
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

        cv2.imshow("Mental Health Detection - Face Module", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()