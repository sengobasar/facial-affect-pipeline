import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetector:

    def __init__(self):

        base_options = python.BaseOptions(
            model_asset_path="blaze_face_short_range.tflite"
        )

        options = vision.FaceDetectorOptions(
            base_options=base_options
        )

        self.detector = vision.FaceDetector.create_from_options(options)

    def detect_faces(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        detection_result = self.detector.detect(mp_image)

        faces = []

        if detection_result.detections:

            for detection in detection_result.detections:

                bbox = detection.bounding_box

                x = bbox.origin_x
                y = bbox.origin_y
                w = bbox.width
                h = bbox.height

                # ensure coordinates stay inside image
                h_frame, w_frame, _ = frame.shape

                x = max(0, x)
                y = max(0, y)
                w = min(w, w_frame - x)
                h = min(h, h_frame - y)

                # crop the face
                face_crop = frame[y:y+h, x:x+w]

                faces.append((x, y, w, h, face_crop))

        return faces