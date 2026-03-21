import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

class FaceLandmarks:

    def __init__(self):
        # Resolve path relative to this file
        model_path = Path(__file__).resolve().parent.parent / "face_landmarker.task"

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract_landmarks(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = []

        h, w, _ = frame.shape

        for lm in result.face_landmarks[0]:

            x = int(lm.x * w)
            y = int(lm.y * h)

            landmarks.append((x, y))

        return landmarks