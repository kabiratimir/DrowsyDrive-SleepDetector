from pathlib import Path

import mediapipe as mp


class FaceDetector:
    """
    MediaPipe Face Landmarker wrapper.

    The first run downloads the official MediaPipe face-landmarker task model
    if it is not already present in models/.
    """

    def __init__(self):
        self.model_path = Path("models/face_landmarker.task")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Missing models/face_landmarker.task. "
                "See README.md for the model download step."
            )

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def process(self, bgr_frame):
        # MediaPipe expects an RGB image.
        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=__import__("cv2").cvtColor(bgr_frame, __import__("cv2").COLOR_BGR2RGB),
        )
        result = self.landmarker.detect(rgb)

        if not result.face_landmarks:
            return None

        # Convert normalized landmarks to simple (x, y, z) tuples.
        landmarks = [
            (lm.x, lm.y, lm.z)
            for lm in result.face_landmarks[0]
        ]
        return [landmarks]

    def close(self):
        self.landmarker.close()
