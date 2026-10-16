# Central configuration for the Student Sleep Detector.

CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# EAR below this value is treated as eye closure.
# This is only a starting point and should be calibrated for the user/camera.
EAR_THRESHOLD = 0.21

# Eye-closure duration thresholds.
DROWSY_SECONDS = 1.0
SLEEP_SECONDS = 3.0

# Later phases will add alarm and GUI settings here.
ALARM_ENABLED = True
ALARM_SECONDS = 3.0
