"""
DrowsyDrive
AI Based Driver Drowsiness Detection and Alert System
Made by - Kabira Timir AND Dudhrejiya Manali

Features:
- Webcam capture
- Single-person / driver detection
- MediaPipe Face Landmarker
- Eye landmark extraction
- EAR calculation
- Blink detection
- Drowsiness detection
- Sleep detection
- Hibernate detection
- Sleep alarm
- Session timer

Run:
    python main.py
"""

import time
from datetime import datetime
import winsound
import threading

import cv2
import mediapipe as mp

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    EAR_THRESHOLD,
    DROWSY_SECONDS,
    SLEEP_SECONDS,
)

from eye_detection import LEFT_EYE, RIGHT_EYE, calculate_ear

from mediapipe.tasks.python import vision


# ---------------------------------------------------------
# HIBERNATE SETTING
# ---------------------------------------------------------

# HIBERNATE must be longer than SLEEP.
HIBERNATE_SECONDS = max(SLEEP_SECONDS + 30, 60)

# Additional drowsiness/yawning settings.
# These can be adjusted if your camera or lighting needs tuning.
YAWN_MOUTH_RATIO = 0.42
YAWN_SECONDS = 0.8



# ---------------------------------------------------------
# SINGLE-PERSON FACE DETECTOR
# ---------------------------------------------------------

class FaceDetector:
    """
    Single-person face detector using MediaPipe Face Landmarker.

    Only one face is processed because this project is intended
    for monitoring the driver.
    """

    def __init__(self, model_path="models/face_landmarker.task"):
        base_options = mp_base_options.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame, timestamp_ms):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        return self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

    def close(self):
        self.detector.close()


# ---------------------------------------------------------
# FORMAT TIME
# ---------------------------------------------------------

def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ---------------------------------------------------------
# MOUTH / YAWN DETECTION
# ---------------------------------------------------------

# MediaPipe Face Mesh landmark indices.
# These are used only for a simple mouth-opening/yawning estimate.
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


def calculate_mouth_ratio(landmarks):
    """
    Estimate how open the mouth is.

    Higher value = mouth more open.
    """

    try:
        left = landmarks[MOUTH_LEFT]
        right = landmarks[MOUTH_RIGHT]
        bottom = landmarks[MOUTH_BOTTOM]

        horizontal = (
            (left.x - right.x) ** 2
            + (left.y - right.y) ** 2
        )

        vertical = (
            (top.x - bottom.x) ** 2
            + (top.y - bottom.y) ** 2
        )

        if horizontal <= 0.0001:
            return 0.0


    except (IndexError, AttributeError):
        return 0.0


# ---------------------------------------------------------
# DROWSINESS LEVEL DETECTION
# ---------------------------------------------------------

def get_drowsiness_level(
    eyes_closed,
    closed_duration,
    mouth_ratio,
    yawn_duration,
):
    """
    Returns a more detailed driver state.

    States:
        AWAKE
        EYES CLOSED
        DROWSY
        YAWNING
        SLEEP
        HIBERNATE

    This is a practical webcam heuristic, not a medical diagnosis.
    """

    # Long continuous eye closure has priority over yawning.
    if eyes_closed:

        if closed_duration >= HIBERNATE_SECONDS:
            return "HIBERNATE"

        if closed_duration >= SLEEP_SECONDS:
            return "SLEEP"

        if closed_duration >= DROWSY_SECONDS:
            return "DROWSY"

        return "EYES CLOSED"

    # Mouth open for a sustained period can indicate a yawn.
    if (
        mouth_ratio >= YAWN_MOUTH_RATIO
    ):
        return "YAWNING"

    # Very low EAR for a short period can also be treated as a
    # drowsiness warning even before the full sleep timer.
    if ear < max(EAR_THRESHOLD * 0.92, 0.12):
        return "DROWSY"

    return "AWAKE"


# ---------------------------------------------------------
# ALARM SYSTEM
# ---------------------------------------------------------

alarm_running = False
alarm_thread = None


def alarm_sound():
    """
    Continuously play a Windows beep alarm until stopped.
    """

    global alarm_running

    while alarm_running:
        try:
            winsound.Beep(1200, 400)

            if not alarm_running:
                break

            time.sleep(0.15)

            winsound.Beep(900, 400)

            if not alarm_running:
                break

            time.sleep(0.15)

        except Exception as error:
            print("Alarm sound error:", error)
            break


def start_alarm():
    """
    Start the alarm in a background thread.
    """

    global alarm_running, alarm_thread

    if alarm_running:
        return

    alarm_running = True

    alarm_thread = threading.Thread(
        target=alarm_sound,
        daemon=True
    )

    alarm_thread.start()

    print("ALARM STARTED!")


def stop_alarm():
    """
    Stop the alarm.
    """

    global alarm_running

    if alarm_running:
        alarm_running = False
        print("ALARM STOPPED!")


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def main():

    print("------------------------------------------")
    print(" DrowsyDrive - Driver Drowsiness Detection")
    print("------------------------------------------")

    # -----------------------------------------------------
    # Initialize Face Detector
    # -----------------------------------------------------

    detector = FaceDetector()

    # -----------------------------------------------------
    # Initialize Webcam
    # -----------------------------------------------------

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open the webcam.")
        print("Check camera permissions or change CAMERA_INDEX in config.py.")

        detector.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # -----------------------------------------------------
    # FULL-SCREEN CAMERA WINDOW
    # -----------------------------------------------------

    window_name = "DrowsyDrive - Driver Monitor"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # -----------------------------------------------------
    # Driver State
    # -----------------------------------------------------

    closed_since = None
    was_closed = False
    closed_duration = 0.0
    yawn_since = None
    yawn_duration = 0.0

    status = "NO FACE"

    blink_count = 0
    session_start = time.monotonic()
    frame_timestamp_ms = 0

    print()
    print("DrowsyDrive started.")
    print("Only one driver/person is monitored.")
    print("States: AWAKE -> EYES CLOSED -> DROWSY -> SLEEP -> HIBERNATE")
    print("Yawning is detected from mouth opening.")
    print("Camera preview is set to full screen.")
    print("Press F to toggle full-screen/window mode.")
    print("Press Q to quit.")
    print()

    # -----------------------------------------------------
    # Main Loop
    # -----------------------------------------------------

    try:

        while True:

            # -------------------------------------------------
            # Capture Frame
            # -------------------------------------------------

            ok, frame = cap.read()

            if not ok:
                print("ERROR: Could not read a frame from webcam.")
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

            # -------------------------------------------------
            # Face Detection - ONE PERSON ONLY
            # -------------------------------------------------

            frame_timestamp_ms += 33

            result = detector.process(
                frame,
                frame_timestamp_ms
            )

            h, w = frame.shape[:2]

            # -------------------------------------------------
            # No Face
            # -------------------------------------------------

            if not result.face_landmarks:

                status = "NO FACE"
            
                # If alarm was already active because the driver
                # was sleeping, keep it active.
                # This prevents the alarm from stopping just because
                # the driver temporarily moves out of the frame.

                if closed_since is not None:
                    closed_duration = (
                        time.monotonic() - closed_since
                    )

                # -------------------------------------------------
                # Overall display
                # -------------------------------------------------

                session_time = (
                    time.monotonic()
                    - session_start
                )

                cv2.putText(
                    frame,
                    f"STATUS: {status}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Driver detected: NO",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 204),
                    2
                )

                cv2.putText(
                    frame,
                    f"Closed: {format_time(closed_duration)}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
(255, 255, 204),
                    2
                )

                cv2.putText(
                    frame,
                    f"Session: {format_time(session_time)}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
(255, 255, 204),
                    2
                )

                if alarm_running:

                    cv2.putText(
                        frame,
                        "ALARM ACTIVE",
                        (20, 205),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow(
                    window_name,
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("f"):

                    current_mode = cv2.getWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN
                    )

                    if current_mode == cv2.WINDOW_FULLSCREEN:

                        cv2.setWindowProperty(
                            window_name,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_NORMAL
                        )

                    else:

                        cv2.setWindowProperty(
                            window_name,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN
                        )

                if key == ord("q"):
                    break

                continue

            # -------------------------------------------------
            # Get First / Only Face
            # -------------------------------------------------

            landmarks = result.face_landmarks[0]

            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]

            x1 = max(
                0,
                int(min(xs) * w)
            )

            x2 = min(
                w - 1,
                int(max(xs) * w)
            )

            y1 = max(
                0,
                int(min(ys) * h)
            )

            y2 = min(
                h - 1,
                int(max(ys) * h)
            )

            # -------------------------------------------------
            # Get Eye Landmarks
            # -------------------------------------------------

            left_points = [
                (
                    landmarks[i].x * w,
                    landmarks[i].y * h
                )
                for i in LEFT_EYE
            ]

            right_points = [
                (
                    landmarks[i].x * w,
                    landmarks[i].y * h
                )
                for i in RIGHT_EYE
            ]

            # -------------------------------------------------
            # Calculate EAR
            # -------------------------------------------------

            left_ear = calculate_ear(
                left_points
            )

            right_ear = calculate_ear(
                right_points
            )

            ear = (
                left_ear
                +
                right_ear
            ) / 2.0

            eyes_closed = (
                ear < EAR_THRESHOLD
            )

            now = time.monotonic()

            # -------------------------------------------------
            # Mouth / Yawn Detection
            # -------------------------------------------------

            mouth_ratio = calculate_mouth_ratio(landmarks)

            if mouth_ratio >= YAWN_MOUTH_RATIO and not eyes_closed:

                if yawn_since is None:
                    yawn_since = now

                yawn_duration = (
                    now
                    -
                    yawn_since
                )

            else:

                yawn_since = None
                yawn_duration = 0.0

            # -------------------------------------------------
            # Check Eye State
            # -------------------------------------------------

            if eyes_closed:

                if closed_since is None:
                    closed_since = now

                closed_duration = (
                    now
                    -
                    closed_since
                )

            else:

                # -------------------------------------------------
                # BLINK DETECTION
                # -------------------------------------------------

                if (
                    was_closed
                    and
                    closed_since is not None
                ):

                    duration = (
                        now
                        -
                        closed_since
                    )

                    if duration < DROWSY_SECONDS:
                        blink_count += 1

                # Reset eye state
                closed_since = None
                closed_duration = 0.0

            # -------------------------------------------------
            # DROWSINESS LEVEL
            # -------------------------------------------------

            status = get_drowsiness_level(
                eyes_closed,
                closed_duration,
                mouth_ratio,
                yawn_duration,
                ear
            )

            was_closed = eyes_closed

            # -------------------------------------------------
            # Face Rectangle Color
            # -------------------------------------------------

            if status in ("SLEEP", "HIBERNATE"):

                rectangle_color = (
                    0,
                    0,
                    255
                )

            elif status in ("DROWSY", "YAWNING", "EYES CLOSED"):

                rectangle_color = (
                    0,
                    165,
                    255
                )

            else:

                rectangle_color = (
                    0,
                    255,
                    0
                )

            # -------------------------------------------------
            # Face Rectangle
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                rectangle_color,
                4
            )

            # -------------------------------------------------
            # Driver Label
            # -------------------------------------------------

            label = f"Driver: {status}"

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        30,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                rectangle_color,
                2
            )

            # -------------------------------------------------
            # Eye Rectangles
            # -------------------------------------------------

            left_xs = [
                int(p[0])
                for p in left_points
            ]

            left_ys = [
                int(p[1])
                for p in left_points
            ]

            right_xs = [
                int(p[0])
                for p in right_points
            ]

            right_ys = [
                int(p[1])
                for p in right_points
            ]

            cv2.rectangle(
                frame,
                (
                    min(left_xs) - 5,
                    min(left_ys) - 5
                ),
                (
                    max(left_xs) + 5,
                    max(left_ys) + 5
                ),
                rectangle_color,
                2
            )

            cv2.rectangle(
                frame,
                (
                    min(right_xs) - 5,
                    min(right_ys) - 5
                ),
                (
                    max(right_xs) + 5,
                    max(right_ys) + 5
                ),
                rectangle_color,
                2
            )

            # -----------------------------------------------------
            # ALARM CONTROL
            # -----------------------------------------------------

            if status in ("DROWSY", "SLEEP", "HIBERNATE"):

                start_alarm()

            elif status == "AWAKE":

                stop_alarm()

            # -----------------------------------------------------
            # SESSION TIMER
            # -----------------------------------------------------

            session_time = (
                time.monotonic()
                -
                session_start
            )

            current_datetime = datetime.now()
            current_date = current_datetime.strftime("%d-%m-%Y")
            current_time = current_datetime.strftime("%I:%M:%S %p")

            # -----------------------------------------------------
            # STATUS COLOR
            # -----------------------------------------------------

            if status in ("SLEEP", "HIBERNATE"):

                status_color = (
                    0,
                    0,
                    255
                )

            elif status in ("DROWSY", "YAWNING", "EYES CLOSED"):

                status_color = (
                    0,
                    165,
                    255
                )

            else:

                status_color = (
                    0,
                    128,
                    0
                )

            # -----------------------------------------------------
            # Display Status
            # -----------------------------------------------------

            cv2.putText(
                frame,
                f"STATUS: {status}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                status_color,
                2
            )

            # -----------------------------------------------------
            # Driver Detection
            # -----------------------------------------------------

            cv2.putText(
                frame,
                "Driver detected: YES",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2
            )

            # -----------------------------------------------------
            # Closed Duration
            # -----------------------------------------------------

            cv2.putText(
                frame,
                f"Closed: {format_time(duration)}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
(255, 255, 204),
                2
            )

            # -----------------------------------------------------
            # Session Time
            # -----------------------------------------------------

            cv2.putText(
                frame,
                f"Session: {format_time(time)}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
(255, 255, 204),
                2
            )

            # -----------------------------------------------------
            # Blink Count
            # -----------------------------------------------------

            cv2.putText(
                frame,
                f"Blink events: {count}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
(255, 255, 204),
                2
            )

            cv2.putText(
                frame,
                f"Date: {date}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 204),
                2
            )

            cv2.putText(
                frame,
                f"Time: {time}",
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 204),
                2
            )

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}  |  Mouth: {mouth_ratio:.2f}",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 204),
                2
            )

            cv2.putText(
                frame,
                "Monitoring: Driver / Single Person",
                (20, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 204),
                2
            )

            cv2.putText(
                frame,
                f"Camera: {FRAME_WIDTH}x{FRAME_HEIGHT}",
                (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0,
                (255, 255, 204),
                2
            )

            # -----------------------------------------------------
            # Sleep / Hibernate Warning
            # -----------------------------------------------------

            if status in (
                "EYES CLOSED",
                "DROWSY",
                "YAWNING",
                "SLEEP",
                "HIBERNATE",
            ):

                if status == "HIBERNATE":
                    warning_text = "HIBERNATE - WAKE UP!"
                    warning_color = (0, 0, 255)
                elif status == "SLEEP":
                    warning_text = "SLEEP - WAKE UP!"
                    warning_color = (0, 0, 255)
                elif status == "DROWSY":
                    warning_text = "DROWSY - BE ALERT!"
                    warning_color = (0, 165, 255)
                elif status == "YAWNING":
                    warning_text = "YAWNING - STAY ALERT!"
                    warning_color = (0, 165, 255)
                else:
                    warning_text = "EYES CLOSED"
                    warning_color = (0, 165, 255)

                cv2.putText(
                    frame,
                    warning_text,
                    (0, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    warning_color,
                    3
                )

                if alarm_running:
                    cv2.putText(
                        frame,
                        "ALARM ACTIVE",
                        (20, 395),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2
                    )

            # -----------------------------------------------------
            # Show Camera Window
            # -----------------------------------------------------

            cv2.imshow(
                window_name,
                frame
            )

            # -----------------------------------------------------
            # Keyboard Input
            # -----------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("f"):

                current_mode = cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN
                )

                if current_mode == cv2.WINDOW_FULLSCREEN:

                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL
                    )

                else:

                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN
                    )

    finally:

        # -----------------------------------------------------
        # Stop Alarm
        # -----------------------------------------------------

        stop_alarm()

        # -----------------------------------------------------
        # Release Camera
        # -----------------------------------------------------

        cap.release()

        # -----------------------------------------------------
        # Close Face Detector
        # -----------------------------------------------------

        detector.close()

        # -----------------------------------------------------
        # Close OpenCV Windows
        # -----------------------------------------------------

        cv2.destroyAllWindows()

        print()
        print("DrowsyDrive stopped.")
        print("Thank you!")


# ---------------------------------------------------------
# PROGRAM START
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
