import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = Path("models/face_landmarker.task")


HEAD_POSE_LANDMARKS = {
    "nose": 1,
    "chin": 199,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


LEFT_EYE = {
    "outer": 33,
    "inner": 133,
    "top": 159,
    "bottom": 145,
    "iris": 468,
}

RIGHT_EYE = {
    "outer": 263,
    "inner": 362,
    "top": 386,
    "bottom": 374,
    "iris": 473,
}


FACE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


def get_head_pose_image_points(
    face_landmarks,
    frame_width,
    frame_height,
):
    """Convert selected normalized landmarks to pixel coordinates."""
    image_points = []

    for landmark_index in HEAD_POSE_LANDMARKS.values():
        landmark = face_landmarks[landmark_index]

        pixel_x = landmark.x * frame_width
        pixel_y = landmark.y * frame_height

        image_points.append((pixel_x, pixel_y))

    return np.array(image_points, dtype=np.float64)


def create_camera_matrix(frame_width, frame_height):
    """Create an approximate camera intrinsic matrix."""
    focal_length = frame_width

    center_x = frame_width / 2
    center_y = frame_height / 2

    return np.array(
        [
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_euler_angles(rotation_matrix):
    """Convert a rotation matrix to yaw, pitch, and roll."""
    sy = math.sqrt(
        rotation_matrix[0, 0] ** 2
        + rotation_matrix[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(
            rotation_matrix[2, 1],
            rotation_matrix[2, 2],
        )

        yaw = math.atan2(
            -rotation_matrix[2, 0],
            sy,
        )

        roll = math.atan2(
            rotation_matrix[1, 0],
            rotation_matrix[0, 0],
        )

    else:
        pitch = math.atan2(
            -rotation_matrix[1, 2],
            rotation_matrix[1, 1],
        )

        yaw = math.atan2(
            -rotation_matrix[2, 0],
            sy,
        )

        roll = 0.0

    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )


def normalize_pitch(pitch):
    """Shift pitch to the centre-relative angle convention."""
    shifted_pitch = pitch - 180.0

    return (shifted_pitch + 180.0) % 360.0 - 180.0


def calculate_horizontal_eye_ratio(
    face_landmarks,
    eye_landmarks,
):
    """Calculate iris position relative to horizontal eye corners."""
    outer = face_landmarks[eye_landmarks["outer"]]
    inner = face_landmarks[eye_landmarks["inner"]]
    iris = face_landmarks[eye_landmarks["iris"]]

    eye_width = inner.x - outer.x

    if abs(eye_width) < 1e-6:
        return None

    return (iris.x - outer.x) / eye_width


def calculate_vertical_eye_ratio(
    face_landmarks,
    eye_landmarks,
):
    """Calculate iris position relative to vertical eye boundaries."""
    top = face_landmarks[eye_landmarks["top"]]
    bottom = face_landmarks[eye_landmarks["bottom"]]
    iris = face_landmarks[eye_landmarks["iris"]]

    eye_height = bottom.y - top.y

    if abs(eye_height) < 1e-6:
        return None

    return (iris.y - top.y) / eye_height


def estimate_head_pose(
    face_landmarks,
    frame_width,
    frame_height,
):
    """Estimate yaw, normalized pitch, and roll."""
    image_points = get_head_pose_image_points(
        face_landmarks,
        frame_width,
        frame_height,
    )

    camera_matrix = create_camera_matrix(
        frame_width,
        frame_height,
    )

    distortion_coefficients = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    success, rotation_vector, _ = cv2.solvePnP(
        FACE_MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    yaw, pitch, roll = rotation_matrix_to_euler_angles(
        rotation_matrix
    )

    pitch = normalize_pitch(pitch)

    return yaw, pitch, roll


def classify_attention(
    relative_yaw,
    relative_horizontal_eye,
    left_boundary,
    right_boundary,
):
    """
    Classify one frame as LOOKING, UNCERTAIN, or AWAY.

    Horizontal head-pose and eye-position evidence are used.
    """

    if left_boundary is None or right_boundary is None:
        return "NOT CALIBRATED"

    if (
        relative_yaw is None
        or relative_horizontal_eye is None
    ):
        return "UNCERTAIN"

    horizontal_yaw_range = (
        right_boundary["yaw"]
        - left_boundary["yaw"]
    )

    horizontal_eye_range = (
        left_boundary["horizontal_eye"]
        - right_boundary["horizontal_eye"]
    )

    # Personalized tolerance based on the calibrated screen range.
    yaw_tolerance = abs(horizontal_yaw_range) * 0.20
    eye_tolerance = abs(horizontal_eye_range) * 0.20

    left_yaw_limit = (
        left_boundary["yaw"]
        - yaw_tolerance
    )

    right_yaw_limit = (
        right_boundary["yaw"]
        + yaw_tolerance
    )

    left_eye_limit = (
        left_boundary["horizontal_eye"]
        + eye_tolerance
    )

    right_eye_limit = (
        right_boundary["horizontal_eye"]
        - eye_tolerance
    )

    head_left_away = relative_yaw < left_yaw_limit
    head_right_away = relative_yaw > right_yaw_limit

    eye_left_away = (
        relative_horizontal_eye > left_eye_limit
    )

    eye_right_away = (
        relative_horizontal_eye < right_eye_limit
    )

    # Strong AWAY evidence requires head and eyes
    # to agree that the user is outside the screen.
    if head_left_away and eye_left_away:
        return "AWAY"

    if head_right_away and eye_right_away:
        return "AWAY"

    # One signal crossing the boundary is not enough
    # for a confident AWAY decision.
    if (
        head_left_away
        or head_right_away
        or eye_left_away
        or eye_right_away
    ):
        return "UNCERTAIN"

    return "LOOKING"


def average_calibration_samples(samples):
    """Calculate average values from calibration samples."""
    sample_count = len(samples)

    return {
        "yaw": sum(
            sample["yaw"] for sample in samples
        ) / sample_count,

        "pitch": sum(
            sample["pitch"] for sample in samples
        ) / sample_count,

        "horizontal_eye": sum(
            sample["horizontal_eye"] for sample in samples
        ) / sample_count,

        "vertical_eye": sum(
            sample["vertical_eye"] for sample in samples
        ) / sample_count,
    }


def create_privacy_frame(width, height):
    """Create a simple black privacy-screen image."""
    privacy_frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    message = "PRIVACY MODE"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    text_size, _ = cv2.getTextSize(
        message,
        font,
        font_scale,
        thickness,
    )

    text_width, text_height = text_size

    text_x = (width - text_width) // 2
    text_y = (height + text_height) // 2

    cv2.putText(
        privacy_frame,
        message,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )

    return privacy_frame


def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found: {MODEL_PATH}")
        return

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started.")
    print("Press C to calibrate the screen centre.")
    print("Press L to calibrate the left screen boundary.")
    print("Press R to calibrate the right screen boundary.")
    print("Press T to calibrate the top screen boundary.")
    print("Press B to calibrate the bottom screen boundary.")
    print("Press Q to quit.")

    previous_time = time.perf_counter()
    start_time = time.perf_counter()

    calibration_target_samples = 90

    # ---------------------------------------------------------
    # Centre calibration
    # ---------------------------------------------------------

    calibration_active = False
    calibration_samples = []
    centre_calibration = None

    # ---------------------------------------------------------
    # Boundary calibration
    # ---------------------------------------------------------

    left_calibration_active = False
    left_calibration_samples = []
    left_boundary = None

    right_calibration_active = False
    right_calibration_samples = []
    right_boundary = None

    top_calibration_active = False
    top_calibration_samples = []
    top_boundary = None

    bottom_calibration_active = False
    bottom_calibration_samples = []
    bottom_boundary = None

    # ---------------------------------------------------------
    # Temporal filtering
    # ---------------------------------------------------------

    stable_attention_state = "LOOKING"

    away_started_at = None
    looking_started_at = None
    no_face_started_at = None

    away_confirmation_seconds = 0.75
    looking_confirmation_seconds = 0.30
    no_face_confirmation_seconds = 1.00

    # ---------------------------------------------------------
    # Privacy overlay
    # ---------------------------------------------------------

    privacy_window_name = "Privacy Screen"
    privacy_window_visible = False

    # Temporary prototype dimensions.
    # OpenCV's fullscreen window will stretch this buffer
    # to the display.
    privacy_frame = create_privacy_frame(
        1920,
        1080,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read frame from webcam.")
                break

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            frame_height, frame_width = frame.shape[:2]

            # Per-frame measurements.
            relative_yaw = None
            relative_pitch = None
            relative_horizontal_eye = None
            relative_vertical_eye = None

            attention_state = "NOT CALIBRATED"

            # =====================================================
            # FACE DETECTED
            # =====================================================

            if result.face_landmarks:
                no_face_started_at = None

                face_landmarks = result.face_landmarks[0]

                # -------------------------------------------------
                # Horizontal eye position
                # -------------------------------------------------

                left_eye_ratio = calculate_horizontal_eye_ratio(
                    face_landmarks,
                    LEFT_EYE,
                )

                right_eye_ratio = calculate_horizontal_eye_ratio(
                    face_landmarks,
                    RIGHT_EYE,
                )

                combined_eye_ratio = None

                if (
                    left_eye_ratio is not None
                    and right_eye_ratio is not None
                ):
                    normalized_right_eye_ratio = (
                        1.0 - right_eye_ratio
                    )

                    combined_eye_ratio = (
                        left_eye_ratio
                        + normalized_right_eye_ratio
                    ) / 2

                # -------------------------------------------------
                # Vertical eye position
                # -------------------------------------------------

                left_vertical_ratio = calculate_vertical_eye_ratio(
                    face_landmarks,
                    LEFT_EYE,
                )

                right_vertical_ratio = calculate_vertical_eye_ratio(
                    face_landmarks,
                    RIGHT_EYE,
                )

                combined_vertical_ratio = None

                if (
                    left_vertical_ratio is not None
                    and right_vertical_ratio is not None
                ):
                    combined_vertical_ratio = (
                        left_vertical_ratio
                        + right_vertical_ratio
                    ) / 2

                # -------------------------------------------------
                # Draw landmarks
                # -------------------------------------------------

                for landmark in face_landmarks:
                    pixel_x = int(
                        landmark.x * frame_width
                    )

                    pixel_y = int(
                        landmark.y * frame_height
                    )

                    cv2.circle(
                        frame,
                        (pixel_x, pixel_y),
                        1,
                        (0, 255, 0),
                        -1,
                    )

                for iris_index in (
                    LEFT_EYE["iris"],
                    RIGHT_EYE["iris"],
                ):
                    iris_landmark = face_landmarks[iris_index]

                    iris_x = int(
                        iris_landmark.x * frame_width
                    )

                    iris_y = int(
                        iris_landmark.y * frame_height
                    )

                    cv2.circle(
                        frame,
                        (iris_x, iris_y),
                        4,
                        (0, 0, 255),
                        -1,
                    )

                # -------------------------------------------------
                # Head pose
                # -------------------------------------------------

                head_pose = estimate_head_pose(
                    face_landmarks,
                    frame_width,
                    frame_height,
                )

                if head_pose is not None:
                    yaw, pitch, roll = head_pose

                    # ---------------------------------------------
                    # Centre calibration
                    # ---------------------------------------------

                    if (
                        calibration_active
                        and combined_eye_ratio is not None
                        and combined_vertical_ratio is not None
                    ):
                        calibration_samples.append(
                            {
                                "yaw": yaw,
                                "pitch": pitch,
                                "horizontal_eye": combined_eye_ratio,
                                "vertical_eye": combined_vertical_ratio,
                            }
                        )

                        if (
                            len(calibration_samples)
                            >= calibration_target_samples
                        ):
                            centre_calibration = (
                                average_calibration_samples(
                                    calibration_samples
                                )
                            )

                            calibration_active = False

                            print(
                                "\nCentre calibration complete:"
                            )
                            print(
                                "Yaw: "
                                f"{centre_calibration['yaw']:.2f}"
                            )
                            print(
                                "Pitch: "
                                f"{centre_calibration['pitch']:.2f}"
                            )
                            print(
                                "Horizontal eye: "
                                f"{centre_calibration['horizontal_eye']:.3f}"
                            )
                            print(
                                "Vertical eye: "
                                f"{centre_calibration['vertical_eye']:.3f}"
                            )

                    # ---------------------------------------------
                    # Relative measurements
                    # ---------------------------------------------

                    if centre_calibration is not None:
                        relative_yaw = (
                            yaw
                            - centre_calibration["yaw"]
                        )

                        relative_pitch = (
                            pitch
                            - centre_calibration["pitch"]
                        )

                        if combined_eye_ratio is not None:
                            relative_horizontal_eye = (
                                combined_eye_ratio
                                - centre_calibration[
                                    "horizontal_eye"
                                ]
                            )

                        if combined_vertical_ratio is not None:
                            relative_vertical_eye = (
                                combined_vertical_ratio
                                - centre_calibration[
                                    "vertical_eye"
                                ]
                            )

                    # ---------------------------------------------
                    # Left calibration
                    # ---------------------------------------------

                    if (
                        left_calibration_active
                        and relative_yaw is not None
                        and relative_pitch is not None
                        and relative_horizontal_eye is not None
                        and relative_vertical_eye is not None
                    ):
                        left_calibration_samples.append(
                            {
                                "yaw": relative_yaw,
                                "pitch": relative_pitch,
                                "horizontal_eye": relative_horizontal_eye,
                                "vertical_eye": relative_vertical_eye,
                            }
                        )

                        if (
                            len(left_calibration_samples)
                            >= calibration_target_samples
                        ):
                            left_boundary = (
                                average_calibration_samples(
                                    left_calibration_samples
                                )
                            )

                            left_calibration_active = False

                            print(
                                "\nLEFT screen boundary calibrated:"
                            )
                            print(
                                "Relative yaw: "
                                f"{left_boundary['yaw']:.2f}"
                            )
                            print(
                                "Relative horizontal eye: "
                                f"{left_boundary['horizontal_eye']:.3f}"
                            )

                    # ---------------------------------------------
                    # Right calibration
                    # ---------------------------------------------

                    if (
                        right_calibration_active
                        and relative_yaw is not None
                        and relative_pitch is not None
                        and relative_horizontal_eye is not None
                        and relative_vertical_eye is not None
                    ):
                        right_calibration_samples.append(
                            {
                                "yaw": relative_yaw,
                                "pitch": relative_pitch,
                                "horizontal_eye": relative_horizontal_eye,
                                "vertical_eye": relative_vertical_eye,
                            }
                        )

                        if (
                            len(right_calibration_samples)
                            >= calibration_target_samples
                        ):
                            right_boundary = (
                                average_calibration_samples(
                                    right_calibration_samples
                                )
                            )

                            right_calibration_active = False

                            print(
                                "\nRIGHT screen boundary calibrated:"
                            )
                            print(
                                "Relative yaw: "
                                f"{right_boundary['yaw']:.2f}"
                            )
                            print(
                                "Relative horizontal eye: "
                                f"{right_boundary['horizontal_eye']:.3f}"
                            )

                    # ---------------------------------------------
                    # Top calibration
                    # ---------------------------------------------

                    if (
                        top_calibration_active
                        and relative_yaw is not None
                        and relative_pitch is not None
                        and relative_horizontal_eye is not None
                        and relative_vertical_eye is not None
                    ):
                        top_calibration_samples.append(
                            {
                                "yaw": relative_yaw,
                                "pitch": relative_pitch,
                                "horizontal_eye": relative_horizontal_eye,
                                "vertical_eye": relative_vertical_eye,
                            }
                        )

                        if (
                            len(top_calibration_samples)
                            >= calibration_target_samples
                        ):
                            top_boundary = (
                                average_calibration_samples(
                                    top_calibration_samples
                                )
                            )

                            top_calibration_active = False

                            print(
                                "\nTOP screen boundary calibrated:"
                            )
                            print(
                                "Relative pitch: "
                                f"{top_boundary['pitch']:.2f}"
                            )
                            print(
                                "Relative vertical eye: "
                                f"{top_boundary['vertical_eye']:.3f}"
                            )

                    # ---------------------------------------------
                    # Bottom calibration
                    # ---------------------------------------------

                    if (
                        bottom_calibration_active
                        and relative_yaw is not None
                        and relative_pitch is not None
                        and relative_horizontal_eye is not None
                        and relative_vertical_eye is not None
                    ):
                        bottom_calibration_samples.append(
                            {
                                "yaw": relative_yaw,
                                "pitch": relative_pitch,
                                "horizontal_eye": relative_horizontal_eye,
                                "vertical_eye": relative_vertical_eye,
                            }
                        )

                        if (
                            len(bottom_calibration_samples)
                            >= calibration_target_samples
                        ):
                            bottom_boundary = (
                                average_calibration_samples(
                                    bottom_calibration_samples
                                )
                            )

                            bottom_calibration_active = False

                            print(
                                "\nBOTTOM screen boundary calibrated:"
                            )
                            print(
                                "Relative pitch: "
                                f"{bottom_boundary['pitch']:.2f}"
                            )
                            print(
                                "Relative vertical eye: "
                                f"{bottom_boundary['vertical_eye']:.3f}"
                            )

                    # ---------------------------------------------
                    # Raw classification
                    # ---------------------------------------------

                    attention_state = classify_attention(
                        relative_yaw,
                        relative_horizontal_eye,
                        left_boundary,
                        right_boundary,
                    )

                    # ---------------------------------------------
                    # Temporal filtering
                    # ---------------------------------------------

                    if attention_state == "AWAY":
                        looking_started_at = None

                        if away_started_at is None:
                            away_started_at = time.perf_counter()

                        away_duration = (
                            time.perf_counter()
                            - away_started_at
                        )

                        if (
                            away_duration
                            >= away_confirmation_seconds
                        ):
                            stable_attention_state = "AWAY"

                    elif attention_state == "LOOKING":
                        away_started_at = None

                        if stable_attention_state == "LOOKING":
                            looking_started_at = None

                        else:
                            if looking_started_at is None:
                                looking_started_at = (
                                    time.perf_counter()
                                )

                            looking_duration = (
                                time.perf_counter()
                                - looking_started_at
                            )

                            if (
                                looking_duration
                                >= looking_confirmation_seconds
                            ):
                                stable_attention_state = "LOOKING"
                                looking_started_at = None

                    else:
                        away_started_at = None
                        looking_started_at = None

                    # ---------------------------------------------
                    # Diagnostic head-direction line
                    # ---------------------------------------------

                    nose_landmark = face_landmarks[
                        HEAD_POSE_LANDMARKS["nose"]
                    ]

                    nose_x = int(
                        nose_landmark.x * frame_width
                    )

                    nose_y = int(
                        nose_landmark.y * frame_height
                    )

                    line_length = 100

                    direction_x = int(
                        nose_x
                        + line_length
                        * math.sin(math.radians(yaw))
                    )

                    direction_y = int(
                        nose_y
                        - line_length
                        * math.sin(math.radians(pitch))
                    )

                    cv2.line(
                        frame,
                        (nose_x, nose_y),
                        (direction_x, direction_y),
                        (255, 0, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Yaw: {yaw:6.1f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Pitch: {pitch:6.1f}",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Roll: {roll:6.1f}",
                        (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                # -------------------------------------------------
                # Eye measurement display
                # -------------------------------------------------

                if left_eye_ratio is not None:
                    cv2.putText(
                        frame,
                        f"Left eye: {left_eye_ratio:.3f}",
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                if right_eye_ratio is not None:
                    cv2.putText(
                        frame,
                        f"Right eye: {right_eye_ratio:.3f}",
                        (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                if combined_eye_ratio is not None:
                    cv2.putText(
                        frame,
                        f"Combined eye: {combined_eye_ratio:.3f}",
                        (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                if combined_vertical_ratio is not None:
                    cv2.putText(
                        frame,
                        f"Vertical eye: {combined_vertical_ratio:.3f}",
                        (20, 270),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

            # =====================================================
            # NO FACE DETECTED
            # =====================================================

            else:
                attention_state = "NO FACE"

                away_started_at = None
                looking_started_at = None

                if no_face_started_at is None:
                    no_face_started_at = time.perf_counter()

                no_face_duration = (
                    time.perf_counter()
                    - no_face_started_at
                )

                if (
                    no_face_duration
                    >= no_face_confirmation_seconds
                ):
                    stable_attention_state = "AWAY"

            # -----------------------------------------------------
            # Centre calibration status
            # -----------------------------------------------------

            if calibration_active:
                centre_status = (
                    "CALIBRATING CENTRE: "
                    f"{len(calibration_samples)}/"
                    f"{calibration_target_samples}"
                )

            elif centre_calibration is not None:
                centre_status = "CENTRE CALIBRATED"

            else:
                centre_status = "CENTRE NOT CALIBRATED"

            cv2.putText(
                frame,
                centre_status,
                (20, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            # -----------------------------------------------------
            # Relative measurements
            # -----------------------------------------------------

            if (
                centre_calibration is not None
                and relative_yaw is not None
                and relative_pitch is not None
            ):
                cv2.putText(
                    frame,
                    f"Rel yaw: {relative_yaw:+.1f}",
                    (20, 335),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

                cv2.putText(
                    frame,
                    f"Rel pitch: {relative_pitch:+.1f}",
                    (20, 355),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

                if relative_horizontal_eye is not None:
                    cv2.putText(
                        frame,
                        (
                            "Rel H-eye: "
                            f"{relative_horizontal_eye:+.3f}"
                        ),
                        (20, 375),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

                if relative_vertical_eye is not None:
                    cv2.putText(
                        frame,
                        (
                            "Rel V-eye: "
                            f"{relative_vertical_eye:+.3f}"
                        ),
                        (20, 395),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

            # -----------------------------------------------------
            # Boundary status
            # -----------------------------------------------------

            boundary_status = (
                f"L:{'Y' if left_boundary is not None else 'N'} "
                f"R:{'Y' if right_boundary is not None else 'N'} "
                f"T:{'Y' if top_boundary is not None else 'N'} "
                f"B:{'Y' if bottom_boundary is not None else 'N'}"
            )

            cv2.putText(
                frame,
                boundary_status,
                (20, 415),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # -----------------------------------------------------
            # Raw state
            # -----------------------------------------------------

            if attention_state == "LOOKING":
                raw_color = (0, 255, 0)

            elif attention_state == "AWAY":
                raw_color = (0, 0, 255)

            elif attention_state == "NO FACE":
                raw_color = (0, 165, 255)

            else:
                raw_color = (0, 255, 255)

            cv2.putText(
                frame,
                f"RAW: {attention_state}",
                (20, 440),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                raw_color,
                2,
            )

            # -----------------------------------------------------
            # Stable state
            # -----------------------------------------------------

            if stable_attention_state == "LOOKING":
                stable_color = (0, 255, 0)

            else:
                stable_color = (0, 0, 255)

            cv2.putText(
                frame,
                f"STABLE: {stable_attention_state}",
                (250, 440),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                stable_color,
                2,
            )

            # -----------------------------------------------------
            # FPS
            # -----------------------------------------------------

            current_time = time.perf_counter()
            elapsed_time = current_time - previous_time

            if elapsed_time > 0:
                fps = 1 / elapsed_time
            else:
                fps = 0.0

            previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # =====================================================
            # BASIC PRIVACY OVERLAY
            # =====================================================

            if stable_attention_state == "AWAY":
                if not privacy_window_visible:
                    cv2.namedWindow(
                        privacy_window_name,
                        cv2.WINDOW_NORMAL,
                    )

                    cv2.setWindowProperty(
                        privacy_window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )

                    privacy_window_visible = True

                cv2.imshow(
                    privacy_window_name,
                    privacy_frame,
                )

            else:
                if privacy_window_visible:
                    try:
                        cv2.destroyWindow(
                            privacy_window_name
                        )
                    except cv2.error:
                        pass

                    privacy_window_visible = False

            # -----------------------------------------------------
            # Webcam debug window
            # -----------------------------------------------------

            cv2.imshow(
                "Privacy Screen - Head and Eye Tracking",
                frame,
            )

            # -----------------------------------------------------
            # Keyboard controls
            # -----------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                calibration_samples.clear()
                centre_calibration = None

                left_calibration_active = False
                left_calibration_samples.clear()
                left_boundary = None

                right_calibration_active = False
                right_calibration_samples.clear()
                right_boundary = None

                top_calibration_active = False
                top_calibration_samples.clear()
                top_boundary = None

                bottom_calibration_active = False
                bottom_calibration_samples.clear()
                bottom_boundary = None

                stable_attention_state = "LOOKING"

                away_started_at = None
                looking_started_at = None
                no_face_started_at = None

                calibration_active = True

                print(
                    "\nCentre calibration started. "
                    "Look naturally at the centre of the screen."
                )

            if key == ord("l"):
                if centre_calibration is None:
                    print(
                        "\nCalibrate the centre first by pressing C."
                    )

                else:
                    left_calibration_samples.clear()
                    left_boundary = None
                    left_calibration_active = True

                    print(
                        "\nLEFT boundary calibration started. "
                        "Look naturally at the left edge of the screen."
                    )

            if key == ord("r"):
                if centre_calibration is None:
                    print(
                        "\nCalibrate the centre first by pressing C."
                    )

                else:
                    right_calibration_samples.clear()
                    right_boundary = None
                    right_calibration_active = True

                    print(
                        "\nRIGHT boundary calibration started. "
                        "Look naturally at the right edge of the screen."
                    )

            if key == ord("t"):
                if centre_calibration is None:
                    print(
                        "\nCalibrate the centre first by pressing C."
                    )

                else:
                    top_calibration_samples.clear()
                    top_boundary = None
                    top_calibration_active = True

                    print(
                        "\nTOP boundary calibration started. "
                        "Look naturally at the top edge of the screen."
                    )

            if key == ord("b"):
                if centre_calibration is None:
                    print(
                        "\nCalibrate the centre first by pressing C."
                    )

                else:
                    bottom_calibration_samples.clear()
                    bottom_boundary = None
                    bottom_calibration_active = True

                    print(
                        "\nBOTTOM boundary calibration started. "
                        "Look naturally at the bottom edge of the screen."
                    )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()