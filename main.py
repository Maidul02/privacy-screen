import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = Path("models/face_landmarker.task")


# MediaPipe landmark indices used for head-pose estimation.
HEAD_POSE_LANDMARKS = {
    "nose": 1,
    "chin": 199,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


# MediaPipe landmark indices used for eye-position estimation.
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


# Approximate 3D reference model of the selected facial points.
# The order must match HEAD_POSE_LANDMARKS.
FACE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye
        (43.3, 32.7, -26.0),      # Right eye
        (-28.9, -28.9, -24.1),    # Left mouth
        (28.9, -28.9, -24.1),     # Right mouth
    ],
    dtype=np.float64,
)


def get_head_pose_image_points(
    face_landmarks,
    frame_width,
    frame_height,
):
    """Convert selected normalized MediaPipe landmarks to pixel coordinates."""
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
    """Convert a rotation matrix to yaw, pitch, and roll angles."""
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
    """Shift the current pitch convention to a centre-relative range."""
    shifted_pitch = pitch - 180.0

    return (shifted_pitch + 180.0) % 360.0 - 180.0


def calculate_horizontal_eye_ratio(
    face_landmarks,
    eye_landmarks,
):
    """Calculate iris position relative to the horizontal eye corners."""
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
    """Calculate iris position relative to the vertical eye boundaries."""
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
    """Estimate yaw, normalized pitch, and roll from facial landmarks."""
    image_points = get_head_pose_image_points(
        face_landmarks,
        frame_width,
        frame_height,
    )

    camera_matrix = create_camera_matrix(
        frame_width,
        frame_height,
    )

    # For this prototype, assume no lens distortion.
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
    print("Press Q to quit.")

    previous_time = time.perf_counter()
    start_time = time.perf_counter()

    # Centre-calibration state.
    calibration_active = False
    calibration_target_samples = 90
    calibration_samples = []
    centre_calibration = None

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read frame from webcam.")
                break

            # OpenCV provides webcam frames in BGR format.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # Convert the NumPy RGB frame to a MediaPipe image.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # VIDEO mode requires increasing timestamps in milliseconds.
            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            frame_height, frame_width = frame.shape[:2]

            if result.face_landmarks:
                # num_faces=1, so use the first detected face.
                face_landmarks = result.face_landmarks[0]

                # ---------------------------------------------------------
                # Horizontal eye-position estimation
                # ---------------------------------------------------------

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
                    # The two eyes use opposite horizontal directions.
                    # Flip the right-eye ratio before combining them.
                    normalized_right_eye_ratio = (
                        1.0 - right_eye_ratio
                    )

                    combined_eye_ratio = (
                        left_eye_ratio
                        + normalized_right_eye_ratio
                    ) / 2

                # ---------------------------------------------------------
                # Vertical eye-position estimation
                # ---------------------------------------------------------

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

                # ---------------------------------------------------------
                # Draw all facial landmarks
                # ---------------------------------------------------------

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

                # ---------------------------------------------------------
                # Highlight iris landmarks
                # ---------------------------------------------------------

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

                # ---------------------------------------------------------
                # Head-pose estimation
                # ---------------------------------------------------------

                head_pose = estimate_head_pose(
                    face_landmarks,
                    frame_width,
                    frame_height,
                )

                if head_pose is not None:
                    yaw, pitch, roll = head_pose

                    # -----------------------------------------------------
                    # Centre calibration
                    # -----------------------------------------------------

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
                            sample_count = len(calibration_samples)

                            centre_calibration = {
                                "yaw": sum(
                                    sample["yaw"]
                                    for sample in calibration_samples
                                ) / sample_count,

                                "pitch": sum(
                                    sample["pitch"]
                                    for sample in calibration_samples
                                ) / sample_count,

                                "horizontal_eye": sum(
                                    sample["horizontal_eye"]
                                    for sample in calibration_samples
                                ) / sample_count,

                                "vertical_eye": sum(
                                    sample["vertical_eye"]
                                    for sample in calibration_samples
                                ) / sample_count,
                            }

                            calibration_active = False

                            print("\nCentre calibration complete:")
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

                    # Draw a simple visual head-direction indicator.
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

                # ---------------------------------------------------------
                # Display eye-position measurements
                # ---------------------------------------------------------

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

            # -------------------------------------------------------------
            # Calibration status
            # -------------------------------------------------------------

            if calibration_active:
                cv2.putText(
                    frame,
                    (
                        "CALIBRATING CENTRE: "
                        f"{len(calibration_samples)}/"
                        f"{calibration_target_samples}"
                    ),
                    (20, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            elif centre_calibration is not None:
                cv2.putText(
                    frame,
                    "CENTRE CALIBRATED",
                    (20, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # -------------------------------------------------------------
            # FPS calculation
            # -------------------------------------------------------------

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

            cv2.imshow(
                "Privacy Screen - Head and Eye Tracking",
                frame,
            )

            # -------------------------------------------------------------
            # Keyboard controls
            # -------------------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                calibration_samples.clear()
                centre_calibration = None
                calibration_active = True

                print(
                    "\nCentre calibration started. "
                    "Look naturally at the centre of the screen."
                )

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()