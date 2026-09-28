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


def get_head_pose_image_points(face_landmarks, frame_width, frame_height):
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


def estimate_head_pose(face_landmarks, frame_width, frame_height):
    """Estimate raw yaw, pitch, and roll from facial landmarks."""
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

    print("Webcam started. Press Q to quit.")

    previous_time = time.perf_counter()
    start_time = time.perf_counter()

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
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

                # Draw all facial landmarks.
                for landmark in face_landmarks:
                    pixel_x = int(landmark.x * frame_width)
                    pixel_y = int(landmark.y * frame_height)

                    cv2.circle(
                        frame,
                        (pixel_x, pixel_y),
                        1,
                        (0, 255, 0),
                        -1,
                    )

                # Estimate raw head orientation.
                head_pose = estimate_head_pose(
                    face_landmarks,
                    frame_width,
                    frame_height,
                )

                if head_pose is not None:
                    yaw, pitch, roll = head_pose

                    nose_landmark = face_landmarks[HEAD_POSE_LANDMARKS["nose"]]
                    nose_x = int(nose_landmark.x * frame_width)
                    nose_y = int(nose_landmark.y * frame_height)

                    line_length = 100

                    direction_x = int(nose_x + line_length * math.sin(math.radians(yaw)))
                    direction_y = int(nose_y - line_length * math.sin(math.radians(pitch)))

                    cv2.line(frame,(nose_x, nose_y),(direction_x, direction_y),(255, 0, 0),2,)

                    

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

            # Calculate approximate instantaneous FPS.
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
                "Privacy Screen - Head Pose",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()