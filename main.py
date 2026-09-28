import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path("models/face_landmarker.task")


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

            # OpenCV provides frames in BGR format.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Wrap the NumPy RGB image for MediaPipe.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # VIDEO mode requires a monotonically increasing timestamp.
            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )
            

            # Draw each detected landmark as a small point.
            if result.face_landmarks:
                frame_height, frame_width = frame.shape[:2]

                for face_landmarks in result.face_landmarks:
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

            cv2.imshow("Privacy Screen - Face Landmarks", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()