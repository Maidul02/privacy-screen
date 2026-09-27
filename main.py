import time

import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press Q to quit.")

    previous_time = time.perf_counter()

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame from webcam.")
            break

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

        cv2.imshow("Privacy Screen - Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()