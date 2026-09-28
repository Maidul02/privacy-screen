from pathlib import Path

import mediapipe as mp


MODEL_PATH = Path("models/face_landmarker.task")




def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options):
        print("Face Landmarker loaded successfully.")


if __name__ == "__main__":
    main()