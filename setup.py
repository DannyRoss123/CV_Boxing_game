"""
Setup script for PunchCV.
Downloads the MediaPipe pose landmarker model and trains the default RF model.

Usage:
    python setup.py
"""

import os
import subprocess
import sys
import urllib.request


TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
    "float16/latest/pose_landmarker_lite.task"
)
TASK_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")


def download_pose_model():
    if os.path.exists(TASK_PATH):
        print(f"[OK] pose_landmarker_lite.task already exists ({os.path.getsize(TASK_PATH) // 1024} KB)")
        return
    print("Downloading pose_landmarker_lite.task ...")
    urllib.request.urlretrieve(TASK_URL, TASK_PATH)
    print(f"[OK] Downloaded to {TASK_PATH}")


def train_models():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    if os.path.exists(rf_path):
        print("[OK] Trained models already exist in models/")
        return
    print("Training models (this takes ~2 minutes) ...")
    subprocess.check_call([sys.executable, os.path.join("scripts", "train_models.py")])
    print("[OK] Models trained and saved to models/")


def main():
    print("=" * 50)
    print("  PunchCV Setup")
    print("=" * 50)
    download_pose_model()
    train_models()
    print("\n[DONE] Setup complete. Run the app:")
    print("  python app.py          (web UI)")
    print("  python scripts/game.py (local Pygame game)")


if __name__ == "__main__":
    main()
