"""
Data collection tool for boxing action recognition.

Usage:
    python scripts/collect_data.py --action jab
    python scripts/collect_data.py --action block_high
    python scripts/collect_data.py --action idle

Controls:
    SPACE  - Start/stop recording a sample
    Q/ESC  - Quit

Each sample records ~1-2 seconds of pose landmarks (30 frames at ~30fps).
Saved as .npy files in data/raw/<action_name>/
"""

import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

ACTIONS = [
    "idle",
    "jab",
    "cross",
    "hook",
    "uppercut",
    "block_high",
    "block_low",
    "front_kick",
]

SEQUENCE_LENGTH = 60  # frames per sample (~2 seconds at 30fps)


def landmarks_to_array(landmarks) -> np.ndarray:
    """Convert a list of 33 landmarks to a flat array of [x, y, z, visibility] * 33 = 132 values."""
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(row, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Record pose data for an action class.")
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=ACTIONS,
        help=f"Action to record. Options: {ACTIONS}",
    )
    args = parser.parse_args()

    action = args.action
    save_dir = os.path.join(DATA_DIR, action)
    os.makedirs(save_dir, exist_ok=True)

    # Count existing samples to continue numbering.
    existing = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    sample_count = len(existing)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    recording = False
    countdown_active = False
    countdown_start = 0
    COUNTDOWN_SECS = 3
    current_sequence = []
    frame_ts = 0

    print(f"\nRecording action: {action}")
    print(f"Existing samples: {sample_count}")
    print(f"Press SPACE to start/stop recording a sample.")
    print(f"Each sample = {SEQUENCE_LENGTH} frames (~1 second).")
    print(f"Press Q or ESC to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        result = landmarker.detect_for_video(mp_image, frame_ts)

        has_pose = result.pose_landmarks and len(result.pose_landmarks) > 0

        # Handle countdown before recording.
        if countdown_active:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECS - elapsed
            if remaining <= 0:
                countdown_active = False
                recording = True
                current_sequence = []
                print(f"  Recording sample {sample_count}...")
            else:
                num = int(remaining) + 1
                cv2.putText(
                    frame,
                    str(num),
                    (w // 2 - 40, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.0,
                    (0, 255, 255),
                    8,
                    cv2.LINE_AA,
                )

        if recording and has_pose:
            landmarks = landmarks_to_array(result.pose_landmarks[0])
            current_sequence.append(landmarks)

            # Draw progress bar.
            progress = len(current_sequence) / SEQUENCE_LENGTH
            bar_w = int(w * 0.6)
            bar_x = int(w * 0.2)
            bar_y = h - 50
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 20), (0, 0, 255), -1)

            if len(current_sequence) >= SEQUENCE_LENGTH:
                # Save the sample.
                sample = np.array(current_sequence, dtype=np.float32)
                filename = f"{action}_{sample_count:04d}.npy"
                filepath = os.path.join(save_dir, filename)
                np.save(filepath, sample)
                sample_count += 1
                print(f"  Saved: {filename} (shape: {sample.shape}) | Total: {sample_count}")
                current_sequence = []
                recording = False

        # Status display.
        if countdown_active:
            color = (0, 255, 255)
            status = "GET READY..."
        elif recording:
            color = (0, 0, 255)
            status = f"RECORDING {action} ({len(current_sequence)}/{SEQUENCE_LENGTH})"
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)  # Red dot.
        else:
            color = (0, 255, 0)
            status = f"READY - action: {action} | samples: {sample_count} | press SPACE"

        cv2.putText(frame, status, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        if not has_pose:
            cv2.putText(
                frame,
                "No pose detected - step back",
                (20, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(f"Data Collection - {action}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            if not recording and not countdown_active:
                countdown_active = True
                countdown_start = time.time()
                print(f"  Starting countdown for sample {sample_count}...")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Total samples for '{action}': {sample_count}")


if __name__ == "__main__":
    main()
