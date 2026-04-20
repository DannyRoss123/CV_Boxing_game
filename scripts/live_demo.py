"""
Real-time boxing action recognition using webcam + Random Forest.

Press SPACE to record 60 frames (same as training), then see the prediction.

Usage:
    python scripts/live_demo.py

Controls:
    SPACE - Record + classify an action
    Q/ESC - Quit
"""

import os
import time
import numpy as np
import cv2
import joblib
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task")
RF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_scaler.joblib")

ACTIONS = ["idle", "jab", "cross", "hook", "uppercut", "block_high"]
SEQUENCE_LENGTH = 60
NUM_LANDMARKS = 33
FEATURES_PER_LM = 4
FEATURE_DIM = NUM_LANDMARKS * FEATURES_PER_LM  # 132

# Landmark indices for normalization.
L_HIP = 23 * 4
R_HIP = 24 * 4
L_SHOULDER = 11 * 4
R_SHOULDER = 12 * 4

ACTION_COLORS = {
    "idle":       (200, 200, 200),
    "jab":        (0, 100, 255),
    "cross":      (0, 0, 255),
    "hook":       (0, 180, 255),
    "uppercut":   (0, 255, 255),
    "block_high": (255, 150, 0),
}


def landmarks_to_array(landmarks) -> np.ndarray:
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(row, dtype=np.float32)


def normalize_frame(frame):
    frame = frame.copy()
    cx = (frame[L_HIP] + frame[R_HIP]) / 2
    cy = (frame[L_HIP + 1] + frame[R_HIP + 1]) / 2
    cz = (frame[L_HIP + 2] + frame[R_HIP + 2]) / 2
    sx = (frame[L_SHOULDER] + frame[R_SHOULDER]) / 2
    sy = (frame[L_SHOULDER + 1] + frame[R_SHOULDER + 1]) / 2
    sz = (frame[L_SHOULDER + 2] + frame[R_SHOULDER + 2]) / 2
    torso_len = np.sqrt((sx - cx)**2 + (sy - cy)**2 + (sz - cz)**2)
    if torso_len < 0.01:
        torso_len = 1.0
    for lm_idx in range(NUM_LANDMARKS):
        idx = lm_idx * FEATURES_PER_LM
        frame[idx]     = (frame[idx]     - cx) / torso_len
        frame[idx + 1] = (frame[idx + 1] - cy) / torso_len
        frame[idx + 2] = (frame[idx + 2] - cz) / torso_len
    return frame


def extract_statistics(frames):
    """Motion-only stats from a list of frames -> (396,)."""
    X = np.array(frames, dtype=np.float32)
    if X.shape[0] < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - X.shape[0], FEATURE_DIM), dtype=np.float32)
        X = np.vstack([X, pad])
    elif X.shape[0] > SEQUENCE_LENGTH:
        X = X[:SEQUENCE_LENGTH]
    feat_std = np.std(X, axis=0)
    feat_range = np.max(X, axis=0) - np.min(X, axis=0)
    feat_delta = X[-1] - X[0]
    return np.concatenate([feat_std, feat_range, feat_delta])


def main():
    print("Loading Random Forest model...")
    rf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model loaded.")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_ts = 0
    state = "READY"
    countdown_start = 0
    COUNTDOWN_SECS = 3
    recording_buffer = []
    current_action = "idle"
    confidence = 0.0
    result_timer = 0
    RESULT_DISPLAY_FRAMES = 60

    print("\nLive demo running (Random Forest).")
    print("Press SPACE to start recording (3s countdown, then perform your action).")
    print("Press Q or ESC to quit.\n")

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

        if state == "COUNTDOWN":
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECS - elapsed
            if remaining <= 0:
                state = "RECORDING"
                recording_buffer = []
                print("  Recording...")
            else:
                num = int(remaining) + 1
                cv2.putText(
                    frame, str(num),
                    (w // 2 - 40, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 8, cv2.LINE_AA,
                )

        elif state == "RECORDING" and has_pose:
            raw = landmarks_to_array(result.pose_landmarks[0])
            norm = normalize_frame(raw)
            recording_buffer.append(norm)

            progress = len(recording_buffer) / SEQUENCE_LENGTH
            bar_w_px = int(w * 0.6)
            bar_x = int(w * 0.2)
            bar_y = h - 50
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_px, bar_y + 20), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w_px * progress), bar_y + 20), (0, 0, 255), -1)
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)

            if len(recording_buffer) >= SEQUENCE_LENGTH:
                features = extract_statistics(recording_buffer)
                features_scaled = scaler.transform(features.reshape(1, -1))
                current_action = rf.predict(features_scaled)[0]
                proba = rf.predict_proba(features_scaled)[0]
                confidence = np.max(proba)

                ranked = sorted(zip(rf.classes_, proba), key=lambda x: -x[1])
                print(f"  >> {current_action.upper():>12s} ({confidence*100:.0f}%)  |  "
                      + "  ".join(f"{a}:{p*100:.0f}%" for a, p in ranked))

                state = "RESULT"
                result_timer = RESULT_DISPLAY_FRAMES

        elif state == "RESULT":
            result_timer -= 1
            if result_timer <= 0:
                state = "READY"
                current_action = "idle"
                confidence = 0.0

        # ── Draw UI ─────────────────────────────────────────────────────
        color = ACTION_COLORS.get(current_action, (255, 255, 255))

        if state == "RESULT" or (state == "READY" and current_action != "idle"):
            cv2.putText(
                frame, current_action.upper(),
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"{confidence*100:.0f}%",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
            )
        elif state == "READY":
            cv2.putText(
                frame, "Press SPACE to record",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )
        elif state == "RECORDING":
            cv2.putText(
                frame, f"RECORDING ({len(recording_buffer)}/{SEQUENCE_LENGTH})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )

        if not has_pose and state != "COUNTDOWN":
            cv2.putText(
                frame, "No pose detected",
                (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA,
            )

        cv2.imshow("Boxing Action Recognition - LIVE (RF)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord(" ") and state == "READY":
            state = "COUNTDOWN"
            countdown_start = time.time()
            print(f"  Countdown started...")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
