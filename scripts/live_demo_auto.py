"""
Auto-detect boxing actions using motion detection + RF classification.

No button press needed - just punch and it classifies automatically.

How it works:
    1. Rolling buffer keeps last ~120 frames
    2. Wrist/elbow velocity tracked every frame
    3. When motion spike detected → rewind to grab idle lead-in
    4. When motion stops → grab idle tail
    5. Pad/trim to 60 frames → classify with RF
    6. If no motion for 3s → auto-classify as idle

Usage:
    python scripts/live_demo_auto.py

Controls:
    Q/ESC - Quit
"""

import os
import time
import collections

import cv2
import joblib
import mediapipe as mp_lib
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task")
RF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_scaler.joblib")

# ── Constants ────────────────────────────────────────────────────────────────
ACTIONS = ["idle", "jab", "cross", "hook", "uppercut", "block_high"]
SEQUENCE_LENGTH = 60
NUM_LANDMARKS = 33
FEATURES_PER_LM = 4
FEATURE_DIM = NUM_LANDMARKS * FEATURES_PER_LM  # 132

L_HIP = 23 * 4
R_HIP = 24 * 4
L_SHOULDER = 11 * 4
R_SHOULDER = 12 * 4

# Key landmark indices for velocity tracking (wrists + elbows, x/y only).
R_WRIST_X = 16 * 4
R_WRIST_Y = 16 * 4 + 1
L_WRIST_X = 15 * 4
L_WRIST_Y = 15 * 4 + 1
R_ELBOW_X = 14 * 4
R_ELBOW_Y = 14 * 4 + 1
L_ELBOW_X = 13 * 4
L_ELBOW_Y = 13 * 4 + 1

# Motion detection tuning.
BUFFER_SIZE = 120            # ~4 seconds at 30fps
VELOCITY_THRESHOLD = 0.015   # normalized landmark units per frame
MOTION_START_FRAMES = 3      # consecutive frames above threshold to trigger
MOTION_STOP_FRAMES = 10      # consecutive frames below threshold to end
LEAD_IN_FRAMES = 18          # idle frames to grab before motion
TAIL_FRAMES = 12             # idle frames to grab after motion
COOLDOWN_SECS = 1.0          # minimum time between classifications
IDLE_TIMEOUT_SECS = 3.0      # no motion for this long → auto idle

ACTION_COLORS = {
    "idle":       (200, 200, 200),
    "jab":        (0, 100, 255),
    "cross":      (0, 0, 255),
    "hook":       (0, 180, 255),
    "uppercut":   (0, 255, 255),
    "block_high": (255, 150, 0),
}


# ── Classification helpers (identical to training) ───────────────────────────

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
    X = np.array(frames, dtype=np.float32)
    if X.shape[0] < SEQUENCE_LENGTH:
        # Pad by repeating first/last frame (real idle pose, NOT zeros).
        deficit = SEQUENCE_LENGTH - X.shape[0]
        pad_before = deficit // 2
        pad_after = deficit - pad_before
        parts = []
        if pad_before > 0:
            parts.append(np.tile(X[0:1], (pad_before, 1)))
        parts.append(X)
        if pad_after > 0:
            parts.append(np.tile(X[-1:], (pad_after, 1)))
        X = np.vstack(parts)
    elif X.shape[0] > SEQUENCE_LENGTH:
        # Center the motion in the window.
        excess = X.shape[0] - SEQUENCE_LENGTH
        start = excess // 2
        X = X[start:start + SEQUENCE_LENGTH]
    feat_std = np.std(X, axis=0)
    feat_range = np.max(X, axis=0) - np.min(X, axis=0)
    feat_delta = X[-1] - X[0]
    return np.concatenate([feat_std, feat_range, feat_delta])


def compute_velocity(frame_a, frame_b):
    """Compute average velocity of key arm landmarks between two frames."""
    if frame_a is None or frame_b is None:
        return 0.0
    indices = [
        (R_WRIST_X, R_WRIST_Y),
        (L_WRIST_X, L_WRIST_Y),
        (R_ELBOW_X, R_ELBOW_Y),
        (L_ELBOW_X, L_ELBOW_Y),
    ]
    total = 0.0
    for ix, iy in indices:
        dx = frame_b[ix] - frame_a[ix]
        dy = frame_b[iy] - frame_a[iy]
        total += np.sqrt(dx*dx + dy*dy)
    return total / len(indices)


def main():
    print("[1/4] Loading RF model from %s ..." % RF_PATH)
    rf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[1/4] RF model loaded OK.")

    print("[2/4] Creating MediaPipe PoseLandmarker...")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    print("[2/4] PoseLandmarker ready.")

    print("[3/4] Opening webcam (index 0)... if this hangs, camera is in use by another app.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Close other apps using the camera.")
    print("[3/4] Webcam opened OK (%dx%d)." % (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Rolling buffer of normalized frames.
    buffer = collections.deque(maxlen=BUFFER_SIZE)
    prev_frame = None
    frame_ts = 0

    # Motion detection state.
    state = "IDLE"  # IDLE, MOTION, COOLDOWN
    motion_count = 0
    still_count = 0
    motion_start_idx = 0
    last_classify_time = 0.0
    last_motion_time = time.time()

    # Display state.
    current_action = "idle"
    confidence = 0.0
    result_timer = 0

    # Velocity history for visualization.
    vel_history = collections.deque(maxlen=200)

    print("[4/4] All systems go!")
    print("Just punch - no buttons needed! Q/ESC to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        # Pose detection.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        result = landmarker.detect_for_video(mp_image, frame_ts)
        has_pose = result.pose_landmarks and len(result.pose_landmarks) > 0

        velocity = 0.0

        if has_pose:
            raw = landmarks_to_array(result.pose_landmarks[0])
            norm = normalize_frame(raw)
            buffer.append(norm)

            if prev_frame is not None:
                velocity = compute_velocity(prev_frame, norm)
            prev_frame = norm
        else:
            prev_frame = None

        vel_history.append(velocity)

        # ── Motion State Machine ────────────────────────────────────────
        if state == "IDLE":
            if velocity > VELOCITY_THRESHOLD:
                motion_count += 1
                if motion_count >= MOTION_START_FRAMES:
                    motion_start_idx = max(0, len(buffer) - motion_count - LEAD_IN_FRAMES)
                    state = "MOTION"
                    still_count = 0
                    last_motion_time = now
                    print("  Motion detected...")
            else:
                motion_count = 0

            if now - last_motion_time > IDLE_TIMEOUT_SECS and current_action != "idle":
                current_action = "idle"
                confidence = 1.0
                result_timer = 30

        elif state == "MOTION":
            last_motion_time = now

            if velocity < VELOCITY_THRESHOLD:
                still_count += 1
                if still_count >= MOTION_STOP_FRAMES:
                    motion_end_idx = len(buffer)
                    buf_list = list(buffer)

                    # Expand window to at least 60 frames using real idle
                    # frames from the buffer (before and after motion).
                    seg_len = motion_end_idx - motion_start_idx
                    if seg_len < SEQUENCE_LENGTH:
                        deficit = SEQUENCE_LENGTH - seg_len
                        expand_before = deficit // 2
                        expand_after = deficit - expand_before
                        new_start = max(0, motion_start_idx - expand_before)
                        new_end = min(len(buf_list), motion_end_idx + expand_after)
                        # If one side hit the edge, expand more on the other.
                        actual_before = motion_start_idx - new_start
                        actual_after = new_end - motion_end_idx
                        if actual_before < expand_before:
                            new_end = min(len(buf_list), new_end + (expand_before - actual_before))
                        elif actual_after < expand_after:
                            new_start = max(0, new_start - (expand_after - actual_after))
                        motion_start_idx = new_start
                        motion_end_idx = new_end

                    segment = buf_list[motion_start_idx:motion_end_idx]

                    if len(segment) >= 10:
                        features = extract_statistics(segment)
                        features_scaled = scaler.transform(features.reshape(1, -1))
                        action = rf.predict(features_scaled)[0]
                        proba = rf.predict_proba(features_scaled)[0]
                        conf = np.max(proba)

                        current_action = action
                        confidence = conf
                        result_timer = 60

                        ranked = sorted(zip(rf.classes_, proba), key=lambda x: -x[1])
                        print(f"  >> {action.upper():>12s} ({conf*100:.0f}%)  "
                              f"[{len(segment)} frames]  |  "
                              + "  ".join(f"{a}:{p*100:.0f}%" for a, p in ranked))

                        last_classify_time = now
                    else:
                        print(f"  Motion too short ({len(segment)} frames), skipped.")

                    state = "COOLDOWN"
                    motion_count = 0
                    still_count = 0
            else:
                still_count = 0

                # Safety: if motion goes on too long (>3 sec), force classify.
                frames_in_motion = len(buffer) - motion_start_idx
                if frames_in_motion > 90:
                    buf_list = list(buffer)
                    segment = buf_list[motion_start_idx:]
                    features = extract_statistics(segment)
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    action = rf.predict(features_scaled)[0]
                    proba = rf.predict_proba(features_scaled)[0]
                    conf = np.max(proba)

                    current_action = action
                    confidence = conf
                    result_timer = 60

                    ranked = sorted(zip(rf.classes_, proba), key=lambda x: -x[1])
                    print(f"  >> {action.upper():>12s} ({conf*100:.0f}%)  "
                          f"[{len(segment)} frames, forced]  |  "
                          + "  ".join(f"{a}:{p*100:.0f}%" for a, p in ranked))

                    last_classify_time = now
                    state = "COOLDOWN"
                    motion_count = 0
                    still_count = 0

        elif state == "COOLDOWN":
            if now - last_classify_time > COOLDOWN_SECS:
                state = "IDLE"
                motion_count = 0

        # ── Draw UI ─────────────────────────────────────────────────────
        # Velocity bar.
        bar_x = 20
        bar_y = h - 40
        bar_max_w = 300
        vel_display = min(velocity / 0.08, 1.0)
        vel_w = int(bar_max_w * vel_display)
        vel_color = (0, 255, 0) if velocity < VELOCITY_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 15), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + vel_w, bar_y + 15), vel_color, -1)
        # Threshold marker.
        thresh_x = bar_x + int(bar_max_w * VELOCITY_THRESHOLD / 0.08)
        cv2.line(frame, (thresh_x, bar_y - 3), (thresh_x, bar_y + 18), (0, 255, 255), 2)
        cv2.putText(frame, "velocity", (bar_x, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # Velocity graph.
        graph_x, graph_y = w - 220, h - 60
        graph_w, graph_h = 200, 50
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (30, 30, 30), -1)
        if len(vel_history) > 1:
            pts = []
            for i, v in enumerate(vel_history):
                px = graph_x + int(i * graph_w / vel_history.maxlen)
                py = graph_y + graph_h - int(min(v / 0.08, 1.0) * graph_h)
                pts.append((px, py))
            for j in range(1, len(pts)):
                c = (0, 0, 255) if list(vel_history)[j] > VELOCITY_THRESHOLD else (0, 180, 0)
                cv2.line(frame, pts[j-1], pts[j], c, 1)
            # Threshold line on graph.
            thresh_gy = graph_y + graph_h - int(VELOCITY_THRESHOLD / 0.08 * graph_h)
            cv2.line(frame, (graph_x, thresh_gy), (graph_x + graph_w, thresh_gy), (0, 255, 255), 1)

        # State indicator.
        state_colors = {"IDLE": (0, 255, 0), "MOTION": (0, 0, 255), "COOLDOWN": (0, 180, 255)}
        state_color = state_colors.get(state, (200, 200, 200))
        cv2.putText(frame, state, (w - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2, cv2.LINE_AA)

        if state == "MOTION":
            cv2.circle(frame, (w - 170, 30), 8, (0, 0, 255), -1)

        # Action result.
        if result_timer > 0:
            result_timer -= 1
            color = ACTION_COLORS.get(current_action, (255, 255, 255))
            cv2.putText(frame, current_action.upper(), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)
            cv2.putText(frame, f"{confidence*100:.0f}%", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        if not has_pose:
            cv2.putText(frame, "No pose detected", (20, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)

        cv2.imshow("Boxing Auto-Detect Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
