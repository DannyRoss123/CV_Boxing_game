"""
CV WebSocket Server — bridges the Python pose-detection / classification
pipeline to the Godot game client.

Runs the camera + MediaPipe + motion-detection + RF classifier in a
background thread and pushes classified actions to all connected Godot
clients over WebSocket (JSON messages).

Usage:
    python scripts/cv_server.py

The server listens on ws://localhost:9090 by default.
"""

import asyncio
import collections
import json
import os
import signal
import sys
import threading
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import websockets

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(ROOT, "pose_landmarker_lite.task")
RF_PATH = os.path.join(ROOT, "models", "random_forest.joblib")
SCALER_PATH = os.path.join(ROOT, "models", "rf_scaler.joblib")

# ── Constants (same as game.py) ──────────────────────────────────────────────
SEQUENCE_LENGTH = 60
NUM_LANDMARKS = 33
FEATURES_PER_LM = 4
FEATURE_DIM = NUM_LANDMARKS * FEATURES_PER_LM

L_HIP = 23 * 4
R_HIP = 24 * 4
L_SHOULDER = 11 * 4
R_SHOULDER = 12 * 4

R_WRIST_X = 16 * 4
R_WRIST_Y = 16 * 4 + 1
L_WRIST_X = 15 * 4
L_WRIST_Y = 15 * 4 + 1
R_ELBOW_X = 14 * 4
R_ELBOW_Y = 14 * 4 + 1
L_ELBOW_X = 13 * 4
L_ELBOW_Y = 13 * 4 + 1

BUFFER_SIZE = 120
VELOCITY_THRESHOLD = 0.022   # raised from 0.015 — ignores body sway/breathing
MIN_CONFIDENCE = 0.65        # don't send low-confidence guesses to Godot
MOTION_START_FRAMES = 3
MOTION_STOP_FRAMES = 10      # must match live_demo_auto.py exactly
LEAD_IN_FRAMES = 18          # must match live_demo_auto.py exactly
TAIL_FRAMES = 12             # idle tail after motion — same as live_demo_auto.py
COOLDOWN_SECS = 1.0          # must match live_demo_auto.py exactly

HOST = "localhost"
PORT = 9090


# ── Feature helpers (identical to game.py / training) ────────────────────────

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
        excess = X.shape[0] - SEQUENCE_LENGTH
        start = excess // 2
        X = X[start:start + SEQUENCE_LENGTH]
    feat_std = np.std(X, axis=0)
    feat_range = np.max(X, axis=0) - np.min(X, axis=0)
    feat_delta = X[-1] - X[0]
    return np.concatenate([feat_std, feat_range, feat_delta])


def compute_velocity(frame_a, frame_b):
    if frame_a is None or frame_b is None:
        return 0.0
    indices = [
        (R_WRIST_X, R_WRIST_Y), (L_WRIST_X, L_WRIST_Y),
        (R_ELBOW_X, R_ELBOW_Y), (L_ELBOW_X, L_ELBOW_Y),
    ]
    total = 0.0
    for ix, iy in indices:
        dx = frame_b[ix] - frame_a[ix]
        dy = frame_b[iy] - frame_a[iy]
        total += np.sqrt(dx*dx + dy*dy)
    return total / len(indices)


# ── Camera / Classification Thread ──────────────────────────────────────────

class CVThread(threading.Thread):
    """Webcam + MediaPipe + motion detection + RF classification.

    Produces a queue of action events that the async WebSocket server
    forwards to Godot.
    """

    def __init__(self, rf, scaler):
        super().__init__(daemon=True)
        self.rf = rf
        self.scaler = scaler
        self.running = True
        self.active = True   # always classify; Godot filters by game state
        self.lock = threading.Lock()

        # Queue of action dicts to send.
        self._action_queue: list[dict] = []

        # Latest tiny JPEG for the Godot webcam preview (160x120, ~3 KB).
        self._latest_jpeg: bytes | None = None
        self._frame_counter: int = 0

        self.has_pose = False

    def pop_actions(self) -> list[dict]:
        with self.lock:
            actions = self._action_queue[:]
            self._action_queue.clear()
        return actions

    def pop_frame(self) -> bytes | None:
        with self.lock:
            frame = self._latest_jpeg
            self._latest_jpeg = None
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam.")
            self.running = False
            return

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = PoseLandmarker.create_from_options(options)
        frame_ts = 0

        buffer = collections.deque(maxlen=BUFFER_SIZE)
        prev_frame = None

        m_state = "IDLE"
        motion_count = 0
        still_count = 0
        motion_start_idx = 0
        last_classify_time = 0.0

        print(f"Camera thread running (webcam index 0)")

        while self.running:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
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

            with self.lock:
                self.has_pose = has_pose

            now = time.time()

            # ── Motion detection state machine ──────────────────────
            if m_state == "IDLE":
                if velocity > VELOCITY_THRESHOLD:
                    motion_count += 1
                    if motion_count >= MOTION_START_FRAMES:
                        motion_start_idx = max(0, len(buffer) - motion_count - LEAD_IN_FRAMES)
                        m_state = "MOTION"
                        still_count = 0
                else:
                    motion_count = 0

            elif m_state == "MOTION":
                if velocity < VELOCITY_THRESHOLD:
                    still_count += 1
                    if still_count >= MOTION_STOP_FRAMES:
                        self._classify_segment(buffer, motion_start_idx, len(buffer))
                        last_classify_time = now
                        m_state = "COOLDOWN"
                        motion_count = 0
                        still_count = 0
                else:
                    still_count = 0
                    if len(buffer) - motion_start_idx > 90:
                        self._classify_segment(buffer, motion_start_idx, len(buffer))
                        last_classify_time = now
                        m_state = "COOLDOWN"
                        motion_count = 0
                        still_count = 0

            elif m_state == "COOLDOWN":
                if now - last_classify_time > COOLDOWN_SECS:
                    m_state = "IDLE"
                    motion_count = 0

            # ── Encode preview with velocity bar overlay ────────────
            self._frame_counter += 1
            if self._frame_counter % 3 == 0 and connected_clients:
                small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_NEAREST)
                # Velocity bar across the bottom (last 8 rows).
                bar_fill = int(min(velocity / (VELOCITY_THRESHOLD * 3), 1.0) * 160)
                bar_color = (0, 0, 220) if m_state == "MOTION" else (0, 200, 0) if m_state == "IDLE" else (0, 180, 255)
                small[112:120, :, :] = 30  # dark background strip
                if bar_fill > 0:
                    small[112:120, :bar_fill, :] = bar_color
                # Threshold marker line.
                thresh_x = int(VELOCITY_THRESHOLD / (VELOCITY_THRESHOLD * 3) * 160)
                small[112:120, thresh_x:thresh_x+2, :] = (0, 220, 220)
                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 55])
                with self.lock:
                    self._latest_jpeg = bytes(buf)

        landmarker.close()
        cap.release()
        print("Camera thread stopped.")

    def _classify_segment(self, buffer, start_idx, end_idx):
        if not self.active:
            return
        buf_list = list(buffer)

        # Include idle tail frames after motion (identical to live_demo_auto.py).
        end_idx = min(len(buf_list), end_idx + TAIL_FRAMES)

        # Expand to at least 60 frames with real surrounding data.
        seg_len = end_idx - start_idx
        if seg_len < SEQUENCE_LENGTH:
            deficit = SEQUENCE_LENGTH - seg_len
            eb = deficit // 2
            ea = deficit - eb
            ns = max(0, start_idx - eb)
            ne = min(len(buf_list), end_idx + ea)
            ab = start_idx - ns
            aa = ne - end_idx
            if ab < eb:
                ne = min(len(buf_list), ne + (eb - ab))
            elif aa < ea:
                ns = max(0, ns - (ea - aa))
            start_idx = ns
            end_idx = ne

        segment = buf_list[start_idx:end_idx]
        if len(segment) < 10:
            return

        features = extract_statistics(segment)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        action = self.rf.predict(features_scaled)[0]
        proba = self.rf.predict_proba(features_scaled)[0]
        conf = float(np.max(proba))

        ranked = sorted(zip(self.rf.classes_, proba), key=lambda x: -x[1])
        print(f"  >> {action.upper():>12s} ({conf*100:.0f}%)  "
              f"[{len(segment)} frames]  |  "
              + "  ".join(f"{a}:{p*100:.0f}%" for a, p in ranked))

        # Don't send idle or low-confidence guesses — they trigger random animations.
        if action == "idle" or conf < MIN_CONFIDENCE:
            return

        msg = {
            "type": "action",
            "action": action,
            "confidence": round(conf, 3),
            "frames": len(segment),
        }

        with self.lock:
            self._action_queue.append(msg)

    def stop(self):
        self.running = False
        self.join(timeout=3)


# ── WebSocket Server ─────────────────────────────────────────────────────────

connected_clients: set = set()
_cv_thread: "CVThread | None" = None  # set in main() so handler can reach it


async def handler(websocket):
    """Handle a single Godot client connection."""
    connected_clients.add(websocket)
    addr = websocket.remote_address
    print(f"  Godot client connected: {addr}")

    welcome = {
        "type": "welcome",
        "actions": ["idle", "jab", "cross", "hook", "uppercut", "block_high"],
        "damage": {"jab": 8, "cross": 12, "hook": 15, "uppercut": 20, "idle": 0, "block_high": 0},
    }
    await websocket.send(json.dumps(welcome))

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "set_active" and _cv_thread is not None:
                    _cv_thread.active = bool(data.get("active", True))
                    print(f"  CV {'ENABLED' if _cv_thread.active else 'DISABLED'}")
            except Exception:
                pass
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"  Godot client disconnected: {addr}")


async def broadcast_actions(cv_thread: CVThread):
    """Poll the CV thread for new actions and frames; broadcast to all clients."""
    while cv_thread.running:
        if not connected_clients:
            await asyncio.sleep(0.05)
            continue

        to_remove: set = set()

        for action_msg in cv_thread.pop_actions():
            payload = json.dumps(action_msg)
            for ws in connected_clients:
                try:
                    await ws.send(payload)
                except websockets.ConnectionClosed:
                    to_remove.add(ws)

        jpeg = cv_thread.pop_frame()
        if jpeg is not None:
            for ws in connected_clients:
                try:
                    await ws.send(jpeg)
                except websockets.ConnectionClosed:
                    to_remove.add(ws)

        for ws in to_remove:
            connected_clients.discard(ws)

        await asyncio.sleep(0.01)  # ~100 Hz poll — less WebSocket latency


async def main():
    print("Loading RF model...")
    rf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model loaded.\n")

    global _cv_thread
    cv_thread = CVThread(rf, scaler)
    _cv_thread = cv_thread
    cv_thread.start()

    print(f"WebSocket server starting on ws://{HOST}:{PORT}")
    print("Waiting for Godot client to connect...\n")

    # Start the broadcast loop as a background task.
    broadcast_task = asyncio.create_task(broadcast_actions(cv_thread))

    stop_event = asyncio.Event()

    # Handle Ctrl+C gracefully.
    def on_signal():
        print("\nShutting down...")
        cv_thread.stop()
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, on_signal)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler.
            pass

    async with websockets.serve(handler, HOST, PORT):
        print(f"Server listening on ws://{HOST}:{PORT}")
        print("Press Ctrl+C to stop.\n")
        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            on_signal()

    broadcast_task.cancel()
    cv_thread.stop()
    print("Server stopped.")


if __name__ == "__main__":
    asyncio.run(main())
