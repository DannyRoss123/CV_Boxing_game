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
VELOCITY_THRESHOLD = 0.015
MOTION_START_FRAMES = 3
MOTION_STOP_FRAMES = 6        # reduced from 10 for faster response
LEAD_IN_FRAMES = 12
COOLDOWN_SECS = 0.4            # reduced from 1.0 for faster response

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
        self.lock = threading.Lock()

        # Queue of action dicts to send.
        self._action_queue: list[dict] = []

        # Pose state for optional forwarding.
        self.has_pose = False

    def pop_actions(self) -> list[dict]:
        with self.lock:
            actions = self._action_queue[:]
            self._action_queue.clear()
        return actions

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

        landmarker.close()
        cap.release()
        print("Camera thread stopped.")

    def _classify_segment(self, buffer, start_idx, end_idx):
        buf_list = list(buffer)

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


async def handler(websocket):
    """Handle a single Godot client connection."""
    connected_clients.add(websocket)
    addr = websocket.remote_address
    print(f"  Godot client connected: {addr}")

    # Send welcome message with action list.
    welcome = {
        "type": "welcome",
        "actions": ["idle", "jab", "cross", "hook", "uppercut", "block_high"],
        "damage": {"jab": 8, "cross": 12, "hook": 15, "uppercut": 20, "idle": 0, "block_high": 0},
    }
    await websocket.send(json.dumps(welcome))

    try:
        async for message in websocket:
            # Future: Godot can send commands back (e.g., reset, game state).
            data = json.loads(message)
            print(f"  From Godot: {data}")
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"  Godot client disconnected: {addr}")


async def broadcast_actions(cv_thread: CVThread):
    """Poll the CV thread for new actions and broadcast to all clients."""
    while cv_thread.running:
        # Send classified actions (text/JSON).
        actions = cv_thread.pop_actions()
        for action_msg in actions:
            payload = json.dumps(action_msg)
            to_remove = []
            for ws in connected_clients:
                try:
                    await ws.send(payload)
                except websockets.ConnectionClosed:
                    to_remove.append(ws)
            for ws in to_remove:
                connected_clients.discard(ws)

        await asyncio.sleep(0.03)  # ~30 Hz poll


async def main():
    print("Loading RF model...")
    rf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model loaded.\n")

    cv_thread = CVThread(rf, scaler)
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
