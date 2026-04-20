"""
Boxing game MVP - your body is the controller.

Classify your punch → fighter plays the animation → deal damage.

Usage:
    python scripts/game.py

Controls:
    SPACE - Record action (3s countdown, then perform punch)
    R     - Reset fight
    Q/ESC - Quit
"""

import os
import time
import threading
import collections
import math

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pygame

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
SCREEN_W, SCREEN_H = 1024, 680
PREVIEW_W, PREVIEW_H = 320, 240

ACTIONS = ["idle", "jab", "cross", "hook", "uppercut", "block_high"]
SEQUENCE_LENGTH = 60
NUM_LANDMARKS = 33
FEATURES_PER_LM = 4
FEATURE_DIM = NUM_LANDMARKS * FEATURES_PER_LM  # 132

L_HIP = 23 * 4
R_HIP = 24 * 4
L_SHOULDER = 11 * 4
R_SHOULDER = 12 * 4

DAMAGE = {"jab": 8, "cross": 12, "hook": 15, "uppercut": 20, "idle": 0, "block_high": 0}

# Velocity tracking indices (wrists + elbows, x/y).
R_WRIST_X = 16 * 4
R_WRIST_Y = 16 * 4 + 1
L_WRIST_X = 15 * 4
L_WRIST_Y = 15 * 4 + 1
R_ELBOW_X = 14 * 4
R_ELBOW_Y = 14 * 4 + 1
L_ELBOW_X = 13 * 4
L_ELBOW_Y = 13 * 4 + 1

# Motion detection tuning.
BUFFER_SIZE = 120
VELOCITY_THRESHOLD = 0.015
MOTION_START_FRAMES = 3
MOTION_STOP_FRAMES = 10
LEAD_IN_FRAMES = 12
COOLDOWN_SECS = 1.0

# ── Colors ───────────────────────────────────────────────────────────────────
C_BG = (18, 18, 28)
C_TEXT = (230, 230, 230)
C_COUNTDOWN = (255, 255, 50)
C_RECORDING = (255, 50, 50)

# Fighter palette.
SKIN = (230, 185, 140)
SKIN_DARK = (195, 150, 110)
SKIN_LIGHT = (245, 210, 175)
HAIR = (40, 30, 25)
SHOE = (45, 45, 50)

# Player colors.
P_GLOVE = (20, 80, 200)
P_GLOVE_HI = (80, 150, 255)
P_SHORTS = (30, 100, 220)
P_SHORTS_HI = (60, 140, 255)
P_TANK = (22, 65, 170)
P_TANK_HI = (40, 90, 210)
P_WRAP = (200, 200, 210)

# Opponent colors.
O_GLOVE = (200, 30, 30)
O_GLOVE_HI = (255, 80, 80)
O_SHORTS = (180, 35, 35)
O_SHORTS_HI = (220, 70, 70)
O_TANK = (150, 25, 25)
O_TANK_HI = (180, 50, 50)
O_WRAP = (210, 200, 190)

ACTION_COLORS = {
    "idle":       (180, 180, 180),
    "jab":        (255, 150, 0),
    "cross":      (255, 50, 50),
    "hook":       (255, 200, 0),
    "uppercut":   (255, 255, 0),
    "block_high": (0, 180, 255),
}


# ── Classification helpers (identical to live_demo / training) ───────────────

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


# ── Camera Thread ────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    """Webcam + MediaPipe + motion detection + auto-classification."""

    def __init__(self, rf, scaler):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.running = True
        self.rf = rf
        self.scaler = scaler

        # Shared with main thread.
        self.preview_rgb = None
        self.has_pose = False

        # Action queue: main thread pops classified actions.
        self.pending_action = None  # (action_str, confidence, proba_dict)

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

        # Motion state.
        m_state = "IDLE"  # IDLE, MOTION, COOLDOWN
        motion_count = 0
        still_count = 0
        motion_start_idx = 0
        last_classify_time = 0.0

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
            small = cv2.resize(rgb, (PREVIEW_W, PREVIEW_H))

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
                self.preview_rgb = small
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
                        # Motion ended - extract and classify.
                        motion_end_idx = len(buffer)
                        buf_list = list(buffer)

                        # Expand to at least 60 frames with real idle.
                        seg_len = motion_end_idx - motion_start_idx
                        if seg_len < SEQUENCE_LENGTH:
                            deficit = SEQUENCE_LENGTH - seg_len
                            eb = deficit // 2
                            ea = deficit - eb
                            ns = max(0, motion_start_idx - eb)
                            ne = min(len(buf_list), motion_end_idx + ea)
                            ab = motion_start_idx - ns
                            aa = ne - motion_end_idx
                            if ab < eb:
                                ne = min(len(buf_list), ne + (eb - ab))
                            elif aa < ea:
                                ns = max(0, ns - (ea - aa))
                            motion_start_idx = ns
                            motion_end_idx = ne

                        segment = buf_list[motion_start_idx:motion_end_idx]

                        if len(segment) >= 10:
                            features = extract_statistics(segment)
                            features_scaled = self.scaler.transform(features.reshape(1, -1))
                            action = self.rf.predict(features_scaled)[0]
                            proba = self.rf.predict_proba(features_scaled)[0]
                            conf = float(np.max(proba))

                            ranked = sorted(zip(self.rf.classes_, proba), key=lambda x: -x[1])
                            print(f"  >> {action.upper():>12s} ({conf*100:.0f}%)  "
                                  f"[{len(segment)} frames]  |  "
                                  + "  ".join(f"{a}:{p*100:.0f}%" for a, p in ranked))

                            with self.lock:
                                self.pending_action = (action, conf)

                            last_classify_time = now

                        m_state = "COOLDOWN"
                        motion_count = 0
                        still_count = 0
                else:
                    still_count = 0
                    # Force classify if motion too long.
                    if len(buffer) - motion_start_idx > 90:
                        buf_list = list(buffer)
                        segment = buf_list[motion_start_idx:]
                        features = extract_statistics(segment)
                        features_scaled = self.scaler.transform(features.reshape(1, -1))
                        action = self.rf.predict(features_scaled)[0]
                        proba = self.rf.predict_proba(features_scaled)[0]
                        conf = float(np.max(proba))

                        ranked = sorted(zip(self.rf.classes_, proba), key=lambda x: -x[1])
                        print(f"  >> {action.upper():>12s} ({conf*100:.0f}%)  "
                              f"[{len(segment)} frames, forced]  |  "
                              + "  ".join(f"{a}:{p*100:.0f}%" for a, p in ranked))

                        with self.lock:
                            self.pending_action = (action, conf)

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

    def stop(self):
        self.running = False
        self.join(timeout=3)


# ── Fighter Renderer ─────────────────────────────────────────────────────────

def lerp(a, b, t):
    return a + (b - a) * t


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_out_back(t):
    """Overshoots slightly then settles - gives punches a snap."""
    c = 1.4
    return 1 + c * (t - 1) ** 3 + c * (t - 1) ** 2


class Fighter:
    """
    Pseudo-3D boxer drawn with pygame primitives.

    The 3D effect comes from:
    - Body rotation (torso twist via shoulder offset)
    - Foreshortening (glove size scales with "depth")
    - Draw order changes based on which arm is in front
    - Limb thickness variation to suggest roundness
    - Shadow/highlight split on body parts
    """

    def __init__(self, x, y, facing_right=True, is_player=True):
        self.x = x
        self.y = y
        self.facing = 1 if facing_right else -1
        self.is_player = is_player

        if is_player:
            self.glove = P_GLOVE
            self.glove_hi = P_GLOVE_HI
            self.shorts = P_SHORTS
            self.shorts_hi = P_SHORTS_HI
            self.tank = P_TANK
            self.tank_hi = P_TANK_HI
            self.wrap = P_WRAP
        else:
            self.glove = O_GLOVE
            self.glove_hi = O_GLOVE_HI
            self.shorts = O_SHORTS
            self.shorts_hi = O_SHORTS_HI
            self.tank = O_TANK
            self.tank_hi = O_TANK_HI
            self.wrap = O_WRAP

        # Animation.
        self.pose = "idle"
        self.anim_t = 0.0
        self.anim_speed = 0.08
        self.anim_done = True

        # Body rotation angle (radians, 0 = facing side, positive = toward viewer).
        self.body_rot = 0.0
        self.target_rot = 0.0

        # Impact effect.
        self.shake_t = 0.0

    def trigger_action(self, action):
        self.pose = action
        self.anim_t = 0.0
        self.anim_done = False

    def update(self):
        if not self.anim_done:
            self.anim_t += self.anim_speed
            if self.anim_t >= 1.0:
                self.anim_t = 1.0
                self.anim_done = True
                self.pose = "idle"
                self.target_rot = 0.0

        # Smooth body rotation.
        self.body_rot = lerp(self.body_rot, self.target_rot, 0.25)

        if self.shake_t > 0:
            self.shake_t -= 0.05

    def take_hit(self):
        self.shake_t = 1.0

    def _get_pose_data(self):
        """
        Returns a dict of body part positions as (x, y) offsets from hip center,
        plus a 'depth' value per glove for foreshortening, and body rotation.

        Coordinate system: x+ = facing direction, y- = up.
        """
        f = self.facing
        t = ease_out_cubic(self.anim_t) if not self.anim_done else 0.0
        tb = ease_out_back(min(self.anim_t * 1.2, 1.0)) if not self.anim_done else 0.0

        # Default guard stance.
        pose = {
            # Shoulders (affected by body rotation).
            "lead_shoulder": (f * 24, -100),
            "rear_shoulder": (f * -24, -100),
            # Lead arm (front hand).
            "lead_elbow": (f * 48, -78),
            "lead_fist":  (f * 42, -105),
            "lead_glove_size": 16,
            # Rear arm (back hand).
            "rear_elbow": (f * -32, -78),
            "rear_fist":  (f * -28, -105),
            "rear_glove_size": 13,  # Slightly smaller = further away.
            # Body rotation for 3D effect.
            "body_rot": 0.0,
            # Which arm is in front for draw order.
            "lead_in_front": True,
        }

        if self.pose == "jab" and not self.anim_done:
            # Quick straight punch with lead hand. Minimal body rotation.
            # The fist extends forward (toward opponent) and gets BIGGER (closer to viewer).
            self.target_rot = 0.1

            pose["lead_elbow"] = (f * lerp(48, 90, tb), lerp(-78, -95, t))
            pose["lead_fist"]  = (f * lerp(42, 145, tb), lerp(-105, -98, t))
            pose["lead_glove_size"] = int(lerp(16, 22, t))  # Gets bigger = closer.
            pose["lead_in_front"] = True

        elif self.pose == "cross" and not self.anim_done:
            # Rear hand drives across with full body rotation.
            # Torso twists significantly - rear shoulder comes forward.
            self.target_rot = 0.45

            rot = t * 0.45
            # Rear shoulder comes forward as body rotates.
            pose["rear_shoulder"] = (f * lerp(-24, 10, t), lerp(-100, -102, t))
            pose["lead_shoulder"] = (f * lerp(24, 5, t), lerp(-100, -98, t))

            # Rear arm extends straight through.
            pose["rear_elbow"] = (f * lerp(-32, 70, tb), lerp(-78, -92, t))
            pose["rear_fist"]  = (f * lerp(-28, 155, tb), lerp(-105, -96, t))
            pose["rear_glove_size"] = int(lerp(13, 24, t))  # Big = power shot.

            # Lead arm tucks back.
            pose["lead_elbow"] = (f * lerp(48, 25, t), lerp(-78, -82, t))
            pose["lead_fist"]  = (f * lerp(42, 18, t), lerp(-105, -108, t))
            pose["lead_glove_size"] = int(lerp(16, 12, t))  # Smaller = pulled back.
            pose["lead_in_front"] = False  # Rear arm is now in front.

        elif self.pose == "hook" and not self.anim_done:
            # Lead arm swings in a wide horizontal arc from the side.
            # Elbow stays bent ~90 degrees. The fist sweeps laterally.
            self.target_rot = 0.25

            # The hook arc - fist travels in a curve.
            # Phase 1 (0-0.4): wind up slightly back.
            # Phase 2 (0.4-1.0): sweep across.
            if t < 0.3:
                st = t / 0.3
                pose["lead_elbow"] = (f * lerp(48, 55, st), lerp(-78, -72, st))
                pose["lead_fist"]  = (f * lerp(42, 50, st), lerp(-105, -95, st))
            else:
                st = ease_out_back(min((t - 0.3) / 0.7 * 1.1, 1.0))
                # Elbow stays wide, fist sweeps in a horizontal arc.
                pose["lead_elbow"] = (f * lerp(55, 75, st), lerp(-72, -90, st))
                # Fist arc: starts wide, sweeps across to center.
                arc_x = lerp(50, 130, st)
                arc_y = lerp(-95, -92, st) + math.sin(st * math.pi) * (-15)
                pose["lead_fist"] = (f * arc_x, arc_y)

            pose["lead_glove_size"] = int(lerp(16, 20, t))
            pose["lead_in_front"] = True

            # Body rotation to drive the hook.
            pose["lead_shoulder"] = (f * lerp(24, 30, t), lerp(-100, -98, t))

        elif self.pose == "uppercut" and not self.anim_done:
            # Rear hand drops low then rips straight up.
            # Very vertical motion. Body dips then springs up.
            self.target_rot = 0.3

            if t < 0.35:
                # Phase 1: Dip down - hand drops, knees bend.
                st = t / 0.35
                pose["rear_elbow"] = (f * lerp(-32, -20, st), lerp(-78, -50, st))
                pose["rear_fist"]  = (f * lerp(-28, -10, st), lerp(-105, -35, st))
                pose["rear_glove_size"] = int(lerp(13, 14, st))
                # Body dips.
                pose["rear_shoulder"] = (f * -24, lerp(-100, -90, st))
                pose["lead_shoulder"] = (f * 24, lerp(-100, -92, st))
            else:
                # Phase 2: Explode upward.
                st = ease_out_back(min((t - 0.35) / 0.65 * 1.1, 1.0))
                pose["rear_elbow"] = (f * lerp(-20, 30, st), lerp(-50, -110, st))
                pose["rear_fist"]  = (f * lerp(-10, 50, st), lerp(-35, -165, st))
                pose["rear_glove_size"] = int(lerp(14, 26, st))  # Huge at peak.
                # Body springs up and rotates.
                pose["rear_shoulder"] = (f * lerp(-24, 5, st), lerp(-90, -105, st))
                pose["lead_shoulder"] = (f * lerp(24, 15, st), lerp(-92, -100, st))

            pose["lead_in_front"] = False
            # Lead stays guarding.
            pose["lead_elbow"] = (f * lerp(48, 35, t), lerp(-78, -85, t))
            pose["lead_fist"]  = (f * lerp(42, 28, t), lerp(-105, -115, t))
            pose["lead_glove_size"] = int(lerp(16, 13, t))

        elif self.pose == "block_high" and not self.anim_done:
            # Both arms come up tight to protect head. Very compact.
            self.target_rot = 0.0

            pose["lead_elbow"] = (f * lerp(48, 30, t), lerp(-78, -100, t))
            pose["lead_fist"]  = (f * lerp(42, 18, t), lerp(-105, -135, t))
            pose["lead_glove_size"] = int(lerp(16, 17, t))

            pose["rear_elbow"] = (f * lerp(-32, -22, t), lerp(-78, -100, t))
            pose["rear_fist"]  = (f * lerp(-28, -12, t), lerp(-105, -135, t))
            pose["rear_glove_size"] = int(lerp(13, 16, t))
            pose["lead_in_front"] = True

        return pose

    def draw(self, surface, hit_flash=False):
        # Apply screen shake on hit.
        shake_x = int(math.sin(self.shake_t * 30) * self.shake_t * 6) if self.shake_t > 0 else 0
        cx = self.x + shake_x
        cy = self.y

        pose = self._get_pose_data()
        f = self.facing

        def p(offset):
            return (int(cx + offset[0]), int(cy + offset[1]))

        def draw_limb(surf, start, end, thickness, color_outer, color_inner=None):
            """Draw a limb with a highlight stripe for roundness."""
            s, e = p(start), p(end)
            pygame.draw.line(surf, color_outer, s, e, thickness)
            if color_inner and thickness > 4:
                # Inner highlight line offset toward light source.
                dx = e[0] - s[0]
                dy = e[1] - s[1]
                length = math.sqrt(dx*dx + dy*dy) + 0.001
                nx, ny = -dy / length, dx / length
                off = max(1, thickness // 5)
                s2 = (int(s[0] + nx * off), int(s[1] + ny * off))
                e2 = (int(e[0] + nx * off), int(e[1] + ny * off))
                pygame.draw.line(surf, color_inner, s2, e2, max(1, thickness // 3))

        def draw_glove(surf, pos, size, color, highlight):
            """Draw a boxing glove with 3D shading."""
            gp = p(pos)
            # Main glove body.
            pygame.draw.circle(surf, color, gp, size)
            # Highlight (upper-left light).
            hi_pos = (gp[0] - size // 4, gp[1] - size // 4)
            pygame.draw.circle(surf, highlight, hi_pos, max(3, size // 3))
            # Outline.
            darker = (max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 40))
            pygame.draw.circle(surf, darker, gp, size, 2)
            # Wrist wrap.
            wr = (gp[0] - f * size // 2, gp[1] + size // 3)
            pygame.draw.circle(surf, self.wrap, wr, max(3, size // 3))

        # ── Legs ─────────────────────────────────────────────────────
        hip_l = (f * 14, 0)
        hip_r = (f * -14, 0)
        knee_l = (f * 22, 58)
        knee_r = (f * -16, 55)
        foot_l = (f * 28, 115)
        foot_r = (f * -12, 112)

        draw_limb(surface, hip_r, knee_r, 11, SKIN_DARK, SKIN)
        draw_limb(surface, knee_r, foot_r, 10, SKIN, SKIN_LIGHT)
        draw_limb(surface, hip_l, knee_l, 11, SKIN_DARK, SKIN)
        draw_limb(surface, knee_l, foot_l, 10, SKIN, SKIN_LIGHT)

        # Shoes.
        for foot in [foot_l, foot_r]:
            fp = p(foot)
            shoe_rect = (fp[0] - 6 + f * 6, fp[1] - 6, 24, 16)
            pygame.draw.ellipse(surface, SHOE, shoe_rect)
            pygame.draw.ellipse(surface, (70, 70, 75), shoe_rect, 2)

        # ── Shorts ───────────────────────────────────────────────────
        shorts_pts = [
            p((f * -22, -8)),
            p((f * 22, -8)),
            p((f * 28, 32)),
            p((f * 6, 38)),
            p((f * -6, 38)),
            p((f * -28, 32)),
        ]
        pygame.draw.polygon(surface, self.shorts, shorts_pts)
        # Waistband.
        pygame.draw.line(surface, self.shorts_hi, p((f * -22, -8)), p((f * 22, -8)), 5)
        # Stripe.
        pygame.draw.line(surface, self.shorts_hi,
                         p((f * 0, -6)), p((f * 0, 35)), 3)

        # ── Torso ────────────────────────────────────────────────────
        ls = pose["lead_shoulder"]
        rs = pose["rear_shoulder"]
        torso_pts = [
            p(rs),
            p(ls),
            p((ls[0] - f * 4, -5)),
            p((rs[0] + f * 4, -5)),
        ]
        pygame.draw.polygon(surface, self.tank, torso_pts)
        # Chest highlight for 3D.
        chest_hi_pts = [
            p((lerp(ls[0], rs[0], 0.2), ls[1] + 5)),
            p(ls),
            p((ls[0] - f * 4, -30)),
            p((lerp(ls[0], rs[0], 0.3), -25)),
        ]
        pygame.draw.polygon(surface, self.tank_hi, chest_hi_pts)

        # Neck.
        neck_x = (ls[0] + rs[0]) / 2
        neck_top = ((ls[0] + rs[0]) / 2, min(ls[1], rs[1]) - 15)
        neck_bot = ((ls[0] + rs[0]) / 2, min(ls[1], rs[1]))
        draw_limb(surface, neck_bot, neck_top, 11, SKIN_DARK, SKIN)

        # ── Head ─────────────────────────────────────────────────────
        head_cx = cx + int(neck_x)
        head_cy = cy + int(min(ls[1], rs[1])) - 35
        head_pos = (head_cx, head_cy)

        # Head shadow.
        pygame.draw.circle(surface, SKIN_DARK, (head_cx + 1, head_cy + 1), 24)
        # Head.
        pygame.draw.circle(surface, SKIN, head_pos, 23)
        # Head highlight.
        pygame.draw.circle(surface, SKIN_LIGHT, (head_cx - 5, head_cy - 8), 9)

        # Hair.
        hair_rect = (head_cx - 25, head_cy - 27, 50, 28)
        pygame.draw.ellipse(surface, HAIR, hair_rect)

        # Eyes - looking toward opponent.
        eye_y = head_cy - 2
        eye_x1 = head_cx + f * 7
        eye_x2 = head_cx + f * -5
        pygame.draw.circle(surface, (255, 255, 255), (eye_x1, eye_y), 5)
        pygame.draw.circle(surface, (255, 255, 255), (eye_x2, eye_y), 4)
        pygame.draw.circle(surface, (30, 30, 30), (eye_x1 + f * 2, eye_y), 3)
        pygame.draw.circle(surface, (30, 30, 30), (eye_x2 + f * 2, eye_y), 2)

        # Mouth.
        if hit_flash:
            pygame.draw.ellipse(surface, (180, 50, 50),
                                (head_cx - 5, head_cy + 8, 10, 8))
        else:
            pygame.draw.line(surface, SKIN_DARK,
                             (head_cx - 4, head_cy + 10),
                             (head_cx + 4, head_cy + 10), 2)

        # ── Arms (draw order depends on which is in front) ───────────
        def draw_back_arm():
            draw_limb(surface, pose["rear_shoulder"], pose["rear_elbow"], 10, SKIN_DARK)
            draw_limb(surface, pose["rear_elbow"], pose["rear_fist"], 9, SKIN_DARK, SKIN)
            is_active = not self.anim_done and self.pose in ("cross", "uppercut")
            gc = self.glove_hi if is_active else self.glove
            ghi = (min(255, gc[0]+50), min(255, gc[1]+50), min(255, gc[2]+50))
            draw_glove(surface, pose["rear_fist"], pose["rear_glove_size"], gc, ghi)

        def draw_front_arm():
            draw_limb(surface, pose["lead_shoulder"], pose["lead_elbow"], 10, SKIN_DARK, SKIN)
            draw_limb(surface, pose["lead_elbow"], pose["lead_fist"], 9, SKIN, SKIN_LIGHT)
            is_active = not self.anim_done and self.pose in ("jab", "hook")
            gc = self.glove_hi if is_active else self.glove
            ghi = (min(255, gc[0]+50), min(255, gc[1]+50), min(255, gc[2]+50))
            draw_glove(surface, pose["lead_fist"], pose["lead_glove_size"], gc, ghi)

        if pose["lead_in_front"]:
            draw_back_arm()
            draw_front_arm()
        else:
            draw_front_arm()
            draw_back_arm()

        # ── Hit flash ────────────────────────────────────────────────
        if hit_flash:
            flash_surf = pygame.Surface((90, 200), pygame.SRCALPHA)
            flash_surf.fill((255, 255, 255, 45))
            surface.blit(flash_surf, (cx - 45, cy - 160))

        # ── Impact burst effect ──────────────────────────────────────
        if self.shake_t > 0.5:
            burst_x = cx - f * 30
            burst_y = cy - 100
            for i in range(6):
                angle = i * math.pi / 3 + self.shake_t * 10
                length = 12 + self.shake_t * 15
                ex = int(burst_x + math.cos(angle) * length)
                ey = int(burst_y + math.sin(angle) * length)
                pygame.draw.line(surface, (255, 255, 200),
                                 (burst_x, burst_y), (ex, ey), 3)


# ── Drawing Helpers ──────────────────────────────────────────────────────────

def draw_health_bars(surface, player_hp, opponent_hp, max_hp=100):
    bar_w = 350
    bar_h = 28
    y = 22
    r = 3

    # Player HP.
    bg = pygame.Rect(30, y, bar_w, bar_h)
    pygame.draw.rect(surface, (40, 40, 50), bg, border_radius=r)
    pw = int(bar_w * max(player_hp, 0) / max_hp)
    if pw > 0:
        color = (30, 180, 60) if player_hp > 30 else (220, 180, 30)
        pygame.draw.rect(surface, color, (30, y, pw, bar_h), border_radius=r)
    pygame.draw.rect(surface, (180, 180, 180), bg, 2, border_radius=r)

    # Opponent HP.
    rx = SCREEN_W - 30 - bar_w
    bg2 = pygame.Rect(rx, y, bar_w, bar_h)
    pygame.draw.rect(surface, (40, 40, 50), bg2, border_radius=r)
    ow = int(bar_w * max(opponent_hp, 0) / max_hp)
    if ow > 0:
        color = (200, 50, 50) if opponent_hp > 30 else (220, 180, 30)
        pygame.draw.rect(surface, color, (rx + bar_w - ow, y, ow, bar_h), border_radius=r)
    pygame.draw.rect(surface, (180, 180, 180), bg2, 2, border_radius=r)


def draw_text(surface, text, x, y, font, color=C_TEXT):
    rendered = font.render(text, True, color)
    surface.blit(rendered, (x, y))


def draw_text_centered(surface, text, cx, y, font, color=C_TEXT):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(center=(cx, y))
    surface.blit(rendered, rect)


def draw_floor(surface):
    floor_y = 525
    # Main floor line.
    pygame.draw.line(surface, (70, 70, 95), (0, floor_y), (SCREEN_W, floor_y), 3)
    # Ring ropes (background detail).
    for ry in [120, 220, 330]:
        pygame.draw.line(surface, (50, 50, 65), (0, ry), (SCREEN_W, ry), 1)
    # Floor gradient.
    for i in range(12):
        c = max(0, 30 - i * 2)
        color = (c + 5, c + 5, c + 15)
        pygame.draw.line(surface, color, (0, floor_y + 3 + i * 5), (SCREEN_W, floor_y + 3 + i * 5), 5)


# ── Main Game Loop ───────────────────────────────────────────────────────────

def main():
    print("Loading RF model...")
    rf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model loaded.")

    camera = CameraThread(rf, scaler)
    camera.start()
    print("Camera thread started.")

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Boxing Game")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("consolas", 72, bold=True)
    font_med = pygame.font.SysFont("consolas", 36, bold=True)
    font_sm = pygame.font.SysFont("consolas", 20)
    font_action = pygame.font.SysFont("consolas", 28, bold=True)

    # Fighter positions.
    CORNER_PLAYER_X = 80       # Left corner of ring.
    CORNER_OPPONENT_X = 944    # Right corner of ring.
    FIGHT_PLAYER_X = 300       # Fighting position.
    FIGHT_OPPONENT_X = 724     # Fighting position.
    FIGHTER_Y = 420

    # Fighters (start in corners).
    player = Fighter(x=CORNER_PLAYER_X, y=FIGHTER_Y, facing_right=True, is_player=True)
    opponent = Fighter(x=CORNER_OPPONENT_X, y=FIGHTER_Y, facing_right=False, is_player=False)

    # Game state machine: INTRO -> COUNTDOWN -> WALK_IN -> FIGHTING -> KO -> (back to COUNTDOWN)
    game_state = "INTRO"
    state_timer = 0.0
    countdown_num = 3
    walk_in_progress = 0.0  # 0.0 = corners, 1.0 = fight positions.

    # Fight state.
    player_hp = 100
    opponent_hp = 100
    last_action = ""
    last_confidence = 0.0
    action_display_until = 0.0
    hit_flash_until = 0.0
    ko_time = 0.0
    round_num = 1

    running = True
    print("\nGame running! Just punch - no buttons needed! R to reset, Q/ESC to quit.\n")

    while running:
        now = time.time()
        dt = clock.get_time() / 1000.0  # Delta time in seconds.

        # ── Events ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    # Reset to intro.
                    game_state = "INTRO"
                    state_timer = 0.0
                    player.x = CORNER_PLAYER_X
                    opponent.x = CORNER_OPPONENT_X
                    opponent_hp = 100
                    player_hp = 100
                    last_action = ""
                    player.pose = "idle"
                    player.anim_done = True
                    opponent.pose = "idle"
                    opponent.anim_done = True
                    walk_in_progress = 0.0
                    round_num += 1
                    print(f"  Reset! Round {round_num}")
                elif event.key == pygame.K_SPACE and game_state == "INTRO":
                    # Start countdown.
                    game_state = "COUNTDOWN"
                    countdown_num = 3
                    state_timer = now
                    print("  Countdown started!")

        # ── Game state machine ──────────────────────────────────────────
        if game_state == "INTRO":
            # Waiting for player to press SPACE.
            # Drain any pending actions so they don't fire later.
            with camera.lock:
                camera.pending_action = None

        elif game_state == "COUNTDOWN":
            elapsed = now - state_timer
            countdown_num = 3 - int(elapsed)
            if countdown_num <= 0:
                game_state = "WALK_IN"
                walk_in_progress = 0.0
                state_timer = now
                print("  FIGHT! Walking in...")

        elif game_state == "WALK_IN":
            # Fighters walk from corners to fight positions over 2 seconds.
            walk_in_progress = min(1.0, (now - state_timer) / 2.0)
            t = ease_out_cubic(walk_in_progress)
            player.x = int(lerp(CORNER_PLAYER_X, FIGHT_PLAYER_X, t))
            opponent.x = int(lerp(CORNER_OPPONENT_X, FIGHT_OPPONENT_X, t))

            if walk_in_progress >= 1.0:
                game_state = "FIGHTING"
                print("  CV active - throw punches!")
                # Drain stale actions from walk-in.
                with camera.lock:
                    camera.pending_action = None

        elif game_state == "FIGHTING":
            # Check for new classified action from camera thread.
            with camera.lock:
                action_data = camera.pending_action
                camera.pending_action = None

            if action_data is not None:
                action, conf = action_data
                last_action = action
                last_confidence = conf

                # Trigger player animation.
                player.trigger_action(action)

                dmg = DAMAGE.get(action, 0)
                if dmg > 0:
                    opponent_hp = max(0, opponent_hp - dmg)
                    hit_flash_until = now + 0.4
                    opponent.take_hit()

                action_display_until = now + 2.0

                if opponent_hp <= 0:
                    ko_time = now
                    game_state = "KO"
                    print(f"  KO! Round {round_num} complete!")

        elif game_state == "KO":
            if now - ko_time > 4.0:
                # Reset for next round.
                round_num += 1
                opponent_hp = 100
                player_hp = 100
                last_action = ""
                player.pose = "idle"
                player.anim_done = True
                opponent.pose = "idle"
                opponent.anim_done = True
                # Go back to corners for next round.
                game_state = "COUNTDOWN"
                countdown_num = 3
                state_timer = now
                player.x = CORNER_PLAYER_X
                opponent.x = CORNER_OPPONENT_X
                walk_in_progress = 0.0
                print(f"  Round {round_num} - FIGHT!")

        # Update animations.
        player.update()
        opponent.update()

        # ── Drawing ─────────────────────────────────────────────────────
        screen.fill(C_BG)
        draw_floor(screen)

        # Health bars (only show during fighting/KO).
        if game_state in ("FIGHTING", "KO"):
            draw_health_bars(screen, player_hp, opponent_hp)
            draw_text(screen, "YOU", 35, 54, font_sm, (50, 220, 80))
            draw_text(screen, "OPPONENT", SCREEN_W - 35 - 120, 54, font_sm, (220, 60, 60))

        draw_text_centered(screen, f"Round {round_num}", SCREEN_W // 2, 36, font_sm)

        # Fighters.
        opp_hit = now < hit_flash_until
        opponent.draw(screen, hit_flash=opp_hit)
        player.draw(screen)

        # ── State-specific overlays ─────────────────────────────────────
        if game_state == "INTRO":
            draw_text_centered(screen, "BOXING GAME", SCREEN_W // 2, SCREEN_H // 2 - 80,
                               font_big, (255, 215, 0))
            draw_text_centered(screen, "Press SPACE to start",
                               SCREEN_W // 2, SCREEN_H // 2, font_med, C_TEXT)
            draw_text_centered(screen, "Your body is the controller",
                               SCREEN_W // 2, SCREEN_H // 2 + 50, font_sm, (150, 150, 150))

        elif game_state == "COUNTDOWN":
            # Big countdown number.
            num = max(1, countdown_num)
            colors = {3: (255, 80, 80), 2: (255, 200, 50), 1: (80, 255, 80)}
            c = colors.get(num, (255, 255, 255))
            draw_text_centered(screen, str(num), SCREEN_W // 2, SCREEN_H // 2, font_big, c)

        elif game_state == "WALK_IN":
            # "FIGHT!" text fading in.
            alpha = int(walk_in_progress * 255)
            fight_color = (min(255, 80 + alpha), max(0, 255 - alpha), 80)
            draw_text_centered(screen, "FIGHT!", SCREEN_W // 2, SCREEN_H // 2 - 30,
                               font_big, fight_color)

        elif game_state == "KO":
            draw_text_centered(screen, "K.O.!", SCREEN_W // 2, SCREEN_H // 2 - 30,
                               font_big, (255, 50, 50))
            draw_text_centered(screen, f"Round {round_num} won!",
                               SCREEN_W // 2, SCREEN_H // 2 + 40, font_med, C_TEXT)

        elif game_state == "FIGHTING":
            # Action display.
            if now < action_display_until and last_action:
                ac = ACTION_COLORS.get(last_action, C_TEXT)
                draw_text_centered(screen, last_action.upper(),
                                   SCREEN_W // 2, SCREEN_H - 55, font_action, ac)
                draw_text_centered(screen, f"{last_confidence*100:.0f}%",
                                   SCREEN_W // 2, SCREEN_H - 25, font_sm, ac)
                dmg = DAMAGE.get(last_action, 0)
                if dmg > 0:
                    draw_text_centered(screen, f"-{dmg} HP",
                                       opponent.x, 100, font_med, (255, 80, 80))
            else:
                draw_text_centered(screen, "Throw a punch!",
                                   SCREEN_W // 2, SCREEN_H - 50, font_sm, (100, 255, 100))

        # Webcam preview (top-left).
        with camera.lock:
            preview = camera.preview_rgb
            has_pose = camera.has_pose

        if preview is not None:
            surf = pygame.surfarray.make_surface(np.transpose(preview, (1, 0, 2)))
            px, py = 10, 80
            pygame.draw.rect(screen, (80, 80, 100),
                             (px - 2, py - 2, PREVIEW_W + 4, PREVIEW_H + 4), border_radius=3)
            screen.blit(surf, (px, py))
            # Pose status indicator.
            indicator_color = (50, 220, 80) if has_pose else (220, 60, 60)
            pygame.draw.circle(screen, indicator_color, (px + PREVIEW_W - 12, py + 12), 6)

        pygame.display.flip()
        clock.tick(30)

    camera.stop()
    pygame.quit()
    print("Game ended.")


if __name__ == "__main__":
    main()
