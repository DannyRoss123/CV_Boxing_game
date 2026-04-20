"""
Boxing Memory — Pygame

Simon Says with punches. A combo flashes on screen, disappears,
then you throw the punches from memory in order.

Controls:
    SPACE  - start / retry
    Q/ESC  - quit
"""

import collections
import json
import math
import os
import random
import sys
import threading
import time

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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(ROOT, "pose_landmarker_lite.task")
RF_PATH    = os.path.join(ROOT, "models", "random_forest.joblib")
SCALER_PATH = os.path.join(ROOT, "models", "rf_scaler.joblib")

# ── CV constants (must match training) ────────────────────────────────────────
SEQ_LEN   = 60
N_LM      = 33
F_PER_LM  = 4

L_HIP  = 23 * 4;  R_HIP  = 24 * 4
L_SHO  = 11 * 4;  R_SHO  = 12 * 4
RWX = 16*4; RWY = 16*4+1
LWX = 15*4; LWY = 15*4+1
REX = 14*4; REY = 14*4+1
LEX = 13*4; LEY = 13*4+1

BUF_SIZE          = 120
VEL_THRESH        = 0.015
MOTION_START      = 3
MOTION_STOP       = 10
LEAD_IN           = 18
TAIL              = 12
COOLDOWN          = 1.0
MIN_CONF          = 0.50

# ── Display ───────────────────────────────────────────────────────────────────
W, H  = 1280, 720
FPS   = 60
CAM_W, CAM_H = 640, 480   # resized camera preview

PUNCH_ACTIONS = ["jab", "cross", "hook", "uppercut"]

COLORS = {
    "jab":        (255, 155,  50),
    "cross":      (255,  55,  55),
    "hook":       (255, 215,  50),
    "uppercut":   ( 50, 220, 255),
    "block_high": ( 50, 130, 255),
    "idle":       (130, 130, 130),
}

BG_DARK   = (10, 10, 18)
WHITE     = (255, 255, 255)
GREY      = (140, 140, 150)
DIM       = ( 45,  45,  55)
DIM2      = ( 65,  65,  75)
GREEN     = ( 60, 220,  90)
RED       = (220,  45,  45)
GOLD      = (255, 210,  50)


SCORES_PATH = os.path.join(ROOT, "data", "boxing_memory_scores.json")


def level_config(lvl: int):
    """Return (sequence_length, show_seconds) for a given level."""
    length   = min(2 + (lvl - 1) // 2, 7)
    show_sec = max(1.5, 3.6 - lvl * 0.14)
    return length, show_sec


def load_high_score() -> dict:
    try:
        with open(SCORES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"hi_score": 0, "hi_level": 1}


def save_high_score(data: dict):
    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── CV helpers ─────────────────────────────────────────────────────────────────

def lm_to_arr(lms):
    row = []
    for lm in lms:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(row, dtype=np.float32)


def normalize(frame):
    frame = frame.copy()
    cx = (frame[L_HIP]   + frame[R_HIP])   / 2
    cy = (frame[L_HIP+1] + frame[R_HIP+1]) / 2
    cz = (frame[L_HIP+2] + frame[R_HIP+2]) / 2
    sx = (frame[L_SHO]   + frame[R_SHO])   / 2
    sy = (frame[L_SHO+1] + frame[R_SHO+1]) / 2
    sz = (frame[L_SHO+2] + frame[R_SHO+2]) / 2
    torso = max(math.sqrt((sx-cx)**2 + (sy-cy)**2 + (sz-cz)**2), 0.01)
    for i in range(N_LM):
        idx = i * F_PER_LM
        frame[idx]   = (frame[idx]   - cx) / torso
        frame[idx+1] = (frame[idx+1] - cy) / torso
        frame[idx+2] = (frame[idx+2] - cz) / torso
    return frame


def extract_stats(frames):
    X = np.array(frames, dtype=np.float32)
    if X.shape[0] < SEQ_LEN:
        deficit = SEQ_LEN - X.shape[0]
        pb = deficit // 2; pa = deficit - pb
        parts = []
        if pb: parts.append(np.tile(X[:1], (pb, 1)))
        parts.append(X)
        if pa: parts.append(np.tile(X[-1:], (pa, 1)))
        X = np.vstack(parts)
    elif X.shape[0] > SEQ_LEN:
        s = (X.shape[0] - SEQ_LEN) // 2
        X = X[s:s + SEQ_LEN]
    return np.concatenate([np.std(X, 0), np.max(X, 0) - np.min(X, 0), X[-1] - X[0]])


def vel(a, b):
    if a is None or b is None:
        return 0.0
    t = 0.0
    for ix, iy in [(RWX,RWY),(LWX,LWY),(REX,REY),(LEX,LEY)]:
        t += math.sqrt((b[ix]-a[ix])**2 + (b[iy]-a[iy])**2)
    return t / 4


# ── CV Thread ──────────────────────────────────────────────────────────────────

class CVThread(threading.Thread):
    def __init__(self, rf, scaler):
        super().__init__(daemon=True)
        self.rf = rf; self.scaler = scaler
        self.running = True
        self._lock  = threading.Lock()
        self._queue = []           # (action, confidence, peak_vel)
        self._vel   = 0.0
        self._mstate = "IDLE"
        self._frame  = None        # latest BGR frame

    def pop_actions(self):
        with self._lock:
            q = self._queue[:]
            self._queue.clear()
        return q

    @property
    def velocity(self):
        with self._lock: return self._vel

    @property
    def motion_state(self):
        with self._lock: return self._mstate

    @property
    def latest_frame(self):
        with self._lock: return self._frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open webcam.")
            return

        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        lmk = PoseLandmarker.create_from_options(opts)
        ts  = 0
        buf = collections.deque(maxlen=BUF_SIZE)
        prev = None
        ms = "IDLE"
        mc = sc = mi = 0
        last_t = 0.0
        peak   = 0.0

        while self.running:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame.copy()

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts  += 33
            res  = lmk.detect_for_video(img, ts)
            has  = bool(res.pose_landmarks)

            v = 0.0
            if has:
                raw  = lm_to_arr(res.pose_landmarks[0])
                norm = normalize(raw)
                buf.append(norm)
                v = vel(prev, norm)
                prev = norm
            else:
                prev = None

            now = time.time()

            if ms == "IDLE":
                if v > VEL_THRESH:
                    mc += 1
                    if mc >= MOTION_START:
                        mi = max(0, len(buf) - mc - LEAD_IN)
                        ms = "MOTION"; sc = 0; peak = 0.0
                else:
                    mc = 0

            elif ms == "MOTION":
                peak = max(peak, v)
                if v < VEL_THRESH:
                    sc += 1
                    if sc >= MOTION_STOP:
                        self._classify(buf, mi, len(buf), peak)
                        last_t = now; ms = "COOLDOWN"; mc = sc = 0
                else:
                    sc = 0
                    if len(buf) - mi > 90:
                        self._classify(buf, mi, len(buf), peak)
                        last_t = now; ms = "COOLDOWN"; mc = sc = 0

            elif ms == "COOLDOWN":
                if now - last_t > COOLDOWN:
                    ms = "IDLE"; mc = 0

            with self._lock:
                self._vel = v
                self._mstate = ms

        lmk.close(); cap.release()

    def _classify(self, buf, si, ei, peak):
        bl = list(buf)
        ei = min(len(bl), ei + TAIL)
        seg = ei - si
        if seg < SEQ_LEN:
            deficit = SEQ_LEN - seg
            eb = deficit // 2; ea = deficit - eb
            ns = max(0, si - eb); ne = min(len(bl), ei + ea)
            ab = si - ns; aa = ne - ei
            if ab < eb: ne = min(len(bl), ne + (eb - ab))
            elif aa < ea: ns = max(0, ns - (ea - aa))
            si, ei = ns, ne
        seg = bl[si:ei]
        if len(seg) < 10: return

        feats = extract_stats(seg)
        fs    = self.scaler.transform(feats.reshape(1, -1))
        act   = self.rf.predict(fs)[0]
        prob  = self.rf.predict_proba(fs)[0]
        conf  = float(np.max(prob))

        ranked = sorted(zip(self.rf.classes_, prob), key=lambda x: -x[1])
        print(f"  >> {act.upper():>12s} ({conf*100:.0f}%)  "
              + "  ".join(f"{a}:{p*100:.0f}%" for a, p in ranked))

        if act == "idle" or conf < MIN_CONF:
            return
        with self._lock:
            self._queue.append((act, conf, peak))


# ── Pygame helpers ─────────────────────────────────────────────────────────────

def make_fonts():
    pygame.font.init()
    names = ["Segoe UI", "Arial", "Helvetica", None]
    def get(size, bold=False):
        for n in names:
            try:
                return pygame.font.SysFont(n, size, bold=bold)
            except Exception:
                pass
        return pygame.font.Font(None, size)
    return {
        "title": get(82, bold=True),
        "big":   get(64, bold=True),
        "med":   get(38, bold=True),
        "sm":    get(24),
        "xs":    get(17),
    }


def blit_centered(surf, font, text, color, cx, cy, alpha=255):
    s = font.render(text, True, color)
    if alpha < 255:
        s = s.copy(); s.set_alpha(alpha)
    surf.blit(s, (cx - s.get_width() // 2, cy - s.get_height() // 2))
    return s.get_width(), s.get_height()


def draw_rounded_rect(surf, color, rect, radius=12, border=0, border_color=None, alpha=255):
    r = pygame.Rect(rect)
    tmp = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
    pygame.draw.rect(tmp, (*color, alpha), tmp.get_rect(), border_radius=radius)
    if border and border_color:
        pygame.draw.rect(tmp, (*border_color, alpha), tmp.get_rect(), border, border_radius=radius)
    surf.blit(tmp, r.topleft)


def frame_to_surf(frame, size):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


# ── Game ───────────────────────────────────────────────────────────────────────

class Game:
    INTRO     = "INTRO"
    SHOW      = "SHOW"
    READY     = "READY"
    PLAYING   = "PLAYING"
    FEEDBACK  = "FEEDBACK"   # brief pause after each correct punch
    WRONG     = "WRONG"
    LEVEL_UP  = "LEVEL_UP"
    GAME_OVER = "GAME_OVER"

    def __init__(self, screen, cv: CVThread, fonts: dict):
        self.screen = screen
        self.cv     = cv
        self.F      = fonts

        self.state  = self.INTRO
        self.timer  = 0.0

        self.level    = 1
        self.score    = 0

        saved = load_high_score()
        self.hi_score = saved["hi_score"]
        self.hi_level = saved["hi_level"]

        self.sequence  = []
        self.progress  = []
        self.show_time = 3.0
        self.last_act  = None   # (action, correct, conf)

        self.flash       = None      # [r,g,b, alpha]
        self.particles   = []        # [x,y,vx,vy,color,life,max_life]
        self.floats      = []        # [text,color,x,y,life,max_life]
        self._cam_surf   = None

    # ── state helpers ─────────────────────────────────────────────────────────

    def _enter(self, state):
        self.state = state
        self.timer = 0.0

    def new_level(self):
        length, self.show_time = level_config(self.level)
        self.sequence = [random.choice(PUNCH_ACTIONS) for _ in range(length)]
        self.progress = []
        self._enter(self.SHOW)

    def _slot_rect(self, idx, n):
        sw = min(150, (W - 120) // n)
        gap = min(14, sw // 8)
        total = n * sw + (n - 1) * gap
        x0 = W // 2 - total // 2
        x  = x0 + idx * (sw + gap)
        return pygame.Rect(x, H - 165, sw, 75)

    def _fx_particles(self, x, y, color, n=22):
        for _ in range(n):
            a = random.uniform(0, math.tau)
            spd = random.uniform(2, 9)
            life = random.uniform(0.35, 0.75)
            self.particles.append([x, y, spd*math.cos(a), spd*math.sin(a), color, life, life])

    def _fx_float(self, text, color, x, y):
        self.floats.append([text, color, float(x), float(y), 0.9, 0.9])

    # ── process incoming punch ────────────────────────────────────────────────

    def on_punch(self, action, conf, peak):
        if self.state != self.PLAYING:
            return
        idx      = len(self.progress)
        expected = self.sequence[idx]
        correct  = (action == expected)
        self.last_act = (action, correct, conf)

        if correct:
            self.progress.append(action)
            self.score += 10 + self.level * 5
            r = self._slot_rect(idx, len(self.sequence))
            self._fx_particles(r.centerx, r.centery, COLORS[action])
            self._fx_float(action.upper(), COLORS[action], r.centerx, r.top - 20)
            self.flash = [50, 210, 80, 130]
            self._enter(self.FEEDBACK)
        else:
            # Save high score then show game over
            new_best = False
            if self.score > self.hi_score or self.level > self.hi_level:
                self.hi_score = max(self.hi_score, self.score)
                self.hi_level = max(self.hi_level, self.level)
                save_high_score({"hi_score": self.hi_score, "hi_level": self.hi_level})
                new_best = True
            self._new_best = new_best
            self.flash = [210, 40, 40, 200]
            self._enter(self.GAME_OVER)

    # ── update ────────────────────────────────────────────────────────────────

    def update(self, dt):
        self.timer += dt

        # particles
        for p in self.particles[:]:
            p[0] += p[2]; p[1] += p[3]; p[3] += 0.18
            p[5] -= dt
            if p[5] <= 0: self.particles.remove(p)

        # floats
        for f in self.floats[:]:
            f[3] -= 45 * dt; f[4] -= dt
            if f[4] <= 0: self.floats.remove(f)

        # screen flash decay
        if self.flash:
            self.flash[3] = max(0, self.flash[3] - 320 * dt)
            if self.flash[3] <= 0: self.flash = None

        if self.state == self.SHOW:
            if self.timer >= self.show_time:
                self._enter(self.READY)

        elif self.state == self.READY:
            if self.timer >= 1.1:
                self._enter(self.PLAYING)

        elif self.state == self.FEEDBACK:
            if self.timer >= 0.45:
                if len(self.progress) == len(self.sequence):
                    self._enter(self.LEVEL_UP)
                else:
                    self._enter(self.PLAYING)

        elif self.state == self.LEVEL_UP:
            if self.timer >= 2.2:
                self.level += 1
                if self.level > self.hi_level:
                    self.hi_level = self.level
                    self.hi_score = max(self.hi_score, self.score)
                    save_high_score({"hi_score": self.hi_score, "hi_level": self.hi_level})
                self.new_level()

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self):
        # Camera background
        frame = self.cv.latest_frame
        if frame is not None:
            self._cam_surf = frame_to_surf(frame, (W, H))
        if self._cam_surf:
            self.screen.blit(self._cam_surf, (0, 0))
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 168))
            self.screen.blit(ov, (0, 0))
        else:
            self.screen.fill(BG_DARK)

        # Screen flash
        if self.flash and self.flash[3] > 0:
            fl = pygame.Surface((W, H), pygame.SRCALPHA)
            fl.fill((*self.flash[:3], int(self.flash[3])))
            self.screen.blit(fl, (0, 0))

        # State-specific drawing
        if   self.state == self.INTRO:     self._d_intro()
        elif self.state == self.SHOW:      self._d_show()
        elif self.state == self.READY:     self._d_ready()
        elif self.state in (self.PLAYING, self.FEEDBACK): self._d_playing()
        elif self.state == self.LEVEL_UP:  self._d_levelup()
        elif self.state == self.GAME_OVER: self._d_gameover()

        # Always-on HUD and velocity bar
        self._d_hud()
        self._d_vel_bar()

        # Particles
        for p in self.particles:
            lr = max(0.0, p[5] / p[6])
            r  = max(2, int(6 * lr))
            pygame.draw.circle(self.screen, p[4], (int(p[0]), int(p[1])), r)

        # Floating labels
        for f in self.floats:
            lr = max(0.0, f[4] / f[5])
            blit_centered(self.screen, self.F["med"], f[0], f[1], int(f[2]), int(f[3]), int(255 * lr))

    # ── individual draw methods ───────────────────────────────────────────────

    def _d_intro(self):
        blit_centered(self.screen, self.F["title"], "BOXING MEMORY", GOLD, W//2, H//2 - 100)
        blit_centered(self.screen, self.F["med"], "Simon Says — With Punches", GREY, W//2, H//2 - 20)
        blit_centered(self.screen, self.F["sm"],
                      "Memorise the combo  ·  It disappears  ·  Throw it from memory",
                      GREY, W//2, H//2 + 40)
        pulse_a = int(180 + 75 * math.sin(time.time() * 2.5))
        blit_centered(self.screen, self.F["med"], "PRESS SPACE TO START",
                      (pulse_a, pulse_a, pulse_a), W//2, H//2 + 115)

    def _d_show(self):
        t = self.timer / self.show_time
        # alpha: fade in 0→0.12, full 0.12→0.85, fade out 0.85→1.0
        if t < 0.12:
            alpha = int(255 * (t / 0.12))
        elif t > 0.85:
            alpha = int(255 * (1.0 - (t - 0.85) / 0.15))
        else:
            alpha = 255

        blit_centered(self.screen, self.F["med"],
                      f"LEVEL {self.level}  —  MEMORISE THIS COMBO",
                      GREY, W//2, 105, alpha)

        n  = len(self.sequence)
        sw = min(160, (W - 120) // n)
        gap = min(16, sw // 7)
        total = n * sw + (n - 1) * gap
        x0 = W // 2 - total // 2

        for i, punch in enumerate(self.sequence):
            x   = x0 + i * (sw + gap)
            col = COLORS[punch]
            rect = pygame.Rect(x, H // 2 - 55, sw, 110)
            draw_rounded_rect(self.screen, col, rect, radius=14,
                              border=3, border_color=WHITE, alpha=int(alpha * 0.85))
            blit_centered(self.screen, self.F["med"], punch.upper(), WHITE,
                          rect.centerx, rect.centery - 8, alpha)
            blit_centered(self.screen, self.F["xs"], str(i + 1), GREY,
                          rect.x + 14, rect.y + 12, alpha)

    def _d_ready(self):
        alpha = min(255, int(255 * self.timer / 0.25))
        blit_centered(self.screen, self.F["big"], "GO!", WHITE, W // 2, H // 2 - 30, alpha)
        self._d_slots()

    def _d_playing(self):
        idx = len(self.progress)
        if idx < len(self.sequence):
            # step counter only — no punch name shown (that's the whole game)
            blit_centered(self.screen, self.F["med"],
                          f"Punch {idx + 1} of {len(self.sequence)}",
                          GREY, W // 2, H // 2 - 30)

            # motion indicator
            if self.cv.motion_state == "MOTION":
                a = int(180 + 75 * math.sin(time.time() * 8))
                blit_centered(self.screen, self.F["sm"],
                              "● CLASSIFYING...", (a, 80, 80), W // 2, H // 2 + 25)
            elif self.cv.motion_state == "COOLDOWN":
                blit_centered(self.screen, self.F["sm"], "● COOLDOWN", (60, 160, 255),
                              W // 2, H // 2 + 25)

        self._d_slots()

    def _d_gameover(self):
        t = self.timer
        alpha = min(255, int(255 * t / 0.2))

        blit_centered(self.screen, self.F["title"], "GAME OVER", RED, W // 2, H // 2 - 130, alpha)

        if self.last_act:
            action, _, conf = self.last_act
            expected = self.sequence[len(self.progress)]
            blit_centered(self.screen, self.F["med"],
                          f"You threw  {action.upper()}  —  Expected  {expected.upper()}",
                          (210, 210, 210), W // 2, H // 2 - 45, alpha)

        blit_centered(self.screen, self.F["med"],
                      f"Level reached:  {self.level}     Score:  {self.score}",
                      GOLD, W // 2, H // 2 + 20, alpha)

        if getattr(self, "_new_best", False):
            pulse = int(180 + 75 * math.sin(time.time() * 4))
            blit_centered(self.screen, self.F["med"], "★  NEW HIGH SCORE  ★",
                          (pulse, 220, 80), W // 2, H // 2 + 80, alpha)
        else:
            blit_centered(self.screen, self.F["sm"],
                          f"Best:  Level {self.hi_level}   Score {self.hi_score}",
                          GREY, W // 2, H // 2 + 80, alpha)

        if t > 1.0:
            pulse_a = int(160 + 95 * math.sin(time.time() * 2.5))
            blit_centered(self.screen, self.F["med"], "PRESS SPACE TO TRY AGAIN",
                          (pulse_a, pulse_a, pulse_a), W // 2, H // 2 + 150)

    def _d_levelup(self):
        t = self.timer
        alpha = min(255, int(255 * t / 0.2))
        blit_centered(self.screen, self.F["big"],
                      f"LEVEL {self.level} COMPLETE!", GREEN, W // 2, H // 2 - 70, alpha)
        blit_centered(self.screen, self.F["med"],
                      f"Score:  {self.score}", GOLD, W // 2, H // 2 + 10, alpha)
        blit_centered(self.screen, self.F["sm"],
                      f"Next up: {level_config(self.level + 1)[0]} punches", GREY,
                      W // 2, H // 2 + 65, alpha)

    def _d_slots(self):
        n = len(self.sequence)
        for i in range(n):
            r   = self._slot_rect(i, n)
            if i < len(self.progress):
                # completed — filled
                punch = self.progress[i]
                col   = COLORS[punch]
                draw_rounded_rect(self.screen, col, r, radius=10)
                pygame.draw.rect(self.screen, WHITE, r, 2, border_radius=10)
                blit_centered(self.screen, self.F["sm"], punch.upper(), (0, 0, 0),
                              r.centerx, r.centery)
                # tick
                blit_centered(self.screen, self.F["xs"], "✓", (0, 0, 0),
                              r.x + 12, r.y + 10)
            elif i == len(self.progress) and self.state in (self.PLAYING, self.READY):
                # current — pulsing border
                pulse = 0.5 + 0.5 * math.sin(time.time() * 6)
                target = self.sequence[i]
                col    = COLORS[target]
                draw_rounded_rect(self.screen, DIM, r, radius=10,
                                  border=max(2, int(3 + 2 * pulse)), border_color=col,
                                  alpha=255)
                blit_centered(self.screen, self.F["med"], "?", col, r.centerx, r.centery)
            else:
                # future — dim
                draw_rounded_rect(self.screen, DIM, r, radius=10,
                                  border=2, border_color=DIM2, alpha=255)

        # progress text under slots
        blit_centered(self.screen, self.F["xs"],
                      f"{len(self.progress)} / {len(self.sequence)}",
                      GREY, W // 2, H - 72)

    def _d_hud(self):
        hud = pygame.Surface((W, 52), pygame.SRCALPHA)
        hud.fill((0, 0, 0, 160))
        self.screen.blit(hud, (0, 0))
        blit_centered(self.screen, self.F["sm"], f"LEVEL  {self.level}", (180, 180, 255), 110, 26)
        blit_centered(self.screen, self.F["sm"], f"SCORE  {self.score}", GOLD, W // 2, 26)
        blit_centered(self.screen, self.F["sm"], f"BEST  {self.hi_score}", (150, 255, 160), W - 110, 26)

    def _d_vel_bar(self):
        v      = self.cv.velocity
        ms     = self.cv.motion_state
        bw, bh = 420, 14
        bx     = W // 2 - bw // 2
        by     = H - 22

        # track
        pygame.draw.rect(self.screen, (28, 28, 38), (bx, by, bw, bh), border_radius=7)

        # fill
        fill = int(min(v / (VEL_THRESH * 3), 1.0) * bw)
        if fill > 0:
            col = (215, 50, 50) if ms == "MOTION" else (50, 195, 75) if ms == "IDLE" else (50, 170, 255)
            pygame.draw.rect(self.screen, col, (bx, by, fill, bh), border_radius=7)

        # threshold tick
        tx = bx + int(bw / 3)
        pygame.draw.rect(self.screen, (0, 210, 210), (tx, by - 2, 2, bh + 4))

        # state label
        state_col = {"IDLE": (70, 195, 75), "MOTION": (215, 50, 50), "COOLDOWN": (50, 170, 255)}
        blit_centered(self.screen, self.F["xs"], ms,
                      state_col.get(ms, GREY), bx - 48, by + bh // 2)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Loading model…")
    rf     = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model loaded.\n")

    cv = CVThread(rf, scaler)
    cv.start()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Boxing Memory")
    clock  = pygame.time.Clock()
    fonts  = make_fonts()
    game   = Game(screen, cv, fonts)

    while True:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cv.running = False
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    cv.running = False
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_SPACE:
                    if game.state == game.INTRO:
                        game.level = 1
                        game.score = 0
                        game.new_level()
                    elif game.state == game.GAME_OVER and game.timer > 1.0:
                        game.level = 1
                        game.score = 0
                        game._new_best = False
                        game.new_level()

        for action, conf, peak in cv.pop_actions():
            game.on_punch(action, conf, peak)

        game.update(dt)
        game.draw()
        pygame.display.flip()


if __name__ == "__main__":
    main()
