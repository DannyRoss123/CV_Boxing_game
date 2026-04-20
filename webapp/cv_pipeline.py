"""
Server-side CV pipeline.
Browser sends raw MediaPipe landmark arrays (132 floats each).
Server normalizes, buffers, runs motion detection, classifies with RF.
No OpenCV or MediaPipe needed on server.
"""

import math
import time
import numpy as np
import joblib
from webapp.config import RF_PATH, SCALER_PATH

# Load models once at import time
_rf     = joblib.load(RF_PATH)
_scaler = joblib.load(SCALER_PATH)

SEQ_LEN = 60
N_LM    = 33
L_HIP=23*4; R_HIP=24*4; L_SHO=11*4; R_SHO=12*4
RWX=16*4; RWY=16*4+1; LWX=15*4; LWY=15*4+1
REX=14*4; REY=14*4+1; LEX=13*4; LEY=13*4+1

VEL_THRESH   = 0.013
MOTION_START = 3
MOTION_STOP  = 8
LEAD_IN      = 18
TAIL         = 12
COOLDOWN     = 0.6
MIN_CONF     = 0.30
BUF_MAX      = 120

PUNCH_ACTIONS = ["jab", "cross", "hook", "uppercut", "block_high"]


def normalize(fr: np.ndarray) -> np.ndarray:
    fr = fr.copy()
    cx = (fr[L_HIP]   + fr[R_HIP])   / 2
    cy = (fr[L_HIP+1] + fr[R_HIP+1]) / 2
    cz = (fr[L_HIP+2] + fr[R_HIP+2]) / 2
    sx = (fr[L_SHO]   + fr[R_SHO])   / 2
    sy = (fr[L_SHO+1] + fr[R_SHO+1]) / 2
    sz = (fr[L_SHO+2] + fr[R_SHO+2]) / 2
    t  = max(math.sqrt((sx-cx)**2 + (sy-cy)**2 + (sz-cz)**2), 0.01)
    for i in range(N_LM):
        idx = i * 4
        fr[idx]   = (fr[idx]   - cx) / t
        fr[idx+1] = (fr[idx+1] - cy) / t
        fr[idx+2] = (fr[idx+2] - cz) / t
    return fr


def _compute_vel(a, b) -> float:
    if a is None or b is None:
        return 0.0
    total = 0.0
    for ix, iy in [(RWX,RWY), (LWX,LWY), (REX,REY), (LEX,LEY)]:
        total += math.sqrt((b[ix]-a[ix])**2 + (b[iy]-a[iy])**2)
    return total / 4


def _extract_stats(frames) -> np.ndarray:
    X = np.array(frames, dtype=np.float32)
    if X.shape[0] < SEQ_LEN:
        d = SEQ_LEN - X.shape[0]
        pb = d // 2; pa = d - pb
        parts = []
        if pb: parts.append(np.tile(X[:1], (pb, 1)))
        parts.append(X)
        if pa: parts.append(np.tile(X[-1:], (pa, 1)))
        X = np.vstack(parts)
    elif X.shape[0] > SEQ_LEN:
        s = (X.shape[0] - SEQ_LEN) // 2
        X = X[s:s+SEQ_LEN]
    return np.concatenate([np.std(X, 0), np.max(X, 0)-np.min(X, 0), X[-1]-X[0]])


def classify_segment(segment) -> tuple[str, float]:
    if len(segment) < 10:
        return "idle", 0.0
    feats  = _extract_stats(segment)
    fs     = _scaler.transform(feats.reshape(1, -1))
    proba  = _rf.predict_proba(fs)[0]
    act    = _rf.classes_[int(np.argmax(proba))]
    conf   = float(np.max(proba))
    if act == "idle" or conf < MIN_CONF:
        return "idle", conf
    return act, conf


class MotionPipeline:
    """Per-connection state machine. Feed normalized landmark frames, get back punch events."""

    def __init__(self):
        self._buf: list  = []
        self._prev       = None
        self._ms         = "IDLE"   # IDLE | MOTION | COOLDOWN
        self._mc         = 0        # motion counter
        self._sc         = 0        # stop counter
        self._mi         = 0        # motion start index in buf
        self._last_t     = 0.0
        self._vel        = 0.0

    @property
    def velocity(self) -> float:
        return self._vel

    @property
    def motion_state(self) -> str:
        return self._ms

    def feed(self, raw_landmarks: list) -> tuple[str, float] | None:
        """
        raw_landmarks: flat list of 132 floats [x0,y0,z0,v0, x1,...].
        Returns (action, confidence) if a punch was classified, else None.
        """
        fr  = np.array(raw_landmarks, dtype=np.float32)
        nrm = normalize(fr)
        v   = _compute_vel(self._prev, nrm)
        self._prev = nrm
        self._vel  = v

        # Rolling buffer
        self._buf.append(nrm)
        if len(self._buf) > BUF_MAX:
            self._buf.pop(0)

        now = time.time()
        result = None

        if self._ms == "IDLE":
            if v > VEL_THRESH:
                self._mc += 1
                if self._mc >= MOTION_START:
                    self._mi = max(0, len(self._buf) - self._mc - LEAD_IN)
                    self._ms = "MOTION"
                    self._sc = 0
            else:
                self._mc = 0

        elif self._ms == "MOTION":
            if v < VEL_THRESH:
                self._sc += 1
                if self._sc >= MOTION_STOP:
                    result = self._classify()
                    self._last_t = now
                    self._ms = "COOLDOWN"
                    self._mc = self._sc = 0
            else:
                self._sc = 0
                if len(self._buf) - self._mi > 90:
                    result = self._classify()
                    self._last_t = now
                    self._ms = "COOLDOWN"
                    self._mc = self._sc = 0

        elif self._ms == "COOLDOWN":
            if now - self._last_t > COOLDOWN:
                self._ms = "IDLE"
                self._mc = 0

        return result

    def _classify(self) -> tuple[str, float] | None:
        buf = self._buf
        ei  = min(len(buf), len(buf) + TAIL)
        seg = buf[self._mi:ei]
        act, conf = classify_segment(seg)
        if act == "idle":
            return None
        return act, conf
