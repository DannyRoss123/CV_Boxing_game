import random
import time

COMBO_ROUNDS = 8
COMBO_POOL = [
    ["jab", "cross"], ["cross", "hook"], ["jab", "uppercut"],
    ["hook", "cross"], ["uppercut", "jab"], ["cross", "jab"],
    ["jab", "cross", "hook"], ["jab", "jab", "cross"],
    ["hook", "uppercut", "jab"], ["cross", "hook", "uppercut"],
    ["jab", "hook", "cross"], ["uppercut", "cross", "jab"],
]


def _rand_combo(prev=None):
    choices = [c for c in COMBO_POOL if c != prev]
    return random.choice(choices)


class ComboRushGame:
    COUNTDOWN = "COUNTDOWN"; THROWING = "THROWING"
    RESULT = "RESULT"; DONE = "DONE"

    def __init__(self):
        self.state       = self.COUNTDOWN
        self.timer       = 0.0
        self.round       = 0
        self.score       = 0
        self.combo: list = []
        self.progress: list = []
        self.combo_start: float | None = None
        self.last_result = None
        self._next_combo(None)

    def _next_combo(self, prev):
        self.combo       = _rand_combo(prev)
        self.progress    = []
        self.combo_start = None

    def on_punch(self, action: str, conf: float, peak: float = 0.0):
        if self.state != self.THROWING:
            return
        idx      = len(self.progress)
        expected = self.combo[idx]
        if self.combo_start is None:
            self.combo_start = time.time()
        if action == expected:
            self.progress.append(action)
            if len(self.progress) == len(self.combo):
                elapsed_ms = int((time.time() - self.combo_start) * 1000)
                pts = max(0, 500 - elapsed_ms // 10)
                self.score += pts
                self.last_result = {"ok": True, "elapsed_ms": elapsed_ms, "pts": pts,
                                    "wrong": None, "expected": None}
                self.state = self.RESULT; self.timer = 0.0
        else:
            elapsed_ms = int((time.time() - self.combo_start) * 1000) if self.combo_start else 0
            self.last_result = {"ok": False, "elapsed_ms": elapsed_ms, "pts": 0,
                                "wrong": action, "expected": expected}
            self.state = self.RESULT; self.timer = 0.0

    def update(self, dt: float):
        self.timer += dt
        if self.state == self.COUNTDOWN and self.timer >= 3.0:
            self.state = self.THROWING; self.timer = 0.0
        elif self.state == self.RESULT and self.timer >= 1.8:
            prev = self.combo; self.round += 1
            if self.round >= COMBO_ROUNDS:
                self.state = self.DONE; self.timer = 0.0
            else:
                self._next_combo(prev); self.state = self.THROWING; self.timer = 0.0

    def get_state(self) -> dict:
        elapsed_ms = None
        if self.combo_start is not None and self.state == self.THROWING:
            elapsed_ms = int((time.time() - self.combo_start) * 1000)
        return {
            "phase":       self.state,
            "round":       self.round,
            "total_rounds": COMBO_ROUNDS,
            "score":       self.score,
            "combo":       self.combo,
            "progress":    self.progress,
            "timer":       round(self.timer, 3),
            "elapsed_ms":  elapsed_ms,
            "last_result": self.last_result,
        }

    @property
    def is_done(self) -> bool:
        return self.state == self.DONE
