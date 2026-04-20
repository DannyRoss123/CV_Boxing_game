import random
import time

COUNTER_MAP = {
    "jab":      {"cross": 100, "hook": 80,      "block_high": 40},
    "cross":    {"hook": 100,  "uppercut": 80,  "block_high": 40},
    "hook":     {"uppercut": 100, "cross": 80,  "block_high": 40},
    "uppercut": {"jab": 100,  "cross": 80,      "block_high": 40},
}

ATTACK_WINDOW  = 3.5
DEFENSE_ROUNDS = 10
DEFENSE_HP     = 5


class DefenseDrillGame:
    GUIDE   = "GUIDE"
    WAITING = "WAITING"
    JUDGING = "JUDGING"
    KO      = "KO"
    DONE    = "DONE"

    def __init__(self):
        self.state       = self.GUIDE
        self.timer       = 0.0
        self.round       = 0
        self.score       = 0
        self.hp          = DEFENSE_HP
        self.attack: str | None = None
        self.last_result = None
        self.results: list = []

    def advance(self):
        if self.state == self.GUIDE:
            self._next_attack()

    def _next_attack(self):
        self.attack = random.choice(list(COUNTER_MAP.keys()))
        self.state  = self.WAITING
        self.timer  = 0.0

    def on_punch(self, action: str, conf: float, peak: float = 0.0):
        if self.state != self.WAITING:
            return
        pts = COUNTER_MAP[self.attack].get(action, 0)
        if pts > 0:
            tag = "hit" if pts >= 80 else "block"
            self.score += pts
        else:
            tag = "wrong"
            self.hp -= 1
        self.last_result = {"tag": tag, "action": action, "pts": pts, "attack": self.attack}
        self.results.append({"tag": tag, "pts": pts})
        self.state = self.JUDGING; self.timer = 0.0

    def update(self, dt: float):
        self.timer += dt
        if self.state == self.WAITING and self.timer >= ATTACK_WINDOW:
            self.hp -= 1
            self.last_result = {"tag": "miss", "action": None, "pts": 0, "attack": self.attack}
            self.results.append({"tag": "miss", "pts": 0})
            self.state = self.JUDGING; self.timer = 0.0
        elif self.state == self.JUDGING and self.timer >= 1.2:
            if self.hp <= 0:
                self.state = self.KO; self.timer = 0.0
            else:
                self.round += 1
                if self.round >= DEFENSE_ROUNDS:
                    self.state = self.DONE; self.timer = 0.0
                else:
                    self._next_attack()

    def get_state(self) -> dict:
        time_left = None
        if self.state == self.WAITING:
            time_left = round(max(0.0, ATTACK_WINDOW - self.timer), 2)
        return {
            "phase":         self.state,
            "round":         self.round,
            "total_rounds":  DEFENSE_ROUNDS,
            "score":         self.score,
            "hp":            self.hp,
            "max_hp":        DEFENSE_HP,
            "attack":        self.attack,
            "time_left":     time_left,
            "attack_window": ATTACK_WINDOW,
            "last_result":   self.last_result,
            "results":       self.results,
        }

    @property
    def is_done(self) -> bool:
        return self.state in (self.DONE, self.KO)
