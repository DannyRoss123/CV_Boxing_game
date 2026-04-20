import random
import time

PUNCH_ACTIONS = ["jab", "cross", "hook", "uppercut"]


def _level_config(lvl: int) -> tuple[int, float]:
    return min(2 + (lvl - 1) // 2, 7), max(1.5, 3.6 - lvl * 0.14)


class MemoryGame:
    SHOW = "SHOW"; READY = "READY"; PLAYING = "PLAYING"
    FEEDBACK = "FEEDBACK"; LEVEL_UP = "LEVEL_UP"; DONE = "DONE"

    def __init__(self):
        self.level    = 1
        self.score    = 0
        self.sequence: list = []
        self.progress: list = []
        self.show_time = 3.0
        self.last_act  = None
        self.state     = self.SHOW
        self.timer     = 0.0
        self._new_best = False
        self._new_level()

    def _new_level(self):
        length, self.show_time = _level_config(self.level)
        self.sequence = [random.choice(PUNCH_ACTIONS) for _ in range(length)]
        self.progress = []
        self._enter(self.SHOW)

    def _enter(self, s: str):
        self.state = s
        self.timer = 0.0

    def on_punch(self, action: str, conf: float, peak: float = 0.0):
        if self.state != self.PLAYING:
            return
        idx      = len(self.progress)
        expected = self.sequence[idx]
        correct  = action == expected
        self.last_act = (action, correct, conf)
        if correct:
            self.progress.append(action)
            self.score += 10 + self.level * 5
            self._enter(self.FEEDBACK)
        else:
            self._enter(self.DONE)

    def update(self, dt: float):
        self.timer += dt
        if self.state == self.SHOW and self.timer >= self.show_time:
            self._enter(self.READY)
        elif self.state == self.READY and self.timer >= 1.1:
            self._enter(self.PLAYING)
        elif self.state == self.FEEDBACK and self.timer >= 0.45:
            if len(self.progress) == len(self.sequence):
                self._enter(self.LEVEL_UP)
            else:
                self._enter(self.PLAYING)
        elif self.state == self.LEVEL_UP and self.timer >= 2.2:
            self.level += 1
            self._new_level()

    def get_state(self) -> dict:
        wrong_expected = None
        if self.state == self.DONE and self.last_act:
            action, _, _ = self.last_act
            wrong_expected = self.sequence[len(self.progress)] if len(self.progress) < len(self.sequence) else None
        return {
            "phase":          self.state,
            "level":          self.level,
            "score":          self.score,
            "sequence":       self.sequence,
            "progress":       self.progress,
            "show_time":      self.show_time,
            "timer":          round(self.timer, 3),
            "last_act":       {"action": self.last_act[0], "correct": self.last_act[1], "conf": self.last_act[2]} if self.last_act else None,
            "wrong_expected": wrong_expected,
        }

    @property
    def is_done(self) -> bool:
        return self.state == self.DONE
