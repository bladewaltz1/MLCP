import math


class CosineScheduler:
    def __init__(self, init_value, total_step, min_value=1e-2):
        self.init_value = init_value
        self.total_step = total_step
        self.min_value = min_value

        self._step = 0
        self._value = init_value

    def step(self):
        self._value = max(self.min_value, (math.cos(math.pi * self._step / self.total_step) + 1) * self.init_value / 2)
        self._step += 1

    @property
    def value(self):
        return self._value
