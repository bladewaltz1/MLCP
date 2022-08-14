import math


# TODO save state
class CosineScheduler:
    def __init__(self, init_value, total_step):
        self.init_value = init_value
        self.total_step = total_step

        self._step = 0
        self._value = init_value

    def step(self):
        self._value = (math.cos(math.pi * self._step / self.total_step) + 1) * self.init_value / 2
        self._step += 1

    @property
    def value(self):
        return self._value
