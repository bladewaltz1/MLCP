import math


class CosineScheduler:
    def __init__(self, init_value, min_value, total_step):
        self.init_value = init_value
        self.total_step = total_step
        self.min_value = min_value

        self._step = 0
        self._value = init_value

    def step(self):
        self._value = (math.cos(math.pi * self._step / self.total_step) 
                       * self.init_value / 2 + self.init_value / 2) \
                      * (1 - self.min_value) + self.min_value
        self._step += 1

    @property
    def value(self):
        return self._value
