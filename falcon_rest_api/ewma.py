import numpy as np


class EWMA(object):  # TODO: define filter interface
    def __init__(self, half_life_count):
        self.values = None
        self.factor = np.exp(np.log(0.5) / half_life_count)

    def reset(self):
        self.values = None

    def step(self, y):
        if self.values is None:
            self.values = y
        else:
            self.values = y * self.factor + (y * (1.0 - self.factor))

    def __call__(self, i=None):
        if self.values is None:
            return None
        if i is None:
            return self.values
        else:
            return self.values[i]

    def values(self):
        return self.values
