class CNT(object):  # TODO: define filter interface
    def __init__(self):
        self.values = None

    def reset(self):
        self.values = None

    def step(self, cnt):
        if self.values is None:
            self.values = cnt
        else:
            self.values = self.values + cnt

    def __call__(self, i=None):
        if self.values is None:
            return None
        if i is None:
            return self.values
        else:
            return self.values[i]

    def values(self):
        return self.values
