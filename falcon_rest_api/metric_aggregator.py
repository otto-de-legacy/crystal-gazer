import os
from time import gmtime, strftime

import numpy as np
import psutil as psutil


class CNT(object):
    """counter"""

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

    def get_values(self):
        return self.values


class EWMA(object):
    """exponential weighted moving average"""

    def __init__(self, half_life_count):
        self.values = None
        self.factor = np.exp(np.log(0.5) / half_life_count)

    def reset(self):
        self.values = None

    def step(self, y):
        if self.values is None:
            self.values = y
        else:
            self.values = (self.values - y) * self.factor + y

    def __call__(self, i=None):
        if self.values is None:
            return None
        if i is None:
            return self.values
        else:
            return self.values[i]

    def get_values(self):
        return self.values


class EWSTD(object):
    """exponential weighted moving standard deviation"""

    def __init__(self, half_life_count):
        self.maFilterAvg = EWMA(half_life_count)
        self.maFilterSquare = EWMA(half_life_count)

    def reset(self):
        self.maFilterAvg.reset()
        self.maFilterSquare.reset()

    def step(self, y):
        self.maFilterAvg.step(y)
        self.maFilterSquare.step(pow(y, 2))

    def get_std(self):
        avgs = self.maFilterSquare.get_values()
        sq_avgs = self.maFilterAvg.get_values()
        if avgs is None or sq_avgs is None:
            return None
        else:
            return np.sqrt(abs(avgs - pow(sq_avgs, 2)))

    def get_avg(self):
        return self.maFilterAvg.get_values()


class MetricAggregator(object):

    def __init__(self, config):

        self.config = config
        self.efSearch = self.config.efSearch
        self.neighbors_counter = CNT()
        self.batch_neighbors_counter = CNT()
        self.neighbors_counter = CNT()
        self.failed_counter = CNT()
        self.batch_neighbors_ewstd = EWSTD(self.config.half_life_count_dt)
        self.neighbors_ewstd = EWSTD(self.config.half_life_count_dt)

        self.start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.latest_error = ""

    def failed_request(self, exception, count=1):
        self.latest_error = exception
        self.failed_counter.step(count)

    def new_batch_request(self, dt, count=1):
        self.batch_neighbors_counter.step(count)
        self.batch_neighbors_ewstd.step(dt)

        self.h.observe(dt)

    def new_neighbors_request(self, dt, count=1):
        self.neighbors_counter.step(count)
        self.neighbors_ewstd.step(dt)

    def get_failed_request_count(self):
        return self.failed_counter.get_values()

    def get_batch_req_count(self):
        return self.batch_neighbors_counter.get_values()

    def get_neighbors_req_count(self):
        return self.neighbors_counter.get_values()

    def get_batch_req_dt_avg(self):
        return self.batch_neighbors_ewstd.get_avg()

    def get_neighbors_dt_avg(self):
        return self.neighbors_ewstd.get_avg()

    def get_batch_req_dt_std(self):
        return self.batch_neighbors_ewstd.get_std()

    def get_neighbors_dt_std(self):
        return self.neighbors_ewstd.get_std()

    @staticmethod
    def memory_usage_psutil():
        """STILL FAST, return the memory usage of a Python module in MB (>~ 10000 per second)"""
        process = psutil.Process(os.getpid())
        mem = process.memory_info()[0] / float(2 ** 20)
        return mem

    @staticmethod
    def cpu_usage_percent():
        """ """
        cpu_perc = psutil.cpu_percent()
        return cpu_perc
