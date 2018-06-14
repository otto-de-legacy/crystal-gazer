# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
from core.data_sampler import DataSampler
from scipy import stats
import core.url_mapper as um
import time


class Event():

    def __init__(self, feature_idx, label_idx):
        self.feature_idx = feature_idx
        self.label_idx = label_idx
        self.count = 1

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.feature_idx == other.feature_idx) and (self.label_idx == other.label_idx)
        else:
            return False

    def __mul__(self, other):
        return self.count * other.count

    def __add__(self, other):
        return self.count + other.count

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.feature_idx < other.feature_idx

    def __le__(self, other):
        return self.feature_idx <= other.feature_idx

    def __str__(self):
        return "feature: " + str(self.feature_idx) + ", label:" + str(self.label_idx)

    def __ge__(self, other):
        return self.feature_idx >= other.feature_idx

    def __gt__(self, other):
        return self.feature_idx > other.feature_idx

    def __hash__(self):
        return hash((self.feature_idx, self.label_idx))

