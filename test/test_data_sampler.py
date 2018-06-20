from unittest import TestCase

import numpy as np

from core.data_sampler import DataSampler


class TestNetwork(TestCase):
    def test_constructor(self):
        np.random.seed(0)
        rs = DataSampler(np.arange(0, 100, 1), np.array([0.01] * 100), 10)
        res = rs.rvs(2)
        np.testing.assert_array_equal(rs.bucket_sizes, np.array([10] * 10), err_msg="check bucket sizes")
        np.testing.assert_array_equal(res, np.array([50, 53]), err_msg="check draw from one bucket")

    def test_get_by_condition(self):
        np.random.seed(0)
        rs = DataSampler(np.arange(0, 100, 1), np.array([0.01] * 100), 10)
        res = rs.get_by_condition(lambda x: x > 80)
        np.testing.assert_array_equal(res, np.arange(81, 100, 1), err_msg="check bucket sizes")
