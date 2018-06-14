from unittest import TestCase
from core.config import Config
import numpy as np
from core.url_mapper import UrlMapper
import tensorflow as tf


class TestURLMapper(TestCase):
    def test_constructor(self):
        cf = Config()
        cf.url_map = './resources/url_map'
        um = UrlMapper(cf)
        expected = 8
        self.assertTrue(um.url_class_cnt == expected,
                        msg=str(um.url_class_cnt) + "!=" + str(expected))

    def test_to_tf(self):
        cf = Config()
        cf.url_map = './resources/url_map'
        um = UrlMapper(cf)
        tensor = um.idxs_to_tf([0, 1])
        np.testing.assert_array_equal(np.array(tensor.indices), np.array([[0, 0], [1, 1]]), err_msg="")
        np.testing.assert_array_equal(np.array(tensor.values), np.array([1, 1]), err_msg="")
        np.testing.assert_array_equal(np.array(tensor.dense_shape), np.array([2, um.url_class_cnt]), err_msg="")
