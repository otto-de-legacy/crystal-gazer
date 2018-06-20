from unittest import TestCase

import numpy as np

from core.config import Config
from core.interaction_mapper import InteractionMapper

class TestInteractionMapper(TestCase):
    def test_constructor(self):
        im = InteractionMapper('./resources/interaction_map')
        expected = 8
        self.assertTrue(im.interaction_class_cnt == expected,
                        msg=str(im.interaction_class_cnt) + "!=" + str(expected))

    def test_to_tf(self):
        im = InteractionMapper('./resources/interaction_map')
        tensor = im.idxs_to_tf([0, 1])
        np.testing.assert_array_equal(np.array(tensor.indices), np.array([[0, 0], [1, 1]]), err_msg="indices incorrect")
        np.testing.assert_array_equal(np.array(tensor.get_values), np.array([1, 1]), err_msg="values incorrect")
        np.testing.assert_array_equal(np.array(tensor.dense_shape), np.array([2, im.interaction_class_cnt]), err_msg="shape incorrect")
