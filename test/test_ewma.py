from unittest import TestCase

import numpy as np

from falcon_rest_api.ewma import EWMA


class TestEWMA(TestCase):

    def test_works_with_vectors(self):
        ewma = EWMA(1)

        ewma.step(np.array([100.0, 100.0], dtype=np.float32))
        ewma.step(np.array([100.0, 0.0], dtype=np.float32))

        np.testing.assert_array_equal(ewma.get_values(), [100, 50])
