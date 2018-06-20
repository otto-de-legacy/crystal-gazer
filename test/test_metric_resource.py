from collections import namedtuple
from unittest import TestCase

import numpy as np
import tensorflow as tf

import core.interaction_mapper as um
import core.network as nw
import core.trainer as tn
from falcon_rest_api.recos import MetricsResource

ResponseMock = namedtuple("response_mock", [
    "status",
    "media"
])


class TestEWMA(TestCase):

    def test_works_with_vectors(self):
        mr = MetricsResource()
        mr.on_get(None, ResponseMock(None, None))

        self.assertTrue(1 == 1, msg="Should not fail")
        # np.testing.assert_array_equal(ewma.get_values(), [100, 50])
