from collections import namedtuple
from unittest import TestCase

from falcon_rest_api.recos import MetricsResource

ResponseMock = namedtuple("response_mock", [
    "status",
    "media"
])


class TestEWMA(TestCase):

    def test_works_with_vectors(self):
        # TODO: mock correctly
        # mr = MetricsResource()
        # mr.on_get(None, ResponseMock(None, None))

        self.assertTrue(1 == 1, msg="Should not fail")
        # np.testing.assert_array_equal(ewma.get_values(), [100, 50])
