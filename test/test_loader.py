from unittest import TestCase
from core.config import Config
from core.event import Event
import core.loader as ld
import core.url_mapper as um
import numpy as np
import tensorflow as tf

url_input = """1,2,3
        2,3,2,1,3
        2,5,2,4,3"""


class TestLoader(TestCase):

    def test_user_journey_to_feature_target(self):
        np.random.seed(0)
        cf = Config()
        cf.url_map = None
        url_mapper = um.UrlMapper()
        test_string = "1,2,3,4,5,6"
        loader = ld.Loader(cf, test_string, url_mapper)
        compare = [Event(1, 2),
                   Event(2, 3),
                   Event(3, 4),
                   Event(4, 5),
                   Event(5, 6)]
        result = loader._user_journey_to_events(test_string)
        np.testing.assert_array_equal(compare, result, err_msg=str(compare) + "!=" + str(result))

    def test_prepare_events(self):
        cf = Config()
        cf.url_map = None
        url_mapper = um.UrlMapper()
        loader = ld.Loader(cf, url_input, url_mapper)
        compare = [Event(1, 2),
                   Event(2, 3),
                   Event(2, 4),
                   Event(4, 2),
                   Event(2, 1),
                   Event(1, 3),
                   Event(2, 5),
                   Event(5, 2),
                   Event(4, 3)]
        result = loader.unique_train_event_cnt
        self.assertTrue(len(compare) == result, msg=str(compare) + "!=" + str(result))

    def test_batching(self):
        np.random.seed(0)
        cf = Config()
        cf.url_map = None
        cf.fake_frac = 0.52

        url_mapper = um.UrlMapper()
        url_mapper.total_url_cnt = 6
        url_mapper.url_class_cnt = 8
        loader = ld.Loader(cf, url_input, url_mapper)

        features, labels, dist_vals = loader.get_next_batch(3)
        self.assertTrue(features.dense_shape == [3, 8], msg=("was: " + str(features.dense_shape)))
        self.assertTrue(labels.dense_shape == [3, 8], msg="was: " + str(labels.dense_shape))
        np.testing.assert_array_equal(dist_vals, [0, 0, 1], err_msg="wrong dist_vals: " + str(dist_vals))
