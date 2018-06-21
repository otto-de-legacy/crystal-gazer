from unittest import TestCase

import numpy as np

import core.interaction_mapper as um
import core.loader as ld
from core.config import Config
from core.event import Event

interaction_input = """1,2,3
        2,3,2,1,3
        2,5,2,4,3"""

cf = Config("./resources", continue_previous_run=False)
cf.neighboring_interactions = 1
interaction_mapper = um.InteractionMapper('./resources/map')
interaction_mapper.total_interaction_cnt = 6
interaction_mapper.interaction_class_cnt = 8


class TestLoader(TestCase):

    def test_user_journey_to_feature_target(self):
        cf.neighboring_interactions = 1
        np.random.seed(0)
        test_string = "1,2,3,3,4,5,6"
        loader = ld.Loader(cf, interaction_mapper, "./resources/train")
        compare = [Event(1, 2),
                   Event(2, 3),
                   Event(3, 4),
                   Event(4, 5),
                   Event(5, 6)]
        result = loader._user_journey_to_events(test_string)
        np.testing.assert_array_equal(compare, result, err_msg=str(compare) + "!=" + str(result))

    def test_prepare_events(self):
        cf.neighboring_interactions = 1
        loader = ld.Loader(cf, interaction_mapper, "./resources/train")
        result = loader.unique_train_event_cnt
        self.assertTrue(507 == result, msg="507 !=" + str(result))

    def test_batching(self):
        cf.neighboring_interactions = 1
        cf.fake_frac = 0.52
        loader = ld.Loader(cf, interaction_mapper, "./resources/train")
        features, labels, dist_vals = loader.get_next_batch(3)
        self.assertTrue(features.dense_shape == [3, 8], msg=("was: " + str(features.dense_shape)))
        self.assertTrue(labels.dense_shape == [3, 8], msg="was: " + str(labels.dense_shape))
        np.testing.assert_array_equal(dist_vals, [0, 0, 1], err_msg="wrong dist_vals: " + str(dist_vals))

    def test__prepare_events(self):
        cf.neighboring_interactions = 4
        cf.fake_frac = 0
        cf.bucket_count = 1
        loader = ld.Loader(cf, interaction_mapper, "./resources/train")
        found_events = loader.random_generator.data_buckets[0]
        duplicate = False
        for event in found_events:
            if event.feature_idx == event.label_idx:
                duplicate = True
        self.assertTrue(duplicate == False)
