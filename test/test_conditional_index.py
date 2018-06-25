from unittest import TestCase
import pandas as pd
from core.conditional_index import ConditionalIndex
from core.interaction_index import InteractionIndex
from core.interaction_mapper import InteractionMapper
import numpy as np


class TestInteractionIndex(TestCase):

    def test_knn_interaction_query(self):
        im = InteractionMapper("./resources/map")
        pd_df = pd.read_csv("./resources/interaction_index/interaction_index.txt", header=None)
        for col in pd_df.columns:
            pd_df[col] = pd_df[col].astype(float)
        lambdas = [
            lambda key: True if len(key) > 1 else False,
            lambda key: True if len(key) <= 1 else False
        ]
        ci = ConditionalIndex(im, pd_df.values, lambdas)

        ii = InteractionIndex(im, pd_df.values)

        dummy = ii.knn_interaction_query("a", k=100)

        np.testing.assert_array_equal(ci.knn_interaction_query("a", 0, k=3)[0], list(filter(lambda s: len(s) > 1, dummy[0]))[0:3])
        np.testing.assert_array_equal(ci.knn_interaction_query("a", 1, k=3)[0], list(filter(lambda s: len(s) <= 1, dummy[0]))[0:3])
        np.testing.assert_array_equal(ci.knn_interaction_query("a", k=3)[0], ii.knn_interaction_query("a", k=3)[0])
