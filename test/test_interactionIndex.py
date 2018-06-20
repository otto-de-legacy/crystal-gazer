from unittest import TestCase

from core.interaction_index import InteractionIndex
from core.interaction_mapper import InteractionMapper


class TestInteractionIndex(TestCase):
    def test_knn_idx_query(self):
        im = InteractionMapper("./resources/map")
        iv = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
        ii = InteractionIndex(im, iv)

        result = ii.knn_idx_query(1)

        self.assertTrue(result[0][0] == "a")

    def test_knn_interaction_query(self):
        im = InteractionMapper("./resources/map")
        iv = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
        ii = InteractionIndex(im, iv)

        result = ii.knn_interaction_query("a")

        self.assertTrue(result[0][0] == "a")

    def test_knn_interaction_query_exception(self):
        im = InteractionMapper("./resources/map")
        iv = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
        ii = InteractionIndex(im, iv)

        result = ii.knn_interaction_query("")

        self.assertTrue(len(result[0]) == 0)
        self.assertTrue(len(result[1]) == 0)
        self.assertTrue(len(result[2]) == 0)
