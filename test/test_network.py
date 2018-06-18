from unittest import TestCase

import tensorflow as tf

import core.interaction_mapper as um
from core.config import Config
from core.network import Network


class TestNetwork(TestCase):
    def test_layer_sizes(self):
        cf = Config()
        cf.embedding_size = 3
        interaction_map = um.InteractionMapper('./resources/interaction_map')
        test_netowrk = Network(cf, interaction_map)

        input_to_layer = tf.sparse_placeholder(tf.float32, shape=[None, interaction_map.interaction_class_cnt], name="interaction_feature")
        out_layer = test_netowrk.predict(input_to_layer)
        out_layer_shape = out_layer.get_shape().as_list()
        self.assertTrue(out_layer_shape == [None, cf.embedding_size], msg="layer shape")
