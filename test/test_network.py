from unittest import TestCase

import numpy as np
import tensorflow as tf

import core.interaction_mapper as um
from core.config import Config
from core.network import Network

cf = Config("'./resources", continue_previous_run=False)
cf.embedding_size = 3
interaction_map = um.InteractionMapper('./resources/map')


class TestNetwork(TestCase):
    def test_layer_sizes(self):
        test_netowrk = Network(cf, interaction_map)

        input_to_layer = tf.sparse_placeholder(tf.float32, shape=[None, interaction_map.interaction_class_cnt],
                                               name="interaction_feature")
        out_layer = test_netowrk.predict(input_to_layer)
        out_layer_shape = out_layer.get_shape().as_list()
        self.assertTrue(out_layer_shape == [None, cf.embedding_size], msg="layer shape")

    def test__initialize_params_(self):
        preheated_embeddings = np.array([
            np.array([-1.0, -1.0, -6.0]),
            np.array([-1.0, -1.0, 5.0]),
            np.array([-1.0, -1.0, 4.0]),
            np.array([-1.0, 3.0, -1.0]),
            np.array([1.0, 1.0, -1.0]),
            np.array([2.0, 1.0, -1.0]),
            np.array([1.0, -1.0, 1.0]),
            np.array([1.0, -1.0, -1.0])])

        interaction_sparse_tensor = interaction_map.idxs_to_tf([2])

        test_netowrk = Network(cf, interaction_map, preheated_embeddings)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(test_netowrk.embedd_interaction_sparse_tensor(interaction_sparse_tensor))

        np.testing.assert_array_equal(result, [[-1.0, -1.0, 4.0]])
