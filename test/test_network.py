from unittest import TestCase
import tensorflow as tf
from core.network import Network
from core.config import Config
import core.url_mapper as um

class TestNetwork(TestCase):
    def test_layer_sizes(self):
        cf = Config()
        cf.embedding_size = 3
        url_map = um.UrlMapper()
        test_netowrk = Network(cf, url_map)

        input_to_layer = tf.sparse_placeholder(tf.float32, shape=[None, url_map.url_class_cnt], name="url_feature")
        out_layer = test_netowrk.predict(input_to_layer)
        out_layer_shape = out_layer.get_shape().as_list()
        self.assertTrue(out_layer_shape == [None, cf.embedding_size], msg="")
