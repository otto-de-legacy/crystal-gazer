from unittest import TestCase
import tensorflow as tf
import core.network as nw
import core.trainer as tn
from core.config import Config
import core.url_mapper as um
import numpy as np

dim = 3
cf = Config()
cf.embedding_size = dim
url_map = um.UrlMapper()
network = nw.Network(cf, url_map)
trainer = tn.Trainer(cf, network)


class TestTrainer(TestCase):

    def test_constructor(self):
        self.assertTrue(trainer.merged is not None, msg="")

    def test__loss_identical_input(self):
        url_vectors = tf.nn.l2_normalize(tf.constant(np.array([
            np.array([1.0, 1.0, 1.0], np.float32),
            np.array([1.0, 1.0, 1.0], np.float32)], np.float32)), 1)

        exp_dist = tf.constant(np.array([0.0, 0.0], np.float32))

        with tf.Session() as sess:
            result = sess.run(trainer._loss_(url_vectors, url_vectors, url_vectors, exp_dist))

        self.assertTrue(result < 0.001, msg="")

    def test__loss_opposite_input(self):
        url_vectors = tf.nn.l2_normalize(tf.constant(np.array([
            np.array([1.0, 1.0, 1.0], np.float32),
            np.array([1.0, 1.0, 1.0], np.float32)], np.float32)), 1)

        target = tf.nn.l2_normalize(tf.constant(np.array([
            np.array([-1.0, -1.0, -1.0], np.float32),
            np.array([-1.0, -1.0, -1.0], np.float32)], np.float32)), 1)

        exp_dist = tf.constant(np.array([1.0, 1.0], np.float32))

        with tf.Session() as sess:
            result = sess.run(trainer._loss_(url_vectors, url_vectors, target, exp_dist))

        self.assertTrue(result < 0.001, msg="")

    def test__single_dist_(self):
        prediction = tf.constant(np.array([
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0])]))

        with tf.Session() as sess:
            result = sess.run(trainer._single_dist_(prediction, prediction))

        self.assertTrue(len(result) == 2, msg="")
        self.assertTrue(sum(abs(result - [0, 0])) < 0.001, msg="")

        target = tf.constant(np.array([
            np.array([-1.0, -1.0, -1.0]),
            np.array([-1.0, -1.0, -1.0])]))

        with tf.Session() as sess:
            result = sess.run(trainer._single_dist_(prediction, target))

        self.assertTrue(len(result) == 2, msg="")
        self.assertTrue(sum(abs(result - [1.0, 1.0])) < 0.001, msg="")

    def _single_loss_(self, single_dist, exp_dist):
        return single_dist - exp_dist
