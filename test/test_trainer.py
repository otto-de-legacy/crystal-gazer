from unittest import TestCase

import numpy as np
import tensorflow as tf

import core.interaction_mapper as um
import core.network as nw
import core.trainer as tn
from core.config import Config

dim = 3
cf = Config('./resources', continue_previous_run=False)
cf.embedding_size = dim
interaction_map = um.InteractionMapper('./resources/map')
network = nw.Network(cf, interaction_map)
trainer = tn.Trainer(cf, network)


class TestTrainer(TestCase):

    def test_constructor(self):
        self.assertTrue(trainer.merged is not None, msg="does not fail")

    def test__loss_identical_input(self):
        interaction_vectors = tf.nn.l2_normalize(tf.constant(np.array([
            np.array([1.0, 1.0, 1.0], np.float32),
            np.array([1.0, 1.0, 1.0], np.float32)], np.float32)), 1)

        exp_dist = tf.constant(np.array([0.0, 0.0], np.float32))

        with tf.Session() as sess:
            result = sess.run(trainer._loss_(interaction_vectors, interaction_vectors, interaction_vectors, exp_dist))

        self.assertTrue(result < 0.001, msg="loss for identical input should be 0")

    def test__loss_opposite_input(self):
        interaction_vectors = tf.nn.l2_normalize(tf.constant(np.array([
            np.array([1.0, 1.0, 1.0], np.float32),
            np.array([1.0, 1.0, 1.0], np.float32)], np.float32)), 1)

        target = tf.nn.l2_normalize(tf.constant(np.array([
            np.array([-1.0, -1.0, -1.0], np.float32),
            np.array([-1.0, -1.0, -1.0], np.float32)], np.float32)), 1)

        exp_dist = tf.constant(np.array([1.0, 1.0], np.float32))

        with tf.Session() as sess:
            result = sess.run(trainer._loss_(interaction_vectors, interaction_vectors, target, exp_dist))

        self.assertTrue(result < 0.001, msg="loss for opposite input should be exp_dist")

    def test__single_dist_(self):
        prediction = tf.constant(np.array([
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0])]))

        with tf.Session() as sess:
            result = sess.run(trainer._single_dist_(prediction, prediction))

        self.assertTrue(len(result) == 2, msg="length incorrect")
        self.assertTrue(sum(abs(result - [0, 0])) < 0.001, msg="result incorrect for identical input")

        target = tf.constant(np.array([
            np.array([-1.0, -1.0, -1.0]),
            np.array([-1.0, -1.0, -1.0])]))

        with tf.Session() as sess:
            result = sess.run(trainer._single_dist_(prediction, target))

        self.assertTrue(sum(abs(result - [1.0, 1.0])) < 0.001, msg="result incorrect for opposite input")
