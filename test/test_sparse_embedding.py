from unittest import TestCase

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.losses.losses_impl import Reduction

from core.config import Config
import core.loader as ld
import core.network as nw


class TestNetwork(TestCase):

    def test_1(self):
        indices = [[0, 0], [1, 2]]
        values = [1., 1.]
        shape = [4, 6]
        test_sparse = tf.SparseTensor(indices, values, shape)
        v = tf.ones((6, 1))

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        mult = tf.sparse_tensor_dense_matmul(test_sparse, v)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print("------------------")
            print(sess.run(mult))

    def test_2(self):
        indices = [[0, 0], [0, 1], [0, 2]]
        values = [3., 6., 9.]
        shape = [1, 10]
        test_sparse = tf.SparseTensor(indices, values, shape)
        v = tf.ones((10, 1))

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        mult = tf.sparse_tensor_dense_matmul(test_sparse, v)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print("------------------")
            print(sess.run(mult))

    def test_3(self):
        indices = [[0, 0], [0, 1], [0, 2]]
        values = [3., 6., 9.]
        shape = [1, 10]

        sparse_feature = tf.SparseTensor(indices, values, shape)
        embedding_matrix = tf.Variable(tf.random_normal([10, 4], stddev=0.1))
        mult = tf.sparse_tensor_dense_matmul(sparse_feature, embedding_matrix)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print("------------------")
            print(sess.run(mult))

    def test_normalization(self):

        dim = 3
        batch = tf.constant(np.array([np.ones(dim),
                                      np.ones(dim),
                                      np.zeros(dim),
                                      np.ones(dim),
                                      np.ones(dim),
                                      np.zeros(dim)]))
        # tf.greater
        # tf.clip_by_norm
        # tf.ones_like()
        norm = tf.norm(batch, axis=1)

        norm_minus_one = norm - tf.ones(norm.shape, tf.float64)
        norm_minus_one = norm - tf.ones_like(norm.shape, tf.float64)
        rm  = tf.reduce_mean(norm_minus_one)

        with tf.Session() as sess:
            result = sess.run([norm_minus_one, rm + rm])
        print(result)
        print(str(np.sqrt(dim) - 1))

    def test_cos_dist(self):
        dim = 5
        batch = tf.constant(np.array([np.ones(dim),
                                      np.ones(dim),
                                      np.ones(dim),
                                      np.ones(dim),
                                      np.ones(dim),
                                      np.ones(dim)]))

        single_dist = tf.losses.cosine_distance(
            tf.nn.l2_normalize(batch,1),
            tf.nn.l2_normalize(batch,1),
            dim=1, reduction=Reduction.NONE)

        with tf.Session() as sess:
            result = sess.run([single_dist])
        print("result single dist calc:")
        print(result)
        print(str(np.sqrt(dim) - 1))

        with tf.Session() as sess:
            result = sess.run([tf.nn.l2_normalize(batch,1)])
        print(result)




