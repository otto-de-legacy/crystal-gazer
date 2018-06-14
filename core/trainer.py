import tensorflow as tf
import numpy as np
from tensorflow.python.ops.losses.losses_impl import Reduction
from tensorflow.python.client import timeline

class Trainer(object):

    def __init__(self, config, network):
        self.cf = config
        self.network = network
        self.batch_cnt = 0

        self.x = None
        self.y = None
        self.train_output = None
        self.test_output = None
        self.merged = None
        self._initialize_graph()
        self._initialize_tensorboard()

    def _initialize_graph(self):
        """build graph"""
        self.x = tf.sparse_placeholder(tf.float32, shape=[None, self.network.um.url_class_cnt], name="url_feature")
        self.y = tf.sparse_placeholder(tf.float32, shape=[None, self.network.um.url_class_cnt], name="url_label")
        self.d = tf.placeholder(tf.float32, shape=[None], name="expected_dist")  # for mixing with falsy predictions

        self.url_vectors = self.network.embedd_url_sparse_tensor(self.x)
        self.pred = self.network.predict(self.x)
        self.cost = self._loss_(
            self.url_vectors,
            self.pred,
            self.network.embedd_url_sparse_tensor(self.y),
            self.d
        )

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.cf.learning_rate).minimize(self.cost)  #

    def _initialize_tensorboard(self):
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        tf.summary.scalar("evaluation_metric: cost", self.cost)
        self.merged = tf.summary.merge_all()

    def _single_dist_(self, prediction, target):
        # return tf.reduce_sum(tf.pow(prediction - target, 2), 1)
        cos_sim = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(prediction, 1), tf.nn.l2_normalize(target, 1)), axis=1)
        cos_dist = tf.divide(tf.ones_like(cos_sim) - cos_sim, 2) # because sim is [-1,1] where 1 is close, so dist is [0,1] where 0 is close
        return cos_dist

    def _single_loss_(self, single_dist, exp_dist):
        return tf.pow(single_dist - exp_dist, 2)

    def _loss_(self, url_vectors, prediction, target, exp_dist):
        single_loss = self._single_loss_(self._single_dist_(prediction, target), exp_dist)

        norm = tf.norm(url_vectors, axis=1)
        norm_minus_one = tf.pow(norm - tf.ones_like(norm, tf.float32), 2)
        self.url_vector_norm_renormalization_condition = tf.reduce_mean(norm_minus_one)

        return tf.reduce_mean(single_loss) + self.url_vector_norm_renormalization_condition

    def train(self, sess, batch_x, batch_y, exp_dist):

        self.train_output = sess.run([self.optimizer, self.merged],  # , self.url_vector_norm_renormalization_condition
                                     run_metadata=self.run_metadata,
                                     feed_dict={self.x: batch_x,
                                                self.y: batch_y,
                                                self.d: exp_dist},
                                     options=self.run_options)

        

        return self.train_output[1]  # returns something to log with tensorboard

    def chrome_trace(self):
        tl = timeline.Timeline(self.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        return ctf

    def test(self, sess, batch_x, batch_y, exp_dist):
        self.test_output = sess.run([self.merged, self.cost],  # , self.url_vector_norm_renormalization_condition
                                    feed_dict={self.x: batch_x,
                                               self.y: batch_y,
                                               self.d: exp_dist})
        print("test cost: " + str(self.test_output[1]))
        return self.test_output[0]

    def get_url_embeddings(self, sess):
        tot_urls = self.network.um.total_url_cnt
        url_sparse_vectors = self.network.um.idxs_to_tf(range(tot_urls))
        return sess.run(self.network.embedd_url_sparse_tensor(self.x), feed_dict={self.x: url_sparse_vectors})

    def print_info_(self):
        print("--------------------------------------------------------------")
        print("merged:", self.train_output[1])
