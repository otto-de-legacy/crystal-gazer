import tensorflow as tf
import numpy as np

class Network:
    weights = None
    biases = None

    def __init__(self, config, interaction_mapper, preheated_embeddings=None):
        self.cf = config
        self.im = interaction_mapper
        self._initialize_params_(preheated_embeddings)

    def _initialize_params_(self, preheated_embeddings=None):
        """preheated_embeddings should be a numpy array"""
        rand_std = 0.01
        num_categories = self.im.interaction_class_cnt

        if preheated_embeddings is not None:
            cf_shape = np.array([num_categories, self.cf.embedding_size])
            pre_shape = preheated_embeddings.shape
            if not np.array_equal(cf_shape, pre_shape):
                raise ValueError('preheated_embeddings, shape= ' + str(pre_shape) + ' has not the shape in the config, shape=' + str(cf_shape))
            initial_embedding = tf.Variable(tf.constant(preheated_embeddings))
        else:
            initial_embedding = tf.Variable(tf.random_normal([num_categories, self.cf.embedding_size], stddev=rand_std))

        self.weights = {
            'w_embedding': initial_embedding,
            'w_core': tf.Variable(tf.random_normal([self.cf.embedding_size, self.cf.embedding_size], stddev=rand_std))
        }

        self.biases = {
            'b_core': tf.Variable(tf.random_normal([self.cf.embedding_size]))
        }

    def predict(self, interaction_sparse_tensor):
        out_embedding = self.embedd_interaction_sparse_tensor(interaction_sparse_tensor)
        # out_core = self.layer_core(out_embedding)

        return out_embedding

    def embedd_interaction_sparse_tensor(self, sparse_interaction_tensor):
        # TODO: check out tensorflow embedding lookup for speed
        embedding_matrix = self.weights['w_embedding']
        result = tf.sparse_tensor_dense_matmul(sparse_interaction_tensor, embedding_matrix)
        return result

    def layer_core(self, in_tensor):
        return tf.matmul(in_tensor, self.weights['w_core']) + self.biases['b_core']

