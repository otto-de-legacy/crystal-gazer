import tensorflow as tf


class Network:
    weights = None
    biases = None

    def __init__(self, config, url_mapper):
        self.cf = config
        self.um = url_mapper
        self._initialize_params_()

    def _initialize_params_(self):
        rand_std = 0.01
        num_categories = self.um.url_class_cnt

        self.weights = {
            'w_embedding': tf.Variable(tf.random_normal([num_categories, self.cf.embedding_size], stddev=rand_std)),
            'w_core': tf.Variable(tf.random_normal([self.cf.embedding_size, self.cf.embedding_size], stddev=rand_std))
        }

        self.biases = {
            'b_core': tf.Variable(tf.random_normal([self.cf.embedding_size]))
        }

    def predict(self, url_sparse_tensor):
        out_embedding = self.embedd_url_sparse_tensor(url_sparse_tensor)
        # out_core = self.layer_core(out_embedding)

        return out_embedding

    def embedd_url_sparse_tensor(self, sparse_url_tensor):
        # TODO: check out tensorflow embedding lookup for speed
        embedding_matrix = self.weights['w_embedding']
        result = tf.sparse_tensor_dense_matmul(sparse_url_tensor, embedding_matrix)
        return result

    def layer_core(self, in_tensor):
        return tf.matmul(in_tensor, self.weights['w_core']) + self.biases['b_core']

