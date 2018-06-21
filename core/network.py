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
        rand_std = 1/self.cf.embedding_size
        num_categories = self.im.interaction_class_cnt

        if preheated_embeddings is not None:

            if not np.array_equal(self.cf.embedding_size, preheated_embeddings.shape[1]):
                raise ValueError('preheated_embeddings embedding size does not match the embedding size: ' + str(self.cf.embedding_size))
            if num_categories < len(preheated_embeddings):
                raise ValueError('preheated_embeddings are larger than the network is constructed for, num_categories: ' + str(num_categories))

            if num_categories > len(preheated_embeddings):
                print("INFO: filling missing embedding vectors with a random normal distribution.")
                missing_vector_cnt = num_categories - len(preheated_embeddings)
                new_rand = np.array(np.random.randn(missing_vector_cnt, self.cf.embedding_size) * rand_std, dtype=np.float32)
                initial_embedding = tf.Variable(tf.constant(np.concatenate((preheated_embeddings, new_rand), axis=0)))
            else:
                print("INFO: Using only the given embeddings.")
                initial_embedding = tf.Variable(tf.constant(preheated_embeddings))
        else:
            print("INFO: Generate random embeddings.")
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

