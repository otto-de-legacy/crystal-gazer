# -*- coding: utf-8 -*-
import nmslib

import numpy as np


class InteractionIndex(object):
    def __init__(self, interaction_mapper, interaction_vectors, method='ghtree', space='cosinesimil'):
        self.im = interaction_mapper
        self.interaction_vectors = interaction_vectors

        # if self.cf.short:
        # self.index = nmslib.init(method='ghtree', space='cosinesimil')
        # self.index = nmslib.init(method='ghtree', space='l2')
        self.index = nmslib.init(method=method, space=space)
        self.index.addDataPointBatch(interaction_vectors)
        self.index.createIndex()
        # else:
        #    # self.index = nmslib.init(method='hnsw', space='cosinesimil')
        #    # self.index = nmslib.init(method='ghtree', space='l2')
        #    self.index = nmslib.init(method='ghtree', space='cosinesimil')
        #    self.index.addDataPointBatch(interaction_vectors)
        #    # self.index.createIndex({'post': 2}, print_progress=True)
        #    self.index.createIndex()

    def knn_idx_query(self, idx_int, k=1):
        try:
            query_vector = self.interaction_vectors[idx_int]
        except:
            print("Error: no corresponding interaction found")
            return [], [], []
        return self.knn_vector_query(query_vector, k=k)

    def knn_vector_query(self, vec, k=1):
        query_vector = vec
        ids, distances = self.index.knnQuery(query_vector, k=k)
        ret_interaction = [self.im.num_to_interaction(id) for id in ids]
        return ret_interaction, ids, distances

    def knn_interaction_query(self, interaction_str, k=1):
        return self.knn_idx_query(self.im.interaction_to_num(interaction_str), k=k)

    def safe(self, path):
        np.savetxt(path + "interaction_index.txt", self.interaction_vectors, delimiter=",")
        self.im.save(path)
