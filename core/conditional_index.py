# -*- coding: utf-8 -*-
import nmslib

import numpy as np


class ConditionalIndex(object):
    def __init__(self, interaction_mapper, interaction_vectors, lambda_on_dict_key, method="ghtree",
                 space="cosinesimil"):
        self.im = interaction_mapper


        conditional_vectors = []
        self.indx_to_full_int = dict()
        cnt = 0
        for i, vec in enumerate(interaction_vectors):
            if i % 10000 == 0:
                print(str(i))
            if lambda_on_dict_key(self.im.num_to_interaction(i)):
                self.indx_to_full_int[cnt] = i
                conditional_vectors.append(vec)
                cnt = cnt + 1

        self.interaction_vectors = interaction_vectors

        self.index = nmslib.init(method=method, space=space)
        self.index.addDataPointBatch(conditional_vectors)
        if method == "hnsw":
            self.index.createIndex({'post': 2}, print_progress=True)
        else:
            self.index.createIndex()

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
        ret_interaction = [self.im.num_to_interaction(self.indx_to_full_int[id]) for id in ids]
        return ret_interaction, ids, distances

    def knn_interaction_query(self, interaction_str, k=1):
        return self.knn_idx_query(self.im.interaction_to_num(interaction_str), k=k)

    def safe(self, path):
        np.savetxt(path + "/interaction_index.txt", self.interaction_vectors, delimiter=",")
        self.im.save(path)
