# -*- coding: utf-8 -*-
import nmslib

import numpy as np


class ConditionalIndex(object):
    def __init__(self,
                 interaction_mapper,
                 interaction_vectors,
                 lambdas_of_key=list([]),
                 method="ghtree",
                 space="cosinesimil",
                 build_full_index=True):

        self.im = interaction_mapper
        self.build_full_index = build_full_index
        self.interaction_vectors = interaction_vectors
        self.vector_size = len(interaction_vectors[0])
        self.tot_object_cnt = len(interaction_vectors)

        conditional_vectors = [[] for _ in range(len(lambdas_of_key))]
        self.conditional_remappings = [dict() for _ in range(len(lambdas_of_key))] #* len(lambdas_of_key)
        cnts = [0] * len(lambdas_of_key)
        for i, vec in enumerate(interaction_vectors):
            key_name = self.im.num_to_interaction(i)
            if i % 10000 == 0:
                print(str(i))
            for j, fun in enumerate(lambdas_of_key):
                if fun(key_name):
                    self.conditional_remappings[j][cnts[j]] = i
                    conditional_vectors[j].append(vec)
                    cnts[j] = cnts[j] + 1

        self.conditional_indices = []
        for j in range(len(lambdas_of_key)):
            print("building conditional index...")
            self.conditional_indices.append(nmslib.init(method=method, space=space))
            self.conditional_indices[j].addDataPointBatch(conditional_vectors[j])
            if method == "hnsw":
                self.conditional_indices[j].createIndex({'post': 2}, print_progress=True)
            else:
                self.conditional_indices[j].createIndex()

        if self.build_full_index:
            print("building full index...")
            self.full_index = nmslib.init(method=method, space=space)
            self.full_index.addDataPointBatch(self.interaction_vectors)
            if method == "hnsw":
                self.full_index.createIndex({'post': 2}, print_progress=True)
            else:
                self.full_index.createIndex()



    def knn_vector_query(self, vec, which_index=None, k=1):
        query_vector = vec
        if which_index is not None:
            ids, distances = self.conditional_indices[which_index].knnQuery(query_vector, k=k)
            ret_interaction = [self.im.num_to_interaction(self.conditional_remappings[which_index][id]) for id in ids]
            return ret_interaction, ids, distances
        elif self.build_full_index:  # return full query
            ids, distances = self.full_index.knnQuery(query_vector, k=k)
            ret_interaction = [self.im.num_to_interaction(id) for id in ids]
            return ret_interaction, ids, distances
        else:
            print("ERROR: did not build the full index, please specify the wanted index by providing an integer")
            return [], [], []

    def knn_idx_query(self, idx_int, which_index=None, k=1):
        try:
            query_vector = self.interaction_vectors[idx_int]
        except:
            print("Error: no corresponding interaction found")
            return [], [], []
        return self.knn_vector_query(query_vector, which_index, k=k)

    def knn_interaction_query(self, interaction_str, which_index=None, k=1):
        return self.knn_idx_query(self.im.interaction_to_num(interaction_str), which_index, k=k)

    def safe(self, path):
        np.savetxt(path + "/interaction_index.txt", self.interaction_vectors, delimiter=",")
        self.im.save(path)
