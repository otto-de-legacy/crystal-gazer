# -*- coding: utf-8 -*-
import nmslib

import numpy as np


class UrlIndex(object):
    def __init__(self, config, url_mapper, url_vectors):
        self.cf = config
        self.um = url_mapper
        self.url_vectors = url_vectors

        if self.cf.short:
            # self.index = nmslib.init(method='ghtree', space='cosinesimil')
            # self.index = nmslib.init(method='ghtree', space='l2')
            self.index = nmslib.init(method='ghtree', space='cosinesimil')
            self.index.addDataPointBatch(url_vectors)
            self.index.createIndex()
        else:
            # self.index = nmslib.init(method='hnsw', space='cosinesimil')
            # self.index = nmslib.init(method='ghtree', space='l2')
            self.index = nmslib.init(method='ghtree', space='cosinesimil')
            self.index.addDataPointBatch(url_vectors)
            # self.index.createIndex({'post': 2}, print_progress=True)
            self.index.createIndex()

    def knn_idx_query(self, idx_int, k=1):
        try:
            query_vector = self.url_vectors[idx_int]
        except:
            print("Error: no corresponding Url found")
            return [], [], []
        return self.knn_vector_query(query_vector, k=k)

    def knn_vector_query(self, vec, k=1):
        query_vector = vec
        ids, distances = self.index.knnQuery(query_vector, k=k)
        ret_urls = [self.um.num_to_url(id) for id in ids]
        return ret_urls, ids, distances

    def knn_url_query(self, url_str, k=1):
        return self.knn_idx_query(self.um.url_to_num(url_str), k=k)

    def safe(self, path):
        np.savetxt(path, self.url_vectors, delimiter=",")
