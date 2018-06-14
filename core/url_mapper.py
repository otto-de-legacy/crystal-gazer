# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import re


class UrlMapper(object):
    def __init__(self, config=None):

        self.cf = config
        self.num_to_url_dict, self.url_to_num_dict, max_url_num = self.load_dictionaries()
        self.total_url_cnt = max_url_num
        self.url_class_cnt = max_url_num + 2  # + default class and cuonting from zero

    def load_dictionaries(self):

        num_to_url_dict = dict()
        url_to_num_dict = dict()
        max_url_num = 0

        if self.cf is not None:
            print()
            with open(self.cf.url_map) as f:
                lines = f.read().splitlines()
                for line in lines:
                    entries = line.split(",")
                    if len(entries) == 2:

                        num = int(entries[0])
                        url = entries[1]

                        if max_url_num < num:
                            max_url_num = num

                        num_to_url_dict[num] = url
                        url_to_num_dict[url] = num
                    else:
                        print("Warn: entry seems corrupted (will be ignored), " + line)
        print("maximum url int found: " + str(max_url_num))
        return num_to_url_dict, url_to_num_dict, max_url_num

    def url_idx_apply_constraints(self, url):
        if url <= self.total_url_cnt:
            return url
        else:
            print("Warn: url was: " + str(url) + ", only maximum of " + str(self.total_url_cnt) + "expected.")
            return self.total_url_cnt

    def url_to_num(self, url):
        return self.url_to_num_dict.get(url, self.url_class_cnt)

    def num_to_url(self, num):
        return self.num_to_url_dict.get(num, "")

    def idxs_to_tf(self, url_int_reps):
        batch_size = len(url_int_reps)
        indices = []
        values = np.ones(batch_size)
        shape = [batch_size, self.url_class_cnt]
        for batch_idx, url in enumerate(url_int_reps):
            url_num = self.url_idx_apply_constraints(url)
            indices.append([batch_idx, url_num])
        return tf.SparseTensorValue(indices, values, shape)

    def events_to_tf(self, events):
        features = self.idxs_to_tf([e.feature for e in events])
        labels = self.idxs_to_tf([e.label for e in events])

        return features, labels

    def to_string(self):
        ret_string = """
        url mapping info:
        
        max url int found: """ + str(self.total_url_cnt) + """ 
        assumed unique urls + 1(default) +1 (index from 0): """ + str(self.url_class_cnt) + """ 
        """
        return ret_string
