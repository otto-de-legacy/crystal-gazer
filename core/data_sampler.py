# -*- coding: utf-8 -*-
import numpy as np


class DataSampler(object):
    def __init__(self, vals, probs, bucket_count=10):
        self.max_prob = max(probs)
        self.bucket_count = min(bucket_count, int(len(probs) / 3))

        print("""RandomSampler: bucketing... 
            Usage warn: draw first bucket then alll from that bucket. 
            Should avarage out at the end, not for a single draw!""")
        sorted_val_probs = sorted(zip(vals, probs), key=lambda x: x[1], reverse=False)

        bucket_prob_supposed = 1 / self.bucket_count

        self.data_buckets = [np.array([], dtype=np.int32)] * self.bucket_count
        self.bucket_probs = [np.array([], dtype=np.float32)] * self.bucket_count

        self.split_idxs = [0] * (self.bucket_count + 1)

        acc_bucket_prob = 0
        current_bucket = 0
        idx_counter = 0

        for val, prob in sorted_val_probs:
            acc_bucket_prob = acc_bucket_prob + prob

            if acc_bucket_prob >= bucket_prob_supposed:
                print("bucket determined")
                self.split_idxs[current_bucket + 1] = idx_counter
                self.bucket_probs[current_bucket] = acc_bucket_prob
                current_bucket = current_bucket + 1
                acc_bucket_prob = prob

            idx_counter = idx_counter + 1

        self.split_idxs[self.bucket_count] = len(sorted_val_probs)

        for i in range(self.bucket_count):
            idx_from = self.split_idxs[i]
            idx_to = self.split_idxs[i + 1]

            self.data_buckets[i] = np.array(sorted_val_probs[idx_from:idx_to])[:, 0]

        self.bucket_sizes = [len(bucket) for bucket in self.data_buckets]

        for i in range(len(self.bucket_sizes)):
            if self.bucket_sizes[i] == 0:
                print("ERROR: one bucket is zero")
            if self.bucket_sizes[i] == 1:
                print("WARNING: one bucket is small! Content:" + str(self.data_buckets[i][0]))

        print(self.to_string())

    def get_by_condition(self, lambda_exp=lambda x: x is not None):
        res = []
        for bucket in self.data_buckets:
            res = res + list(filter(lambda_exp, bucket))
        return res

    def rvs(self, size=1):
        chosen_bucket_idx = np.random.randint(0, self.bucket_count)
        bucket_size = self.bucket_sizes[chosen_bucket_idx]
        if bucket_size is 1:
            random_bucket_idxs = np.array([0] * size, np.int32)
        else:
            random_bucket_idxs = np.random.randint(0, bucket_size - 1, size=size)
        return self.data_buckets[chosen_bucket_idx][random_bucket_idxs]
        # return np.random.choice(self.vals, size, p=self.probs) SLOW!
        # return np.random.choice(self.vals, size) SLOW!

    def get_top_bucket(self):

        ret_dic = dict([
            ("bucket_events", self.data_buckets[-1]),
            ("bucket_cnt", self.bucket_sizes[-1]),
            ("bucket_prob", self.bucket_probs[-1])])

        return ret_dic

    def to_string(self):

        ret_string = """
        Result bucket organization:
        
        maximum probability: """ + str(self.max_prob) + """
        bucket count: """ + str(self.bucket_count) + """
        bucket event counts: """ + str(self.bucket_sizes) + """
        bucket probabilities: """ + str(self.bucket_probs) + """ 
        """
        #
        # for i in range(len(self.data_buckets)):
        #     to_idx = min(10, len(self.data_buckets[i]))
        #     events = ", ".join([str(e) for e in self.data_buckets[i][0:to_idx]])
        #     approx_prob = str(self.bucket_probs[i] / self.bucket_sizes[i])
        #     ret_string = ret_string + "\n\nnapprox prob: " + approx_prob + ",   events: " + events

        return ret_string
