# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple
from itertools import groupby

MetricEntry = namedtuple("metric_entry", [
    "url",
    "truth_corresp_url",
    "truth_corresp_url_idx",
    "model_url_idx",
    "rel_position",
    "weight",
    "found_q"
])


class MetricProfiler(object):
    def __init__(self, config, sess, tb_writer, loader, url_mapper, url_index):
        self.cf = config
        self.tbw = tb_writer
        self.ld = loader
        self.um = url_mapper
        self.ui = url_index
        self.sess = sess

    def calc_metrics(self, urls_considered_cnt, k, result_cnt):
        metric_entries = []

        for cnt in range(urls_considered_cnt):

            random_url_idx = self.ld.random_generator.rvs(size=1)[0].feature_idx
            random_url_str = self.um.num_to_url(random_url_idx)

            events_from_true_data = sorted(self.ld.random_generator.get_by_condition(
                lambda event: event.feature_idx == random_url_idx),
                key=lambda e: e.count, reverse=True)[0:result_cnt]
            _, query_res_from_model_idxs, _ = self.ui.knn_idx_query(random_url_idx, k=k)

            for truth_url_idx, re in enumerate(events_from_true_data):
                if re.label_idx in query_res_from_model_idxs:
                    found_q = True
                    pos_actual = list(query_res_from_model_idxs).index(re.label_idx)
                else:
                    found_q = False
                    pos_actual = k + 1

                metric_entries = metric_entries + [
                    MetricEntry(url=random_url_str,
                                truth_corresp_url=self.um.num_to_url(re.label_idx),
                                truth_corresp_url_idx=truth_url_idx,
                                model_url_idx=pos_actual,
                                rel_position=max(pos_actual - truth_url_idx + 1, 0),
                                weight=re.count,
                                found_q=found_q)]
        return metric_entries

    def log_plots(self, x_label):
        metric_res = self.calc_metrics(self.cf.result_cnt_plots, self.cf.knn_plots, self.cf.events_from_true_data)

        weighted_results = [r.rel_position * r.weight for r in metric_res]
        weights = [r.weight for r in metric_res]
        weighted_pos_avg = sum(weighted_results) / (sum(weights) + 0.000000001)

        self.tbw.log_scalar(weighted_pos_avg, x_label, tag="evaluation_metric: weighted_pos_avg k=" + str(self.cf.knn_plots) + ", res=" + str(self.cf.result_cnt_plots))


    def log_url_results(self):
        metric_res = self.calc_metrics(self.cf.result_cnt_final, self.cf.knn_plots, self.cf.events_from_true_data)

        for key, group in groupby(metric_res, lambda x: x.url):
            self.log_url_group(group)

    def log_url_group(self, group):
        group_list = list(group)
        log_str = "truth idx,model idx,relative position,weight,url\t\t\n"
        cnt = 0
        for metric_entry in group_list:
            cnt = cnt + 1
            if cnt <= 50:
                log_str = log_str + str(metric_entry.truth_corresp_url_idx) + "\t\t,"
                if metric_entry.found_q:
                    log_str = log_str + str(metric_entry.model_url_idx) + "\t\t,"
                    log_str = log_str + str(metric_entry.rel_position) + "\t\t,"
                else:
                    log_str = log_str + "---" + "\t\t,"
                    log_str = log_str + "---" + "\t\t,"
                log_str = log_str + str(metric_entry.weight) + "\t\t,"
                log_str = log_str + metric_entry.truth_corresp_url + "\t\t\n"

        weighted_results = [r.rel_position * r.weight for r in group_list]
        weights = [r.weight for r in group_list]
        weighted_pos_avg = sum(weighted_results) / (sum(weights) + 0.000000001)

        log_str = "total weight: " + str(sum(weights)) + "\t\t\n" + log_str
        log_str = "weighted pos avg: " + str(weighted_pos_avg) + "\t\t\n" + log_str
        log_str = "our url: " + group_list[0].url + "\t\t\n" + log_str

        self.tbw.log_info(self.sess, log_str)
        print(log_str)



    #
    #
    # def write_metrics(self, x_label, k=100, log_single_urls=False):
    #     pos_diffs = []
    #     weights = []
    #
    #     cnt = 0
    #     while cnt < self.cf.analyze_result_cnt:
    #         cnt = cnt + 1
    #         log_str = ""
    #         random_url_idx = self.ld.random_generator.rvs(size=1)[0].feature_idx
    #         random_url_str = self.um.num_to_url(random_url_idx)
    #
    #         log_str = log_str + "truth idx,model idx,relative position,weight,url\t\t\n"
    #
    #         events_from_true_data = self.ld.random_generator.get_by_condition(
    #             lambda event: event.feature_idx == random_url_idx)
    #         events_from_true_data = sorted(events_from_true_data, key=lambda e: e.count, reverse=True)
    #         _, query_res_from_model_idxs, _ = self.ui.knn_idx_query(random_url_idx, k=k)
    #
    #         weighted_pos_avg_sum = 0
    #         weight_sum = 0
    #
    #         for idx, re in enumerate(events_from_true_data):
    #             pos_should_be = idx + 1
    #
    #             if re.label_idx in query_res_from_model_idxs:
    #                 found_q = True
    #                 pos_actual = list(query_res_from_model_idxs).index(re.label_idx)
    #             else:
    #                 found_q = False
    #                 pos_actual = k + 1
    #
    #             rel_pos = max(pos_actual - pos_should_be, 0)
    #             pos_diffs = pos_diffs + [rel_pos]
    #             weight = re.count
    #             weights = weights + [weight]
    #
    #             weighted_pos_avg_sum = weighted_pos_avg_sum + rel_pos * weight
    #             weight_sum = weight_sum + weight
    #
    #
    #             me = MetricEntry(5, 6)
    #
    #         log_str = "total weight: " + str(weight_sum) + "\t\t\n" + log_str
    #         log_str = "weighted pos avg: " + str(
    #             weighted_pos_avg_sum / (weight_sum + 0.0000000001)) + "\t\t\n" + log_str
    #         log_str = "our url: " + random_url_str + "\t\t\n" + log_str
    #         if log_single_urls:
    #             self.tbw.log_info(self.sess, log_str)
    #             print(log_str)
    #     quer_res = np.array(np.array(list(zip(pos_diffs, weights))))
    #     weightet_pos_avg = sum(quer_res[:, 0] * quer_res[:, 1]) / sum(quer_res[:, 1])
    #     self.tbw.log_scalar(weightet_pos_avg, x_label, tag="evaluation_metric: weighted_pos_avg k=" + str(k))
