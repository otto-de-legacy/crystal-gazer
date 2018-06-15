# -*- coding: utf-8 -*-
from collections import namedtuple
from itertools import groupby

MetricEntry = namedtuple("metric_entry", [
    "interaction",
    "truth_corresp_interaction",
    "truth_corresp_interaction_idx",
    "model_interaction_idx",
    "rel_position",
    "weight",
    "found_q"
])


class MetricProfiler(object):
    def __init__(self, config, sess, tb_writer, loader, interaction_mapper, interaction_index):
        self.cf = config
        self.tbw = tb_writer
        self.ld = loader
        self.im = interaction_mapper
        self.ii = interaction_index
        self.sess = sess

    def calc_metrics(self, interactions_considered_cnt, k, result_cnt):
        metric_entries = []

        for cnt in range(interactions_considered_cnt):

            random_interaction_idx = self.ld.random_generator.rvs(size=1)[0].feature_idx
            random_interaction_str = self.im.num_to_interaction(random_interaction_idx)

            events_from_true_data = sorted(self.ld.random_generator.get_by_condition(
                lambda event: event.feature_idx == random_interaction_idx),
                key=lambda e: e.count, reverse=True)[0:result_cnt]
            _, query_res_from_model_idxs, _ = self.ii.knn_idx_query(random_interaction_idx, k=k)

            for truth_interaction_idx, re in enumerate(events_from_true_data):
                if re.label_idx in query_res_from_model_idxs:
                    found_q = True
                    pos_actual = list(query_res_from_model_idxs).index(re.label_idx)
                else:
                    found_q = False
                    pos_actual = k + 1

                metric_entries = metric_entries + [
                    MetricEntry(interaction=random_interaction_str,
                                truth_corresp_interaction=self.im.num_to_interaction(re.label_idx),
                                truth_corresp_interaction_idx=truth_interaction_idx,
                                model_interaction_idx=pos_actual,
                                rel_position=max(pos_actual - truth_interaction_idx + 1, 0),
                                weight=re.count,
                                found_q=found_q)]
        return metric_entries

    def log_plots(self, x_label):
        metric_res = self.calc_metrics(self.cf.result_cnt_plots, self.cf.knn_plots, self.cf.events_from_true_data)

        weighted_results = [r.rel_position * r.weight for r in metric_res]
        weights = [r.weight for r in metric_res]
        weighted_pos_avg = sum(weighted_results) / (sum(weights) + 0.000000001)

        self.tbw.log_scalar(weighted_pos_avg, x_label, tag="evaluation_metric: weighted_pos_avg k=" + str(self.cf.knn_plots) + ", res=" + str(self.cf.result_cnt_plots))


    def log_results(self):
        metric_res = self.calc_metrics(self.cf.result_cnt_final, self.cf.knn_plots, self.cf.events_from_true_data)

        for key, group in groupby(metric_res, lambda x: x.interaction):
            self.log_group(group)

    def log_group(self, group):
        group_list = list(group)
        log_str = "truth idx,model idx,relative position,weight,interaction\t\t\n"
        cnt = 0
        for metric_entry in group_list:
            cnt = cnt + 1
            if cnt <= 50:
                log_str = log_str + str(metric_entry.truth_corresp_interaction_idx) + "\t\t,"
                if metric_entry.found_q:
                    log_str = log_str + str(metric_entry.model_interaction_idx) + "\t\t,"
                    log_str = log_str + str(metric_entry.rel_position) + "\t\t,"
                else:
                    log_str = log_str + "---" + "\t\t,"
                    log_str = log_str + "---" + "\t\t,"
                log_str = log_str + str(metric_entry.weight) + "\t\t,"
                log_str = log_str + metric_entry.truth_corresp_interaction + "\t\t\n"

        weighted_results = [r.rel_position * r.weight for r in group_list]
        weights = [r.weight for r in group_list]
        weighted_pos_avg = sum(weighted_results) / (sum(weights) + 0.000000001)

        log_str = "total weight: " + str(sum(weights)) + "\t\t\n" + log_str
        log_str = "weighted pos avg: " + str(weighted_pos_avg) + "\t\t\n" + log_str
        log_str = "our interaction: " + group_list[0].interaction + "\t\t\n" + log_str

        self.tbw.log_info(self.sess, log_str)
        print(log_str)


