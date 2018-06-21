import glob
import random

import numpy as np

from core.data_sampler import DataSampler
from core.event import Event


class Loader(object):
    def __init__(self, config, interaction_mapper, path):
        """will try to read ALL files in path!"""
        self.cf = config
        self.root_path = path
        self.random_generator, self.tot_event_cnt, self.unique_train_event_cnt = self._prepare_events()
        self.im = interaction_mapper
        self.epoch_cnt = 0
        self.batch_cnt = 0
        self.event_cnt = 0
        self.new_epoch = True

    def _user_journey_to_events(self, journey_string):
        """journey_string needs to be comma separated integers, i.e., 15,35,37,38,..."""
        user_interaction = journey_string.split(",")
        events = []
        for i in range(len(user_interaction) - 1):
            interaction = user_interaction[i]
            f_idx = int(interaction)

            for n in range(self.cf.neighboring_interactions + 1):
                if n + i > len(user_interaction) - 1:
                    break
                interaction_plus_n = user_interaction[i + n]
                t_idx = int(interaction_plus_n)
                if f_idx is not t_idx:
                    events.append(Event(f_idx, t_idx))
        return events

    def _prepare_events(self):
        events = dict()
        cnt_tot_events = 0
        cnt_uniqe_events = 0

        for file_url in glob.glob(self.root_path + "/*"):
            with open(file_url) as f:
                for line in f:
                    for new_event in self._user_journey_to_events(line):
                        cnt_tot_events = cnt_tot_events + 1
                        if new_event in events:
                            events[new_event] = events[new_event] + 1
                        else:
                            events[new_event] = 1
                            cnt_uniqe_events = cnt_uniqe_events + 1
                        if cnt_tot_events % 100000 is 0:
                            print("Events in list: " + str(cnt_tot_events))
                            print("Events in dict: " + str(len(events)))

        xk = list(events.keys())
        for e, cnt in zip(events, events.values()):  # TODO: make this nicer by aggregation and sum elements
            e.count = cnt
        pk = list(np.array(list(events.values())) / cnt_tot_events)

        random_generator = DataSampler(xk, pk, bucket_count=self.cf.bucket_count)  #
        return random_generator, cnt_tot_events, cnt_uniqe_events

    def _update_processed_state(self, batch_size):
        if self.event_cnt % self.unique_train_event_cnt >= (self.event_cnt + batch_size) % self.unique_train_event_cnt:
            self.epoch_cnt += 1
            self.new_epoch = True
        self.batch_cnt += 1
        self.event_cnt += batch_size

    def get_random_events(self, size):
        return self.random_generator.rvs(size=size)

    def get_next_batch(self, batch_size, fake_factor=None):
        if fake_factor is None:
            fake_factor = self.cf.fake_frac

        eff_batch_size = min(batch_size, self.unique_train_event_cnt)

        self.new_epoch = False

        fake_batch_size = int(eff_batch_size * fake_factor)
        real_batch_size = eff_batch_size - fake_batch_size

        real_batch_events = self.get_random_events(real_batch_size)
        fake_batch_events = self.get_random_events(fake_batch_size)

        fake_batch_events_clone = [Event(e.feature_idx, random.randint(0, self.im.total_interaction_cnt)) for e in
                                   fake_batch_events]

        full_events = np.concatenate((real_batch_events, fake_batch_events_clone), axis=0)

        features = self.im.idxs_to_tf([e.feature_idx for e in full_events])
        labels = self.im.idxs_to_tf([e.label_idx for e in full_events])
        target_distance = np.concatenate((np.zeros(len(real_batch_events), np.float32),
                                          np.ones(len(fake_batch_events), np.float32)), axis=0)

        self._update_processed_state(eff_batch_size)
        return features, labels, target_distance

    def get_all_data(self):
        return self.get_next_batch(self.unique_train_event_cnt)

    def to_string(self):
        ret_string = """
        General parameters of the loader:
        
        events considered in total: """ + str(self.tot_event_cnt) + """
        unique train events: """ + str(self.unique_train_event_cnt) + """
        epochs processed: """ + str(self.epoch_cnt) + """
        batches processed: """ + str(self.batch_cnt) + """
        events processed: """ + str(self.event_cnt) + """
        
        
        """ + self.random_generator.to_string() + """\n\ntop bucket: \n\n"""

        top_bucket_info = ""
        top_bucket = self.random_generator.get_top_bucket()
        cnt = top_bucket["bucket_cnt"]
        prob = top_bucket["bucket_prob"]
        single_event_prob = prob / cnt
        avg_occurence_per_top_event = self.tot_event_cnt * single_event_prob

        top_bucket_info = top_bucket_info + "avg event occurence: " + str(avg_occurence_per_top_event) + "\n\n"
        top_bucket_info = top_bucket_info + """----------------------------------------------------
            """
        for e in top_bucket["bucket_events"][0:20]:
            top_bucket_info = top_bucket_info + "feature: " + self.im.num_to_interaction(
                e.feature_idx) + ", feature_idx: " + str(e.feature_idx) + """
            """
            top_bucket_info = top_bucket_info + "target: " + self.im.num_to_interaction(
                e.label_idx) + ", label_idx: " + str(e.label_idx) + """"
            """
            top_bucket_info = top_bucket_info + "occurence count: " + str(e.count) + """
            """
            top_bucket_info = top_bucket_info + """----------------------------------------------------
            """

        return ret_string + top_bucket_info
