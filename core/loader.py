# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
from core.data_sampler import DataSampler
from core.event import Event
from scipy import stats
import core.url_mapper as um
import time


class Loader(object):
    def __init__(self, config, in_data_text, url_mapper):
        self.cf = config
        self.random_generator, self.tot_event_cnt, self.unique_train_event_cnt = self._prepare_events(
            in_data_text.replace(" ", ""))
        self.um = url_mapper
        self.epoch_cnt = 0
        self.batche_cnt = 0
        self.event_cnt = 0
        self.new_epoch = True

    def _user_journey_to_events(self, journey_string):
        urls = journey_string.split(",")
        events = []
        for i in range(len(urls) - 1):
            url = urls[i]
            url_plus_one = urls[i + 1]
            f_idx = int(url)
            t_idx = int(url_plus_one)
            events.append(Event(f_idx, t_idx))
        return events

    def _prepare_events(self, text_content):
        events = dict()
        cnt_tot_events = 0
        cnt_uniqe_events = 0
        user_journeys = text_content.split("\n")
        for strng in user_journeys:
            for new_event in self._user_journey_to_events(strng):
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

    def _prepare_eventsOLD(self, text_content):
        events = []
        user_journeys = text_content.split("\n")
        for str in user_journeys:
            events = events + self._user_journey_to_events(str)
        return np.array(events)

    def _update_processed_state(self, batch_size):
        if self.event_cnt % self.unique_train_event_cnt >= (self.event_cnt + batch_size) % self.unique_train_event_cnt:
            self.epoch_cnt += 1
            self.new_epoch = True
        self.batche_cnt += 1
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

        fake_batch_events_clone = [Event(e.feature_idx, random.randint(0, self.um.total_url_cnt)) for e in fake_batch_events]

        full_events = np.concatenate((real_batch_events, fake_batch_events_clone), axis=0)

        features = self.um.idxs_to_tf([e.feature_idx for e in full_events])
        labels = self.um.idxs_to_tf([e.label_idx for e in full_events])
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
        batches processed: """ + str(self.batche_cnt) + """
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
            top_bucket_info = top_bucket_info + "feature: " + self.um.num_to_url(e.feature_idx) + """
            """
            top_bucket_info = top_bucket_info + "target: " + self.um.num_to_url(e.label_idx) + """
            """
            top_bucket_info = top_bucket_info + "occurence count: " + str(e.count) + """
            """
            top_bucket_info = top_bucket_info + """----------------------------------------------------
            """

        return ret_string + top_bucket_info
