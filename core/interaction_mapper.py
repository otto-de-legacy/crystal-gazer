# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from shutil import copyfile


class InteractionMapper(object):
    def __init__(self, map_path):

        self.interaction_map_path = map_path
        self.num_to_interaction_dict, self.interaction_to_num_dict, max_interaction_num = self.load_dictionaries()
        self.total_interaction_cnt = max_interaction_num
        self.interaction_class_cnt = max_interaction_num + 2  # + default class and cuonting from zero

    def load_dictionaries(self):

        num_to_interaction_dict = dict()
        interaction_to_num_dict = dict()
        max_interaction_num = 0

        with open(self.interaction_map_path + "/map") as f:
            lines = f.read().splitlines()
            for line in lines:
                entries = line.split(",")
                if len(entries) == 2:

                    num = int(entries[0])
                    interaction = entries[1]

                    if max_interaction_num < num:
                        max_interaction_num = num

                    num_to_interaction_dict[num] = interaction
                    interaction_to_num_dict[interaction] = num
                else:
                    print("Warn: entry seems corrupted (will be ignored), " + line)
        print("maximum interaction int found: " + str(max_interaction_num))
        return num_to_interaction_dict, interaction_to_num_dict, max_interaction_num

    def interaction_idx_apply_constraints(self, interaction):
        if interaction <= self.total_interaction_cnt:
            return interaction
        else:
            print("Warn: interaction was: " + str(interaction) + ", only maximum of " + str(
                self.total_interaction_cnt) + "expected.")
            return self.total_interaction_cnt

    def interaction_to_num(self, interaction):
        return self.interaction_to_num_dict.get(interaction, self.interaction_class_cnt)

    def num_to_interaction(self, num):
        return self.num_to_interaction_dict.get(num, "")

    def idxs_to_tf(self, interaction_int_reps):
        batch_size = len(interaction_int_reps)
        indices = []
        values = np.ones(batch_size)
        shape = [batch_size, self.interaction_class_cnt]
        for batch_idx, interaction in enumerate(interaction_int_reps):
            interaction_num = self.interaction_idx_apply_constraints(interaction)
            indices.append([batch_idx, interaction_num])
        return tf.SparseTensorValue(indices, values, shape)

    def events_to_tf(self, events):
        features = self.idxs_to_tf([e.feature for e in events])
        labels = self.idxs_to_tf([e.label for e in events])

        return features, labels

    def to_string(self):
        ret_string = """
        interaction mapping info:
        
        max interaction int found: """ + str(self.total_interaction_cnt) + """ 
        assumed unique interactions + 1(default) +1 (index from 0): """ + str(self.interaction_class_cnt) + """ 
        """
        return ret_string

    def save(self, path):
        copyfile(self.interaction_map_path + "/map", path + "map.txt")
