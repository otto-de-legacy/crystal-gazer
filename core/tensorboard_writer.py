# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
from scipy import stats
import os
import core.url_mapper as um
from multiprocessing import Process


class TensorboardWriter(object):

    def __init__(self, config):
        self.cf = config

        def launchTensorBoard(cmd, path):
            os.system(cmd + path + "\"")
            return




        p = Process(target=launchTensorBoard, args=(self.cf.tb_command, self.cf.tb_dir))
        p.start()

        self.tensorboard_train_writer = tf.summary.FileWriter(self.cf.tb_dir + "/train")
        self.tensorboard_test_writer = tf.summary.FileWriter(self.cf.tb_dir + "/test")
        self.tensorboard_log_writer = tf.summary.FileWriter(self.cf.tb_dir + "/log")

    def log_info(self, sess, log_txt):
        log_tensor = sess.run(tf.summary.text('info', tf.convert_to_tensor(log_txt)))
        self.tensorboard_log_writer.add_summary(log_tensor)
        self.tensorboard_log_writer.flush()

    def log_scalar(self, y_val, x_val, tag=""):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=y_val)])
        self.tensorboard_log_writer.add_summary(summary, x_val)
        # log_tensor = sess.run(tf.summary.scalar('info', tf.convert_to_tensor(val)))
        # self.tensorboard_log_writer.add_summary(log_tensor)

    def add_train_summary(self, train_output, counter):
        self.tensorboard_train_writer.add_summary(train_output, counter)

    def add_test_summary(self, test_output, counter):
        self.tensorboard_test_writer.add_summary(test_output, counter)

    def flush(self):
        self.tensorboard_train_writer.flush()
        self.tensorboard_test_writer.flush()
