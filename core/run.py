import time

import tensorflow as tf

import numpy as np
import pandas as pd
import core.loader as ld
from core.config import Config
from core.interaction_index import InteractionIndex
from core.interaction_mapper import InteractionMapper
from core.metric_profiler import MetricProfiler
from core.network import Network
from core.tensorboard_writer import TensorboardWriter
from core.trainer import Trainer
from pathlib import Path

cf = Config(root_folder="./resources", output_folder="/.././output")
um = InteractionMapper(cf.path_interaction_map)
if cf.continnue_previous_run:
    pd_df = pd.read_csv(cf.previous_successful_output_run_dir + "/interaction_indexing/interaction_index.txt", header=None)
    for col in pd_df.columns:
        pd_df[col] = pd_df[col].astype(np.float32)
    network = Network(cf, um, preheated_embeddings=pd_df.values)
else:
    network = Network(cf, um)

train_loader = ld.Loader(cf, um, cf.path_train_data)
test_loader = ld.Loader(cf, um, cf.path_test_data)

trainer = Trainer(cf, network)
ii = None
mp = False
x_label = None

log_txt = "Config: " + cf.to_string() + "\n\n" + \
          "Interaction mapper: " + um.to_string() + "\n\n" + \
          "Train Loader @start: " + train_loader.to_string() + "\n\n" + \
          "Test Loader @start: " + test_loader.to_string()
cf.make_dirs()
tbw = TensorboardWriter(cf)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tbw.log_info(sess, log_txt)

    t0_seconds = time.time()
    while train_loader.epoch_cnt < cf.epochs:
        tb = time.time()
        batch_x, batch_y, target_distance = train_loader.get_next_batch(cf.batch_size)
        x_label = 1000 * train_loader.event_cnt / train_loader.tot_event_cnt + train_loader.epoch_cnt

        dt_batching = time.time() - tb
        tt = time.time()
        tensorboard_log_entry = trainer.train(sess, batch_x, batch_y, target_distance)
        dt_tensorflow = time.time() - tt
        dt_all = time.time() - tb
        events_per_sec_in_thousand = cf.batch_size / dt_all / 1000

        tbw.add_train_summary(tensorboard_log_entry, x_label)
        tbw.log_scalar(events_per_sec_in_thousand, x_label, tag="performance_metric: 1000 events per second")
        tbw.log_scalar(dt_tensorflow / dt_batching, x_label,
                       tag="performance_metric: delta time tensorflow / delta time batch processing")

        if train_loader.new_epoch:
            batch_x, batch_y, target_distance = test_loader.get_next_batch(cf.batch_size * 100, fake_factor=0)
            tensorboard_log_entry = trainer.test(sess, batch_x, batch_y, target_distance)

            tbw.add_test_summary(tensorboard_log_entry, x_label)
            tbw.flush()
            print("epochs: " + str(train_loader.epoch_cnt))

            ii = InteractionIndex(um, trainer.get_interaction_embeddings(sess))
            mp = MetricProfiler(cf, sess, tbw, train_loader, um, ii)
            mp.log_plots(x_label)
            # print(np.linalg.norm(trainer.get_interaction_embeddings(sess) - pd_df.values))

    mp.log_results()
    with open(cf.timeline_profile_path, 'w') as f:
        f.write(trainer.chrome_trace())

    tbw.flush()

ii.safe(cf.index_safe_path)

Path(cf.output_run_dir + '/_SUCCESS').touch()
input("Press Enter to continue...")
