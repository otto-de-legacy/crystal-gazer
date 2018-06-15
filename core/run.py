import time

import tensorflow as tf

import core.loader as ld
from core.config import Config
from core.interaction_index import InteractionIndex
from core.interaction_mapper import InteractionMapper
from core.metric_profiler import MetricProfiler
from core.network import Network
from core.tensorboard_writer import TensorboardWriter
from core.trainer import Trainer

cf = Config(short=True)
cf.make_dirs()
tbw = TensorboardWriter(cf)
um = InteractionMapper(cf)

with open(cf.url_train_data, 'r') as f:
    txt = str(f.read())
train_loader = ld.Loader(cf, txt, um)
with open(cf.url_test_data, 'r') as f:
    txt = str(f.read())
test_loader = ld.Loader(cf, txt, um)

network = Network(cf, um)
trainer = Trainer(cf, network)
ui = None
mp = False
x_label = None

log_txt = "Config: " + cf.to_string() + "\n\n" + \
          "Interaction mapper: " + um.to_string() + "\n\n" + \
          "Train Loader @start: " + train_loader.to_string() + "\n\n" + \
          "Test Loader @start: " + test_loader.to_string()


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

            ui = InteractionIndex(cf, um, trainer.get_interaction_embeddings(sess))
            mp = MetricProfiler(cf, sess, tbw, train_loader, um, ui)
            mp.log_plots(x_label)

    mp.log_results()
    with open(cf.timeline_profile_path, 'w') as f:
        f.write(trainer.chrome_trace())

    tbw.flush()

ui.safe(cf.index_safe_path)
input("Press Enter to continue...")
