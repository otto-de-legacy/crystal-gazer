import datetime
import os


class Config:
    """Contains static configurations"""

    def __init__(self, short=True):
        self.short = short

        if short:
            data_folder = "./resources"
        else:
            data_folder = "./resources_full"

        self.embedding_size = 100  # vector length of user interaction representation

        self.epochs = 50  # number of epochs (epoch = whole train data processed) to train

        self.batch_size = 100000  # number of events processed in single step in tensorflow
        self.fake_frac = 0.7  # fraction of generated fake events for triplet loss
        self.bucket_count = 10  # buckets of the self-made event randomizer
        self.neighboring_interactions = 4  # distance of user_interactions still considered

        self.learning_rate = 0.001  # learning rate for tensorflow

        self.url_train_data = data_folder + '/train_data'
        self.url_test_data = data_folder + '/test_data'

        self.url_interaction_map = data_folder + '/interaction_map'

        self.tb_command = "bash -c \"source /home/chambroc/miniconda3/bin/activate crystal && tensorboard --logdir="

        # self.print_top_queries_cnt = 10  # for logging
        self.events_from_true_data = 100  # top events considered from true data when calculating weighted_pos_avg
        self.knn_plots = 200  # nearest neighbor search for tensorboard plots (weighted_pos_avg)
        self.result_cnt_plots = 50  # weighted random interactions considered for weighted_pos_avg when plotting
        self.knn_final = 200  # nearest neighbor search explicit logging  after entire training
        self.result_cnt_final = 200  # weighted random interactions considered for weighted_pos_avg for final log

        self.run_dir = os.getcwd() + "/.././output/run_" + datetime.datetime.now().strftime("%Y_%B_%d_%H:%M:%S")
        self.tb_dir = self.run_dir + "/tensorboard"
        self.index_safe_path = self.run_dir + "/interaction_indexing/"
        self.timeline_profile_path = self.run_dir + "/timeline_profile/timeline.json"

    def make_dirs(self):
        os.mkdir(self.run_dir)

        os.mkdir(self.run_dir + "/tensorboard")
        os.mkdir(self.run_dir + "/tensorboard/train")
        os.mkdir(self.run_dir + "/tensorboard/test")
        os.mkdir(self.run_dir + "/tensorboard/log")

        os.mkdir(self.run_dir + "/timeline_profile")
        os.mkdir(self.run_dir + "/interaction_indexing")

    def to_string(self):
        ret_string = """
        General parameters of the config:
        
        fast run through: """ + str(self.short) + """
        epochs: """ + str(self.epochs) + """
        batch size: """ + str(self.batch_size) + """
        fake fraction: """ + str(self.fake_frac) + """
        bucket count: """ + str(self.bucket_count) + """
        learning rate: """ + str(self.learning_rate) + """
        """
        return ret_string
