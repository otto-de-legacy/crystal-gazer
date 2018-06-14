import datetime
import os


class Config:
    """Contains static configurations"""

    def __init__(self, short=True):
        self.short = short

        if short:
            append = "_small"
        else:
            append = ""

        self.embedding_size = 100

        self.epochs = 20  # number of epochs (epoch = whole train data processed) to train

        self.batch_size = 100000
        self.fake_frac = 0.7
        self.bucket_count = 10

        self.learning_rate = 0.001

        self.url_train_data = './resources/train_data' + append
        self.url_test_data = './resources/test_data' + append

        self.url_unique = './resources/unique_urls31mb'
        self.url_map = './resources/url_map' + append

        self.tb_command = "bash -c \"source /home/chambroc/miniconda3/bin/activate crystal && tensorboard --logdir="

        self.print_top_queries_cnt = 10
        self.result_cnt_plots = 50
        self.result_cnt_final = 200
        self.events_from_true_data = 100
        self.knn_plots = 200
        self.knn_final = 200

        self.run_dir = os.getcwd() + "/.././output/run_" + datetime.datetime.now().strftime("%Y_%I_%B_%H:%M:%S")
        self.tb_dir = self.run_dir + "/tensorboard"
        self.index_safe_path = self.run_dir + "/url_indexing/url_index.txt"
        self.timeline_profile_path = self.run_dir + "/timeline_profile/timeline.json"
        self.make_dirs()

    def make_dirs(self):
        os.mkdir(self.run_dir)

        os.mkdir(self.run_dir + "/tensorboard")
        os.mkdir(self.run_dir + "/tensorboard/train")
        os.mkdir(self.run_dir + "/tensorboard/test")
        os.mkdir(self.run_dir + "/tensorboard/log")

        os.mkdir(self.run_dir + "/timeline_profile")
        os.mkdir(self.run_dir + "/url_indexing")

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
