import datetime
import glob
import os


class Config:
    """Contains static configurations"""

    def __init__(self, root_folder, continue_previous_run=True, output_folder=""):
        # DIRECTORIES:
        self.root_folder = root_folder
        self.path_train_data = self.root_folder + '/train'
        self.path_test_data = self.root_folder + '/test'
        self.path_interaction_map = self.root_folder + '/map'
        self.output_run_dir = os.getcwd() + output_folder + "/run_" + datetime.datetime.now().strftime(
            "%Y_%B_%d_%H:%M:%S")
        self.tb_dir = self.output_run_dir + "/tensorboard"
        self.index_safe_path = self.output_run_dir + "/interaction_indexing"
        self.timeline_profile_path = self.output_run_dir + "/timeline_profile/timeline.json"
        self.tb_command = "bash -c \"source /home/chambroc/miniconda3/bin/activate crystal && tensorboard --logdir="

        # PREVIOUS RUN DATA
        self.continnue_previous_run = continue_previous_run
        self.previous_successful_output_run_dir = None
        all_outputs = sorted(glob.glob(os.getcwd() + output_folder + "/*"), reverse=False)
        if len(all_outputs) > 0:
            for folder in all_outputs:
                if len(glob.glob(folder + "/_SUCCESS")) > 0:
                    self.previous_successful_output_run_dir = folder
        if self.continnue_previous_run and (self.previous_successful_output_run_dir is None):
            self.continnue_previous_run = False
            print("WARN: no successful previous run found!")
            input("Press Enter to continue with new random initialization...")

        # NETWORK SETUP:
        self.embedding_size = 3  # vector length of user interaction representation

        # TRAINING SETUP:
        self.epochs = 10  # number of epochs (epoch = whole train data processed) to train
        self.batch_size = 100000  # number of events processed in single step in tensorflow
        self.fake_frac = 0.7  # fraction of generated fake events for triplet loss
        self.bucket_count = 10  # buckets of the self-made event randomizer
        self.neighboring_interactions = 4  # distance of user_interactions still considered
        self.learning_rate = 0.001  # learning rate for tensorflow

        # EVALUATION SETUP:
        self.events_from_true_data = 100  # top events considered from true data when calculating weighted_pos_avg
        self.knn_plots = 200  # nearest neighbor search for tensorboard plots (weighted_pos_avg)
        self.result_cnt_plots = 50  # weighted random interactions considered for weighted_pos_avg when plotting
        self.knn_final = 200  # nearest neighbor search explicit logging  after entire training
        self.result_cnt_final = 200  # weighted random interactions considered for weighted_pos_avg for final log

    def make_dirs(self):
        os.mkdir(self.output_run_dir)

        os.mkdir(self.output_run_dir + "/tensorboard")
        os.mkdir(self.output_run_dir + "/tensorboard/train")
        os.mkdir(self.output_run_dir + "/tensorboard/test")
        os.mkdir(self.output_run_dir + "/tensorboard/log")

        os.mkdir(self.output_run_dir + "/timeline_profile")
        os.mkdir(self.output_run_dir + "/interaction_indexing")

    def to_string(self):
        ret_string = """
        Paths for in and output:
        
        root_folder: """ + self.root_folder + """
        path_train_data: """ + self.path_train_data + """
        path_test_data: """ + self.path_test_data + """
        path_interaction_map: """ + self.path_interaction_map + """
        
        output_run_dir: """ + self.output_run_dir + """
        tb_dir: """ + self.tb_dir + """
        index_safe_path: """ + self.index_safe_path + """
        timeline_profile_path: """ + self.timeline_profile_path + """
        
        Previous model dependence:
        
        previous_successful_output_run_dir: """ + str(self.previous_successful_output_run_dir) + """
        
        General parameters of the config:
        
        epochs: """ + str(self.epochs) + """
        embedding_size: """ + str(self.embedding_size) + """
        batch size: """ + str(self.batch_size) + """
        fake fraction: """ + str(self.fake_frac) + """
        bucket count: """ + str(self.bucket_count) + """
        learning rate: """ + str(self.learning_rate) + """
        neighboring_interactions: """ + str(self.neighboring_interactions) + """
        
        Evaluation setup:
        
        events_from_true_data: """ + str(self.events_from_true_data) + """
        knn_plots: """ + str(self.knn_plots) + """
        result_cnt_plots: """ + str(self.result_cnt_plots) + """
        knn_final: """ + str(self.knn_final) + """
        """
        return ret_string
