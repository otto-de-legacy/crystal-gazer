class MultiConfig:
    """Contains static configurations"""

    def __init__(self, source=[]):
        self.source_dirs = source  # can be several source paths for indices... also only one
        # self.interaction_vectors_urls = [dir + "/interaction_index.txt" for dir in self.source_dirs]
        # self.interaction_map_urls = [dir + "/map" for dir in self.source_dirs]

        # self.method = "hnsw"
        self.method = "ghtree"
        self.space = "cosinesimil"

    def to_string(self):
        ret_string = """
        General parameters of the config:
        
        """
        return ret_string
