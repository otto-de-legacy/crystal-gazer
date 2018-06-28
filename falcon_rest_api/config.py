class Config:
    """Contains static configurations"""

    def __init__(self, source):
        self.source_dir = source  # can be several source paths for indices... also only one
        # self.interaction_vectors_url = self.source_dirs + "interaction_index"
        # self.interaction_map_url = self.source_dirs + "map"

        self.method = "hnsw"
        # self.method = "ghtree"
        self.space = "cosinesimil"

    def to_string(self):
        ret_string = """
        General parameters of the config:
        
        """
        return ret_string
