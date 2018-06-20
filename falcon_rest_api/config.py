class Config:
    """Contains static configurations"""

    def __init__(self):
        self.source_dir = "/home/chambroc/Desktop/run_2018_June_18_17:40:38/interaction_indexing/"
        self.interaction_vectors_url = self.source_dir + "interaction_index"
        self.interaction_map_url = self.source_dir + "map"

        self.method = "hnsw"
        #self.method = "ghtree"
        self.space = "cosinesimil"

    def to_string(self):
        ret_string = """
        General parameters of the config:
        
        """
        return ret_string
