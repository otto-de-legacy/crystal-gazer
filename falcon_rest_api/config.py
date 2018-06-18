class Config:
    """Contains static configurations"""

    def __init__(self):
        self.source_dir = "/home/chambroc/github-projects/crystal-gazer/output/run_2018_June_18_09:52:32/interaction_indexing/"
        self.interaction_vectors_url = self.source_dir + "interaction_index.txt"
        self.interaction_map_url = self.source_dir + "interaction_map.txt"

        self.method = "hnsw"
        #self.method = "ghtree"
        self.space = "cosinesimil"

    def to_string(self):
        ret_string = """
        General parameters of the config:
        
        """
        return ret_string
