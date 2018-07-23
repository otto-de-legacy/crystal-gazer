# reco.py
import os
import random
import sys
import time
import numpy as np

import falcon
import pandas as pd

print(os.getcwd())
sys.path.insert(0, os.getcwd() + '/../')

from core.interaction_index import InteractionIndex
from core.interaction_mapper import InteractionMapper
from falcon_rest_api.config import Config
from falcon_rest_api.ewma import EWMA
from falcon_rest_api.cnt import CNT
from core.conditional_index import ConditionalIndex
import falcon_rest_api.machine_metrics as met

cf = Config("/home/chambroc/Desktop/RecoResults/ThreeInARowMoreEvents/day3/interaction_indexing")
cf.method = "hnsw"
print("loading data...")
pd_df = pd.read_csv(cf.source_dir + "/interaction_index.txt", header=None)
for col in pd_df.columns:
    pd_df[col] = pd_df[col].astype(float)
print("building conditional index...")

filter_funs = [
    lambda key: True if "suche" in key else False,
    lambda key: True if "/p/" in key else False,
    lambda key: False if ("suche" in key) or ("/p/" in key) else True,
]

multi_index = ConditionalIndex(
    InteractionMapper(map_path=cf.source_dir),
    pd_df.values,
    lambdas_of_key=filter_funs,
    method=cf.method,
    space=cf.space)
# print("building index full...")
# main_index = InteractionIndex(im, pd_df.values, method=cf.method, space=cf.space)

print("...index ready")
ewma_dt = EWMA(100)
ewma_frac = EWMA(10000)
cnt = CNT()
num_classes = multi_index.im.interaction_class_cnt

class RecoResource(object):
    def on_get(self, req, resp):
        t = time.time()
        resp.status = falcon.HTTP_200
        request_url = req.get_param('url', default=None)
        if request_url is None:
            random_url_idx = random.randint(0, num_classes)
            request_url = multi_index.im.num_to_interaction(random_url_idx)
        k = int(req.get_param('k', default=100))
        url_result = multi_index.knn_interaction_query(request_url, k=k)[0]
        search_result = multi_index.knn_interaction_query(request_url, 0, k=k)[0]
        product_result = multi_index.knn_interaction_query(request_url, 1, k=k)[0]
        other_result = multi_index.knn_interaction_query(request_url, 2, k=k)[0]
        resp.media = {
            'url': request_url,
            'knn_urls': list(url_result),
            'search_result': list(search_result),
            'product': list(product_result),
            'other': list(other_result)
        }
        dt = time.time() - t
        if dt > 0.05:
            bucket = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif dt > 0.01:
            bucket = np.array([100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif dt > 0.005:
            bucket = np.array([100.0, 100.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif dt > 0.001:
            bucket = np.array([100.0, 100.0, 100.0, 0.0, 0.0], dtype=np.float32)
        elif dt > 0.0005:
            bucket = np.array([100.0, 100.0, 100.0, 100.0, 0.0], dtype=np.float32)
        else:
            bucket = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        ewma_dt.step(dt)
        ewma_frac.step(bucket)
        cnt.step(1)


class MetricsResource(object):
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200
        dt_avg = ewma_dt.values


        dt_avg_in_s = ewma_dt.values
        perc_dt_50 = float(ewma_frac(0))
        perc_dt_10 = float(ewma_frac(1))
        if dt_avg_in_s is None:
            statement = "no traffic means, it's fast enough"
        elif dt_avg_in_s * 1000 < 5 and perc_dt_50 > 99.99 and perc_dt_10 > 99.8:
            statement = "excellent job"
        elif dt_avg_in_s * 1000 < 10 and perc_dt_50 > 99.8:
            statement = "you need to drill deeper \n - thats what she said"
        else:
            statement = "your mama is faster than this"

        resp.media = {
            'num_classes': num_classes,
            'embedding_matrix_size': (multi_index.tot_object_cnt, multi_index.vector_size),
            'space': cf.space,
            'method': cf.method,
            'average_last_100_request_duration_in_s': dt_avg,
            'total_calls': cnt.values,
            'mem_usage_mb': met.memory_usage_psutil(),
            'cpu_usage_perc': met.cpu_usage_percent(),
            'perc_last_10000_below_50ms': float(ewma_frac(0)),
            'perc_last_10000_below_10ms':  float(ewma_frac(1)),
            'perc_last_10000_below_5ms':  float(ewma_frac(2)),
            'perc_last_10000_below_1ms':   float(ewma_frac(3)),
            'perc_last_10000_below_0.5ms':   float(ewma_frac(4)),
            'what_would_carl_say':   statement
        }


class WhatWouldCarlSay(object):
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200
        dt_avg_in_s = ewma_dt.values
        if dt_avg_in_s is None:
            statement = "no traffic means, it's fast enough"
        elif dt_avg_in_s * 1000 < 5:
            statement = "this is fine"
        elif dt_avg_in_s * 1000 < 10:
            statement = "you can do better"
        else:
            statement = "your mama is faster than this"
        resp.media = {
            'statement': statement
        }


# falcon.API instances are callable WSGI apps
application = falcon.API()

# Resources are represented by long-lived class instances
# will handle all requests to the URL path
application.add_route('/reco', RecoResource())
application.add_route('/metrics', MetricsResource())
application.add_route('/whatwouldcarlsay', WhatWouldCarlSay())
