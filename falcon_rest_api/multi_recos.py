# recos.py
import os
import sys
import time
import numpy as np

import falcon
import pandas as pd

print(os.getcwd())
sys.path.insert(0, os.getcwd() + '/../')

from core.interaction_index import InteractionIndex
from core.interaction_mapper import InteractionMapper
from falcon_rest_api.multi_config import MultiConfig
from falcon_rest_api.ewma import EWMA
from falcon_rest_api.cnt import CNT

cf = MultiConfig([
    "/home/chambroc/github-projects/crystal-gazer/output/run_2018_June_21_14:04:14/interaction_indexing",
    "/home/chambroc/github-projects/crystal-gazer/output/run_2018_June_21_14:04:22/interaction_indexing",
])
print("building maps and indices......")
iis = []
for dir in cf.source_dirs:
    print(dir)
    im = InteractionMapper(map_path=dir)
    print("...map ready")
    pd_df = pd.read_csv(dir + "/interaction_index.txt", header=None)
    for col in pd_df.columns:
        pd_df[col] = pd_df[col].astype(float)
    iis = iis + [InteractionIndex(im, pd_df.values, method=cf.method, space=cf.space)]
    print("...index ready")

ewma_dt = EWMA(100)
ewma_frac = EWMA(10000)
cnt = CNT()


class RecoResource(object):
    def on_get(self, req, resp):
        t = time.time()

        resp.status = falcon.HTTP_200
        request_url = req.get_param('url')
        k_val = int(req.get_param('k', default=10))
        multi_results = [ii.knn_interaction_query(request_url, k=k_val) for ii in iis]

        ret_dict = {'url': request_url,
                    'sources': cf.source_dirs}
        for idx, res in enumerate(multi_results):
            ret_dict['knn_urls_' + str(idx)] = list(res[0])
        resp.media = ret_dict

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


class JaccardResource(object):

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        request_url = req.get_param('url')
        k_val = int(req.get_param('k', default=10))

        multi_results = [ii.knn_interaction_query(request_url, k=k_val) for ii in iis]

        ret_dict = {'url': request_url,
                    'sources': cf.source_dirs}

        for i in range(len(multi_results)):
            for j in range(i + 1, len(multi_results)):
                a = set(multi_results[i][1])
                b = set(multi_results[j][1])
                union_len = len(a.union(a, b))
                intersec_len = len(a.intersection(b))

                ret_dict[str(i) + "vs" + str(j)] = 1 - intersec_len / union_len

            resp.media = ret_dict


class MetricsResource(object):
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200
        dt_avg = ewma_dt.values
        resp.media = {
            'average_last_100_request_duration_in_s': dt_avg,
            'total_calls': cnt.values,
            'perc_last_10000_below_50ms': float(ewma_frac(0)),
            'perc_last_10000_below_10ms': float(ewma_frac(1)),
            'perc_last_10000_below_5ms': float(ewma_frac(2)),
            'perc_last_10000_below_1ms': float(ewma_frac(3)),
            'perc_last_10000_below_0.5ms': float(ewma_frac(4))
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
app = falcon.API()

# Resources are represented by long-lived class instances
# will handle all requests to the URL path
app.add_route('/recos', RecoResource())
app.add_route('/metrics', MetricsResource())
app.add_route('/jaccard', JaccardResource())
app.add_route('/whatwouldcarlsay', WhatWouldCarlSay())