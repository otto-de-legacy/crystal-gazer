# recos.py

# Let's get this party started!
import os
import time

import pandas as pd
import falcon
import sys

print(os.getcwd())
sys.path.insert(0, os.getcwd() + '/../')

from core.interaction_index import InteractionIndex
from core.interaction_mapper import InteractionMapper
from falcon_rest_api.config import Config
from falcon_rest_api.ewma import EWMA

cf = Config()
im = InteractionMapper(map_path=cf.interaction_map_url)
ii = InteractionIndex(im, pd.read_csv(cf.interaction_vectors_url, header=None).values)
ewma = EWMA(100)


class RecoResource(object):
    def on_get(self, req, resp):
        t = time.time()

        resp.status = falcon.HTTP_200
        request_url = req.get_param('url')
        # k = int(req.get_param('k'))
        result = ii.knn_interaction_query(request_url, k=150)
        resp.media = {
            'url': request_url,
            'knn_urls': list(result[0]),
            # 'knn_distances': str(result[2]),
        }
        ewma.step(time.time() - t)


class MetricsResource(object):
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200
        dt_avg = ewma.values
        resp.media = {
            'average_last_100_request_duration_in_s': dt_avg
        }


class WhatWouldCarlSay(object):
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200
        dt_avg_in_ms = ewma.values * 1000
        if dt_avg_in_ms < 5:
            statement = "its ok"
        elif dt_avg_in_ms < 10:
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
app.add_route('/whatwouldcarlsay', WhatWouldCarlSay())
