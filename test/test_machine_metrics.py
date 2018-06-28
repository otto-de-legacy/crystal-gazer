from collections import namedtuple
from unittest import TestCase
import falcon_rest_api.machine_metrics as met

class TestMetricsResource(TestCase):

    def test_memory_usage_psutil(self):
        mem = met.memory_usage_psutil()
        print("mem usage psutil [MB]: " + str(mem))
        self.assertTrue(mem > 0)

    def test_memory_usage_resource(self):
        mem = met.memory_usage_resource()
        print("mem usage resource [?]: " + str(mem))
        self.assertTrue(mem > 0)

    def test_memory_usage_ps(self):
        mem = met.memory_usage_ps()
        print("mem usage ps [?]: " + str(mem))
        self.assertTrue(mem > 0)

    def test_memory_usage_ps(self):
        cpu_perc = met.cpu_usage_percent()
        print("cpu usage [%]: " + str(cpu_perc))
        self.assertTrue(cpu_perc >= 0)
