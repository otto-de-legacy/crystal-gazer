from unittest import TestCase
from core.config import Config
import core.loader as ld
import core.url_mapper as um
import numpy as np
import tensorflow as tf

url_input = """1,2,3
        2,3,2,1,3
        2,5,2,4,3"""


class TestEvent(TestCase):
    def test_event(self):
        event1 = ld.Event("1", "2")
        event2 = ld.Event("1", "3")

        self.assertTrue(event1 == event1, msg='')
        self.assertFalse(event1 == event2, msg='')
