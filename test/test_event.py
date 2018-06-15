from unittest import TestCase

import core.loader as ld


class TestEvent(TestCase):
    def test_event(self):
        event1 = ld.Event("1", "2")
        event2 = ld.Event("1", "3")

        self.assertTrue(event1 == event1, msg='equality comparer')
        self.assertFalse(event1 == event2, msg='inequality comparer')
