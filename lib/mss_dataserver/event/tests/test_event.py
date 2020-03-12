'''
Created on May 17, 2011

@author: Stefan Mertl
'''
import unittest
import logging
import os

from obspy.core.utcdatetime import UTCDateTime

from mss_dataserver.event.core import Event
import mss_dataserver.core.test_util as test_util


class EventTestCase(unittest.TestCase):
    """
    Test suite for psysmon.packages.geometry.editGeometry.EditGeometryDlg
    """
    @classmethod
    def setUpClass(cls):
        # Configure the logger.
        logger = logging.getLogger('psysmon')
        logger.setLevel('INFO')

        cls.project = test_util.create_db_test_project()

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_event_creation(self):
        ''' Test the pSysmon Event class.
        '''
        # Test the control of None values of the time limits.
        self.assertRaises(ValueError, Event,
                          start_time = None,
                          end_time = None)
        self.assertRaises(ValueError, Event,
                          start_time = '2000-01-01',
                          end_time = None)
        self.assertRaises(ValueError, Event,
                          start_time = None,
                          end_time = '2000-01-01')

        # Test the control of the time limits.
        self.assertRaises(ValueError, Event,
                          start_time = '2000-01-01',
                          end_time = '1999-01-01')
        self.assertRaises(ValueError, Event,
                          start_time = '2000-01-01',
                          end_time = '2000-01-01')

        # Create an event with valid time limits.
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T01:00:00'
        event = Event(start_time = start_time, end_time = end_time)
        self.assertIsInstance(event, Event)
        self.assertEqual(event.start_time, UTCDateTime(start_time))
        self.assertEqual(event.end_time, UTCDateTime(end_time))
        self.assertTrue(event.changed)


def suite():
#    tests = ['testXmlImport']
#    return unittest.TestSuite(map(InventoryTestCase, tests))
    return unittest.makeSuite(EventTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

