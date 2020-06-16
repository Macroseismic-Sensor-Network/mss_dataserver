'''
Created on May 17, 2011

@author: Stefan Mertl
'''
import unittest
import logging
import os

from obspy.core.utcdatetime import UTCDateTime

from mss_dataserver.event.core import Event
import mss_dataserver.event.detection as detection
import mss_dataserver.core.test_util as test_util


class EventTestCase(unittest.TestCase):
    """
    Test suite for psysmon.packages.geometry.editGeometry.EditGeometryDlg
    """
    @classmethod
    def setUpClass(cls):
        # Configure the logger.
        logger = logging.getLogger('mss_dataserver')
        logger.addHandler(logging.StreamHandler())
        logging.basicConfig(level = logging.INFO,
                            format = "LOG - %(asctime)s - %(process)d - %(levelname)s - %(name)s: %(message)s")

        cls.project = test_util.create_db_test_project()
        test_util.clear_project_database_tables(cls.project)
        #test_util.drop_project_database_tables(cls.project)
        #cls.project.create_database_tables()
        cls.project.load_inventory()

    @classmethod
    def tearDownClass(cls):
        #test_util.drop_project_database_tables(cls.project)
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

    def test_write_event_to_database(self):
        ''' Test the writing of an event to the database.
        '''
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T01:00:00'
        creation_time = UTCDateTime()
        event = Event(start_time = start_time,
                      end_time = end_time,
                      creation_time = creation_time)
        event.write_to_database(self.project)

        db_event_orm = self.project.db_tables['event']
        db_session = self.project.get_db_session()
        result = db_session.query(db_event_orm).\
            filter(db_event_orm.id == event.db_id).all()
        db_session.close()
        self.assertEqual(len(result), 1)
        tmp = result[0]
        self.assertEqual(tmp.start_time, event.start_time.timestamp)
        self.assertEqual(tmp.end_time, event.end_time.timestamp)
        self.assertEqual(tmp.creation_time, event.creation_time.isoformat())

    #@unittest.skip("temporary disabled")
    def test_add_detection_to_event(self):
        ''' Test the adding of detections.
        '''
        # Create a detection.
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T01:00:00'
        creation_time = UTCDateTime()
        det = detection.Detection(start_time = start_time,
                                  end_time = end_time,
                                  stat1_id = 1,
                                  stat2_id = 2,
                                  stat3_id = 3,
                                  max_pgv1 = 0.1,
                                  max_pgv2 = 0.2,
                                  max_pgv3 = 0.3,
                                  creation_time = creation_time)
        # Write the detection to the database. Only detections in a database
        # can be associated with the event in the database.
        det.write_to_database(self.project)

        # Check for a valid database id.
        self.assertEqual(det.db_id, 1)

        # Create an event.
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T02:00:00'
        creation_time = UTCDateTime()
        event = Event(start_time = start_time,
                      end_time = end_time,
                      creation_time = creation_time,
                      detections = [det, ])

        # Write the event to the database.
        event.write_to_database(self.project)

        # Now reload the event and check if the detections were linked
        # correctly with the event.
        db_event_orm = self.project.db_tables['event']
        try:
            db_session = self.project.get_db_session()
            result = db_session.query(db_event_orm).filter(db_event_orm.id == event.db_id).all()
            cur_event = Event.from_db_event(result[0])
            self.assertEqual(len(cur_event.detections), 1)
            self.assertEqual(cur_event.detections[0].start_time, det.start_time)
            self.assertEqual(cur_event.detections[0].end_time, det.end_time)
        finally:
            db_session.close()


def suite():
#    tests = ['testXmlImport']
#    return unittest.TestSuite(map(InventoryTestCase, tests))
    return unittest.makeSuite(EventTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

