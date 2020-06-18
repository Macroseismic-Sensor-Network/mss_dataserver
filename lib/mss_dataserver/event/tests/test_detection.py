'''
Created on May 17, 2011

@author: Stefan Mertl
'''
import unittest
import logging
import os

from obspy.core.utcdatetime import UTCDateTime
import sqlalchemy.orm

import mss_dataserver.event.detection as detection
import mss_dataserver.core.test_util as test_util


class DetectionTestCase(unittest.TestCase):
    """
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
        cls.project.load_inventory()

    @classmethod
    def tearDownClass(cls):
        #test_util.drop_project_database_tables(cls.project)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_detection_creation(self):
        ''' Test the pSysmon Event class.
        '''
        # Test the control of None values of the time limits.
        self.assertRaises(ValueError, detection.Detection,
                          start_time = None,
                          end_time = None,
                          stations = [],
                          max_pgv = [])
        self.assertRaises(ValueError, detection.Detection,
                          start_time = '2000-01-01',
                          end_time = None,
                          stations = [],
                          max_pgv = [])
        self.assertRaises(ValueError, detection.Detection,
                          start_time = None,
                          end_time = '2000-01-01',
                          stations = [],
                          max_pgv = [])

        # Test the control of the time limits.
        self.assertRaises(ValueError, detection.Detection,
                          start_time = '2000-01-01',
                          end_time = '1999-01-01',
                          stations = [],
                          max_pgv = [])
        self.assertRaises(ValueError, detection.Detection,
                          start_time = '2000-01-01',
                          end_time = '2000-01-01',
                          stations = [],
                          max_pgv = [])

        # Create a detection.
        # Set the date values.
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T01:00:00'
        creation_time = UTCDateTime()

        # Get the stations from the inventory.
        inventory = self.project.db_inventory
        stat1 = inventory.get_station(name = 'DUBA')[0]
        stat2 = inventory.get_station(name = 'WADU')[0]
        stat3 = inventory.get_station(name = 'WAPE')[0]

        det = detection.Detection(start_time = start_time,
                                  end_time = end_time,
                                  creation_time = creation_time,
                                  stations = [stat1, stat2, stat3],
                                  max_pgv = [0.1, 0.2, 0.3])
        self.assertIsInstance(det, detection.Detection)
        self.assertEqual(det.start_time, UTCDateTime(start_time))
        self.assertEqual(det.end_time, UTCDateTime(end_time))

    def test_write_detection_to_database(self):
        ''' Test the writing to the database.
        '''
        # Create a detection.
        # Set the date values.
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T01:00:00'
        creation_time = UTCDateTime()

        # Get the stations from the inventory.
        inventory = self.project.db_inventory
        stat1 = inventory.get_station(name = 'DUBA')[0]
        stat2 = inventory.get_station(name = 'WADU')[0]
        stat3 = inventory.get_station(name = 'WAPE')[0]

        det = detection.Detection(start_time = start_time,
                                  end_time = end_time,
                                  creation_time = creation_time,
                                  stations = [stat1, stat2, stat3],
                                  max_pgv = [0.1, 0.2, 0.3])
        det.write_to_database(self.project)

        detection_orm = self.project.db_tables['detection']
        db_session = self.project.get_db_session()
        try:
            result = db_session.query(detection_orm).\
                options(sqlalchemy.orm.subqueryload(detection_orm.stat1)).\
                options(sqlalchemy.orm.subqueryload(detection_orm.stat2)).\
                options(sqlalchemy.orm.subqueryload(detection_orm.stat3)).\
                filter(detection_orm.id == det.db_id).all()
        finally:
            db_session.close()

        self.assertEqual(len(result), 1)
        tmp = result[0]
        self.assertEqual(tmp.start_time, UTCDateTime(start_time).timestamp)
        self.assertEqual(tmp.end_time, UTCDateTime(end_time).timestamp)
        self.assertEqual(tmp.creation_time, creation_time.isoformat())
        self.assertEqual(tmp.stat1_id, stat1.id)
        self.assertEqual(tmp.stat2_id, stat2.id)
        self.assertEqual(tmp.stat3_id, stat3.id)
        self.assertEqual(tmp.stat1.id, stat1.id)
        self.assertEqual(tmp.stat2.id, stat2.id)
        self.assertEqual(tmp.stat3.id, stat3.id)


def suite():
    return unittest.makeSuite(DetectionTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

