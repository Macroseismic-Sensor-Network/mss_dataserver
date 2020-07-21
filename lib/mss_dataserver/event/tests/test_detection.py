# -*- coding: utf-8 -*-
##############################################################################
 # LICENSE
 #
 # This file is part of mss_dataserver.
 # 
 # If you use mss_dataserver in any program or publication, please inform and
 # acknowledge its authors.
 # 
 # mss_dataserver is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 # 
 # mss_dataserver is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 # 
 # You should have received a copy of the GNU General Public License
 # along with mss_dataserver. If not, see <http://www.gnu.org/licenses/>.
 #
 # Copyright 2019 Stefan Mertl
##############################################################################

'''
Created on May 17, 2011

@author: Stefan Mertl
'''
import unittest
import logging
import os

from obspy.core.utcdatetime import UTCDateTime
import sqlalchemy.orm

import mss_dataserver
import mss_dataserver.event.detection as detection
import mss_dataserver.test.util as test_util


class DetectionTestCase(unittest.TestCase):
    """
    """
    @classmethod
    def setUpClass(cls):
        # Configure the logger.
        # Configure the logger.
        cls.logger = logging.getLogger('mss_dataserver')
        cls.logger.addHandler(mss_dataserver.get_logger_handler(log_level = 'DEBUG'))

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
                                  max_pgv = {stat1.snl: 0.1,
                                             stat2.snl: 0.2,
                                             stat3.snl: 0.3})
        self.assertIsInstance(det, detection.Detection)
        self.assertEqual(det.start_time, UTCDateTime(start_time))
        self.assertEqual(det.end_time, UTCDateTime(end_time))
        self.assertIsInstance(det.max_pgv, dict)
        self.assertEqual(det.max_pgv[stat1.snl], 0.1)
        self.assertEqual(det.max_pgv[stat2.snl], 0.2)
        self.assertEqual(det.max_pgv[stat3.snl], 0.3)


    def test_update_detection(self):
        ''' Test the updating of a detection.
        '''
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
                                  max_pgv = {stat1.snl: 0.1,
                                             stat2.snl: 0.2,
                                             stat3.snl: 0.3})

        new_start_time = '2020-01-01T00:00:00'
        new_end_time = '2020-01-01T02:00:00'
        det.update(start_time = new_start_time,
                   end_time = new_end_time)

        self.assertEqual(det.start_time, UTCDateTime(new_start_time))
        self.assertEqual(det.end_time, UTCDateTime(new_end_time))

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
                                  max_pgv = {stat1.snl: 0.1,
                                             stat2.snl: 0.2,
                                             stat3.snl: 0.3})
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

