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

import mss_dataserver
from mss_dataserver.event.core import Event
import mss_dataserver.event.detection as detection
import mss_dataserver.test.util as test_util


class EventTestCase(unittest.TestCase):
    """
    Test suite for psysmon.packages.geometry.editGeometry.EditGeometryDlg
    """
    @classmethod
    def setUpClass(cls):
        # Configure the logger.
        cls.logger = logging.getLogger('mss_dataserver')
        cls.logger.addHandler(mss_dataserver.get_logger_handler(log_level = 'DEBUG'))

        cls.project = test_util.create_db_test_project()
        test_util.clear_project_database_tables(cls.project)
        cls.project.load_inventory(update_from_xml = True)

    @classmethod
    def tearDownClass(cls):
        #test_util.drop_project_database_tables(cls.project)
        pass

    def setUp(self):
        #test_util.clear_project_database_tables(self.project,
        #                                        tables = ['event',
        #                                                  'detection'])
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

    def test_update_detection(self):
        ''' The the updating of existing detections.
        '''
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
                                  max_pgv = {stat1.snl_string: 0.1,
                                             stat2.snl_string: 0.2,
                                             stat3.snl_string: 0.3})

        # Create an event.
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T02:00:00'
        creation_time = UTCDateTime()
        event = Event(start_time = start_time,
                      end_time = end_time,
                      creation_time = creation_time,
                      detections = [det, ])

        # Check for existing detections.
        res = event.has_detection([stat1, stat2, stat3])
        self.assertEqual(res, True)
        res = event.has_detection([stat2, stat1, stat3])
        self.assertEqual(res, True)
        res = event.has_detection([stat1, stat1, stat3])
        self.assertEqual(res, False)

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
                                  max_pgv = {stat1.snl_string: 0.1,
                                             stat2.snl_string: 0.2,
                                             stat3.snl_string: 0.3})
        # Write the detection to the database. Only detections in a database
        # can be associated with the event in the database.
        det.write_to_database(self.project)

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
            cur_event = Event.from_orm(result[0],
                                       inventory = inventory)
            self.assertEqual(len(cur_event.detections), 1)
            self.assertEqual(cur_event.detections[0].start_time, det.start_time)
            self.assertEqual(cur_event.detections[0].end_time, det.end_time)
            self.assertEqual(len(cur_event.detections[0].stations), 3)
            self.assertEqual(cur_event.detections[0].stations[0].snl,
                             stat1.snl)
            self.assertEqual(cur_event.detections[0].stations[1].snl,
                             stat2.snl)
            self.assertEqual(cur_event.detections[0].stations[2].snl,
                             stat3.snl)
        finally:
            db_session.close()

    def test_add_multiple_detections_to_event(self):
        ''' Test the adding of multiple detections.
        '''
        # Create a detection.
        # Set the date values.
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T01:00:00'
        creation_time = UTCDateTime()

        # Get the stations from the inventory.
        inventory = self.project.db_inventory
        stat_11 = inventory.get_station(name = 'DUBA')[0]
        stat_12 = inventory.get_station(name = 'WADU')[0]
        stat_13 = inventory.get_station(name = 'WAPE')[0]

        det1 = detection.Detection(start_time = start_time,
                                   end_time = end_time,
                                   creation_time = creation_time,
                                   stations = [stat_11, stat_12, stat_13],
                                   max_pgv = {stat_11.snl_string: 0.1,
                                              stat_12.snl_string: 0.2,
                                              stat_13.snl_string: 0.3})
        det1.write_to_database(self.project)

        stat_21 = inventory.get_station(name = 'HOWA')[0]
        stat_22 = inventory.get_station(name = 'WEIK')[0]
        stat_23 = inventory.get_station(name = 'BAFI')[0]

        det2 = detection.Detection(start_time = start_time,
                                   end_time = end_time,
                                   creation_time = creation_time,
                                   stations = [stat_21, stat_22, stat_23],
                                   max_pgv = {stat_21.snl_string: 0.11,
                                              stat_22.snl_string: 0.22,
                                              stat_23.snl_string: 0.33})
        det2.write_to_database(self.project)

        # Create an event.
        start_time = '2000-02-01T00:00:00'
        end_time = '2000-02-01T02:00:00'
        creation_time = UTCDateTime()
        event = Event(start_time = start_time,
                      end_time = end_time,
                      creation_time = creation_time,
                      detections = [det1, det2])

        # Write the event to the database.
        event.write_to_database(self.project)

        # Now reload the event and check if the detections were linked
        # correctly with the event.
        db_event_orm = self.project.db_tables['event']
        try:
            db_session = self.project.get_db_session()
            result = db_session.query(db_event_orm).filter(db_event_orm.id == event.db_id).all()
            cur_event = Event.from_orm(result[0],
                                       inventory = inventory)
            self.assertEqual(len(cur_event.detections), 2)

            cur_det = cur_event.detections[0]
            self.assertEqual(cur_det.start_time, det1.start_time)
            self.assertEqual(cur_det.end_time, det1.end_time)
            self.assertEqual(len(cur_det.stations), 3)
            self.assertEqual(cur_det.stations[0].snl,
                             stat_11.snl)
            self.assertEqual(cur_det.stations[1].snl,
                             stat_12.snl)
            self.assertEqual(cur_det.stations[2].snl,
                             stat_13.snl)

            cur_det = cur_event.detections[1]
            self.assertEqual(cur_det.start_time, det1.start_time)
            self.assertEqual(cur_det.end_time, det1.end_time)
            self.assertEqual(len(cur_det.stations), 3)
            self.assertEqual(cur_det.stations[0].snl,
                             stat_21.snl)
            self.assertEqual(cur_det.stations[1].snl,
                             stat_22.snl)
            self.assertEqual(cur_det.stations[2].snl,
                             stat_23.snl)
        finally:
            db_session.close()

def suite():
#    tests = ['testXmlImport']
#    return unittest.TestSuite(map(InventoryTestCase, tests))
    return unittest.makeSuite(EventTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

