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

from obspy.core.utcdatetime import UTCDateTime

import mss_dataserver
import mss_dataserver.event.core as ev_core
from mss_dataserver.event.core import Event
import mss_dataserver.event.detection as detection
import mss_dataserver.event.event_type as ev_type
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

        cls.project = test_util.create_test_project_no_db()

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

        
    def test_event_type(self):
        ''' Test the event type handling.
        '''
        name = "Testtype"
        description = "The description of the event type."
        author_uri = "test_author"
        agency_uri = "test_agency"
        creation_time = UTCDateTime()
        event_type = ev_type.EventType(name = name,
                                       description = description,
                                       author_uri = author_uri,
                                       agency_uri = agency_uri,
                                       creation_time = creation_time)
        self.assertIsInstance(event_type, ev_type.EventType)
        self.assertEqual(event_type.name, name)
        self.assertEqual(event_type.description, description)
        self.assertEqual(event_type.author_uri, author_uri)
        self.assertEqual(event_type.agency_uri, agency_uri)
        self.assertEqual(event_type.creation_time, creation_time)
        self.assertIsNone(event_type.db_id)
        self.assertIsNone(event_type.parent)
        
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T01:00:00'
        event = Event(start_time = start_time, end_time = end_time)
        event.set_event_type(event_type = event_type)
        self.assertEqual(event.event_type, event_type)

        # Add children to the event type.
        name = "Testtype"
        description = "The description of the event type."
        author_uri = "test_author"
        agency_uri = "test_agency"
        creation_time = UTCDateTime()
        root = ev_type.EventType(name = name,
                                 description = description,
                                 author_uri = author_uri,
                                 agency_uri = agency_uri,
                                 creation_time = creation_time)
        
        name = "Testtype Child 1"
        description = "The description of child 1."
        author_uri = "test_author_1"
        agency_uri = "test_agency_1"
        creation_time = UTCDateTime()
        child_1 = ev_type.EventType(name = name,
                                    description = description,
                                    author_uri = author_uri,
                                    agency_uri = agency_uri,
                                    creation_time = creation_time)
        root.add_child(child_1)

        name = "Testtype Child 2"
        description = "The description of child 2."
        author_uri = "test_author_2"
        agency_uri = "test_agency_2"
        creation_time = UTCDateTime()
        child_2 = ev_type.EventType(name = name,
                                    description = description,
                                    author_uri = author_uri,
                                    agency_uri = agency_uri,
                                    creation_time = creation_time)
        root.add_child(child_2)
        
        

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
                                  max_pgv = {stat1.nsl_string: 0.1,
                                             stat2.nsl_string: 0.2,
                                             stat3.nsl_string: 0.3})

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


def suite():
#    tests = ['testXmlImport']
#    return unittest.TestSuite(map(InventoryTestCase, tests))
    return unittest.makeSuite(EventTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

