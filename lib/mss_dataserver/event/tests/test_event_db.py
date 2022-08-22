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
        

    def test_write_event_type_to_database(self):
        ''' Test the event type handling.
        '''
        # The event types in the database are build as a tree structure.
        # 
        # Add the event type root.
        name = "Root"
        description = "The description of the root event type."
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

        # Add a child to the child_2 event type.
        name = "Testtype Child 2 - Child 1"
        description = "The description of child 1 of child 2."
        author_uri = "test_author_3"
        agency_uri = "test_agency_3"
        creation_time = UTCDateTime()
        child_2_1 = ev_type.EventType(name = name,
                                      description = description,
                                      author_uri = author_uri,
                                      agency_uri = agency_uri,
                                      creation_time = creation_time)
        child_2.add_child(child_2_1)

        root.write_to_database(self.project)

        # Test for the database ids.
        self.assertIsNotNone(root.db_id)
        self.assertIsNotNone(child_1.db_id)
        self.assertIsNotNone(child_2.db_id)
        self.assertIsNotNone(child_2_1.db_id)
        self.assertEqual(child_1.parent.db_id,
                         root.db_id)
        self.assertEqual(child_2.parent.db_id,
                         root.db_id)
        self.assertEqual(child_2_1.parent.db_id,
                         root.event_types[1].db_id)
        
        children = [child_1, child_2]

        # Load the event type tree from the database.
        roots = ev_type.EventType.load_from_db(project = self.project)
        self.assertEqual(len(roots), 1)
        self.assertIsInstance(roots[0], ev_type.EventType)
        self.assertEqual(len(roots[0].event_types), 2)
        for k, cur_child in enumerate(roots[0].event_types):
            test_child = children[k]
            self.assertIsInstance(cur_child, ev_type.EventType)
            self.assertEqual(cur_child.parent,
                             roots[0])
            self.assertEqual(cur_child.name,
                             test_child.name)
            self.assertEqual(cur_child.description,
                             test_child.description)

        # Test for the children of child 2.
        test_child = roots[0].event_types[1]
        self.assertEqual(len(test_child.event_types), 1)
        self.assertEqual(test_child.event_types[0].name,
                         child_2_1.name)
        self.assertEqual(test_child.event_types[0].parent,
                         test_child)


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

        
    def test_add_event_type_to_event(self):
        ''' Test the classification of an event.
        '''
        # Build the event types tree.
        name = "Root"
        description = "The description of the root event type."
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
        root.write_to_database(self.project)

        # Create an event.
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T01:00:00'
        creation_time = UTCDateTime()
        event = Event(start_time = start_time,
                      end_time = end_time,
                      creation_time = creation_time)
        event.set_event_type(child_1)
        event.write_to_database(self.project)

        # Load the event from database.
        db_event_orm = self.project.db_tables['event']
        db_session = self.project.get_db_session()
        result = db_session.query(db_event_orm).\
            filter(db_event_orm.id == event.db_id).all()
        self.assertEqual(len(result), 1)
        tmp = result[0]
        loaded_event = Event.from_orm(db_event = tmp,
                                      inventory = self.project.inventory)
        db_session.close()
        self.assertIsInstance(loaded_event.event_type,
                              ev_type.EventType)
        self.assertEqual(loaded_event.event_type.name,
                         child_1.name)
        

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
                                  max_pgv = {stat1.nsl_string: 0.1,
                                             stat2.nsl_string: 0.2,
                                             stat3.nsl_string: 0.3})
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
            self.assertEqual(cur_event.detections[0].stations[0].nsl,
                             stat1.nsl)
            self.assertEqual(cur_event.detections[0].stations[1].nsl,
                             stat2.nsl)
            self.assertEqual(cur_event.detections[0].stations[2].nsl,
                             stat3.nsl)
            self.assertEqual(cur_event.max_pgv, 0.3)
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
                                   max_pgv = {stat_11.nsl_string: 0.1,
                                              stat_12.nsl_string: 0.2,
                                              stat_13.nsl_string: 0.3})
        det1.write_to_database(self.project)

        stat_21 = inventory.get_station(name = 'HOWA')[0]
        stat_22 = inventory.get_station(name = 'WEIK')[0]
        stat_23 = inventory.get_station(name = 'BAFI')[0]

        det2 = detection.Detection(start_time = start_time,
                                   end_time = end_time,
                                   creation_time = creation_time,
                                   stations = [stat_21, stat_22, stat_23],
                                   max_pgv = {stat_21.nsl_string: 0.11,
                                              stat_22.nsl_string: 0.22,
                                              stat_23.nsl_string: 0.33})
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
            self.assertEqual(cur_det.stations[0].nsl,
                             stat_11.nsl)
            self.assertEqual(cur_det.stations[1].nsl,
                             stat_12.nsl)
            self.assertEqual(cur_det.stations[2].nsl,
                             stat_13.nsl)

            cur_det = cur_event.detections[1]
            self.assertEqual(cur_det.start_time, det1.start_time)
            self.assertEqual(cur_det.end_time, det1.end_time)
            self.assertEqual(len(cur_det.stations), 3)
            self.assertEqual(cur_det.stations[0].nsl,
                             stat_21.nsl)
            self.assertEqual(cur_det.stations[1].nsl,
                             stat_22.nsl)
            self.assertEqual(cur_det.stations[2].nsl,
                             stat_23.nsl)

            self.assertEqual(cur_event.max_pgv, 0.33)
        finally:
            db_session.close()


    def test_event_library(self):
        ''' Test event library and event catalogs.
        '''
        catalogs = self.project.get_event_catalog_names()
        self.assertEqual(catalogs, [])

        cat_1 = self.project.create_event_catalog(name = '2020-08-13',
                                                  description = 'test description 1')
        catalogs = self.project.get_event_catalog_names()
        self.assertEqual(catalogs, [cat_1.name])
        self.assertEqual(len(self.project.event_library.catalogs), 1)

        cat = self.project.get_event_catalog(name = cat_1.name)
        self.assertEqual(cat.name, cat_1.name)
        self.assertEqual(cat.description, cat_1.description)

        cat = self.project.load_event_catalog(name = cat_1.name)
        self.assertIsNotNone(cat)
        self.assertEqual(cat.name, cat_1.name)
        self.assertEqual(cat.description, cat_1.description)

        # Create a second catalog.
        cat_2 = self.project.create_event_catalog(name = '2020-08-14',
                                                  description = 'test description 2')
        catalogs = self.project.get_event_catalog_names()
        self.assertEqual(catalogs, [cat_1.name, cat_2.name])

        # Add events to the catalogs.
        cat = self.project.get_event_catalog(name = '2020-08-13')
        self.assertIsNotNone(cat)
        self.assertEqual(cat.name, cat_1.name)
        self.assertEqual(cat.description, cat_1.description)

        start_time = UTCDateTime('2020-08-13T00:00:00')
        n_events = 10
        event_list_1 = []
        for k in range(n_events):
            cur_start_time = start_time + k * 3600
            cur_end_time = cur_start_time + 10
            cur_creation_time = UTCDateTime()
            cur_event = Event(start_time = cur_start_time,
                              end_time = cur_end_time,
                              creation_time = cur_creation_time)
            event_list_1.append(cur_event)
        cat.add_events(event_list_1)
        self.assertEqual(len(cat.events), 10)

        cat = self.project.get_event_catalog(name = '2020-08-14')
        self.assertIsNotNone(cat)
        self.assertEqual(cat.name, cat_2.name)
        self.assertEqual(cat.description, cat_2.description)

        start_time = UTCDateTime('2020-08-14T00:00:00')
        n_events = 10
        event_list_2 = []
        for k in range(n_events):
            cur_start_time = start_time + k * 3600
            cur_end_time = cur_start_time + 10
            cur_creation_time = UTCDateTime()
            cur_event = Event(start_time = cur_start_time,
                              end_time = cur_end_time,
                              creation_time = cur_creation_time)
            event_list_2.append(cur_event)
        cat.add_events(event_list_2)
        self.assertEqual(len(cat.events), 10)

        # Get all events in the library.
        events = self.project.get_events()
        self.assertEqual(len(events), 20)

        # Get events for a limited timespan.
        request_start = UTCDateTime('2020-08-13')
        request_end = request_start + 86400
        events = self.project.get_events(start_time = request_start,
                                         end_time = request_end)
        self.assertEqual(len(events), len(event_list_1))
        for k in range(len(events)):
            self.assertEqual(events[k].start_time,
                             event_list_1[k].start_time)

        # Get events for a limited timespan.
        request_start = UTCDateTime('2020-08-14')
        request_end = request_start + 86400
        events = self.project.get_events(start_time = request_start,
                                         end_time = request_end)
        self.assertEqual(len(events), len(event_list_2))
        for k in range(len(events)):
            self.assertEqual(events[k].start_time,
                             event_list_2[k].start_time)

        # Get events for a catalog.
        events = self.project.get_events(catalog_names = ['2020-08-13'])
        self.assertEqual(len(events), len(event_list_1))
        for k in range(len(events)):
            self.assertEqual(events[k].start_time,
                             event_list_1[k].start_time)

    def test_detection_library(self):
        ''' Test detection library and detection catalogs.
        '''
        catalogs = self.project.get_detection_catalog_names()
        self.assertEqual(catalogs, [])

        cat = self.project.create_detection_catalog(name = 'test name',
                                                    description = 'test description')

        catalogs = self.project.get_detection_catalog_names()
        self.assertEqual(catalogs, ['test name'])

        cat = self.project.load_detection_catalog(name = 'test name')
        self.assertIsNotNone(cat)
        self.assertEqual(cat.name, 'test name')
        self.assertEqual(cat.description, 'test description')




def suite():
#    tests = ['testXmlImport']
#    return unittest.TestSuite(map(InventoryTestCase, tests))
    return unittest.makeSuite(EventTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

