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

import unittest
import logging

import obspy

import mss_dataserver
import mss_dataserver.event.core as ev_core
import mss_dataserver.localize.origin as mssds_origin
import mss_dataserver.test.util as test_util


class OriginTestCase(unittest.TestCase):
    """
    Test suite for psysmon.packages.geometry.editGeometry.EditGeometryDlg
    """
    @classmethod
    def setUpClass(cls):
        # Configure the logger.
        cls.logger = logging.getLogger('mss_dataserver')
        handler = mss_dataserver.get_logger_handler(log_level = 'DEBUG')
        cls.logger.addHandler(handler)

        cls.project = test_util.create_db_test_project()
        test_util.clear_project_database_tables(cls.project)
        cls.project.load_inventory(update_from_xml = True)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_write_origin_to_database(self):
        ''' Test the adding of an origin to the database.
        '''
        origin_time = obspy.UTCDateTime('2022-01-01T00:00:00')
        author_uri = self.project.agency_uri
        agency_uri = self.project.author_uri
        creation_time = obspy.UTCDateTime()

        origin = mssds_origin.Origin(time = origin_time,
                                     x = 16.37,
                                     y = 48.21,
                                     z = 172,
                                     coord_system = 'epsg:4326',
                                     method = 'apollonius_circle',
                                     comment = 'Test comment',
                                     agency_uri = agency_uri,
                                     author_uri = author_uri,
                                     creation_time = creation_time)

        origin.write_to_database(project = self.project)

        # Load the origin from the database.
        db_orm = self.project.db_tables['origin']
        db_session = self.project.get_db_session()
        result = db_session.query(db_orm).\
            filter(db_orm.id == origin.db_id).all()
        db_session.close()
        self.assertEqual(len(result), 1)
        loaded_origin = mssds_origin.Origin.from_orm(result[0])
        self.assertIsInstance(loaded_origin,
                              mssds_origin.Origin)
        self.assertEqual(loaded_origin.db_id,
                         origin.db_id)
        self.assertEqual(loaded_origin.time,
                         origin_time)
        self.assertEqual(loaded_origin.x,
                         origin.x)
        self.assertEqual(loaded_origin.y,
                         origin.y)
        self.assertEqual(loaded_origin.z,
                         origin.z)
        self.assertEqual(loaded_origin.coord_system,
                         origin.coord_system)
        self.assertEqual(loaded_origin.method,
                         origin.method)
        self.assertEqual(loaded_origin.comment,
                         origin.comment)
        self.assertEqual(loaded_origin.agency_uri,
                         agency_uri)
        self.assertEqual(loaded_origin.author_uri,
                         author_uri)
        self.assertEqual(loaded_origin.creation_time,
                         creation_time)

        
    def test_add_origin_to_event(self):
        ''' Test adding an origin to an event.
        '''
        start_time = '2000-01-01T00:00:00'
        end_time = '2000-01-01T01:00:00'
        creation_time = obspy.UTCDateTime()
        event = ev_core.Event(start_time = start_time,
                              end_time = end_time,
                              creation_time = creation_time)

        origin_time = obspy.UTCDateTime('2022-01-01T00:00:00')
        author_uri = self.project.agency_uri
        agency_uri = self.project.author_uri
        creation_time = obspy.UTCDateTime()
        origin = mssds_origin.Origin(time = origin_time,
                                     x = 16.37,
                                     y = 48.21,
                                     z = 172,
                                     coord_system = 'epsg:4326',
                                     method = 'apollonius_circle',
                                     comment = 'Test comment',
                                     agency_uri = agency_uri,
                                     author_uri = author_uri,
                                     creation_time = creation_time)

        event.add_origin(origin)
        event.set_preferred_origin(origin)
        event.write_to_database(self.project)

        # Load the event from the database.
        db_orm = self.project.db_tables['event']
        db_session = self.project.get_db_session()
        result = db_session.query(db_orm).\
            filter(db_orm.id == event.db_id).all()
        self.assertEqual(len(result), 1)

        inventory = self.project.db_inventory
        loaded_event_orm = result[0]
        self.assertEqual(len(loaded_event_orm.origins),
                         1)
        self.assertEqual(loaded_event_orm.pref_origin_id,
                         origin.db_id)
        loaded_event = ev_core.Event.from_orm(loaded_event_orm,
                                              inventory = inventory)
        db_session.close()
        
        self.assertIsInstance(loaded_event,
                              ev_core.Event)
        self.assertEqual(len(loaded_event.origins), 1)
        loaded_origin = loaded_event.origins[0]
        self.assertIsInstance(loaded_origin,
                              mssds_origin.Origin)
        self.assertEqual(loaded_origin.db_id,
                         origin.db_id)
        self.assertEqual(loaded_origin.time,
                         origin_time)
        self.assertEqual(loaded_origin.x,
                         origin.x)
        self.assertEqual(loaded_origin.y,
                         origin.y)
        self.assertEqual(loaded_origin.z,
                         origin.z)
        self.assertEqual(loaded_origin.coord_system,
                         origin.coord_system)
        self.assertEqual(loaded_origin.method,
                         origin.method)
        self.assertEqual(loaded_origin.comment,
                         origin.comment)
        self.assertEqual(loaded_origin.agency_uri,
                         agency_uri)
        self.assertEqual(loaded_origin.author_uri,
                         author_uri)
        self.assertEqual(loaded_origin.creation_time,
                         creation_time)

        # Check the loaded preferred origin.
        self.assertIsInstance(loaded_event.preferred_origin,
                              mssds_origin.Origin)
        

def suite():
    return unittest.makeSuite(OriginTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

