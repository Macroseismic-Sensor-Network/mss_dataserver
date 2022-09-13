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
import mss_dataserver.localize.magnitude as mssds_mag
import mss_dataserver.localize.origin as mssds_origin
import mss_dataserver.test.util as test_util


class MagnitudeTestCase(unittest.TestCase):
    """
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

    def test_write_magnitude_to_database(self):
        ''' Test the adding of a magnitude to the database.
        '''
        author_uri = self.project.agency_uri
        agency_uri = self.project.author_uri
        creation_time = obspy.UTCDateTime()

        mag = mssds_mag.Magnitude(mag = 5.4,
                                  mag_type = 'mb',
                                  comment = 'Test comment',
                                  agency_uri = agency_uri,
                                  author_uri = author_uri,
                                  creation_time = creation_time)

        mag.write_to_database(project = self.project)

        # Load the magnitude from the database.
        db_orm = self.project.db_tables['magnitude']
        db_session = self.project.get_db_session()
        result = db_session.query(db_orm).\
            filter(db_orm.id == mag.db_id).all()
        db_session.close()

        # Check the loaded data.
        self.assertEqual(len(result), 1)
        loaded_mag = mssds_mag.Magnitude.from_orm(result[0])
        self.assertIsInstance(loaded_mag,
                              mssds_mag.Magnitude)
        self.assertEqual(loaded_mag.db_id,
                         mag.db_id)
        self.assertEqual(loaded_mag.mag,
                         mag.mag)
        self.assertEqual(loaded_mag.mag_type,
                         mag.mag_type)
        self.assertEqual(loaded_mag.comment,
                         mag.comment)
        self.assertEqual(loaded_mag.agency_uri,
                         agency_uri)
        self.assertEqual(loaded_mag.author_uri,
                         author_uri)
        self.assertEqual(loaded_mag.creation_time,
                         creation_time)


    def test_add_magnitude_to_origin(self):
        ''' Test adding a magnitude to an origin.
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

        mag = mssds_mag.Magnitude(mag = 5.4,
                                  mag_type = 'mb',
                                  comment = 'Test comment',
                                  agency_uri = agency_uri,
                                  author_uri = author_uri,
                                  creation_time = creation_time)

        origin.add_magnitude(mag = mag)
        origin.set_preferred_magnitude(mag = mag)
        origin.write_to_database(project = self.project)

        # Load the origin from the dtabase.
        db_orm = self.project.db_tables['origin']
        db_session = self.project.get_db_session()
        result = db_session.query(db_orm).\
            filter(db_orm.id == origin.db_id).all()
        self.assertEqual(len(result), 1)
        loaded_origin_orm = result[0]
        self.assertEqual(len(loaded_origin_orm.magnitudes),
                         1)
        self.assertEqual(loaded_origin_orm.pref_magnitude_id,
                         mag.db_id)
        
        # Create an origin instance.
        loaded_origin = mssds_origin.Origin.from_orm(loaded_origin_orm)
        db_session.close()

        # Check the loaded instances.
        self.assertIsInstance(loaded_origin,
                              mssds_origin.Origin)
        self.assertEqual(len(loaded_origin.magnitudes),
                         1)

        # Check the origin magnitudes.
        loaded_mag = loaded_origin.magnitudes[0]
        self.assertIsInstance(loaded_mag,
                              mssds_mag.Magnitude)
        self.assertEqual(loaded_mag.mag,
                         mag.mag)
        self.assertEqual(loaded_mag.mag_type,
                         mag.mag_type)
        self.assertEqual(loaded_mag.comment,
                         mag.comment)
        self.assertEqual(loaded_mag.agency_uri,
                         mag.agency_uri)
        self.assertEqual(loaded_mag.author_uri,
                         mag.author_uri)
        self.assertEqual(loaded_mag.creation_time,
                         mag.creation_time)

        # Check the preferred magnitude.
        pref_mag = loaded_origin.pref_magnitude
        self.assertEqual(pref_mag.db_id,
                         mag.db_id)

        
        

def suite():
    return unittest.makeSuite(MagnitudeTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

