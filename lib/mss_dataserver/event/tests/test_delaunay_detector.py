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
import mss_dataserver.event.delaunay_detection as delaunay_detection
import mss_dataserver.test.util as test_util


class DelaunayDetectorTestCase(unittest.TestCase):
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
        # test_util.drop_project_database_tables(cls.project)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_detection_creation(self):
        ''' Test the pSysmon Event class.
        '''
        detector = delaunay_detection.DelaunayDetector()

        self.assertIsInstance(detector, delaunay_detection.DelaunayDetector)
        self.assertIsNone(detector.current_event)
        self.assertEqual(detector.min_trigger_window, 2)
        self.assertEqual(len(detector.stations), 0)
        self.assertIsNone(detector.tri)
        self.assertEqual(len(detector.edge_length), 0)
        self.assertTrue(detector.max_edge_length > 0)

    def test_compute_delaunay_triangulation(self):
        ''' Test the computation of the Delaunay triangles.
        '''
        # Get the stations from the inventory.
        inventory = self.project.db_inventory
        inventory.compute_utm_coordinates()
        stations = []
        stations.append(inventory.get_station(name = 'OBWA')[0])
        stations.append(inventory.get_station(name = 'PFAF')[0])
        stations.append(inventory.get_station(name = 'MUDO')[0])
        stations.append(inventory.get_station(name = 'EBDO')[0])
        stations.append(inventory.get_station(name = 'PODO')[0])
        stations.append(inventory.get_station(name = 'SOLL')[0])
        stations.append(inventory.get_station(name = 'BAVO')[0])

        expected_triangles = [['OBWA', 'PFAF', 'MUDO'],
                              ['OBWA', 'MUDO', 'EBDO'],
                              ['OBWA', 'EBDO', 'PODO'],
                              ['OBWA', 'PODO', 'SOLL'],
                              ['OBWA', 'SOLL', 'BAVO'],
                              ['OBWA', 'BAVO', 'PFAF']]
        expected_triangles = [sorted(x) for x in expected_triangles]

        detector = delaunay_detection.DelaunayDetector()
        detector.set_stations(stations)
        detector.compute_delaunay_triangulation()

        self.assertIsNotNone(detector.tri)
        self.assertEqual(len(detector.tri.simplices), 6)

        for cur_simp in detector.tri.simplices:
            cur_stations = [stations[int(x)] for x in cur_simp]
            cur_stat_names = sorted([x.name for x in cur_stations])
            self.assertTrue(cur_stat_names in expected_triangles)

    def test_compute_edge_length(self):
        ''' Test the computation of the triangle edge lengths.
        '''
        inventory = self.project.db_inventory
        inventory.compute_utm_coordinates()
        stations = []
        stations.append(inventory.get_station(name = 'OBWA')[0])
        stations.append(inventory.get_station(name = 'PFAF')[0])
        stations.append(inventory.get_station(name = 'MUDO')[0])
        stations.append(inventory.get_station(name = 'EBDO')[0])
        stations.append(inventory.get_station(name = 'PODO')[0])
        stations.append(inventory.get_station(name = 'SOLL')[0])
        stations.append(inventory.get_station(name = 'BAVO')[0])

        detector = delaunay_detection.DelaunayDetector()
        detector.set_stations(stations)
        detector.compute_delaunay_triangulation()
        detector.compute_edge_length()

        self.assertEqual(len(detector.edge_length), 6)
        self.logger.info("edge_length: %s", detector.edge_length)


def suite():
    return unittest.makeSuite(DelaunayDetectorTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
