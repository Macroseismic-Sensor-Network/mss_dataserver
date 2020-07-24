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

import numpy as np
import obspy
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
        inventory = self.project.db_inventory
        inventory.compute_utm_coordinates()
        stations = inventory.get_station()
        detector = delaunay_detection.DelaunayDetector(network_stations = stations)

        self.assertIsInstance(detector, delaunay_detection.DelaunayDetector)
        self.assertEqual(len(detector.network_stations), len(stations))
        self.assertEqual(detector.network_stations, stations)
        self.assertIsNone(detector.current_event)
        self.assertEqual(detector.min_trigger_window, 2)
        self.assertEqual(len(detector.detect_stations), 0)
        self.assertIsNone(detector.tri)
        self.assertEqual(len(detector.edge_length), 0)
        self.assertTrue(detector.max_edge_length > 0)
        self.assertTrue(detector.max_time_window > 0)

    def test_prepare_detection_stream(self):
        ''' Test the computation of the delaunay triangle max. pgv values.
        '''
        inventory = self.project.db_inventory
        inventory.compute_utm_coordinates()
        all_stations = inventory.get_station()
        stations = []
        stations.append(inventory.get_station(name = 'OBWA')[0])
        stations.append(inventory.get_station(name = 'PFAF')[0])
        stations.append(inventory.get_station(name = 'MUDO')[0])
        stations.append(inventory.get_station(name = 'EBDO')[0])
        stations.append(inventory.get_station(name = 'PODO')[0])
        stations.append(inventory.get_station(name = 'SOLL')[0])
        stations.append(inventory.get_station(name = 'BAVO')[0])

        # Create the test seismogram stream.
        sps = 100
        signal_length = 120
        traces = []
        starttime = obspy.UTCDateTime('2020-07-20T10:00:00')
        for cur_station in stations:
            cur_data = np.random.random(signal_length * sps)
            cur_stats = {'network': cur_station.network,
                         'station': cur_station.name,
                         'location': '',
                         'channel': 'pgv',
                         'sampling_rate': sps,
                         'npts': len(cur_data),
                         'starttime': starttime}
            cur_trace = obspy.Trace(data = cur_data,
                                    header = cur_stats)
            traces.append(cur_trace)
        stream = obspy.Stream(traces)
        self.logger.info("Stream: %s", stream)

        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations)
        detector.prepare_detection_stream(stream = stream)

        self.assertEqual(len(detector.detect_stream), len(stations))
        self.assertEqual(len(detector.detect_stations), len(stations))
        self.assertEqual(detector.detect_stations, stations)
        for cur_trace in detector.detect_stream:
            self.assertEqual(cur_trace.stats.starttime, starttime)
            self.assertEqual(cur_trace.stats.endtime,
                             starttime + signal_length - detector.window_length - detector.safety_time - 1 / sps)
            self.assertEqual(cur_trace.stats.npts,
                             signal_length * sps - (detector.window_length + detector.safety_time) * sps)

    #@unittest.skip("temporary disabled")
    def test_compute_delaunay_triangulation(self):
        ''' Test the computation of the Delaunay triangles.
        '''
        # Get the stations from the inventory.
        inventory = self.project.db_inventory
        inventory.compute_utm_coordinates()
        all_stations = inventory.get_station()
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

        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations)
        tri = detector.compute_delaunay_triangulation(stations)

        self.assertIsNotNone(tri)
        self.assertEqual(len(tri.simplices), 6)

        for cur_simp in tri.simplices:
            cur_stations = [stations[int(x)] for x in cur_simp]
            cur_stat_names = sorted([x.name for x in cur_stations])
            self.assertTrue(cur_stat_names in expected_triangles)

    #@unittest.skip("temporary disabled")
    def test_compute_edge_length(self):
        ''' Test the computation of the triangle edge lengths.
        '''
        inventory = self.project.db_inventory
        inventory.compute_utm_coordinates()
        all_stations = inventory.get_station()
        stations = []
        stations.append(inventory.get_station(name = 'OBWA')[0])
        stations.append(inventory.get_station(name = 'PFAF')[0])
        stations.append(inventory.get_station(name = 'MUDO')[0])
        stations.append(inventory.get_station(name = 'EBDO')[0])
        stations.append(inventory.get_station(name = 'PODO')[0])
        stations.append(inventory.get_station(name = 'SOLL')[0])
        stations.append(inventory.get_station(name = 'BAVO')[0])

        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations)
        tri = detector.compute_delaunay_triangulation(stations)
        edge_length = detector.compute_edge_length(stations, tri)

        self.assertEqual(len(edge_length), 6)
        self.assertEqual(sorted(edge_length.keys()),
                         sorted([tuple(x.tolist()) for x in tri.simplices]))
        self.logger.info("edge_length: %s", edge_length)


    @unittest.skip("temporary disabled")
    def test_compute_max_pv(self):
        ''' Test the computation of the max. PGV values.
        '''
        inventory = self.project.db_inventory
        inventory.compute_utm_coordinates()
        all_stations = inventory.get_station()

        stations = []
        stations.append(inventory.get_station(name = 'OBWA')[0])
        stations.append(inventory.get_station(name = 'PFAF')[0])
        stations.append(inventory.get_station(name = 'MUDO')[0])
        stations.append(inventory.get_station(name = 'EBDO')[0])
        stations.append(inventory.get_station(name = 'PODO')[0])
        stations.append(inventory.get_station(name = 'SOLL')[0])
        stations.append(inventory.get_station(name = 'BAVO')[0])

        # Create the test seismogram stream.
        sps = 100
        signal_length = 120
        traces = []
        starttime = obspy.UTCDateTime('2020-07-20T10:00:00')
        for cur_station in stations:
            cur_data = np.random.random(signal_length * sps)
            cur_stats = {'network': cur_station.network,
                         'station': cur_station.name,
                         'location': '',
                         'channel': 'pgv',
                         'sampling_rate': sps,
                         'npts': len(cur_data),
                         'starttime': starttime}
            cur_trace = obspy.Trace(data = cur_data,
                                    header = cur_stats)
            traces.append(cur_trace)
        stream = obspy.Stream(traces)

        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations)
        detector.init_detection_run(stream = stream)
        time, pgv = detector.compute_max_pgv()

        self.logger.info("time: %s", time)
        self.logger.info("pgv: %s", pgv)

        self.assertEqual(pgv.shape[1], len(stations))


def suite():
    return unittest.makeSuite(DelaunayDetectorTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
