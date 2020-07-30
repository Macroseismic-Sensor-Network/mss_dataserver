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
import mss_dataserver.event
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

    def create_detection_test_data(self,
                                   start_time = '2020-07-20T10:00:00',
                                   sps = 10,
                                   signal_length = 60,
                                   amp = 0.001e-3,
                                   event_start = 30,
                                   event_length = 2,
                                   event_amp = 0.5e-3):
        ''' Create a test data set for the detection algorithm.
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

        # Create the test PGV stream.
        traces = []
        n_samples = signal_length * sps
        starttime = obspy.UTCDateTime(start_time)
        event_delay = {'OBWA': 0,
                       'PFAF': 0.2,
                       'MUDO': 0.4,
                       'PODO': 1,
                       'SOLL': 1.5}
        for cur_station in stations:
            cur_data = np.random.random(n_samples) * amp
            cur_stats = {'network': cur_station.network,
                         'station': cur_station.name,
                         'location': '',
                         'channel': 'pgv',
                         'sampling_rate': sps,
                         'npts': len(cur_data),
                         'starttime': starttime}

            if cur_station.name in event_delay.keys():
                cur_event_start = event_start + event_delay[cur_station.name]
                cur_start_ind = int(cur_event_start * sps)
                cur_end_ind = int((cur_event_start + event_length) * sps)
                cur_data[cur_start_ind:cur_end_ind] = event_amp

            cur_trace = obspy.Trace(data = cur_data,
                                    header = cur_stats)
            traces.append(cur_trace)
        stream = obspy.Stream(traces)

        return {'all_stations': all_stations,
                'stations': stations,
                'stream': stream,
                'event_delay': event_delay}

    def test_detection_creation(self):
        ''' Test the pSysmon Event class.
        '''
        test_data = self.create_detection_test_data()
        all_stations = test_data['all_stations']

        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations)

        self.assertIsInstance(detector, delaunay_detection.DelaunayDetector)
        self.assertEqual(len(detector.network_stations), len(all_stations))
        self.assertEqual(detector.network_stations, all_stations)
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
        start_time = obspy.UTCDateTime('2020-07-20T10:00:00')
        signal_length = 60
        sps = 10
        test_data = self.create_detection_test_data(start_time = start_time,
                                                    signal_length = signal_length,
                                                    sps = sps)
        all_stations = test_data['all_stations']
        stations = test_data['stations']
        stream = test_data['stream']
        self.logger.info("Stream: %s", stream)

        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations)
        detector.prepare_detection_stream(stream = stream)

        self.assertEqual(len(detector.detect_stream), len(stations))
        self.assertEqual(len(detector.detect_stations), len(stations))
        self.assertEqual(detector.detect_stations, stations)
        for cur_trace in detector.detect_stream:
            self.assertEqual(cur_trace.stats.starttime, start_time)
            self.assertEqual(cur_trace.stats.endtime,
                             start_time + signal_length - detector.safety_time - 1 / sps)
            self.assertEqual(cur_trace.stats.npts,
                             signal_length * sps - detector.safety_time * sps)

    #@unittest.skip("temporary disabled")
    def test_compute_delaunay_triangulation(self):
        ''' Test the computation of the Delaunay triangles.
        '''
        test_data = self.create_detection_test_data()
        all_stations = test_data['all_stations']
        stations = test_data['stations']

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
        test_data = self.create_detection_test_data()
        all_stations = test_data['all_stations']
        stations = test_data['stations']

        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations)
        tri = detector.compute_delaunay_triangulation(stations)
        edge_length = detector.compute_edge_length(stations, tri)

        self.assertEqual(len(edge_length), 6)
        keys = []
        for cur_simp in tri.simplices:
            cur_stations = [stations[x] for x in cur_simp]
            keys.append(tuple(sorted([x.snl_string for x in cur_stations])))
        self.assertEqual(sorted(edge_length.keys()),
                         sorted(keys))

        # Check the removal of simplices using max_edge_length.
        detector.max_edge_length = 9000
        tri = detector.compute_delaunay_triangulation(stations)
        edge_length = detector.compute_edge_length(stations, tri)

        self.assertEqual(len(edge_length), 3)
        keys = []
        for cur_simp in tri.simplices:
            cur_stations = [stations[x] for x in cur_simp]
            keys.append(tuple(sorted([x.snl_string for x in cur_stations])))
        self.assertEqual(sorted(edge_length.keys()),
                         sorted(keys))
        max_edge_length = np.max(list(edge_length.values()))
        self.assertTrue(max_edge_length <= 9000)


    #@unittest.skip("temporary disabled")
    def test_compute_triangle_max_pv(self):
        ''' Test the computation of the max. PGV values.
        '''
        start_time = obspy.UTCDateTime('2020-07-20T10:00:00')
        signal_length = 60
        sps = 10
        event_start = 30
        event_length = 2
        test_data = self.create_detection_test_data(start_time = start_time,
                                                    signal_length = signal_length,
                                                    sps = sps,
                                                    event_start = event_start,
                                                    event_length = event_length)
        all_stations = test_data['all_stations']
        stations = test_data['stations']
        stream = test_data['stream']
        event_delay = test_data['event_delay']

        window_length = 10
        safety_time = 10
        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations,
                                                       window_length = window_length,
                                                       safety_time = safety_time,
                                                       p_vel = 3500)
        detector.init_detection_run(stream = stream)
        desired_triangle = [x for x in detector.tri.simplices if sorted(x) == [0, 1, 2]][0]
        time, pgv, simp_stations = detector.compute_triangle_max_pgv(desired_triangle)

        # Check the correct stations of the used triangle.
        simp_stat_snl = [x.snl for x in simp_stations]
        expected_stations = [stations[x] for x in desired_triangle]
        expected_snl = [x.snl for x in expected_stations]
        self.assertEqual(simp_stat_snl, expected_snl)

        # The PGV values for three stations should have been computed.
        self.assertEqual(pgv.shape[1], 3)

        # The length of the PGV array should equal the total length of the pgv
        # data stream minus the detectors maximum time window and the safety
        # time.
        expected_length = ((signal_length // window_length) * window_length) * sps - (detector.max_time_window + safety_time) * sps
        self.assertEqual(pgv.shape[0], expected_length)

        # Check the pgv start time.
        self.assertEqual(time[0], start_time + detector.max_time_window)

        # Check if the last detection end time has been set correctly.
        self.assertEqual(detector.last_detection_end,
                         start_time + detector.max_time_window + expected_length / sps - 1 / sps)

        # Check the correct time of the event pgv value.
        key = tuple(sorted([x.snl_string for x in simp_stations]))
        edge_length = detector.edge_length[key]
        time_window = np.ceil(edge_length / detector.p_vel)
        for k, cur_stat in enumerate(simp_stations):
            cur_delay = event_delay[cur_stat.name]
            cur_mask = pgv[:, k] >= 0.5e-3
            expected_start = start_time + event_start + cur_delay
            # The expected lenght of computed max PGV is the event length plus the computation
            # time window minus 2 samples.
            expected_end = start_time + event_start + cur_delay + event_length + time_window - 2 / sps
            self.assertEqual(time[cur_mask][0], expected_start)
            self.assertEqual(time[cur_mask][-1], expected_end)


    def test_compute_trigger_data(self):
        ''' Test the computation of the trigger data.
        '''
        start_time = obspy.UTCDateTime('2020-07-20T10:00:00')
        signal_length = 60
        sps = 10
        test_data = self.create_detection_test_data(start_time = start_time,
                                                    signal_length = signal_length,
                                                    sps = sps)
        all_stations = test_data['all_stations']
        stream = test_data['stream']

        window_length = 10
        safety_time = 10
        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations,
                                                       window_length = window_length,
                                                       safety_time = safety_time,
                                                       p_vel = 3500)
        detector.init_detection_run(stream = stream)
        detector.compute_trigger_data()

        self.assertEqual(len(detector.trigger_data), 2)
        expected_stations = []
        expected_stations.append(sorted(['OBWA', 'PFAF', 'MUDO']))
        expected_stations.append(sorted(['OBWA', 'PODO', 'SOLL']))
        available_stations = []
        for cur_data in detector.trigger_data:
            available_stations.append(sorted([x.name for x in cur_data['simp_stations']]))

        for cur_exp_stations in expected_stations:
            self.assertTrue(cur_exp_stations in available_stations)

    #@unittest.skip("temporary disabled")
    def test_check_for_event_trigger(self):
        ''' Test the computation of the trigger start- and endtimes.
        '''
        start_time = obspy.UTCDateTime('2020-07-20T10:00:00')
        signal_length = 60
        sps = 10
        event_start = 30
        event_length = 2
        test_data = self.create_detection_test_data(start_time = start_time,
                                                    signal_length = signal_length,
                                                    sps = sps,
                                                    event_start = event_start,
                                                    event_length = event_length)
        all_stations = test_data['all_stations']
        stream = test_data['stream']
        event_delay = test_data['event_delay']

        window_length = 10
        safety_time = 10
        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations,
                                                       window_length = window_length,
                                                       safety_time = safety_time,
                                                       p_vel = 3500)
        detector.init_detection_run(stream = stream)
        detector.compute_trigger_data()
        trigger_start, trigger_end = detector.check_for_event_trigger()

        expected_start = start_time + event_start + event_delay['MUDO']
        edge_length = detector.edge_length[('OBWA:MSSNet:00',
                                            'PODO:MSSNet:00',
                                            'SOLL:MSSNet:00')]
        time_window = np.ceil(edge_length / detector.p_vel)
        expected_end = start_time + event_start + event_length + time_window - 2 / sps
        self.assertEqual(trigger_start, expected_start)
        self.assertEqual(trigger_end, expected_end)

    def test_evaluate_event_trigger(self):
        ''' Test handling of event triggers.
        '''
        start_time = obspy.UTCDateTime('2020-07-20T10:00:00')
        signal_length = 120
        sps = 10
        event_start = 60
        event_length = 2
        test_data = self.create_detection_test_data(start_time = start_time,
                                                    signal_length = signal_length,
                                                    sps = sps,
                                                    event_start = event_start,
                                                    event_length = event_length)
        all_stations = test_data['all_stations']
        stream = test_data['stream']
        event_delay = test_data['event_delay']

        window_length = 10
        safety_time = 10
        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations,
                                                       window_length = window_length,
                                                       safety_time = safety_time,
                                                       p_vel = 3500)

        # Test with a stream containing no event. Use a sliced part of the test
        # data for initialization, otherwise, the whole data of the stream
        # would be used for detection.
        test_stream = stream.slice(starttime = start_time,
                                   endtime = start_time + 50)
        detector.init_detection_run(stream = test_stream)
        detector.compute_trigger_data()
        detector.evaluate_event_trigger()

        self.assertFalse(detector.event_triggered)
        self.assertIsNone(detector.current_event)
        stream_start = np.min([x.stats.starttime for x in test_stream])
        stream_end = np.max([x.stats.endtime for x in test_stream])
        stream_length = stream_end - stream_start
        expected_length = ((stream_length // window_length) * window_length) * sps - (detector.max_time_window + safety_time) * sps
        self.assertEqual(detector.last_detection_end,
                         stream_start + detector.max_time_window + expected_length / sps - 1 / sps)

        # Test with the next data chunk containing no event.
        test_stream = stream.slice(starttime = start_time,
                                   endtime = start_time + 60)
        detector.init_detection_run(stream = test_stream)
        detector.compute_trigger_data()
        detector.evaluate_event_trigger()

        self.assertIsNone(detector.current_event)
        self.assertFalse(detector.event_triggered)
        self.assertFalse(detector.new_event_available)
        stream_start = np.min([x.stats.starttime for x in test_stream])
        stream_end = np.max([x.stats.endtime for x in test_stream])
        stream_length = stream_end - stream_start
        expected_length = ((stream_length // window_length) * window_length) * sps - (detector.max_time_window + safety_time) * sps
        self.assertEqual(detector.last_detection_end,
                         stream_start + detector.max_time_window + expected_length / sps - 1 / sps)

        # Test with the next data chunk containing an event, but the event is
        # within the safety window, so it should not be detected.
        test_stream = stream.slice(starttime = start_time,
                                   endtime = start_time + 70)
        detector.init_detection_run(stream = test_stream)
        detector.compute_trigger_data()
        detector.evaluate_event_trigger()

        self.assertFalse(detector.event_triggered)
        self.assertFalse(detector.new_event_available)
        self.assertIsNone(detector.current_event)
        stream_start = np.min([x.stats.starttime for x in test_stream])
        stream_end = np.max([x.stats.endtime for x in test_stream])
        stream_length = stream_end - stream_start
        expected_length = ((stream_length // window_length) * window_length) * sps - (detector.max_time_window + safety_time) * sps
        self.assertEqual(detector.last_detection_end,
                         stream_start + detector.max_time_window + expected_length / sps - 1 / sps)

        # Test with the next data chunk containing an event.
        test_stream = stream.slice(starttime = start_time,
                                   endtime = start_time + 80)
        detector.init_detection_run(stream = test_stream)
        detector.compute_trigger_data()
        detector.evaluate_event_trigger()

        self.assertTrue(detector.event_triggered)
        self.assertTrue(detector.new_event_available)
        self.assertIsNotNone(detector.current_event)
        self.assertEqual(detector.current_event.detection_state, 'new')
        self.assertEqual(len(detector.current_event.detections), 2)
        stream_start = np.min([x.stats.starttime for x in test_stream])
        stream_end = np.max([x.stats.endtime for x in test_stream])
        stream_length = stream_end - stream_start
        expected_length = ((stream_length // window_length) * window_length) * sps - (detector.max_time_window + safety_time) * sps
        self.assertEqual(detector.last_detection_end,
                         stream_start + detector.max_time_window + expected_length / sps - 1 / sps)

        expected_start = start_time + event_start + event_delay['MUDO']
        edge_length = detector.edge_length[('OBWA:MSSNet:00',
                                            'PODO:MSSNet:00',
                                            'SOLL:MSSNet:00')]
        time_window = np.ceil(edge_length / detector.p_vel)
        expected_end = start_time + event_start + event_length + time_window - 2 / sps
        self.assertEqual(detector.current_event.start_time, expected_start)
        self.assertEqual(detector.current_event.end_time, expected_end)

        # Test with the next data chunk containing no event. The current event
        # should be closed.
        test_stream = stream.slice(starttime = start_time,
                                   endtime = start_time + 90)
        detector.init_detection_run(stream = test_stream)
        detector.compute_trigger_data()
        detector.evaluate_event_trigger()

        self.assertFalse(detector.event_triggered)
        self.assertTrue(detector.new_event_available)
        self.assertEqual(detector.current_event.detection_state, 'closed')
        stream_start = np.min([x.stats.starttime for x in test_stream])
        stream_end = np.max([x.stats.endtime for x in test_stream])
        stream_length = stream_end - stream_start
        expected_length = ((stream_length // window_length) * window_length) * sps - (detector.max_time_window + safety_time) * sps
        self.assertEqual(detector.last_detection_end,
                         stream_start + detector.max_time_window + expected_length / sps - 1 / sps)

        # Get the event.
        event = detector.get_event()
        self.assertFalse(detector.new_event_available)
        self.assertIsInstance(event, mss_dataserver.event.core.Event)

    def test_evaluate_event_trigger_long_event(self):
        ''' Test handling of event triggers.
        '''
        start_time = obspy.UTCDateTime('2020-07-20T10:00:00')
        signal_length = 160
        sps = 10
        event_start = 65
        event_length = 10
        test_data = self.create_detection_test_data(start_time = start_time,
                                                    signal_length = signal_length,
                                                    sps = sps,
                                                    event_start = event_start,
                                                    event_length = event_length)
        all_stations = test_data['all_stations']
        stream = test_data['stream']
        event_delay = test_data['event_delay']

        window_length = 10
        safety_time = 10
        detector = delaunay_detection.DelaunayDetector(network_stations = all_stations,
                                                       window_length = window_length,
                                                       safety_time = safety_time,
                                                       p_vel = 3500)

        # Test with a stream containing no event. Use a sliced part of the test
        # data for initialization, otherwise, the whole data of the stream
        # would be used for detection.
        test_stream = stream.slice(starttime = start_time,
                                   endtime = start_time + 70)
        detector.init_detection_run(stream = test_stream)
        detector.compute_trigger_data()
        detector.evaluate_event_trigger()

        self.assertFalse(detector.event_triggered)
        self.assertFalse(detector.new_event_available)
        self.assertIsNone(detector.current_event)
        stream_start = np.min([x.stats.starttime for x in test_stream])
        stream_end = np.max([x.stats.endtime for x in test_stream])
        stream_length = stream_end - stream_start
        expected_length = ((stream_length // window_length) * window_length) * sps - (detector.max_time_window + safety_time) * sps
        self.assertEqual(detector.last_detection_end,
                         stream_start + detector.max_time_window + expected_length / sps - 1 / sps)


        # Test with the next data chunk containing an event.
        test_stream = stream.slice(starttime = start_time,
                                   endtime = start_time + 80)
        detector.init_detection_run(stream = test_stream)
        detector.compute_trigger_data()
        detector.evaluate_event_trigger()

        self.assertTrue(detector.event_triggered)
        self.assertTrue(detector.new_event_available)
        self.assertIsNotNone(detector.current_event)
        self.assertEqual(detector.current_event.detection_state, 'new')
        self.assertEqual(len(detector.current_event.detections), 2)
        stream_start = np.min([x.stats.starttime for x in test_stream])
        stream_end = np.max([x.stats.endtime for x in test_stream])
        stream_length = stream_end - stream_start
        expected_length = ((stream_length // window_length) * window_length) * sps - (detector.max_time_window + safety_time) * sps
        self.assertEqual(detector.last_detection_end,
                         stream_start + detector.max_time_window + expected_length / sps - 1 / sps)

        expected_start = start_time + event_start + event_delay['MUDO']
        expected_end = start_time + 70 - 1 / sps
        self.assertEqual(detector.current_event.start_time, expected_start)
        self.assertEqual(detector.current_event.end_time, expected_end)

        # Test with the next data chunk containing the still active event.
        test_stream = stream.slice(starttime = start_time,
                                   endtime = start_time + 90)
        detector.init_detection_run(stream = test_stream)
        detector.compute_trigger_data()
        detector.evaluate_event_trigger()

        self.assertTrue(detector.event_triggered)
        self.assertTrue(detector.new_event_available)
        self.assertIsNotNone(detector.current_event)
        self.assertEqual(detector.current_event.detection_state, 'updated')
        self.assertEqual(len(detector.current_event.detections), 2)
        stream_start = np.min([x.stats.starttime for x in test_stream])
        stream_end = np.max([x.stats.endtime for x in test_stream])
        stream_length = stream_end - stream_start
        expected_length = ((stream_length // window_length) * window_length) * sps - (detector.max_time_window + safety_time) * sps
        self.assertEqual(detector.last_detection_end,
                         stream_start + detector.max_time_window + expected_length / sps - 1 / sps)

        # Test with the next data chunk containing no event. The current event
        # should be closed.
        test_stream = stream.slice(starttime = start_time,
                                   endtime = start_time + 100)
        detector.init_detection_run(stream = test_stream)
        detector.compute_trigger_data()
        detector.evaluate_event_trigger()

        self.assertFalse(detector.event_triggered)
        self.assertTrue(detector.new_event_available)
        self.assertIsNotNone(detector.current_event)
        self.assertEqual(detector.current_event.detection_state, 'closed')
        stream_start = np.min([x.stats.starttime for x in test_stream])
        stream_end = np.max([x.stats.endtime for x in test_stream])
        stream_length = stream_end - stream_start
        expected_length = ((stream_length // window_length) * window_length) * sps - (detector.max_time_window + safety_time) * sps
        self.assertEqual(detector.last_detection_end,
                         stream_start + detector.max_time_window + expected_length / sps - 1 / sps)

        # Get the event.
        event = detector.get_event()
        self.assertFalse(detector.new_event_available)
        self.assertIsInstance(event, mss_dataserver.event.core.Event)

def suite():
    return unittest.makeSuite(DelaunayDetectorTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
