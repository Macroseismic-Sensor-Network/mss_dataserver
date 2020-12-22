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
@author: Stefan Mertl
'''
import copy
import unittest
import logging
import threading

import obspy.core
from obspy.core.utcdatetime import UTCDateTime

import mss_dataserver
from mss_dataserver.event.core import Event
import mss_dataserver.monitorclient.monitorclient as monitorclient
import mss_dataserver.test.util as test_util
from mss_dataserver.event.core import Event
import mss_dataserver.event.detection as detection


class MonitorClientTestCase(unittest.TestCase):
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

        cls.create_events(cls)

        host = cls.project.config['seedlink']['host']
        port = cls.project.config['seedlink']['port']
        stations = cls.project.config['process']['stations']
        data_dir = cls.project.config['output']['data_dir']
        event_dir = cls.project.config['output']['event_dir']
        process_interval = cls.project.config['process']['interval']
        pgv_sps = cls.project.config['process']['pgv_sps']
        pgv_archive_time = cls.project.config['process']['pgv_archive_time']
        trigger_thr = cls.project.config['process']['trigger_threshold']
        warn_thr = cls.project.config['process']['warn_threshold']
        valid_event_thr = cls.project.config['process']['valid_event_threshold']
        event_archive_size = cls.project.config['process']['event_archive_size']

        if len(stations) == 0:
            stations = None

        server_url = host + ':' + str(port)

        monitor_stream = obspy.core.Stream()
        stream_lock = threading.Lock()
        stop_event = None
        loop = None

        cls.client = monitorclient.MonitorClient(project = cls.project,
                                                 asyncio_loop = loop,
                                                 server_url = server_url,
                                                 stations = stations,
                                                 monitor_stream = monitor_stream,
                                                 stream_lock = stream_lock,
                                                 data_dir = data_dir,
                                                 event_dir = event_dir,
                                                 process_interval = process_interval,
                                                 pgv_sps = pgv_sps,
                                                 stop_event = stop_event,
                                                 pgv_archive_time = pgv_archive_time,
                                                 trigger_thr = trigger_thr,
                                                 warn_thr = warn_thr,
                                                 valid_event_thr = valid_event_thr,
                                                 event_archive_size = event_archive_size)

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


    def create_events(self):
        ''' Create a set of test events in the database.
        '''
        # Get the stations from the inventory.
        inventory = self.project.db_inventory
        stat1 = inventory.get_station(name = 'DUBA')[0]
        stat2 = inventory.get_station(name = 'WADU')[0]
        stat3 = inventory.get_station(name = 'WAPE')[0]

        start_time = UTCDateTime()
        start_time.hour = 0
        start_time.minute = 0
        start_time.second = 0
        start_time.microsecond = 0

        cat = self.project.create_event_catalog(name = "{0:4d}-{1:02d}-{2:02d}".format(start_time.year,
                                                                                       start_time.month,
                                                                                       start_time.day),
                                                description = 'Catalog of the current day.')

        n_events = 10
        event_list_1 = []
        for k in range(n_events):
            cur_start_time = start_time + k * 3600
            cur_end_time = cur_start_time + 10
            cur_creation_time = UTCDateTime()

            det = detection.Detection(start_time = cur_start_time,
                                      end_time = cur_end_time,
                                      creation_time = cur_creation_time,
                                      stations = [stat1, stat2, stat3],
                                      max_pgv = {stat1.snl_string: 0.1 + 0.01 * k,
                                                 stat2.snl_string: 0.2 + 0.01 * k,
                                                 stat3.snl_string: 0.3 + 0.01 * k})

            # Write the detection to the database. Only detections in a
            # database can be associated with the event in the database.
            det.write_to_database(self.project)

            cur_event = Event(start_time = cur_start_time,
                              end_time = cur_end_time,
                              creation_time = cur_creation_time,
                              detections = [det, ])
            event_list_1.append(cur_event)
        cat.add_events(event_list_1)

        # Write the event to the database.
        cat.write_to_database(self.project)

        self.project.event_library.clear()

        # Store the created catalog for later use in the tests.
        self.test_cat = cat


    def test_event_catalog(self):
        ''' Test the event catalog.
        '''
        self.logger.info("Available catalogs: %s.",
                         list(self.project.event_library.catalogs.keys()))
        self.assertEqual(len(self.project.event_library.catalogs), 1)
        self.assertEqual(list(self.project.event_library.catalogs.keys())[0],
                         self.test_cat.name)
        cat = self.project.event_library.catalogs[self.test_cat.name]
        self.assertEqual(len(cat.events), len(self.test_cat.events))


    def test_get_recent_events(self):
        ''' Test the request for the event archive list.
        '''
        event_list = self.client.get_recent_events()

        self.assertEqual(len(event_list), 10)
        for cur_event in event_list:
            self.assertIsInstance(cur_event, dict)


    def test_process_monitor_stream(self):
        ''' Test the processing of a data stream.
        '''
        # Load and prepare the input data.
        filename = './data/earthquake_20190614T1234_neunkirchen.mseed'
        input_stream = obspy.read(filename)
        # Remove the DUBAM data from the stream.
        mss_stream = obspy.Stream()
        for cur_trace in input_stream:
            if cur_trace.stats.station != 'DUBAM':
                mss_stream.append(cur_trace)

        mss_stream.merge()
        mss_stream.sort()
        mss_stream.trim(starttime = obspy.UTCDateTime('2019-06-14T12:33:50'),
                        endtime = obspy.UTCDateTime('2019-06-14T12:36:00'))

        # Feed the stream in 10 second intervals.
        for win_st in mss_stream.slide(window_length = 10.0,
                                       step = 10.0):
            for cur_trace in copy.deepcopy(win_st):
                self.client.on_data(cur_trace)

            # Process the monitor stream.
            self.client.process_monitor_stream()


def suite():
    return unittest.makeSuite(MonitorClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
