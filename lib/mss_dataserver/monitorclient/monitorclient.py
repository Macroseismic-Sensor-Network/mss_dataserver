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

import copy
import datetime
import gc
import json
import logging
import os
import threading
import time

import asyncio
import numpy as np
import obspy
import obspy.core.utcdatetime as utcdatetime
import obspy.clients.seedlink.easyseedlink as easyseedlink
from obspy.clients.seedlink.slpacket import SLPacket
import pyproj
import scipy
import scipy.spatial

import mss_dataserver.core.validation as validation
import mss_dataserver.event.core as event_core
import mss_dataserver.event.detection as event_detection
import mss_dataserver.event.delaunay_detection as event_ddet


class EasySeedLinkClientException(Exception):
    """
    A base exception for all errors triggered explicitly by EasySeedLinkClient.
    """
    # XXX Base on SeedLinkException?
    pass


class MonitorClient(easyseedlink.EasySeedLinkClient):
    """
    A custom SeedLink client
    """
    def __init__(self, project, server_url, stations,
                 monitor_stream, stream_lock, data_dir, event_dir,
                 process_interval, stop_event, asyncio_loop,
                 pgv_sps = 1, autoconnect = False, pgv_archive_time = 1800,
                 trigger_thr = 0.01e-3, warn_thr = 0.01e-3,
                 valid_event_thr = 0.1e-3, felt_thr = 0.1e-3,
                 event_archive_size = 5):
        ''' Initialize the instance.
        '''
        easyseedlink.EasySeedLinkClient.__init__(self,
                                                 server_url = server_url,
                                                 autoconnect = autoconnect)
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # The project instance.
        self.project = project

        # The asyncio event loop. Used to stop the loop if an error occures.
        self.asyncio_loop = asyncio_loop

        # The stations to stream from the seedlink server.
        self.stations = stations

        # The recent, not yet processed data.
        self.monitor_stream = monitor_stream

        # The stream used for processing.
        self.process_stream = obspy.core.Stream()

        # The pgv stream holding the most recent data which has not been sent
        # over the websocket.
        self.pgv_stream = obspy.core.Stream()

        # The pgv archive stream holding the PGV data of the specified archive
        # time.
        self.pgv_archive_stream = obspy.core.Stream()

        # The velocity waveform data archive stream with the same length as the
        # PGV archive stream.
        self.vel_archive_stream = obspy.core.Stream()

        # The length of the archive stream to keep [s].
        self.pgv_archive_time = pgv_archive_time
        self.vel_archive_time = pgv_archive_time

        # The no-data value.
        self.nodata_value = -999999

        self.stream_lock = stream_lock
        self.archive_lock = threading.Lock()
        self.project_lock = threading.Lock()

        # The common data output directory.
        self.data_dir = data_dir

        # The event data output directory.
        self.event_dir = event_dir

        # The time interval [s] used to process the received data.
        self.process_interval = process_interval

        # The samples per second of the PGV data stream.
        self.pgv_sps = pgv_sps

        self.stop_event = stop_event

        # The trigger parameters.
        self.trigger_thr = trigger_thr
        self.warn_thr = warn_thr
        self.valid_event_thr = valid_event_thr
        self.felt_thr = felt_thr

        # The most recent detected event.
        self.event_triggered = False
        self.current_event = {}

        # The last detected events.
        self.event_archive_size = event_archive_size
        self.event_archive = []

        # The event warning.
        self.event_warning = {}

        # The latest event detection result.
        self.last_detection_result = {}

        # The PGV data state.
        self.pgv_data_available = asyncio.Event()

        # The event detection result state.
        self.event_detection_result_available = asyncio.Event()

        # The event warning state.
        self.event_warning_available = asyncio.Event()

        # The event trigger state.
        self.current_event_available = asyncio.Event()

        # The event archive state.
        self.event_archive_changed = asyncio.Event()

        # The keydata event.
        self.event_keydata_available = asyncio.Event()

        # The psysmon geometry inventory.
        self.inventory = self.project.inventory
        self.inventory.compute_utm_coordinates()

        # The delaunay detector instance.
        all_stations = self.inventory.get_station()
        self.detector = event_ddet.DelaunayDetector(network_stations = all_stations,
                                                    trigger_thr = self.trigger_thr,
                                                    window_length = 10,
                                                    safety_time = 10,
                                                    p_vel = 3500,
                                                    min_trigger_window = 3,
                                                    max_edge_length = 40000,
                                                    author_uri = self.project.project_config['author_uri'],
                                                    agency_uri = self.project.project_config['agency_uri'])

        self.recorder_map = self.get_recorder_mappings(station_nsl = self.stations)

        self.conn.timeout = 10

        # Load the archived data.
        self.event_archive_timespan = 48
        self.load_archive_catalogs(hours = self.event_archive_timespan)


    def seedlink_connect(self):
        ''' Connect to the seedlink server.
        '''
        self.connect()

        for cur_mss in self.recorder_map:
            self.select_stream(cur_mss[0],
                               cur_mss[1],
                               cur_mss[2] + cur_mss[3])


    def load_archive_catalogs(self, hours = 48):
        ''' Load the event catalogs of the specified last days.
        '''
        print("load_archive_catalogs")
        now = utcdatetime.UTCDateTime()
        start_time = now - hours * 3600
        start_day = utcdatetime.UTCDateTime(year = start_time.year,
                                            julday = start_time.julday)
        n_days = np.ceil((now - start_day) / 86400)
        n_days = int(n_days)
        for k in range(n_days):
            cur_cat_date = now - k * 86400
            cur_name = "{0:04d}-{1:02d}-{2:02d}".format(cur_cat_date.year,
                                                        cur_cat_date.month,
                                                        cur_cat_date.day)
            self.logger.info("Requesting catalog %s.", cur_name)
            with self.project_lock:
                cur_cat = self.project.load_event_catalog(name = cur_name,
                                                          load_events = True)
                if cur_cat:
                    self.logger.info("events in catalog: %s", cur_cat.events)
                self.logger.info("Catalog keys: %s", self.project.event_library.catalogs.keys())


    def load_archive_file(self):
        ''' Load data from the JSON archive file.
        '''
        archive_filename = os.path.join(self.data_dir,
                                        'mss_dataserver_archive.json')
        archive = {}
        if os.path.exists(archive_filename):
            try:
                with open(archive_filename) as fp:
                    archive = json.load(fp)
                    self.logger.info('Loaded the archive from file %s.',
                                     archive_filename)
            except Exception as e:
                self.logger.exception("Couldn't load the archive file %s.",
                                      archive_filename)
        return archive

    def load_from_archive(self):
        ''' Load data from a saved archive.
        '''
        archive = self.load_archive_file()
        if archive:
            if 'current_event' in archive.keys():
                self.current_event = archive['current_event']
                self.current_event['start_time'] = utcdatetime.UTCDateTime(self.current_event['start_time'])
                self.current_event['end_time'] = utcdatetime.UTCDateTime(self.current_event['end_time'])
                # TODO: Handle the loading of the pgv data stream.
                self.current_event['pgv'] = obspy.core.Stream()
            if 'event_archive' in archive.keys():
                self.event_archive = archive['event_archive']

                for cur_event in self.event_archive:
                    cur_event['start_time'] = utcdatetime.UTCDateTime(cur_event['start_time'])
                    cur_event['end_time'] = utcdatetime.UTCDateTime(cur_event['end_time'])

    def save_to_archive(self, key):
        ''' Save data to the JSON archive.
        '''
        archive_filename = os.path.join(self.data_dir,
                                        'mss_dataserver_archive.json')
        archive = self.load_archive_file()

        if key == 'current_event':
            data = self.get_current_event()
            # TODO: Handle the saving of the event PGV data.
        if key == 'event_archive':
            data = self.get_event_archive()

        try:
            archive['last_write'] = utcdatetime.UTCDateTime().isoformat()
            archive[key] = data

            with open(archive_filename, 'w') as fp:
                pref = json.dump(archive, fp, indent = 4, sort_keys = True)
                self.logger.info('Saved the archive to file %s.',
                                 archive_filename)
        except Exception as e:
            self.logger.exception("Error saving the data to the archive. data: %s, archive: %s",
                                  data, archive)


    def get_recorder_mappings(self, station_nsl = None):
        ''' Get the mappings of the requested NSLC.

        Return the matching NSLC codes of the MSS units relating their
        serial numbers to the actual station locations.
        '''
        recorder_map = {}
        if station_nsl is None:
            station_list = self.inventory.get_station()
        else:
            station_list = []
            for cur_nsl in station_nsl:
                cur_station = self.inventory.get_station(network = cur_nsl[0],
                                                         name = cur_nsl[1],
                                                         location = cur_nsl[2])
                if len(cur_station) > 1:
                    raise ValueError('There are more than one stations. This is not yet supported.')
                if len(cur_station) == 0:
                    raise ValueError('No station found for {0:s}. Check the input file.'.format(cur_nsl))
                cur_station = cur_station[0]
                station_list.append(cur_station)

        for station in station_list:
            for cur_channel in station.channels:
                stream_tb = cur_channel.get_stream(start_time = obspy.UTCDateTime())
                cur_loc = stream_tb[0].item.name.split(':')[0]
                cur_chan = stream_tb[0].item.name.split(':')[1]

                cur_key = ('XX',
                           stream_tb[0].item.serial,
                           cur_loc,
                           cur_chan)
                recorder_map[cur_key] = cur_channel.nslc

        self.logger.debug(recorder_map)
        return recorder_map

    def reconnect(self):
        ''' Reconnect to the server.'''
        self.logger.info('Reconnecting to the server.')
        self.close()

        self.connect()
        for cur_station in self.stations:
            self.select_stream(cur_station[0], cur_station[1], cur_station[2])

    def on_data(self, trace):
        """
        Override the on_data callback
        """
        #self.logger.debug('Received trace:')
        #self.logger.debug(str(trace))
        #self.logger.debug("on_data trace data: %s", trace.data)
        self.stream_lock.acquire()
        cur_nslc = self.recorder_map[tuple(trace.id.split('.'))]
        trace.stats.network = cur_nslc[0]
        trace.stats.station = cur_nslc[1]
        trace.stats.location = cur_nslc[2]
        trace.stats.channel = cur_nslc[3]
        self.monitor_stream.append(trace)
        #self.monitor_stream.merge(method = 1, fill_value = 'interpolate')
        #self.logger.debug(self.monitor_stream)
        self.stream_lock.release()
        #self.logger.debug('Leaving on_data.')

    def run(self):
        """
        Start streaming data from the SeedLink server.

        Streams need to be selected using
        :meth:`~.EasySeedLinkClient.select_stream` before this is called.

        This method enters an infinite loop, calling the client's callbacks
        when events occur.
        """
        # Note: This somewhat resembles the run() method in SLClient.

        # Check if any streams have been specified (otherwise this will result
        # in an infinite reconnect loop in the SeedLinkConnection)
        if not len(self.conn.streams):
            msg = 'No streams specified. Use select_stream() to select ' + \
                  'a stream.'
            raise EasySeedLinkClientException(msg)

        self.__streaming_started = True

        # Start the processing timer thread.
        #self.process_timer_thread = threading.Thread(name = 'process_timer',
        #                                             target = self.task_timer,
        #                                             args = (self.process_monitor_stream, ))

        #self.process_timer_thread.start()


        # Start the collection loop
        start = time.time()
        end = None
        while not self.stop_event.is_set():
            self.logger.debug('Waiting to collect data....')
            data = self.conn.collect()
            self.logger.debug('... received some data.')

            if data == SLPacket.SLTERMINATE:
                self.on_terminate()
                continue
            elif data == SLPacket.SLERROR:
                self.on_seedlink_error()
                continue

            # At this point the received data should be a SeedLink packet
            # XXX In SLClient there is a check for data == None, but I think
            #     there is no way that self.conn.collect() can ever return None
            assert(isinstance(data, SLPacket))

            packet_type = data.get_type()

            # Ignore in-stream INFO packets (not supported)
            if packet_type not in (SLPacket.TYPE_SLINF, SLPacket.TYPE_SLINFT):
                # The packet should be a data packet
                trace = data.get_trace()
                # Pass the trace to the on_data callback
                self.on_data(trace)
                end = time.time()

            # Start the data export thread if the write_interval has been
            # reached.
            #if end and (end - start) > self.write_interval:
                #process_thread = threading.Thread(target = process_monitor_stream, args = (self.monitor_stream, self.stream_lock, self.data_dir))
                #logging.info('Starting the processing thread.')
                #process_thread.start()
                #start = end
                #end = None

    def on_terminate(self):
        ''' Handle termination of the client.
        '''
        self.logger.warning('Terminating the client.')
        # Clean the state.
        self.stream_lock.acquire()
        self.monitor_stream.clear()
        self.stream_lock.release()

        # Reconnect to the server.
        try:
            self.reconnect()
        except Exception:
            self.logger.error("Can't reconnect to the seedlink server.")
            self.stop_event.set()
            # Stop the asyncio loop.
            self.logger.error("Stopping the asyncio event loop because there where troubles reconnecting to the seedlink server.")
            self.asyncio_loop.stop()

    def on_seedlink_error(self):
        ''' Handle client errors.
        '''
        self.logger.error('Got a seedlink error.')

    async def task_timer(self, callback):
        ''' A timer executing a task at regular intervals.
        '''
        self.logger.debug('Starting the timer.')
        interval = int(self.process_interval)
        now = obspy.UTCDateTime()
        delay_to_next_interval = interval - (now.timestamp % interval)
        self.logger.debug('Sleeping for %f seconds.', delay_to_next_interval)
        time.sleep(delay_to_next_interval)

        while not self.stop_event.is_set():
            try:
                self.logger.info('task_timer: Executing callback.')
                callback()
            except Exception as e:
                self.logger.exception(e)
                self.stop()

            now = obspy.UTCDateTime()
            delay_to_next_interval = interval - (now.timestamp % interval)
            self.logger.info('task_timer: Sleeping for %f seconds.', delay_to_next_interval)
            #time.sleep(delay_to_next_interval)
            await asyncio.sleep(delay_to_next_interval)

        self.logger.debug("Leaving the task_timer method.")

    def detect_event_warning(self):
        ''' Run the Voronoi detection with the most recent PGV data only.
        This is a first crude detection of an event and is used for a
        first warning notification of a possible event. The warning has
        has to be confirmed by a detected event.
        '''
        self.logger.info("Running detect_event_warning.")
        now = obspy.UTCDateTime()

        with self.stream_lock:
            working_stream = self.pgv_stream.copy()

        # Get the max. PGV and the related stations.
        max_pgv = []
        stations = []
        for cur_trace in working_stream:
            max_pgv.append(np.nanmax(cur_trace.data))
            stations.append(self.inventory.get_station(name = cur_trace.stats.station)[0])

        # Compute the Delaunay triangles.
        tri = self.compute_delaunay(stations)

        # Convert the lists to numpy arrays.
        max_pgv = np.array(max_pgv)
        stations = np.array(stations)

        # Compute the Delaunay trigger.
        trigger_pgv = []
        simp_stations = []
        if tri:
            edge_length = self.compute_edge_length(tri,
                                                   stations)
            valid_tri = np.argwhere(edge_length < 30000).flatten()
            edge_length = edge_length[valid_tri]
            for k, cur_simp in enumerate(tri.simplices[valid_tri]):
                cur_tri_pgv = max_pgv[cur_simp]
                simp_stations.append([x.name for x in stations[cur_simp]])
                trigger_pgv.append(np.nanmin(cur_tri_pgv))

        trigger_pgv = np.array(trigger_pgv)
        simp_stations = np.array(simp_stations)

        # Detect the event warning.
        mask = trigger_pgv >= self.warn_thr
        if mask.any():
            self.event_warning['time'] = now
            self.event_warning['simp_stations'] = simp_stations[mask]
            self.event_warning['trigger_pgv'] = trigger_pgv[mask]
            self.event_warning_available.set()
            self.logger.info("Event warning issued.")
            self.logger.debug("Event warning data: %s.", self.event_warning)
        else:
            self.event_warning['time'] = now
            self.event_warning['simp_stations'] = np.array([])
            self.event_warning['trigger_pgv'] = np.array([])
            self.event_warning_available.set()
            self.logger.info("No event warning issued.")
        self.logger.info("Finished the event warning computation.")

    def detect_event(self):
        ''' Run the Voronoi event detection.
        '''
        self.logger.info('Running the event detection.')
        detect_win_length = 10
        safety_win = 10
        trigger_thr = self.trigger_thr
        min_trigger_window = 3

        now = obspy.UTCDateTime()
        with self.archive_lock:
            working_stream = self.pgv_archive_stream.copy()
        self.logger.info("event detection working_stream: %s", working_stream)

        # Run the Delaunay detection.
        self.detector.init_detection_run(stream = working_stream)
        if self.detector.detection_run_initialized:
            self.detector.compute_trigger_data()
            self.detector.evaluate_event_trigger()

            # Evaluate the detection result.
            if self.detector.new_event_available:
                self.current_event = self.detector.get_event()

                # TODO: Add the original waveform data to the event.

                self.current_event_available.set()

                if self.current_event.detection_state == 'closed':
                    try:

                        # Save the event and its metadata in a thread to
                        # prevent blocking the data acquisition.
                        export_event = self.current_event
                        export_event_thread = threading.Thread(name = 'export_event',
                                                               target = self.export_event,
                                                               args = (export_event, ))
                        self.logger.info("Starting the export_event_thread.")
                        export_event_thread.start()
                        #export_event_thread.join()
                        self.logger.info("Continue the program execution.")
                    finally:
                        # Clear the event instance.
                        self.current_event = None
                        # Clear the detector flag.
                        self.detector.new_event_available = False
        else:
            self.logger.warning("Failed to initialize the detection run.")

    def detect_event_old(self):
        ''' Run the Voronoi event detections.
        '''
        logger = logging.getLogger('mss_data_server.detect_event')
        logger.info('Running the event detection.')
        detect_win_length = 10
        safety_win = 10
        trigger_thr = self.trigger_thr
        min_trigger_window = 3

        now = obspy.UTCDateTime()
        with self.archive_lock:
            working_stream = self.pgv_archive_stream.copy()

        self.logger.info("event detection working_stream: %s", working_stream)
        max_end_time = np.max([x.stats.endtime for x in working_stream])
        detect_win_begin = (max_end_time.timestamp - (detect_win_length + safety_win)) // detect_win_length * detect_win_length
        detect_win_begin = obspy.UTCDateTime(detect_win_begin)
        detect_win_end = detect_win_begin + 10
        logger.debug("max_end_time: %s", max_end_time)
        logger.debug("now: %s", now)
        logger.debug("detect_win_begin: %s", detect_win_begin)
        logger.debug("detect_win_end: %s", detect_win_end)

        # Compute the maximum possible time window length.
        tri = self.compute_delaunay(self.inventory.get_station())
        edge_length = self.compute_edge_length(tri, self.inventory.get_station())
        # Use only triangles with an edge length smaller than a threshold.
        if tri:
            valid_tri = np.argwhere(edge_length < 40000).flatten()
            logger.debug("valid_tri: %s", valid_tri)
            edge_length = edge_length[valid_tri]

        max_time_window = np.max(edge_length) / 3500
        max_time_window = np.ceil(max_time_window)

        logger.debug("max_time_window: %s", max_time_window)
        detect_stream = working_stream.slice(starttime = detect_win_begin - max_time_window,
                                             endtime = detect_win_end,
                                             nearest_sample = False)
        self.logger.info("event detection detect_stream: %s", detect_stream)

        detect_stations = []
        for cur_trace in detect_stream:
            cur_station = self.inventory.get_station(name = cur_trace.stats.station)
            if cur_station:
                detect_stations.append(cur_station[0])
        self.last_detection_result['computation_time'] = now.isoformat()
        self.last_detection_result['used_stations'] = [x.nsl for x in detect_stations]

        logger.debug("number of detect_stream traces: %d.", len(detect_stream))
        logger.debug("number of detect_stations: %d.", len(detect_stations))
        #logger.debug("x_utm: %s", [x.x_utm for x in detect_stations])

        trigger_data = []
        if detect_stations:
            # Compute the Delaunay triangulation.
            tri = self.compute_delaunay(detect_stations)

            if tri:
                # Compute the max. edge lengths of the triangles.
                edge_length = self.compute_edge_length(tri,
                                                       detect_stations)
                valid_tri = np.argwhere(edge_length < 30000).flatten()
                edge_length = edge_length[valid_tri]

                for k, cur_simp in enumerate(tri.simplices[valid_tri]):
                    cur_simp_stations = [detect_stations[x] for x in cur_simp]
                    logger.debug("Computing max PGV for stations: %s.", [x.name for x in cur_simp_stations])
                    cur_time, cur_pgv = self.compute_max_pgv(stream = detect_stream,
                                                             stations = cur_simp_stations,
                                                             edge_lengths = edge_length[k],
                                                             offset = max_time_window,
                                                             min_trigger_window = min_trigger_window,
                                                             compute_interval = 1)
                    if len(cur_pgv) > 0:
                        if np.any(np.isnan(cur_pgv)):
                            logger.warning("There is a NaN value in the cur_pgv.")
                            logger.debug("cur_pgv: %s.", cur_pgv)
                            # TODO: JSON can't handle NaN values. Ignore them right
                            # now until I find a good solution.
                            continue
                        cur_trig = np.nanmin(cur_pgv, axis = 1) >= trigger_thr
                        tmp = {}
                        #tmp['simp_stations'] = [x.name for x in cur_simp_stations]
                        #tmp['time'] = [x.isoformat() for x in cur_time]
                        #tmp['pgv'] = cur_pgv.tolist()
                        #tmp['trigger'] = cur_trig.tolist()
                        tmp['simp_stations'] = cur_simp_stations
                        tmp['time'] = cur_time
                        tmp['pgv'] = cur_pgv
                        tmp['trigger'] = cur_trig
                        trigger_data.append(tmp)

            # Check if any of the triangles has triggered and get the earliest
            # time of the triggering.
            trigger_times = []
            event_start = None
            event_end = None
            for cur_trigger_data in trigger_data:
                if np.any(cur_trigger_data['trigger']):
                    cur_mask = cur_trigger_data['trigger']
                    cur_trigger_start = np.array(cur_trigger_data['time'])[cur_mask][0]
                    cur_trigger_end = np.array(cur_trigger_data['time'])[cur_mask][-1]
                    trigger_times.append([obspy.UTCDateTime(cur_trigger_start),
                                          obspy.UTCDateTime(cur_trigger_end)])
            trigger_times = np.array(trigger_times)
            if len(trigger_times) > 0:
                logger.debug("trigger_times: %s", trigger_times)
                event_start = np.min(trigger_times[:, 0])
                event_end = np.max(trigger_times[:, 1])

            # If not in event mode and an event has been declared, set the
            # event mode.
            if not self.event_triggered and event_start is not None:
                logger.info("Event triggered.")
                try:
                    # Use the Event class to create an event.
                    cur_event_obj = event_core.Event(start_time = event_start,
                                                     end_time = event_end,
                                                     author_uri = self.project.author_uri,
                                                     agency_uri = self.project.agency_uri
                                                     )

                    # Compute the max. PGV of each triggered station.
                    cur_max_station_pgv = {}
                    for cur_data in [x for x in trigger_data if np.any(x['trigger'])]:
                        # Create a detection instance.
                        pgv_max = np.max(cur_data['pgv'], axis = 0)
                        cur_detection = event_detection.Detection(start_time = cur_data['time'][0],
                                                                  end_time = cur_data['time'][-1],
                                                                  stations = cur_simp_stations,
                                                                  pgv_max = {cur_simp_stations[0].nsl: pgv_max[0],
                                                                             cur_simp_stations[1].nsl: pgv_max[1],
                                                                             cur_simp_stations[2].nsl: pgv_max[2]})
                        cur_event_obj.add_detection(cur_detection)

                    self.current_event = cur_event
                    self.current_event_obj = cur_event_obj
                    self.event_triggered = True
                    self.current_event_available.set()
                except Exception as e:
                    logger.exception("Error processing the event trigger.")
                    self.event_triggered = False
                    self.current_event = {}
                    self.current_event_obj = None
            elif self.event_triggered and event_start is not None:
                logger.info("Updating triggered event.")
                try:
                    self.current_event['end_time'] = event_end
                    self.current_event['pgv'] = self.current_event['pgv'] + detect_stream
                    self.current_event['trigger_data'][detect_win_end.isoformat()] = trigger_data

                    self.current_event_obj.end_time = event_end

                    overall_trigger_data = self.current_event['overall_trigger_data']
                    for cur_trigger_data in [x for x in trigger_data if np.any(x['trigger'])]:
                        cur_simp_stations = cur_trigger_data['simp_stations']
                        pgv_max = np.max(cur_data['pgv'], axis = 0)
                        if self.current_event_obj.has_detection(cur_simp_stations):
                            # Update the detection.
                            cur_detection = self.current_event_obj.get_detection(cur_simp_stations)
                            if len(cur_detectin == 1):
                                cur_detection = cur_detection[0]
                            else:
                                logger.error("Expected exactly one detection. Got: %s.", cur_detection)
                            cur_detection.update(end_time = cur_data['time'][-1],
                                                 pgv_max = {cur_simp_stations[0].nsl: pgv_max[0],
                                                            cur_simp_stations[1].nsl: pgv_max[1],
                                                            cur_simp_stations[2].nsl: pgv_max[2]})
                        else:
                            # Add the detection.
                            cur_detection = event_detection.Detection(start_time = cur_data['time'][0],
                                                                      end_time = cur_data['time'][-1],
                                                                      stations = cur_simp_stations,
                                                                      pgv_max = {cur_simp_stations[0].nsl: pgv_max[0],
                                                                                 cur_simp_stations[1].nsl: pgv_max[1],
                                                                                 cur_simp_stations[2].nsl: pgv_max[2]})
                            self.current_event_obj.add_detection(cur_detection)

                    self.current_event_available.set()
                except Exception as e:
                    logger.exception("Error updating the current event.")
            elif self.event_triggered and event_start is None:
                logger.info("Event end declared.")
                try:
                    self.current_event['pgv'].merge()
                    self.current_event['state'] = 'closed'
                    self.event_triggered = False
                    self.logger.info('overall_trigger_data: %s', self.current_event['overall_trigger_data'])
                    # Compute the max. PGV of the event.
                    max_pgv = []
                    for cur_pgv in [x['pgv'] for x in self.current_event['overall_trigger_data']]:
                        max_pgv.append(np.nanmax(cur_pgv))
                    max_pgv = np.nanmax(max_pgv)
                    self.current_event['max_pgv'] = max_pgv
                    self.logger.info('max_pgv: %f', self.current_event['max_pgv'])
                    if max_pgv >= self.valid_event_thr:
                        self.current_event_to_archive()

                    # Write the event to the database.
                    self.current_event_obj.write_to_database(self.project)
                    self.current_event = {}
                    self.current_event_obj = None
                    self.event_data_available.set()
                except Exception as e:
                    logger.exception("Error processing the event end.")
                    self.current_event = {}

        logger.info("Finished event detection.")
        logger.debug("Number of trigger_data: %d.", len(trigger_data))
        self.last_detection_result['trigger_data'] = trigger_data

        # Delete unused instances.
        del working_stream
        del detect_stream
        logger.info('# gc.get_objects: %d', len(gc.get_objects()))

        self.event_detection_result_available.set()

    def compute_delaunay_old(self, stations):
        x = [x.x_utm for x in stations]
        y = [x.y_utm for x in stations]
        coords = np.array(list(zip(x, y)))
        try:
            tri = scipy.spatial.Delaunay(coords)
        except Exception as e:
            tri = None

        return tri

    def compute_edge_length_old(self, tri, stations):
        x = [x.x_utm for x in stations]
        y = [x.y_utm for x in stations]
        coords = np.array(list(zip(x, y)))

        dist = []
        for cur_simp in tri.simplices:
            cur_vert = coords[cur_simp]
            cur_dist = [np.linalg.norm(x - y) for x in cur_vert for y in cur_vert]
            dist.append(cur_dist)
        edge_length = np.max(dist, axis=1)

        return np.array(edge_length)

    def compute_utm_coordinates_old(self, stations, inventory):
        code = inventory.get_utm_epsg()
        proj = pyproj.Proj(init = 'epsg:' + code[0][0])

        for cur_station in stations:
            x, y = proj(cur_station.get_lon_lat()[0],
                        cur_station.get_lon_lat()[1])
            cur_station.x_utm = x
            cur_station.y_utm = y

    def compute_max_pgv_old(self, stream, stations, edge_lengths, offset,
                        min_trigger_window = 2, compute_interval = 1):
        time_window = np.max(edge_lengths) / 3500
        time_window = np.ceil(time_window)

        if time_window < min_trigger_window:
            self.logger.debug("Time window too small. Edge lengths: %s.", edge_lengths)
            time_window = min_trigger_window

        tri_stream = obspy.Stream()
        for cur_station in stations:
            tri_stream = tri_stream + stream.select(station = cur_station.name)

        pgv = []
        time = []
        self.logger.debug("compute_max_pgv:  tri_stream: %s.", tri_stream.__str__(extended = True))

        pgv_strided = []
        time_strided = []
        def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
            nrows = ((a.size-L)//S)+1
            n = a.strides[0]
            return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))
        for cur_trace in tri_stream:
            self.logger.debug("cur_trace.id: %s", cur_trace.id)
            cur_win_length = int(np.floor(time_window / cur_trace.stats.sampling_rate))
            cur_offset = int(np.floor(offset / cur_trace.stats.sampling_rate))
            self.logger.debug("cur_offset: %s", cur_offset)
            self.logger.debug("cur_win_length: %d", cur_win_length)
            self.logger.debug("cur_trace.data: %s", cur_trace.data)
            if len(cur_trace.data) < cur_win_length:
                self.logger.error("The window length is smaller than the data size.")
                continue
            cur_data = strided_app(cur_trace.data, cur_win_length,  1)
            self.logger.debug("cur_data: %s", cur_data)
            cur_max_pgv = np.max(cur_data, axis = 1)
            self.logger.debug("sp filter: %s", cur_max_pgv)
            cur_max_pgv = cur_max_pgv[(cur_offset - cur_win_length + compute_interval):]
            cur_start = cur_trace.stats.starttime + (cur_offset - cur_win_length + compute_interval + cur_win_length - 1) * cur_trace.stats.delta
            cur_time = [cur_start + x * compute_interval for x in range(len(cur_max_pgv))]
            pgv_strided.append(cur_max_pgv)
            time_strided.append(cur_time)

#        for cur_stream in tri_stream.slide(window_length = time_window,
#                                           step = compute_interval,
#                                           offset = offset - time_window + compute_interval,
#                                           include_partial_windows = False):
#            if len(cur_stream) != 3:
#                self.logger.error("Not exactly 3 traces in the stream: %s.", cur_stream)
#                continue
#            cur_pgv = []
#            for cur_trace in cur_stream:
#                cur_pgv.append(np.max(cur_trace.data))
#
#            pgv.append(cur_pgv)
#            time.append(cur_trace.stats.endtime)

        # Delete unused instances.
        del tri_stream

#        self.logger.info("obspy slide pgv: %s", np.array(pgv))
#        self.logger.info("obspy slide time: %s", np.array(time))

        if len(set([len(x) for x in pgv_strided])) == 1:
            pgv_strided = np.array(pgv_strided).transpose()
            time_strided = np.array(time_strided).transpose()[:,0]
        else:
            self.logger.error("The size of the computed PGV max don't match: %s.", pgv_strided)
            pgv_strided = []
        self.logger.debug("pgv_strides: %s", pgv_strided)
        self.logger.debug("time_strided: %s", time_strided)

        #return np.array(time), np.array(pgv)
        return time_strided, pgv_strided

    def current_event_to_archive_old(self):
        ''' Save the current event in the archive.
        '''
        self.logger.debug("Saving the current event to the archive.")
        if self.current_event:
            self.event_archive.append(self.current_event)
            self.event_archive = self.event_archive[-self.event_archive_size:]
            self.event_archive_changed.set()
            self.save_to_archive('event_archive')

    def process_monitor_stream(self):
        ''' Process the data in the monitor stream.

        '''
        self.logger.info('Processing the monitor_stream.')
        self.logger.info('# gc.get_objects: %d', len(gc.get_objects()))
        self.stream_lock.acquire()
        monitor_stream_length = len(self.monitor_stream)
        self.monitor_stream.merge()
        self.stream_lock.release()
        if monitor_stream_length > 0:
            #now = obspy.UTCDateTime()
            #process_end_time = now - now.timestamp % self.process_interval
            #logger.debug('Trimming to end time: %s.', process_end_time)
            self.stream_lock.acquire()
            for cur_trace in self.monitor_stream:
                sec_remain = cur_trace.stats.endtime.timestamp % self.process_interval
                cur_end_time = obspy.UTCDateTime(round(cur_trace.stats.endtime.timestamp - sec_remain))
                cur_end_time = cur_end_time - cur_trace.stats.delta
                self.logger.debug('Trimming %s to end time: %f.', cur_trace.id, cur_end_time.timestamp)
                cur_slice_trace = cur_trace.slice(endtime = cur_end_time)
                if len(cur_slice_trace) > 0:
                    self.process_stream.append(cur_slice_trace)
                    cur_trace.trim(starttime = cur_end_time + cur_trace.stats.delta)
            self.stream_lock.release()

            self.logger.info('process_stream: %s', self.process_stream.__str__(extended = True))

            with self.stream_lock:
                self.logger.info('monitor_stream: %s', str(self.monitor_stream.__str__(extended = True)))

            # Get a stream containing only equal length traces per station.
            el_stream = self.get_equal_length_traces()
            self.logger.debug('Got el_stream: %s.', el_stream.__str__(extended = True))

            # Detrend to remove eventual offset.
            el_stream = el_stream.split()
            el_stream.detrend(type = 'constant')
            el_stream.merge()

            # Convert the amplitudes from counts to m/s.
            self.convert_to_physical_units(el_stream)

            # Add the velocity waveform data to an archive stream.
            with self.archive_lock:
                self.vel_archive_stream = self.vel_archive_stream + el_stream

            # Compute the PGV values.
            self.compute_pgv(el_stream)

            # Trim the archive stream.
            self.trim_archive()

            # Start the event alarm detection using the most recent pgv values.
            #try:
            #    self.detect_event_warning()
            #except Exception as e:
            #    self.logger.exception("Error computing the event warning.")

            # Signal available PGV data.
            self.pgv_data_available.set()

            # Start the event detection.
            try:
                self.detect_event()
            except Exception as e:
                self.logger.exception("Error computing the event detection.")

            # Set the flag to mark another SOH state available.
            self.event_keydata_available.set()
            return


            # TODO: Check if something could be interesting and remove the code
            # below.
            if len(set([len(x) for x in export_stream])) != 1:
                self.logger.warning('The length of the traces in the stream is not equal. No export.')
                self.logger.debug('Add the stream back to the monitor stream.')
                stream_lock.acquire()
                monitor_stream += export_stream
                monitor_stream.merge()
                self.logger.debug(monitor_stream)
                stream_lock.release()
            else:
                if len(export_stream) > 0:
                    self.logger.debug('Exporting the stream.')
                    for cur_trace in export_stream:
                        self.logger.debug(cur_trace.get_id().replace('.', '_'))
                        self.logger.debug(cur_trace.stats.starttime.isoformat())
                        cur_filename = 'syscom_{0:s}_{1:s}.msd'.format(cur_trace.get_id().replace('.', '_'), cur_trace.stats.starttime.isoformat().replace('-', '').replace(':', '').replace('.', '_').replace('T', '_'))
                        cur_filename = os.path.join(data_dir, cur_filename)
                        export_stream.write(cur_filename, format = 'MSEED', reclen = 512, encoding = 11)
                    self.logger.debug('Done.')
                else:
                    self.logger.debug('No data in stream.')

            # Delete old files.
            now = utcdatetime.UTCDateTime()
            for cur_file in os.listdir(data_dir):
                if os.stat(os.path.join(data_dir, cur_file)).st_mtime < now - 60:
                    os.remove(os.path.join(data_dir, cur_file))

    def get_equal_length_traces(self):
        ''' Get a stream containing traces with equal length per station.
        '''
        unique_stations = [(x.stats.network, x.stats.station, x.stats.location) for x in self.process_stream]
        unique_stations = list(set(unique_stations))
        el_stream = obspy.core.Stream()
        for cur_station in unique_stations:
            self.logger.debug('Getting stream for %s.', cur_station)
            # Get all traces for the station.
            cur_stream = self.process_stream.select(network = cur_station[0],
                                                    station = cur_station[1],
                                                    location = cur_station[2])

            self.logger.debug('Selected stream: %s.', cur_stream)
            for cur_trace in cur_stream:
                self.process_stream.remove(cur_trace)
            cur_stream = cur_stream.merge()

            # Check if all required channels are present:
            missing_data = False
            required_nslc = [x for x in self.recorder_map.values()
                             if x[0] == cur_station[1]]
            available_nslc = [(x.stats.network,
                               x.stats.station,
                               x.stats.location,
                               x.stats.channel) for x in cur_stream]
            self.logger.debug('available_nslc: %s', available_nslc)
            self.logger.debug('required_nslc: %s', required_nslc)

            for cur_required_nslc in required_nslc:
                if cur_required_nslc not in available_nslc:
                    self.logger.debug('Data for channel %s is missing.',
                                      cur_required_nslc)
                    missing_data = True
                    break

            if missing_data:
                self.logger.debug('Skipping station because of missing channel data.')
                self.process_stream.extend(cur_stream)
                continue

            # Select a timespan with equal end time.
            min_end = np.min([x.stats.endtime for x in cur_stream])
            cur_el_stream = cur_stream.slice(endtime = min_end)
            cur_rem_stream = cur_stream.trim(starttime = min_end + cur_el_stream[0].stats.delta)
            self.process_stream.extend(cur_rem_stream)
            self.logger.debug('el_stream: %s', cur_el_stream)
            self.logger.debug('rem_stream: %s', cur_rem_stream)
            el_stream.extend(cur_el_stream)

        return el_stream

    def compute_pgv(self, stream):
        ''' Compute the PGV values of the stream.
        '''
        self.logger.info('Computing the PGV.')
        unique_stations = [(x.stats.network, x.stats.station, x.stats.location) for x in stream]
        unique_stations = list(set(unique_stations))
        samp_interval = 1 / self.pgv_sps

        for cur_station in unique_stations:
            self.logger.debug('Getting stream for %s.', cur_station)
            # Get all traces for the station.
            cur_stream = stream.select(network = cur_station[0],
                                       station = cur_station[1],
                                       location = cur_station[2])

            self.logger.debug('Selected stream: %s.', cur_stream)

            min_start = np.min([x.stats.starttime for x in cur_stream])
            sec_remain = min_start.timestamp % samp_interval
            cur_offset = -sec_remain
            self.logger.debug('offset: %f.', cur_offset)
            cur_delta = cur_stream[0].stats.delta

            cur_pgv = []
            for win_st in cur_stream.slide(window_length = samp_interval - cur_delta,
                                           step = samp_interval,
                                           offset = cur_offset):
                self.logger.debug('Processing stream slice: %s.', win_st)
                x_st = win_st.select(channel = 'Hparallel')
                y_st = win_st.select(channel = 'Hnormal')

                # TODO: It is assumed, that there are no gaps in the data.
                # Handle this problem.
                if len(x_st.traces) > 1:
                    self.logger.error("Too many traces in the stream slice.")
                for cur_x_trace, cur_y_trace in zip(x_st.traces, y_st.traces):
                    cur_x = cur_x_trace.data
                    cur_y = cur_y_trace.data

                    # Handle masked arrays.
                    if isinstance(cur_x, np.ma.MaskedArray):
                        cur_x = cur_x.data
                    if isinstance(cur_y, np.ma.MaskedArray):
                        cur_y = cur_y.data

                    if len(cur_x) != len(cur_y):
                        self.logger.error("The x and y data lenght dont't match. Can't compute the res. PGV for this trace.")
                        continue

                    cur_sec_remain = cur_x_trace.stats.starttime.timestamp % samp_interval
                    cur_win_start = cur_x_trace.stats.starttime - cur_sec_remain
                    #cur_win_end = cur_win_start + self.process_interval - cur_x_trace.stats.delta
                    #cur_pgv_time = cur_x_trace.stats.starttime + \
                    #               (cur_x_trace.stats.endtime - cur_x_trace.stats.starttime) / 2
                    cur_pgv_time = cur_win_start + samp_interval / 2
                    cur_pgv_value = np.nanmax(np.sqrt(cur_x**2 + cur_y**2))
                    cur_pgv.append([cur_pgv_time, cur_pgv_value])

            if cur_pgv:
                cur_pgv = np.array(cur_pgv)
                #self.logger.debug('cur_x: %s', cur_x)
                #self.logger.debug('cur_y: %s', cur_y)
                self.logger.debug('cur_pgv: %s', cur_pgv)
                cur_stats = {'network': cur_x_trace.stats.network,
                             'station': cur_x_trace.stats.station,
                             'location': cur_x_trace.stats.location,
                             'channel': 'pgv',
                             'sampling_rate': self.pgv_sps,
                             'starttime': cur_pgv[0, 0]}
                self.logger.debug('data type: %s.', cur_pgv[:, 1].dtype)
                cur_pgv_trace = obspy.core.Trace(data = cur_pgv[:, 1].astype(np.float32),
                                                 header = cur_stats)
                # Write the data to the current pgv stream.
                self.pgv_stream.append(cur_pgv_trace)

        # Merge the current pgv stream.
        self.pgv_stream.merge()
        self.logger.debug('pgv_stream: %s.', self.pgv_stream.__str__(extended = True))

        with self.archive_lock:
            self.pgv_archive_stream = self.pgv_archive_stream + self.pgv_stream

        # Merge the archive stream.
        with self.archive_lock:
            self.pgv_archive_stream.merge()
        self.logger.debug("pgv_archive_stream: %s", self.pgv_archive_stream.__str__(extended = True))
        self.logger.info("Finished compute_pgv.")

        #self.pgv_archive_stream.write("/home/stefan/Schreibtisch/pgv_beben_neunkirchen.msd",
        #                             format = "MSEED",
        #                             reclen = 512)

    def convert_to_physical_units(self, stream):
        ''' Convert the counts to physical units.
        '''
        for tr in stream.traces:
            station = self.inventory.get_station(network = tr.stats.network,
                                                 name = tr.stats.station,
                                                 location = tr.stats.location)
            if len(station) > 1:
                raise ValueError('There are more than one stations. This is not yet supported.')
            station = station[0]

            channel = station.get_channel(name = tr.stats.channel)

            if len(channel) > 1:
                raise ValueError('There are more than one channels. This is not yet supported.')
            channel = channel[0]

            stream_tb = channel.get_stream(start_time = tr.stats.starttime,
                                           end_time = tr.stats.endtime)

            if len(stream_tb) > 1:
                raise ValueError('There are more than one recorder streams. This is not yet supported.')
            rec_stream = stream_tb[0].item

            rec_stream_param = rec_stream.get_parameter(start_time = tr.stats.starttime,
                                                        end_time = tr.stats.endtime)
            if len(rec_stream_param) > 1:
                raise ValueError('There are more than one recorder stream parameters. This is not yet supported.')
            rec_stream_param = rec_stream_param[0]


            components_tb = rec_stream.get_component(start_time = tr.stats.starttime,
                                                     end_time = tr.stats.endtime)

            if len(components_tb) > 1:
                raise ValueError('There are more than one components. This is not yet supported.')
            component = components_tb[0].item
            comp_param = component.get_parameter(start_time = tr.stats.starttime,
                                                 end_time = tr.stats.endtime)

            if len(comp_param) > 1:
                raise ValueError('There are more than one parameters for this component. This is not yet supported.')

            comp_param = comp_param[0]

            self.logger.debug("bitweight: %f", rec_stream_param.bitweight)
            self.logger.debug("gain: %f", rec_stream_param.gain)
            self.logger.debug("sensitivity: %f", comp_param.sensitivity)
            tr.data = tr.data * rec_stream_param.bitweight / (rec_stream_param.gain * comp_param.sensitivity)
            tr.stats.unit = component.output_unit.strip()

    def compute_pgv_res(st):
        ''' Compute the resultant of the peak-ground-velocity.
        '''
        x_st = st.select(channel = 'Hparallel').merge()
        y_st = st.select(channel = 'Hnormal').merge()

        res_st = obspy.core.Stream()
        for cur_x_trace, cur_y_trace in zip(x_st.traces, y_st.traces):
            cur_x = cur_x_trace.data
            cur_y = cur_y_trace.data

            if len(cur_x) != len(cur_y):
                self.logger.error("The x and y data lenght dont't match. Can't compute the res. PGV for this trace.")
                continue

            self.logger.debug("x: %s", cur_x)
            self.logger.debug("y: %s", cur_y)

            cur_res = np.sqrt(cur_x**2 + cur_y**2)

            cur_stats = {'network': cur_x_trace.stats['network'],
                         'station': cur_x_trace.stats['station'],
                         'location': cur_x_trace.stats['location'],
                         'channel': 'res',
                         'sampling_rate': cur_x_trace.stats['sampling_rate'],
                         'starttime': cur_x_trace.stats['starttime']}
            res_trace = obspy.core.Trace(data = cur_res, header = cur_stats)
            res_st.append(res_trace)

        res_st.split()

        return res_st

    def get_triggered_stations(self):
        ''' Get the stations which have a PGV exceeding the threshold value.
        '''
        now = utcdatetime.UTCDateTime()
        window = 120

        with self.archive_lock:
            working_stream = self.pgv_archive_stream.copy()

        working_stream.trim(starttime = now - window)
        all_stations = {}
        triggered_stations = {}
        for cur_trace in working_stream:
            cur_max_pgv = np.nanmax(cur_trace.data)
            if np.isnan(cur_max_pgv):
                all_stations[cur_trace.stats.station] = self.nodata_value
            else:
                all_stations[cur_trace.stats.station] = float(cur_max_pgv)
            if cur_max_pgv >= self.felt_thr:
                triggered_stations[cur_trace.stats.station] = float(cur_max_pgv)
        self.logger.debug("triggered_stations: %s", triggered_stations)

        return (all_stations, triggered_stations)

    def get_triggered_event_stations(self):
        ''' Get the stations which have triggered during the last event.
        '''
        # TODO: Fix the method to work with the new event objects.
        now = utcdatetime.UTCDateTime()
        window = 300
        triggered_stations = {}

        if self.current_event and self.current_event['start_time'] >= (now - window):
            for cur_station, cur_max_station_pgv in self.current_event['max_station_pgv'].items():
                if cur_max_station_pgv >= self.felt_thr:
                    triggered_stations[cur_station] = float(cur_max_station_pgv)
        elif self.event_archive and self.event_archive[-1]['start_time'] >= (now - window):
            for cur_station, cur_max_station_pgv in self.event_archive[-1]['max_station_pgv'].items():
                if cur_max_station_pgv >= self.felt_thr:
                    triggered_stations[cur_station] = float(cur_max_station_pgv)

        return triggered_stations

    def trim_archive(self):
        ''' Crop the archive to a specified length.
        '''
        with self.archive_lock:
            if len(self.pgv_archive_stream) > 0:
                max_end_time = np.max([x.stats.endtime for x in self.pgv_archive_stream if x])
                self.logger.debug("max_end_time: %s.", max_end_time)
                crop_start_time = max_end_time - self.pgv_archive_time
                self.pgv_archive_stream.trim(starttime = crop_start_time)
                self.logger.debug("Trimmed the pgv archive stream to %s.",
                                  crop_start_time)

            if len(self.vel_archive_stream) > 0:
                max_end_time = np.max([x.stats.endtime for x in self.vel_archive_stream if x])
                self.logger.debug("max_end_time: %s.", max_end_time)
                crop_start_time = max_end_time - self.vel_archive_time
                self.vel_archive_stream.trim(starttime = crop_start_time)
                self.logger.debug("Trimmed the velocity archive stream to %s.",
                                  crop_start_time)


    def export_event(self, export_event):
        ''' Save the event.
        '''
        # Get or create the event catalog and add the event to it.
        cat_name = "{0:04d}-{1:02d}-{2:02d}".format(export_event.start_time.year,
                                                    export_event.start_time.month,
                                                    export_event.start_time.day)

        with self.project_lock:
            cur_cat = self.project.get_event_catalog(cat_name)
            cur_cat.add_events([export_event, ])

            # Get or create the detection catalog and add the event
            # detections to it.
            det_catalogs = self.project.get_detection_catalog_names()
            if cat_name not in det_catalogs:
                cur_det_cat = self.project.create_detection_catalog(name = cat_name)
            else:
                cur_det_cat = self.project.load_detection_catalog(name = cat_name)
            cur_det_cat.add_detections(export_event.detections)

            # Write the detections to the database. This has to be done
            # separately. Adding the detection to the event and then
            # writing the event to the database doesn't write the
            # detections to the database.
            # TODO: An error occured, because the detection already had
            # a database id assigned, and the write_to_database method
            # tried to update the existing detection. This part of the
            # method is not yet working, thus an error was thrown. This
            # should not happen, because only fresh detections with no
            # database id should be available at this point!!!!!
            for cur_detection in export_event.detections:
                cur_detection.write_to_database(self.project)

            # Write the event to the database.
            self.logger.info("Writing the event to the database.")
            export_event.write_to_database(self.project)

        # Export the event data to disk files.
        self.logger.info("Exporting the event data.")
        self.export_event_data(export_event)

        # TODO: Trigger a thread to compute the event results (e.g.
        # localization, geojson layers, ...).


    def export_event_data(self, export_event):
        ''' Write the data of the passed event to a directory structure.
        '''
        # Build the output directory.
        date_dir = "{0:04d}_{1:02d}_{2:02d}".format(export_event.start_time.year,
                                                    export_event.start_time.month,
                                                    export_event.start_time.day)
        output_dir = os.path.join(self.event_dir,
                                  date_dir,
                                  export_event.public_id)

        self.logger.info("Exporting the event data to folder %s.", output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        export_start_time = export_event.start_time - 20
        export_end_time = export_event.end_time + 20

        # Write the PGV data to the event directory.
        with self.archive_lock:
            pgv_stream = self.pgv_archive_stream.slice(starttime = export_start_time,
                                                       endtime = export_end_time)
        filename = "{0:s}_{1:d}_pgv.msd".format(export_event.public_id,
                                                export_event.db_id)
        filepath = os.path.join(output_dir, filename)
        self.logger.info("Writing the PGV data to file %s.", filepath)
        pgv_stream.write(filepath,
                         format = 'MSEED',
                         blocksize = 512)

        # Write the velocity data to the event directory.
        with self.archive_lock:
            vel_stream = self.vel_archive_stream.slice(starttime = export_start_time,
                                                       endtime = export_end_time)
        vel_stream.merge()
        vel_stream = vel_stream.split()
        self.logger.info("vel_stream: %s", vel_stream.__str__(extended = True))
        filename = "{0:s}_{1:d}_vel.msd".format(export_event.public_id,
                                                export_event.db_id)
        filepath = os.path.join(output_dir, filename)
        self.logger.info("Writing the velocity data to file %s.", filepath)
        vel_stream.write(filepath,
                         format = 'MSEED',
                         blocksize = 512)


        # Collect the event metadata.
        pgv_stream = pgv_stream.slice(starttime = export_event.start_time,
                                      endtime = export_event.end_time)
        max_network_pgv = {}
        for cur_trace in pgv_stream:
            cur_nsl = '{0:s}:{1:s}:{2:s}'.format(cur_trace.stats.network,
                                                 cur_trace.stats.station,
                                                 cur_trace.stats.location)
            max_network_pgv[cur_nsl] = float(np.nanmax(cur_trace.data))

        max_event_pgv = export_event.get_max_pgv_per_station()

        event_meta = {}
        event_meta['max_network_pgv'] = max_network_pgv
        event_meta['max_event_pgv'] = max_event_pgv


        # Write the event metadata to a json file.
        try:
            filename = "{0:s}_{1:d}_detection_data.json".format(export_event.public_id,
                                                                export_event.db_id)
            filepath = os.path.join(output_dir, filename)
            self.logger.info("Writing the detection data to file %s.",
                             filepath)
            with open(filepath, 'w') as json_file:
                pref = json.dump(event_meta,
                                 json_file,
                                 indent = 4,
                                 sort_keys = True)
        except Exception as e:
            self.logger.exception("Error saving the detection data to json file.")

        # TODO: Export the original seismograms.
        # TODO: Export the max. PGV data of each available station.

    def get_pgv_data(self):
        ''' Get the latest PGV data.
        '''
        pgv_data = {}
        for cur_trace in self.pgv_stream:
            cur_start_time = cur_trace.stats.starttime
            tmp = {}
            try:
                # There have been some NaN values in the data. I couldn't
                # figure out where they came from. Make sure to replace them
                # with the nodata value before creating the pgv data.
                cur_trace.data[np.isnan(cur_trace.data)] = self.nodata_value
                tmp['time'] = [x.isoformat() for (x, y) in zip(cur_trace.times(type = "utcdatetime"), cur_trace.data.tolist()) if y != self.nodata_value]
                tmp['data'] = [x for x in cur_trace.data.tolist() if x != self.nodata_value]
            except Exception as e:
                self.logger.exception("Error getting the PGV data.")
            pgv_data[cur_trace.get_id()] = tmp

        # Clear the pgv stream.
        self.pgv_stream.clear()
        return pgv_data

    def get_pgv_archive(self):
        ''' Get the archived PGV data.
        '''
        pgv_data = {}
        with self.archive_lock:
            working_stream = self.pgv_archive_stream.copy()

        for cur_trace in working_stream:
            try:
                # There have been some NaN values in the data. I couldn't
                # figure out where they came from. Make sure to replace them
                # with the nodata value before creating the pgv data.
                cur_trace.data[np.isnan(cur_trace.data)] = self.nodata_value
                tmp = {}
                tmp['time'] = [x.isoformat() for (x, y) in zip(cur_trace.times(type = "utcdatetime"), cur_trace.data.tolist()) if y != self.nodata_value]
                tmp['data'] = [x for x in cur_trace.data.tolist() if x != self.nodata_value]
                pgv_data[cur_trace.get_id()] = tmp
            except Exception as e:
                self.logger.exception("Error while preparing the pgv archive.")

        return pgv_data

    def get_current_event(self):
        ''' Return the current event in serializable form.
        '''
        cur_event = {}
        if self.current_event:
            cur_event = validation.Event(id = self.current_event.db_id,
                                         start_time = self.current_event.start_time.isoformat(),
                                         end_time = self.current_event.end_time.isoformat(),
                                         description = self.current_event.description,
                                         comment = self.current_event.comment,
                                         max_pgv = self.current_event.max_pgv,
                                         state = self.current_event.detection_state)
            cur_event = cur_event.dict()

            # TODO: Serve the detailed event data only on request.
            #cur_event['trigger_data'] = self.current_event['trigger_data']
            #cur_event['overall_trigger_data'] = self.current_event['overall_trigger_data']
            #cur_archive_event['max_station_pgv'] = cur_event['max_station_pgv']

        return cur_event

    def get_event_warning(self):
        ''' Return the current event warning in serializable form.
        '''
        cur_warning = {}
        if self.event_warning:
            cur_warning['time'] = self.event_warning['time'].isoformat()
            cur_warning['simp_stations'] = self.event_warning['simp_stations'].tolist()
            cur_warning['trigger_pgv'] = self.event_warning['trigger_pgv'].tolist()

        return cur_warning

    def get_recent_events(self):
        ''' Return the current event archive in serializable form.
        '''
        now = utcdatetime.UTCDateTime()
        today = utcdatetime.UTCDateTime(now.timestamp // 86400 * 86400)
        request_start = today - self.event_archive_timespan * 3600
        with self.project_lock:
            events = self.project.get_events(start_time = request_start)
        cur_archive = []
        if len(events) > 0:
            for cur_event in events:
                cur_archive_event = validation.Event(id = cur_event.db_id,
                                                     start_time = cur_event.start_time.isoformat(),
                                                     end_time = cur_event.end_time.isoformat(),
                                                     description = cur_event.description,
                                                     comment = cur_event.comment,
                                                     max_pgv = cur_event.max_pgv,
                                                     state = cur_event.detection_state)

                # TODO: Get the data of the events only on request using
                # websocket.
                #
                #cur_archive_event['trigger_data'] = cur_event['trigger_data']
                #cur_archive_event['state'] = cur_event.state
                #cur_archive_event['overall_trigger_data'] = cur_event['overall_trigger_data']
                #try:
                #    cur_archive_event['max_station_pgv'] = cur_event['max_station_pgv']
                #except Exception:
                #    cur_archive_event['max_station_pgv'] = {}
                cur_archive.append(cur_archive_event.dict())

        return cur_archive

    def get_keydata(self):
        ''' Return the keydata.
        '''
        all_stations, triggered_stations = self.get_triggered_stations()
        keydata = {}
        keydata['time'] = utcdatetime.UTCDateTime().isoformat()
        keydata['all_stations'] = all_stations
        keydata['triggered_stations'] = triggered_stations
        keydata['triggered_event_stations'] = self.get_triggered_event_stations()

        return keydata
