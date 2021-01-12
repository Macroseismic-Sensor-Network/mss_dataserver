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

import logging
import numpy as np
import obspy
import scipy
import scipy.spatial

import mss_dataserver.event.core as event_core
import mss_dataserver.event.detection as event_detection


class DelaunayDetector(object):
    ''' Event detection using amplitudes and Delaunay Triangulation.

    '''
    def __init__(self, network_stations,
                 trigger_thr = 0.01e-3,
                 window_length = 10,
                 safety_time = 10,
                 p_vel = 3500,
                 min_trigger_window = 2,
                 max_edge_length = 40000,
                 author_uri = None,
                 agency_uri = None):
        ''' Initialization of the instance.
        '''
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # All available network stations.
        self.network_stations = network_stations

        # The trigger threshold value [m/s].
        self.trigger_thr = trigger_thr

        # The length of the detection window [s].
        self.window_length = window_length

        # The length of the timespan used as a safety period when selecting
        # computing the end-time of the detection timespan [s].
        self.safety_time = safety_time

        # The p wave velocity [m/s].
        self.p_vel = p_vel

        # The length of the minimum trigger window [s].
        self.min_trigger_window = min_trigger_window

        # The maximum edge length of a triangle used for the detection [m].
        self.max_edge_length = max_edge_length

        # The URI of the author.
        self.author_uri = author_uri

        # The URI of the agency
        self.agency_uri = agency_uri

        # The maximum maximum time that a wave needs to pass the triangles [s].
        self.max_time_window = None

        # The currently detected event.
        self.current_event = None

        # The stations used for the event detection.
        self.detect_stations = []

        # The stations not used for the event detection.
        self.unused_stations = []

        # The seismogram stream used for the event detection.
        self.detect_stream = None

        # The delaunay triangulation result.
        self.tri = None

        # The triangle max. edge lenghts [m].
        self.edge_length = {}

        # The time of the last data used for detection.
        self.last_detection_start = None
        self.last_detection_end = None

        # The current trigger data.
        self.trigger_data = []

        # The state of the event detector.
        # A new detection run has been initialized.
        self.detection_run_initialized = False
        # An event has been triggered.
        self.event_triggered = False
        # A new event is available.
        self.new_event_available = False

        # Compute the maximum time window based on all available stations.
        self.compute_max_time_window()


    def reset(self):
        ''' Reset the detector to an initial state.
        '''
        self.max_time_window = None
        self.current_event = None
        self.detect_stations = []
        self.unused_stations = []
        self.detect_stream = None
        self.tri = None
        self.edge_length = {}
        self.last_detection_start = None
        self.last_detection_end = None
        self.trigger_data = []
        self.detection_run_initialized = False
        self.event_triggered = False
        self.new_event_available = False
        self.compute_max_time_window()


    def init_detection_run(self, stream):
        ''' Initialize a new detection run.
        '''
        self.logger.info('Initializing the detection run.')
        self.trigger_data = []
        if self.detect_stream is not None:
            del self.detect_stream
            self.detect_stream = None

        if len(stream) == 0:
            self.logger.warning("The passed stream contains no traces. Aborting the initialisation of the detection run.")
            self.detection_run_initialized = False
        else:
            self.prepare_detection_stream(stream)
            if self.detect_stream is not None:
                self.tri = self.compute_delaunay_triangulation(self.detect_stations)
                self.edge_length = self.compute_edge_length(stations = self.detect_stations,
                                                            tri = self.tri)
                self.detection_run_initialized = True
            else:
                self.detection_run_initialized = False
            self.logger.debug('detect_stations: %s', self.detect_stations)
            self.logger.debug('tri: %s', self.tri)
            self.logger.debug('edge_length: %s', self.edge_length)
            self.logger.debug('detection_run_initialized: %s', self.detection_run_initialized)


    def compute_max_time_window(self):
        ''' Compute the maximum time window that a wave needs to pass the triangles.
        '''
        tri = self.compute_delaunay_triangulation(self.network_stations)
        edge_length = self.compute_edge_length(self.network_stations, tri)
        if len(edge_length) > 0:
            self.max_time_window = np.max(list(edge_length.values())) / self.p_vel
            self.max_time_window = np.ceil(self.max_time_window)

    def compute_delaunay_triangulation(self, stations):
        x = [stat.x_utm for stat in stations]
        y = [stat.y_utm for stat in stations]
        coords = np.array(list(zip(x, y)))
        try:
            tri = scipy.spatial.Delaunay(coords)
        except Exception:
            self.logger.exception("Error computing the delaunay triangulation.")
            tri = None
        return tri

    def compute_edge_length(self, stations, tri, clean_tri = True):
        x = [stat.x_utm for stat in stations]
        y = [stat.y_utm for stat in stations]
        coords = np.array(list(zip(x, y)))

        edge_length = {}
        valid_simp = []
        if tri is not None:
            for cur_simp in tri.simplices:
                cur_stations = [stations[k] for k in cur_simp]
                cur_key = tuple(sorted([stat.nsl_string for stat in cur_stations]))
                cur_vert = coords[cur_simp]
                cur_dist = [np.linalg.norm(a - b) for a in cur_vert for b in cur_vert]
                cur_max_edge = np.max(cur_dist)
                if clean_tri and cur_max_edge <= self.max_edge_length:
                    edge_length[cur_key] = cur_max_edge
                    valid_simp.append(cur_simp)
                elif not clean_tri:
                    edge_length[cur_key] = cur_max_edge
                    valid_simp.append(cur_simp)

            if clean_tri:
                tri.simplices = valid_simp

        return edge_length


    def prepare_detection_stream(self, stream):
        ''' Prepare the data stream used for the detection run.
        '''
        self.logger.info("passed stream: %s", stream.__str__(extended = True))
        max_end_time = np.max([x.stats.endtime for x in stream])
        # Assume, that all traces have the same sampling rate.
        # TODO: Add a check to validate that the sampling rate of all traces is
        # the same.
        sps = stream[0].stats.sampling_rate
        if self.last_detection_end is None:
            detect_win_start = np.min([x.stats.starttime for x in stream]) + self.max_time_window
            #detect_win_start = np.max([x.stats.starttime for x in stream]) + self.max_time_window
        else:
            detect_win_start = self.last_detection_end + 1 / sps

        min_delta = np.min([x.stats.delta for x in stream])
        detect_win_end = ((max_end_time.timestamp + min_delta) - self.safety_time) // self.window_length * self.window_length
        detect_win_end = obspy.UTCDateTime(detect_win_end) - min_delta
        self.logger.info("max_time_window: %s", self.max_time_window)
        self.logger.info("detect_win_start: %s", detect_win_start)
        self.logger.info("detect_win_end: %s", detect_win_end)

        if detect_win_end <= detect_win_start:
            self.logger.warning("The detection window end is smaller than the detection window start. Skipping the preparation of the detection stream.")
            self.detect_stream = None
        elif (detect_win_end - detect_win_start) < (self.window_length - 1/sps):
            self.logger.warning("The length of the available data is smaller than the detection window length %f. Skipping the preparation of the detection stream.", (self.window_length - 1/sps))
            self.detect_stream = None
        else:
            self.detect_stream = stream.slice(starttime = detect_win_start - self.max_time_window,
                                              endtime = detect_win_end,
                                              nearest_sample = False)
            self.logger.info("detect_stream: %s", self.detect_stream.__str__(extended = True))
            # Set the last detection end time.
            self.last_detection_start = detect_win_start
            self.last_detection_end = detect_win_end

            self.detect_stations = []
            self.unused_stations = []
            for cur_trace in self.detect_stream:
                cur_station = [x for x in self.network_stations if x.name == cur_trace.stats.station]
                if cur_station:
                    self.detect_stations.append(cur_station[0])
            self.unused_stations = [x for x in self.network_stations if x not in self.detect_stations]


    def compute_triangle_max_pgv(self, simp):
        ''' Compute the maximal PGV values of a delaunay triangle.
        '''
        self.logger.info("Computing the triangle max pgv.")
        offset = self.max_time_window
        simp_keys = tuple(sorted([self.detect_stations[x].nsl_string for x in simp]))
        simp_edge_length = self.edge_length[simp_keys]

        self.logger.info("Stations involved: %s.", simp_keys)

        # Compute the length of the search time window using a default velocity
        # of 3500 m/s.
        time_window = simp_edge_length / self.p_vel
        time_window = np.ceil(time_window)

        # Use the minimum time window if the computed time window is smaller
        # than the minimum time window.
        if time_window < self.min_trigger_window:
            self.logger.debug("Time window too small. Edge lengths: %s.",
                              self.edge_length)
            time_window = self.min_trigger_window

        # Split an array into chunks using numpy stride_tricks. This is much
        # faster than using a loop.
        def strided_app(a, length, stepsize):
            ''' Create overlapping subarrays using numpy stride_tricks.

            The strides is the step in bytes in each dimension of the array.

            a: the array to split into subarrays
            length: lenght of the subarrays [samples]
            stepsize: stepsize [samples]
            '''
            nrows = ((a.size - length) // stepsize) + 1
            n = a.strides[0]
            return np.lib.stride_tricks.as_strided(a,
                                                   shape=(nrows, length),
                                                   strides=(stepsize * n, n))

        # Get the stations of the triangle corners.
        simp_stations = [self.detect_stations[x] for x in simp]

        # Select the data of the triangle stations from the detection stream.
        tri_stream = obspy.Stream()
        for cur_station in simp_stations:
            tri_stream = tri_stream + self.detect_stream.select(station = cur_station.name)

        # Trim the stream to equal lengths padding too short streams.
        min_start = np.min([x.stats.starttime for x in tri_stream])
        max_end = np.min([x.stats.endtime for x in tri_stream])
        tri_stream.trim(starttime = min_start,
                        endtime = max_end,
                        pad = True,
                        fill_value = None)

        self.logger.debug("tri_stream: %s", tri_stream)

        pgv = []
        time = []
        for cur_trace in tri_stream:
            self.logger.info("cur_trace: %s", cur_trace)
            self.logger.debug("time_window: %s", time_window)
            cur_win_length = int(np.floor(time_window * cur_trace.stats.sampling_rate))
            cur_offset = int(np.floor(offset * cur_trace.stats.sampling_rate))
            self.logger.debug("cur_offset: %s", cur_offset)
            self.logger.debug("cur_win_length: %d", cur_win_length)
            self.logger.debug("cur_trace.data: %s", cur_trace.data)
            if len(cur_trace.data) < cur_win_length:
                self.logger.error("The data size is smaller than the window length.")
                continue
            # Create overlapping windows with the computed length with 1 sample
            # step size.
            # Handle masked trace data.
            cur_tr_data = cur_trace.data
            if isinstance(cur_tr_data, np.ma.MaskedArray):
                cur_tr_data = cur_tr_data.data
            cur_data = strided_app(cur_tr_data, cur_win_length, 1)
            self.logger.debug("cur_data: %s", cur_data)
            cur_max_pgv = np.max(cur_data, axis = 1)
            self.logger.debug("cur_max_pgv: %s", cur_max_pgv)
            # Set max. PGV sample is assigned to the last time of the
            # computation window. I'm using the past data values to compute the
            # max. PGV of a given time value.
            cur_max_pgv = cur_max_pgv[(cur_offset - cur_win_length + 1):]
            cur_start = cur_trace.stats.starttime + cur_offset * cur_trace.stats.delta
            cur_time = [cur_start + x * cur_trace.stats.delta for x in range(len(cur_max_pgv))]
            pgv.append(cur_max_pgv)
            time.append(cur_time)

        if len(set([len(x) for x in pgv])) == 1:
            pgv = np.array(pgv).transpose()
            time = np.array(time).transpose()[:, 0]
        else:
            self.logger.error("The size of the computed PGV max don't match: %s.", pgv)
            pgv = []
            time = []

        self.logger.debug("pgv: %s", pgv)
        self.logger.debug("time: %s", time)

        return time, pgv, simp_stations

    def compute_trigger_data(self):
        ''' Compute the trigger data for all Delaunay triangles.
        '''
        self.logger.info("Computing the trigger data.")
        self.trigger_data = []
        if self.tri is not None:
            for cur_simp in self.tri.simplices:
                cur_time, cur_pgv, cur_simp_stations = self.compute_triangle_max_pgv(cur_simp)

                if len(cur_pgv) > 0:
                    #if np.any(np.isnan(cur_pgv)):
                    #    self.logger.warning("There is a NaN value in the cur_pgv. Skipping this triangle.")
                    #    self.logger.info("cur_pgv: %s.", cur_pgv)

                    cur_trig = np.min(cur_pgv, axis = 1) >= self.trigger_thr

                    tmp = {}
                    tmp['simplices_stations'] = cur_simp_stations
                    tmp['time'] = cur_time
                    tmp['pgv'] = cur_pgv
                    tmp['trigger'] = cur_trig
                    self.trigger_data.append(tmp)
        else:
            self.logger.warning("The delaunay triangles are not available.")

    def check_for_event_trigger(self):
        ''' Compute if an event trigger is available.
        '''
        trigger_times = []
        trigger_start = None
        trigger_end = None
        for cur_trigger_data in self.trigger_data:
            if np.any(cur_trigger_data['trigger']):
                cur_mask = cur_trigger_data['trigger']
                cur_trigger_start = np.array(cur_trigger_data['time'])[cur_mask][0]
                cur_trigger_end = np.array(cur_trigger_data['time'])[cur_mask][-1]
                cur_trigger_limits = [obspy.UTCDateTime(cur_trigger_start),
                                      obspy.UTCDateTime(cur_trigger_end)]
                # Add the trigger limits to the trigger data for later use.
                cur_trigger_data['trigger_limits'] = cur_trigger_limits
                trigger_times.append(cur_trigger_limits)

        trigger_times = np.array(trigger_times)
        if len(trigger_times) > 0:
            self.logger.debug("trigger_times: %s", trigger_times)
            trigger_start = np.min(trigger_times[:, 0])
            trigger_end = np.max(trigger_times[:, 1])

        return trigger_start, trigger_end

    def evaluate_event_trigger(self):
        ''' Evaluate the event trigger data and declare or update an event.
        '''
        self.logger.info("Evaluating the event trigger.")
        trigger_start, trigger_end = self.check_for_event_trigger()
        self.logger.info("trigger_start: %s", trigger_start)
        self.logger.info("trigger_end: %s", trigger_end)

        if not self.event_triggered and trigger_start is not None:
            # A new event has been triggered.
            self.logger.info("New Event triggered.")
            try:
                cur_event = event_core.Event(start_time = trigger_start,
                                             end_time = trigger_end,
                                             author_uri = self.author_uri,
                                             agency_uri = self.agency_uri)

                # Compute the max. PGV of each triggered station.
                for cur_data in [x for x in self.trigger_data if np.any(x['trigger'])]:
                    # Create a detection instance.
                    cur_mask = cur_data['trigger']
                    max_pgv = np.max(cur_data['pgv'][cur_mask], axis = 0)
                    cur_simp_stations = cur_data['simplices_stations']
                    cur_detection = event_detection.Detection(start_time = cur_data['trigger_limits'][0],
                                                              end_time = cur_data['trigger_limits'][1],
                                                              stations = cur_simp_stations,
                                                              max_pgv = {cur_simp_stations[0].nsl_string: max_pgv[0],
                                                                         cur_simp_stations[1].nsl_string: max_pgv[1],
                                                                         cur_simp_stations[2].nsl_string: max_pgv[2]},
                                                              author_uri = self.author_uri,
                                                              agency_uri = self.agency_uri)
                    cur_event.add_detection(cur_detection)

                self.current_event = cur_event
                self.current_event.detection_state = 'new'
                self.event_triggered = True
                self.new_event_available = True
            except Exception:
                self.logger.exception("Error processing the event trigger.")
                self.event_triggered = False
                self.new_event_available = False

        elif self.event_triggered and trigger_start is not None:
            # A trigger occured during an existing event.
            self.logger.info("Updating an existing event.")
            self.current_event.end_time = trigger_end
            self.current_event.detection_state = 'updated'


            for cur_data in [x for x in self.trigger_data if np.any(x['trigger'])]:
                cur_simp_stations = cur_data['simplices_stations']
                cur_mask = cur_data['trigger']
                max_pgv = np.max(cur_data['pgv'][cur_mask], axis = 0)
                if self.current_event.has_detection(cur_simp_stations):
                    # Update the detection.
                    cur_detection = self.current_event.get_detection(cur_simp_stations)
                    if len(cur_detection) == 1:
                        cur_detection = cur_detection[0]
                        cur_detection.update(end_time = cur_data['trigger_limits'][1],
                                             max_pgv = {cur_simp_stations[0].nsl_string: max_pgv[0],
                                                        cur_simp_stations[1].nsl_string: max_pgv[1],
                                                        cur_simp_stations[2].nsl_string: max_pgv[2]})
                    else:
                        self.logger.error("Expected exactly one detection. Got: %s.", cur_detection)
                else:
                    # Add the detection.
                    cur_detection = event_detection.Detection(start_time = cur_data['trigger_limits'][0],
                                                              end_time = cur_data['trigger_limits'][1],
                                                              stations = cur_simp_stations,
                                                              max_pgv = {cur_simp_stations[0].nsl_string: max_pgv[0],
                                                                         cur_simp_stations[1].nsl_string: max_pgv[1],
                                                                         cur_simp_stations[2].nsl_string: max_pgv[2]},
                                                              author_uri = self.author_uri,
                                                              agency_uri = self.agency_uri)
                    self.current_event.add_detection(cur_detection)

        # Check if the event has to be closed because the time from the event
        # end to the currently processed time is large than the keep listening
        # time.
        if self.event_triggered:
            keep_listening = self.max_time_window

            # Add the PGV stream to the event pgv stream.
            self.current_event.pgv_stream += copy.deepcopy(self.detect_stream)
            self.current_event.pgv_stream.merge()

            # Add the trigger data to the event.
            cur_key = self.last_detection_end.isoformat()
            self.current_event.detection_data[cur_key] = {}
            self.current_event.detection_data[cur_key]['detection_start'] = self.last_detection_start
            self.current_event.detection_data[cur_key]['detection_end'] = self.last_detection_end
            self.current_event.detection_data[cur_key]['trigger_data'] = self.trigger_data
            self.current_event.detection_data[cur_key]['detect_stations'] = self.detect_stations
            self.current_event.detection_data[cur_key]['unused_stations'] = self.unused_stations

            if (self.last_detection_end - self.current_event.end_time) > keep_listening:
                self.logger.info("Closing an event.")
                self.event_triggered = False
                self.current_event.detection_state = 'closed'
            else:
                self.logger.info("Event is over, but keep_listening: %s", keep_listening)


    def get_event(self):
        ''' Return a copy of the current event.
        '''
        return copy.copy(self.current_event)


    def run_detection(self, stream):
        ''' Run the event detection using the passed stream.

        The detection algrithm is run using the passed stream and the state and
        data of the current event is updated.
        '''
        pass
