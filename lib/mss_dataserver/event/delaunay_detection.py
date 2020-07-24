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


import numpy as np
import obspy
import scipy
import scipy.spatial

class DelaunayDetector(object):
    ''' Event detection using amplitudes and Delaunay Triangulation.

    '''
    def __init__(self, network_stations,
                 p_vel = 3500,
                 min_trigger_window = 2,
                 max_edge_length = 40000):
        ''' Initialization of the instance.
        '''
        # All available network stations.
        self.network_stations = network_stations

        # The p wave velocity [m/s].
        self.p_vel = p_vel

        # The length of the minimum trigger window [s].
        self.min_trigger_window = min_trigger_window

        # The maximum edge length of a triangle used for the detection [m].
        self.max_edge_length = max_edge_length

        # The currently detected event.
        self.current_event = None

        # The stations used for the event detection.
        self.detect_stations = []

        # The delaunay triangulation result.
        self.tri = None

        # The triangle max. edge lenghts [m].
        self.edge_length = []

        # The maximum maximum time that a wave needs to pass the triangles [s].
        self.max_time_window = None

    def init_detection_run(self, stations):
        self.detect_stations = stations
        self.tri = None
        self.edge_length = []

    def compute_max_time_window(self):
        ''' Compute the maximum time window that a wave needs to pass the triangles.
        '''
        tri = self.compute_delaunay_triangulation(self.network_stations)
        edge_length = self.compute_edge_length(self.network_stations, tri)
        if len(edge_length) > 0:
            self.max_time_window = np.max(edge_length) / self.p_vel
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

    def compute_edge_length(self, stations, tri):
        x = [stat.x_utm for stat in stations]
        y = [stat.y_utm for stat in stations]
        coords = np.array(list(zip(x, y)))

        dist = []
        for cur_simp in tri.simplices:
            cur_vert = coords[cur_simp]
            cur_dist = [np.linalg.norm(a - b) for a in cur_vert for b in cur_vert]
            dist.append(cur_dist)
        edge_length = np.max(dist, axis=1)
        return edge_length

    def compute_max_pgv(self, stream, stations, edge_lengths, offset):
        ''' Compute the maximal PGV values of the delaunay-triangles.
        '''
        # Compute the length of the search time window using a default velocity
        # of 3500 m/s.
        time_window = np.max(edge_lengths) / 3500
        time_window = np.ceil(time_window)

        # Use the minimum time window if the computed time window is smaller
        # than the minimum time window.
        if time_window < self.min_trigger_window:
            self.logger.debug("Time window too small. Edge lengths: %s.",
                              edge_lengths)
            time_window = self.min_trigger_window

        # Get the data for the stations used to compute the max. PGV.
        tri_stream = obspy.Stream()
        for cur_station in stations:
            tri_stream = tri_stream + stream.select(station = cur_station.name)
        self.logger.debug("compute_max_pgv:  tri_stream: %s.", tri_stream)

        # Split an array into chunks using numpy stride_tricks. This is much
        # faster than using a loop.
        def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
            nrows = ((a.size-L)//S)+1
            n = a.strides[0]
            return np.lib.stride_tricks.as_strided(a,
                                                   shape=(nrows, L),
                                                   strides=(S*n, n))

        pgv = []
        time = []
        for cur_trace in tri_stream:
            self.logger.debug("cur_trace.id: %s", cur_trace.id)
            cur_win_length = int(np.floor(time_window / cur_trace.stats.sampling_rate))
            cur_offset = int(np.floor(offset / cur_trace.stats.sampling_rate))
            self.logger.debug("cur_offset: %s", cur_offset)
            self.logger.debug("cur_win_length: %d", cur_win_length)
            self.logger.debug("cur_trace.data: %s", cur_trace.data)
            if len(cur_trace.data) < cur_win_length:
                self.logger.error("The data size is smaller than the window length.")
                continue
            cur_data = strided_app(cur_trace.data, cur_win_length,  1)
            self.logger.debug("cur_data: %s", cur_data)
            cur_max_pgv = np.max(cur_data, axis = 1)
            self.logger.debug("cur_max_pgv: %s", cur_max_pgv)
            # TODO: The compute_interval variable seems useless. It is a relict
            # of the old code using a non-strided version. Check if the
            # selection of the PGV values and the pgv sample times are correct
            # without the compute_interval.
            #cur_max_pgv = cur_max_pgv[(cur_offset - cur_win_length + compute_interval):]
            #cur_start = cur_trace.stats.starttime + (cur_offset - cur_win_length + compute_interval + cur_win_length - 1) * cur_trace.stats.delta
            #cur_time = [cur_start + x * compute_interval for x in range(len(cur_max_pgv))]
            cur_max_pgv = cur_max_pgv[(cur_offset - cur_win_length):]
            cur_start = cur_trace.stats.starttime + cur_offset * cur_trace.stats.delta
            cur_time = [cur_start + x for x in range(len(cur_max_pgv))]
            pgv.append(cur_max_pgv)
            time.append(cur_time)

        # Delete unused instances.
        del tri_stream

        if len(set([len(x) for x in pgv])) == 1:
            pgv = np.array(pgv).transpose()
            time = np.array(time).transpose()[:, 0]
        else:
            self.logger.error("The size of the computed PGV max don't match: %s.", pgv)
            pgv = []
        self.logger.debug("pgv: %s", pgv)
        self.logger.debug("time: %s", time)

        return time, pgv

    def run_detection(self, stream):
        ''' Run the event detection using the passed stream.

        The detection algrithm is run using the passed stream and the state and
        data of the current event is updated.
        '''
        pass
