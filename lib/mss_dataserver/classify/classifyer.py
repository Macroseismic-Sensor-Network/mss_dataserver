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
# Copyright 2021 Stefan Mertl
##############################################################################
''' The event classifyer.
'''

import logging

import numpy as np
import obspy
import scipy.spatial
import scipy.stats


class EventClassifyer(object):
    ''' Classify an event.

    '''

    def __init__(self, public_id, meta, pgv_df, project,
                 event = None, event_types = None):
        ''' Initialize the instance.
        '''
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # The public id of the event to classify.
        self.public_id = public_id

        # The metadata of the event to classify.
        self.meta = meta

        # The pgv data of the event.
        self.pgv_df = pgv_df

        # The related project.
        self.project = project

        # The station inventory.
        self.inventory = self.project.inventory

        # The event instance to classify.
        self.event = event

        # The available event types.
        self.event_types = event_types


    def classify(self):
        ''' Run the event classification.
        '''
        msg = 'Classifying event {}.'.format(self.public_id)
        self.logger.info(msg)

        self.classification = []
        
        # Check if the event is a quarry blast.
        if len(self.classification) == 0:
            self.logger.info('Testing for quarry blast.')
            self.test_for_quarry_blast()

        # Check if the event is a quarry blast in Hainburg.
        # This quarry needs special treatment because of the
        # bad station coverage in the area.
        if len(self.classification) == 0:
            self.logger.info('Testing for quarry blast Hainburg.')
            self.test_for_quarry_blast_hainburg()

        # Check if the event is noise.
        if len(self.classification) == 0:
            self.logger.info('Testing for noise.')
            self.test_for_noise()

        # Check if the event is an earthquake.
        if len(self.classification) == 0:
            self.logger.info('Testing for earthquake.')
            self.test_for_earthquake()

        self.logger.info(self.classification)

        # Write the classificaton to the database.
        if len(self.classification) == 1:
            classification = self.classification[0]
            event_type = classification['event_type']

            # Set the event classification if an event instance
            # is available.
            if self.event is not None:
                self.event.set_event_type(event_type)
                tag = 'automatic'
                if tag not in self.event.tags:
                    self.event.tags.append(tag)

                if 'event_region' in classification:
                    region = classification['event_region']
                    tag = 'class_region:' + region.strip()
                    if tag not in self.event.tags:
                        self.event.tags.append(tag)

                if 'max_station' in classification:
                    max_station = classification['max_station']
                    tag = 'class_maxstat:' + max_station.strip()
                    if tag not in self.event.tags:
                        self.event.tags.append(tag)

            ret_val = classification['event_type']
        else:
            ret_val = None

        return ret_val


    def test_for_quarry_blast(self):
        ''' Test if the event is a quarry blast.
        '''
        # Test for quarry blasts of the quarry
        quarries = {'duernbach': {'nearest_station': 'MSSNet:DUBA:00',
                                  'dist_thr': 3000,
                                  'region': 'Steinbruch Dürnbach'}}

        classification = self.classification

        meta = self.meta
        blast_event_type = [x for x in self.event_types[0].event_types if x.name == 'blast']
        blast_event_type = blast_event_type[0]

        for key, quarry in quarries.items():
            stat_nsl = quarry['nearest_station']
            dist_thr = quarry['dist_thr']
            nearest_station = self.inventory.get_station(nsl_string = stat_nsl)
            nearest_station = nearest_station[0]
            neighbor_stations = self.compute_neighbor_stations(ref_station = nearest_station,
                                                               dist_thr = dist_thr)

            # Check if the nearest station is among the triggered stations.
            recorded_on_nearest = False
            if nearest_station.nsl_string in meta['max_event_pgv'].keys():
                recorded_on_nearest = True

            # Check if the neighboring stations have been triggered.
            neighbors_triggered = False
            triggered_stations = list(meta['max_event_pgv'].keys())
            in_triggered_stations = [x.nsl_string in triggered_stations for x in neighbor_stations]

            if np.all(in_triggered_stations):
                neighbors_triggered = True

            # Check if the max. PGV was recorded at the nearest station.
            max_on_nearest = False
            sorted_pgv = sorted(meta['max_event_pgv'].items(),
                                key = lambda item: item[1],
                                reverse = True)
            max_station_nsl = sorted_pgv[0][0]

            if (sorted_pgv[0][0]) == nearest_station.nsl_string:
                max_on_nearest = True

            # Check if DUBA was among the first triggered stations.
            min_time_on_nearest = False
            det_time_nearest = None
            sorted_trigger = sorted(meta['detection_limits'].items(),
                                    key = lambda item: item[1][0])
            min_detection_time = sorted_trigger[0][1][0]
            if nearest_station.nsl_string in meta['detection_limits'].keys():
                det_time_nearest = meta['detection_limits'][nearest_station.nsl_string][0]

            if det_time_nearest == min_detection_time:
                min_time_on_nearest = True

            event_type = None
            event_region = None
            res = None
            if (recorded_on_nearest and neighbors_triggered and max_on_nearest and min_time_on_nearest):
                event_type = blast_event_type
                event_region = quarry['region']
                res = {'event_type': event_type,
                       'event_region': event_region,
                       'max_station': max_station_nsl}
                classification.append(res)

            # Create the flags dictionary.
            flags = {'recorded_on_nearest': recorded_on_nearest,
                     'neighbors_triggered': neighbors_triggered,
                     'max_on_nearest': max_on_nearest,
                     'min_time_on_nearest': min_time_on_nearest}

            self.logger.debug('Tested for quarry blast: %s; %s',
                              key, quarry)
                              
            self.logger.debug('Classification result: %s',
                              res)
            self.logger.debug('Classification flags: %s',
                              flags)

        self.classification = classification

    
    def test_for_quarry_blast_hainburg(self):
        ''' Test for a quarry blast in Hainburg.

        '''
        project = self.project
        meta = self.meta
        blast_event_type = [x for x in self.event_types[0].event_types if x.name == 'blast']
        blast_event_type = blast_event_type[0]
        classification = self.classification
        
        # The nearest station to the quarry blast location.
        nearest_stations_nsl = ['MSSNet:HAMG:00',
                                'MSSNet:HAHN:00',
                                'MSSNet:DABU:00']
        
        nearest_stations = []
        for cur_nsl in nearest_stations_nsl:
            cur_stat = project.inventory.get_station(nsl_string = cur_nsl)
            nearest_stations.extend(cur_stat)

        all_stations = project.inventory.get_station()
        ref_station = nearest_stations[0]
        for cur_station in all_stations:
            ref_coord = [ref_station.x_utm,
                         ref_station.y_utm]
            comp_coord = [cur_station.x_utm,
                          cur_station.y_utm]
            ref_coord = np.array(ref_coord)
            comp_coord = np.array(comp_coord)
            cur_dist = np.linalg.norm(ref_coord - comp_coord)
            cur_station.rel_dist = cur_dist

        # Check if the nearest_stations are among the triggered stations.
        tmp = []
        for cur_stat in nearest_stations:
            found_match = False
            if cur_stat.nsl_string in meta['max_event_pgv'].keys():
                found_match = True
            tmp.append(found_match)
        recorded_on_nearest = np.all(tmp)

        # Check if the max. PGV was recorded at one of the nearest stations.
        max_on_nearest = False
        sorted_pgv = sorted(meta['max_event_pgv'].items(),
                            key = lambda item: item[1],
                            reverse = True)
        max_station_nsl = sorted_pgv[0][0]

        if (sorted_pgv[0][0]) in nearest_stations_nsl:
            max_on_nearest = True

        # Check if all nearest stations are above a pgv threshold.
        thr = 0.1e-3
        nearest_above_thr = False
        if recorded_on_nearest:
            nearest_pgv = [meta['max_event_pgv'][x] for x in nearest_stations_nsl]
            nearest_pgv = np.array(nearest_pgv)
            nearest_above_thr = np.all(nearest_pgv >= thr)

        if (recorded_on_nearest and max_on_nearest and nearest_above_thr):
            event_type = blast_event_type
            event_region = 'Steinbruch Pfaffenberg'
            res = {'event_type': event_type,
                   'event_region': event_region,
                   'max_station': max_station_nsl}
            classification.append(res)

        self.classification = classification


    def test_for_earthquake(self):
        ''' Classify earthquake signals.
        '''
        pgv_df = self.pgv_df
        project = self.project
        quake_event_type = [x for x in self.event_types[0].event_types if x.name == 'earthquake']
        quake_event_type = quake_event_type[0]
        quake_inside = quake_event_type.get_child('inside network')
        quake_outside = quake_event_type.get_child('outside network')
        classification = self.classification

        # Get the triggered data.
        triggered_mask = pgv_df['triggered']
        pgv_triggered_df = pgv_df[triggered_mask]

        # Get the number of triggered stations.
        n_triggered_stations = len(pgv_triggered_df)

        n_stations_thr = 10
        # Test for enough triggered stations.
        if n_triggered_stations < n_stations_thr:
            return
        
        # Get the station coordinates.
        stat_coords = pgv_triggered_df[['x_utm', 'y_utm']].values

        # Apply the station correction.
        pgv_corr = (pgv_triggered_df['pgv']/pgv_triggered_df['sa']).values

        # Compute a preliminary epicenter.
        percentile = 90
        p_epi = np.percentile(pgv_corr, percentile)
        mask_p_epi = pgv_corr >= p_epi
        p_epi_stat_coord = stat_coords[mask_p_epi, :]
        epi_prelim = np.median(p_epi_stat_coord,
                               axis = 0)

        # Compute the amplitude decay for the preliminary epicenter.
        prelim_epidist = np.sqrt(np.sum((epi_prelim - stat_coords)**2,
                                        axis = 1))
        zero_mask = np.isclose(prelim_epidist, 0)
        prelim_epidist[zero_mask] = 100
        # Linear regression of the pgv data.
        linreg = scipy.stats.linregress(np.log10(prelim_epidist),
                                        np.log10(pgv_corr))

        linreg_fits = False
        if (linreg.rvalue <= -0.75) and (linreg.slope <= -1.2 and linreg.slope >= -2.8):
            linreg_fits = True

        res = {}
        if linreg_fits:
            res = {'event_type': quake_inside}
        else:
            res = {'event_type': quake_outside}

        if res:
            classification.append(res)

        self.classification = classification


    def test_for_noise(self):
        ''' Classify noise signals.
        '''
        meta = self.meta
        pgv_df = self.pgv_df
        project = self.project
        noise_event_type = [x for x in self.event_types[0].event_types if x.name == 'noise']
        noise_event_type = noise_event_type[0]
        classification = self.classification
        
        # Get the station with the maximum pgv value.
        sorted_pgv = sorted(meta['max_event_pgv'].items(),
                            key = lambda item: item[1],
                            reverse = True)
        max_station_nsl = sorted_pgv[0][0]
        max_station = project.inventory.get_station(nsl_string = max_station_nsl)
        max_station = max_station[0]

        # Get the triggered stations.
        triggered_stations = []
        for cur_nsl in meta['max_event_pgv'].keys():
            cur_stat = project.inventory.get_station(nsl_string = cur_nsl)
            triggered_stations.extend(cur_stat)

        # Compute the relative distances to the reference station.
        ref_station = max_station
        for cur_station in triggered_stations:
            ref_coord = [ref_station.x_utm,
                         ref_station.y_utm]
            comp_coord = [cur_station.x_utm,
                          cur_station.y_utm]
            ref_coord = np.array(ref_coord)
            comp_coord = np.array(comp_coord)
            cur_dist = np.linalg.norm(ref_coord - comp_coord)
            cur_station.rel_dist = cur_dist

        # Get the voronoi cell neighbors of the reference station.
        stat_coords = pgv_df[['x_utm', 'y_utm']].values
        tri = scipy.spatial.Delaunay(stat_coords)
        indptr, indices = tri.vertex_neighbor_vertices

        ref_ind = (pgv_df['nsl'] == ref_station.nsl_string).values
        ref_ind = np.argwhere(ref_ind)
        ref_ind = ref_ind[0][0]
        neighbor_ind = indices[indptr[ref_ind]:indptr[ref_ind + 1]]
        neighbor_nsl = pgv_df.iloc[neighbor_ind]['nsl'].values
        neighbor_coords = pgv_df.iloc[neighbor_ind][['x_utm', 'y_utm']].values

        # Compute the triggered neighbors.
        triggered_nsl = [x.nsl_string for x in triggered_stations]
        triggered_neighbors = [x for x in triggered_nsl if x in neighbor_nsl]
        n_triggered_neighbors = len(triggered_neighbors)

        n_stations = len(meta['max_event_pgv'])

        # The maximum number of detected stations.
        n_stations_thr = 6
        
        # The minimum number of triggered neighbors.
        n_neighbors_thr = 2
        
        # The minimum event length [s].
        event_length_thr = 6
        
        # The maximum allowed distance of the farthest station [m].
        neighbor_dist = np.sqrt(np.sum((ref_coord - neighbor_coords)**2,
                                       axis = 1))
        stat_dist_thr = np.mean(neighbor_dist) + np.std(neighbor_dist)

        is_noise = False
        if n_stations <= n_stations_thr:
            event_start = obspy.UTCDateTime(meta['start_time'])
            event_end = obspy.UTCDateTime(meta['end_time'])
            event_length = event_end - event_start
            rel_dist = sorted([x.rel_dist for x in triggered_stations])
            max_rel_dist = np.max(rel_dist)
            
            # Find neighboring stations using the delaunay network
            stat_coords = pgv_df[['x_utm', 'y_utm']].values
            tri = scipy.spatial.Delaunay(stat_coords)
            indptr, indices = tri.vertex_neighbor_vertices
            
            ref_ind = (pgv_df['nsl'] == ref_station.nsl_string).values
            ref_ind = np.argwhere(ref_ind)
            ref_ind = ref_ind[0][0]
            neighbor_ind = indices[indptr[ref_ind]:indptr[ref_ind + 1]]
            neighbor_nsl = pgv_df.iloc[neighbor_ind]['nsl'].values
            triggered_nsl = [x.nsl_string for x in triggered_stations]
            triggered_neighbors = [x for x in triggered_nsl if x in neighbor_nsl]
            n_triggered_neighbors = len(triggered_neighbors)
            
            if n_triggered_neighbors <= n_neighbors_thr and event_length >= event_length_thr:
                is_noise = True
            elif n_triggered_neighbors <= n_neighbors_thr and max_rel_dist >= stat_dist_thr:
                is_noise = True

        if is_noise:
            event_region = max_station.description
            res = {'event_type': noise_event_type,
                   'event_region': event_region,
                   'max_station': max_station_nsl}
            classification.append(res)

        self.classification = classification
            
        
        
    def compute_station_dist(self, ref_station):
        ''' Compute the distance relative to a reference station.
        '''
        all_stations = self.inventory.get_station()

        for cur_station in all_stations:
            ref_coord = [ref_station.x_utm,
                         ref_station.y_utm]
            comp_coord = [cur_station.x_utm,
                          cur_station.y_utm]
            ref_coord = np.array(ref_coord)
            comp_coord = np.array(comp_coord)
            cur_dist = np.linalg.norm(ref_coord - comp_coord)
            cur_station.rel_dist = cur_dist

            
    def compute_neighbor_stations(self, ref_station, dist_thr = 3000):
        ''' Compute the neighbor stations to a reference station.
        '''
        self.compute_station_dist(ref_station = ref_station)
        all_stations = self.inventory.get_station()
        sorted_stations = sorted(all_stations,
                                 key = lambda stat: stat.rel_dist)
        neighbor_stations = [x for x in sorted_stations if x.rel_dist <= dist_thr]
        
        # Remove neighbor stations with no data.
        no_data_nsl = self.pgv_df[self.pgv_df['pgv'].isna()]['nsl']
        no_data_nsl = no_data_nsl.values
        neighbor_stations = [x for x in neighbor_stations if x.nsl_string not in no_data_nsl]

        return neighbor_stations
        
