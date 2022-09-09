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
        
        # Check if the event is a quarry blast.
        classification = self.test_for_quarry_blast()

        # Write the classificaton to the database.
        if len(classification) == 1:
            event_type = classification[0]['event_type']
            event_region = classification[0]['event_region']
            self.event.set_event_type(event_type)
            tag = 'region:{}'.format(event_region)
            if tag not in self.event.tags:
                self.event.tags.append(tag)
            tag = 'mode:automatic'
            if tag not in self.event.tags:
                self.event.tags.append(tag)
            self.event.write_to_database(self.project)
            


    def test_for_quarry_blast(self):
        ''' Test if the event is a quarry blast.
        '''
        # Test for quarry blasts of the quarry
        quarries = {'duernbach': {'nearest_station': 'MSSNet:DUBA:00',
                                  'dist_thr': 3000,
                                  'region': 'Dürnbach'}}

        classification = []

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

            if (sorted_pgv[0][0]) == nearest_station.nsl_string:
                max_on_nearest = True

            # Check if DUBA was among the first triggered stations.
            min_time_on_nearest = False
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
                       'event_region': event_region}
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

        return classification


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
        
