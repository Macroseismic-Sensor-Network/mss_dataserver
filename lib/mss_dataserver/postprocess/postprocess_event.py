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

import csv
import logging
import os

import geopandas as gpd
import numpy as np
import shapely

import mss_dataserver.postprocess.util as util
import mss_dataserver.postprocess.voronoi as voronoi


class EventPostProcessor(object):
    ''' Process a detected event.
    '''

    def __init__(self, project):
        ''' Initialize the instance.
        '''
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)


        # mss_dataserver project.
        self.project = project

        # The author information.
        self.author_uri = self.project.author_uri
        self.agency_uri = self.project.agency_uri

        # The event to process.
        self.event = None

        # The supplement directory.
        self.supplement_dir = self.project.config['output']['event_dir']

        # The map data directory.
        self.map_dir = self.project.config['postprocess']['map_dir']

        # The common data directory.
        self.data_dir = self.project.config['postprocess']['data_dir']

        # Compute the UTM coordinates of the stations.
        self.project.inventory.compute_utm_coordinates()

        # Load common data.
        self.network_boundary = self.load_network_boundary()
        self.station_amp = self.load_station_amplification()
        self.sup_map = util.get_supplement_map()


    def set_event(self, public_id):
        ''' Set the event to process.
        '''
        self.event = self.project.load_event_by_id(public_id = public_id)

    def load_network_boundary(self):
        ''' Load the boundary of the MSS network.
        '''
        # Load the MSS boundary.
        boundary_filename = self.project.config['postprocess']['boundary_filename']
        boundary_filepath = os.path.join(self.map_dir,
                                         boundary_filename)
        return gpd.read_file(boundary_filepath)

    def load_station_amplification(self):
        ''' Load the station amplification data.
        '''
        sa_filename = self.project.config['postprocess']['station_amplification_filename']
        filepath = os.path.join(self.data_dir,
                                sa_filename)
        station_amp = {}
        with open(filepath, 'r') as csv_file:
            reader = csv.DictReader(csv_file)

            for cur_row in reader:
                cur_serial = cur_row['serial']
                cur_amp = float(cur_row['amplification'])
                cur_snl = cur_row['snl']
                cur_snl = cur_snl.split(':')
                cur_nsl = ':'.join([cur_snl[1], cur_snl[0], cur_snl[2]])
                station_amp[cur_nsl] = {'serial': cur_serial,
                                        'amp': cur_amp}

        return station_amp

    def add_station_amplification(self, df):
        ''' Apply the station amplification values to a dataframe.
        '''
        station_amp = self.station_amp
        sorted_sa = [station_amp[row.nsl]['amp'] if row.nsl in station_amp.keys() else np.nan for index, row in df.iterrows()]
        df['sa'] = sorted_sa

    def compute_pgv_df(self, meta):
        ''' Create a dataframe of pgv values with station coordinates.
        '''
        inventory = self.project.inventory
        triggered_nsl = list(meta['max_event_pgv'].keys())
        untriggered_nsl = list(meta['max_network_pgv'].keys())
        untriggered_nsl = [x for x in untriggered_nsl if x not in triggered_nsl]
        nodata_nsl = [x.nsl_string for x in inventory.get_station() if (x.nsl_string not in triggered_nsl and x.nsl_string not in untriggered_nsl)]

        pgv_data = []
        for cur_nsl in triggered_nsl:
            cur_station = inventory.get_station(nsl_string = cur_nsl)[0]
            cur_pgv = meta['max_event_pgv'][cur_nsl]
            cur_trigger = True
            cur_data = [cur_nsl, cur_station.x, cur_station.y,
                        cur_station.x_utm, cur_station.y_utm,
                        cur_pgv, cur_trigger]
            pgv_data.append(cur_data)

        for cur_nsl in untriggered_nsl:
            cur_station = inventory.get_station(nsl_string = cur_nsl)[0]
            cur_pgv = meta['max_network_pgv'][cur_nsl]
            cur_trigger = False
            cur_data = [cur_nsl, cur_station.x, cur_station.y,
                        cur_station.x_utm, cur_station.y_utm,
                        cur_pgv, cur_trigger]
            pgv_data.append(cur_data)

        for cur_nsl in nodata_nsl:
            cur_station = inventory.get_station(nsl_string = cur_nsl)[0]
            cur_pgv = None
            cur_trigger = False
            cur_data = [cur_nsl, cur_station.x, cur_station.y,
                        cur_station.x_utm, cur_station.y_utm,
                        cur_pgv, cur_trigger]
            pgv_data.append(cur_data)

        x_coord = [x[1] for x in pgv_data]
        y_coord = [x[2] for x in pgv_data]
        pgv_data = {'geom_stat': [shapely.geometry.Point([x[0], x[1]]) for x in zip(x_coord, y_coord)],
                    'geom_vor': [shapely.geometry.Polygon([])] * len(pgv_data),
                    'nsl': [x[0] for x in pgv_data],
                    'x': x_coord,
                    'y': y_coord,
                    'x_utm': [x[3] for x in pgv_data],
                    'y_utm': [x[4] for x in pgv_data],
                    'pgv': [x[5] for x in pgv_data],
                    'triggered': [x[6] for x in pgv_data]}

        df = gpd.GeoDataFrame(data = pgv_data,
                              crs = 'epsg:4326',
                              geometry = 'geom_stat')

        return df


    def compute_event_metadata_supplement(self):
        ''' Compute the supplement data based on the event metadata.
        '''
        # Load the event metadata from the supplement file.
        meta = util.get_supplement_data(self.event.public_id,
                                        category = 'detectiondata',
                                        name = 'metadata',
                                        directory = self.supplement_dir)
        meta = meta['metadata']

        # Compute a PGV geodataframe using the event metadata.
        pgv_df = self.compute_pgv_df(meta)
        self.add_station_amplification(pgv_df)

        # Compute the voronoi cells and add them as a geometry to the dataframe.
        # Compute the boundary clipping using shapely because geopandas clipping 
        # throws exceptions e.g. because of None values.
        voronoi.compute_voronoi_geometry(pgv_df,
                                         boundary = self.network_boundary.loc[0, 'geometry'])

        # Get some event properties to add to the properties of the feature collections.
        props = {'db_id': meta['db_id'],
                 'event_start': meta['start_time'].isoformat(),
                 'event_end': meta['end_time'].isoformat(),
                 'author_uri': self.project.author_uri,
                 'agency_uri': self.project.agency_uri}

        filepath = util.save_df_to_geojson(self.event.public_id,
                                           pgv_df.loc[:, ['geom_stat', 'nsl',
                                                          'pgv', 'sa', 'triggered']],
                                           output_dir = self.supplement_dir,
                                           name = self.sup_map['eventpgv']['pgvstation']['name'],
                                           subdir = self.sup_map['eventpgv']['pgvstation']['subdir'],
                                           props = props)
        self.logger.info('Saved station pgv to file %s.', filepath)


    def compute_pgv_sequence_supplement(self):
        ''' Compute the supplement data representing the PGV sequence.
        '''
        # Load the event metadata from the supplement file.
        meta = util.get_supplement_data(self.event.public_id,
                                        category = 'detectiondata',
                                        name = 'metadata',
                                        directory = self.supplement_dir)
        meta = meta['metadata']

        # Load the PGV data stream.
        pgv_stream = util.get_supplement_data(self.event.public_id,
                                                  category = 'detectiondata',
                                                  name = 'pgv',
                                                  directory = self.supplement_dir)

        # Trim the stream.
        pgv_stream.trim(starttime = meta['start_time'] - 6,
                        endtime = meta['end_time'] + 6,
                        pad = True)

        inventory = self.project.inventory

        station_nsl = [('MSSNet', x.stats.station, x.stats.location) for x in pgv_stream]
        station_nsl = [':'.join(x) for x in station_nsl]
        stations = [inventory.get_station(nsl_string = x)[0] for x in station_nsl]
        times = pgv_stream[0].times("utcdatetime")
        data = np.array([x.data for x in pgv_stream]).transpose()

        # Get the stations with no available data.
        available_stations = inventory.get_station()
        no_data_stations = [x for x in available_stations if x.nsl_string not in station_nsl]

        detection_limits = meta['detection_limits']

        sequence_df = None

        for k in range(len(times)):
            cur_time = times[k]
            triggered = []
            for cur_station in stations:
                if cur_station.nsl_string not in detection_limits.keys():
                    cur_trigger = False
                else:
                    cur_detection_limit = detection_limits[cur_station.nsl_string]
                    if cur_time >= cur_detection_limit[0] and cur_time <= cur_detection_limit[1]:
                        cur_trigger = True
                    else:
                        cur_trigger = False
                triggered.append(cur_trigger)

            cur_points = [shapely.geometry.Point(x.x, x.y) for x in stations]
            cur_df = gpd.GeoDataFrame({'geom_vor': [shapely.geometry.Polygon([])] * len(stations),
                                       'geom_stat': cur_points,
                                       'time': [cur_time.isoformat()] * len(stations),
                                       'nsl': [x.nsl_string for x in stations],
                                       'x': [x.x for x in stations],
                                       'y': [x.y for x in stations],
                                       'x_utm': [x.x_utm for x in stations],
                                       'y_utm': [x.y_utm for x in stations],
                                       'pgv': data[k, :],
                                       'triggered': triggered},
                                      crs = "epsg:4326",
                                      geometry = 'geom_stat')

            # Add the station amplification factors.
            self.add_station_amplification(cur_df)

            # Compute the voronoi cells and add them as a geometry to the dataframe.
            # Compute the boundary clipping using shapely because geopandas clipping 
            # throws exceptions e.g. because of None values.
            voronoi.compute_voronoi_geometry(cur_df,
                                             boundary = self.network_boundary.loc[0, 'geometry'])

            # Add the no-data stations
            cur_nd_points = [shapely.geometry.Point(x.x, x.y) for x in no_data_stations]
            cur_nd_df = gpd.GeoDataFrame({'geom_vor': [shapely.geometry.Polygon([])] * len(no_data_stations),
                                          'geom_stat': cur_nd_points,
                                          'time': [cur_time.isoformat()] * len(no_data_stations),
                                          'nsl': [x.nsl_string for x in no_data_stations],
                                          'x': [x.x for x in no_data_stations],
                                          'y': [x.y for x in no_data_stations],
                                          'x_utm': [x.x_utm for x in no_data_stations],
                                          'y_utm': [x.y_utm for x in no_data_stations],
                                          'pgv': [None] * len(no_data_stations),
                                          'triggered': [None] * len(no_data_stations)},
                                         crs = "epsg:4326",
                                         geometry = 'geom_stat')

            # Add the station amplification factors to the no-data stations.
            self.add_station_amplification(cur_nd_df)

            # Append the no-data stations.
            cur_df = cur_df.append(cur_nd_df)

            # Add the dataframe to the sequence.
            if sequence_df is None:
                sequence_df = cur_df
            else:
                sequence_df = sequence_df.append(cur_df)

        # Get some event properties to add to the properties of the feature collections.
        props = {'db_id': meta['db_id'],
                     'event_start': meta['start_time'].isoformat(),
                     'event_end': meta['end_time'].isoformat(),
                     'sequence_start': min(sequence_df.time),
                     'sequence_end': max(sequence_df.time),
                     'author_uri': self.project.author_uri,
                     'agency_uri': self.project.agency_uri}

        # Write the voronoi dataframe to a geojson file.
        sequence_df = sequence_df.set_geometry('geom_vor')
        filepath = util.save_df_to_geojson(self.event.public_id,
                                           sequence_df.loc[:, ['geom_vor', 'time', 'nsl',
                                                               'pgv', 'sa', 'triggered']],
                                           output_dir = self.supplement_dir,
                                           name = self.sup_map['pgvsequence']['pgvvoronoi']['name'],
                                           subdir = self.sup_map['pgvsequence']['pgvvoronoi']['subdir'],
                                           props = props)
        self.logger.info('Saved pgv voronoi sequence to file %s.', filepath)

        sequence_df = sequence_df.set_geometry('geom_stat')
        filepath = util.save_df_to_geojson(self.event.public_id,
                                           sequence_df.loc[:, ['geom_stat', 'time', 'nsl',
                                                               'pgv', 'sa', 'triggered']],
                                           output_dir = self.supplement_dir,
                                           name = self.sup_map['pgvsequence']['pgvstation']['name'],
                                           subdir = self.sup_map['pgvsequence']['pgvstation']['subdir'],
                                           props = props)
        self.logger.info('Saved pgv station marker sequence to file %s.', filepath)
