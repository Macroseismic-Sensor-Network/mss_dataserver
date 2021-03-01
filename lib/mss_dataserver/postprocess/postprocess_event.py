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
import math
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pyproj
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

        # The event detection supplement data.
        self._meta = None
        self._pgv_stream = None
        self._detection_data = None

    @property
    def meta(self):
        ''' Load the metadata supplement.
        '''
        if self._meta is None:
            # Load the event metadata from the supplement file.
            self._meta = util.get_supplement_data(self.event.public_id,
                                                  category = 'detectiondata',
                                                  name = 'metadata',
                                                  directory = self.supplement_dir)
        return self._meta['metadata']

    def set_event(self, public_id):
        ''' Set the event to process.
        '''
        self.event = self.project.load_event_by_id(public_id = public_id)
        self._meta = None
        self._pgv_stream = None
        self._detection_data = None



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


    def compute_detection_data_df(self, trigger_data):
        ''' Compute the detection frames for a common time.
        '''
        df = None

        for cur_data in trigger_data:
            cur_coord = [(x.x, x.y) for x in cur_data['simplices_stations']]
            cur_simp_poly = shapely.geometry.Polygon(cur_coord)
            cur_time = [util.isoformat_tz(obspy.UTCDateTime(x)) for x in cur_data['time']]
            cur_added_to_event = np.zeros(len(cur_time), dtype = bool)
            tmp = np.where(cur_data['trigger'])[0]
            if len(tmp) > 0:
                cur_first_trigger = int(tmp[0])
                cur_added_to_event[cur_first_trigger:] = True
            cur_df = gpd.GeoDataFrame({'geom_simp': [cur_simp_poly] * len(cur_data['time']),
                                       'time': cur_time,
                                       'pgv': map(lambda a: [x if not math.isnan(x) else None for x in a], cur_data['pgv'].tolist()),
                                       'pgv_min': np.nanmin(cur_data['pgv'], axis = 1),
                                       'pgv_max': np.nanmax(cur_data['pgv'], axis = 1),
                                       'triggered': cur_data['trigger'],
                                       'added_to_event': cur_added_to_event},
                                      crs = "epsg:4326",
                                      geometry = 'geom_simp')

            if df is None:
                df = cur_df
            else:
                df = df.append(cur_df,
                               ignore_index = True)

        df = df.sort_values('time',
                            ignore_index = True)
        return df


    def compute_event_metadata_supplement(self):
        ''' Compute the supplement data based on the event metadata.
        '''
        # Load the event metadata from the supplement file.
        meta = self.meta

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
                 'event_start': util.isoformat_tz(meta['start_time']),
                 'event_end': util.isoformat_tz(meta['end_time']),
                 'author_uri': self.project.author_uri,
                 'agency_uri': self.project.agency_uri}

        filepath = util.save_supplement(self.event.public_id,
                                        pgv_df.loc[:, ['geom_stat', 'nsl',
                                                       'pgv', 'sa', 'triggered']],
                                        output_dir = self.supplement_dir,
                                        category = 'eventpgv',
                                        name = 'pgvstation',
                                        props = props)
        self.logger.info('Saved station pgv points to file %s.', filepath)

        pgv_df = pgv_df.set_geometry('geom_vor')
        filepath = util.save_supplement(self.event.public_id,
                                        pgv_df.loc[:, ['geom_vor', 'nsl',
                                                       'pgv', 'sa', 'triggered']],
                                        output_dir = self.supplement_dir,
                                        category = 'eventpgv',
                                        name = 'pgvvoronoi',
                                        props = props)
        self.logger.info('Saved station pgv voronoi cells to file %s.',
                         filepath)


    def compute_pgv_sequence_supplement(self):
        ''' Compute the supplement data representing the PGV sequence.
        '''
        # Load the event metadata from the supplement file.
        meta = self.meta

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
                                       'time': [util.isoformat_tz(cur_time)] * len(stations),
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
                                          'time': [util.isoformat_tz(cur_time)] * len(no_data_stations),
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
                 'event_start': util.isoformat_tz(meta['start_time']),
                 'event_end': util.isoformat_tz(meta['end_time']),
                 'sequence_start': min(sequence_df.time),
                 'sequence_end': max(sequence_df.time),
                 'author_uri': self.project.author_uri,
                 'agency_uri': self.project.agency_uri}

        # Write the voronoi dataframe to a geojson file.
        sequence_df = sequence_df.set_geometry('geom_vor')
        filepath = util.save_supplement(self.event.public_id,
                                        sequence_df.loc[:, ['geom_vor', 'time', 'nsl',
                                                            'pgv', 'sa', 'triggered']],
                                        output_dir = self.supplement_dir,
                                        category = 'pgvsequence',
                                        name = 'pgvvoronoi',
                                        props = props)
        self.logger.info('Saved pgv voronoi sequence to file %s.', filepath)

        sequence_df = sequence_df.set_geometry('geom_stat')
        filepath = util.save_supplement(self.event.public_id,
                                        sequence_df.loc[:, ['geom_stat', 'time', 'nsl',
                                                            'pgv', 'sa', 'triggered']],
                                        output_dir = self.supplement_dir,
                                        category = 'pgvsequence',
                                        name = 'pgvstation',
                                        props = props)
        self.logger.info('Saved pgv station marker sequence to file %s.',
                         filepath)


    def compute_detection_sequence_supplement(self):
        ''' Compute the supplement data representing the detection sequence triangles.
        '''
        # Load the event detection data from the supplement file.
        detection_data = util.get_supplement_data(self.event.public_id,
                                                  category = 'detectiondata',
                                                  name = 'detectiondata',
                                                  directory = self.supplement_dir)
        detection_data = detection_data['detection_data']

        # Load the event metadata from the supplement file.
        meta = self.meta

        # Compute the dataframe using the trigger data.
        sequence_df = None
        for cur_pw_key, cur_process_window in detection_data.items():
            trigger_data = cur_process_window['trigger_data']
            cur_df = self.compute_detection_data_df(trigger_data)
            if sequence_df is None:
                sequence_df = cur_df
            else:
                sequence_df = sequence_df.append(cur_df,
                                                 ignore_index = True)

        # Limit the data to the event timespan.
        pre_window = 6
        end_window = 6
        win_start = meta['start_time'] - pre_window
        win_end = meta['end_time'] + end_window
        df_utctime = np.array([obspy.UTCDateTime(x) for x in sequence_df.time])
        mask = (df_utctime >= win_start) & (df_utctime <= win_end)
        sequence_df = sequence_df.loc[mask, :]

        # Get some event properties to add to the properties of the feature collections.
        props = {'db_id': meta['db_id'],
                 'event_start': util.isoformat_tz(meta['start_time']),
                 'event_end': util.isoformat_tz(meta['end_time']),
                 'sequence_start': min(sequence_df.time),
                 'sequence_end': max(sequence_df.time),
                 'author_uri': self.project.author_uri,
                 'agency_uri': self.project.agency_uri}

        # Write the sequence dataframe to a geojson file.
        filepath = util.save_supplement(self.event.public_id,
                                        sequence_df.loc[:, ['geom_simp', 'time', 'pgv',
                                                            'pgv_min', 'pgv_max', 'triggered',
                                                            'added_to_event']],
                                        output_dir = self.supplement_dir,
                                        category = 'detectionsequence',
                                        name = 'simplices',
                                        props = props)
        self.logger.info('Saved detectioin sequence simplices to file %s.',
                         filepath)


    def intensity_to_pgv(self, intensity = None):
        ''' Compute the pgv and intensity values based on the MSS relationship.
        '''
        if intensity is None:
            return

        k_low = 0.5
        d_low = -5      # np.log10(0.00001)
        k_high = 1
        d_high = -7
        kink = 4

        intensity_low = intensity[intensity <= kink]
        pgv_low = d_low + k_low * intensity_low

        intensity_high = intensity[intensity > kink]
        pgv_high = d_high + k_high * intensity_high

        intensity = np.hstack([intensity_low, intensity_high])
        intensity_pgv = np.hstack([pgv_low, pgv_high])
        intensity_pgv = 10**intensity_pgv

        return np.hstack([intensity[:, np.newaxis],
                          intensity_pgv[:, np.newaxis]])


    def pgv_to_intensity(self, pgv = None):
        ''' Compute the pgv and intensity values based on the MSS relationship.
        '''
        if pgv is None:
            return

        pgv = np.log10(pgv)

        k_low = 2
        d_low = 10
        k_high = 1
        d_high = 7
        kink = np.log10(1e-3)

        pgv_low = pgv[pgv <= kink]
        intensity_low = d_low + k_low * pgv_low

        pgv_high = pgv[pgv > kink]
        intensity_high = d_high + k_high * pgv_high

        pgv = np.hstack([pgv_low, pgv_high])
        intensity = np.hstack([intensity_low, intensity_high])

        return np.hstack([10**pgv[:, np.newaxis],
                          intensity[:, np.newaxis]])


    def compute_isoseismal_supplement(self):
        ''' Compute the isoseismal contour lines using kriging.
        '''
        # Load the event metadata from the supplement file.
        meta = self.meta

        # Compute a PGV geodataframe using the event metadata.
        pgv_df = self.compute_pgv_df(meta)
        self.add_station_amplification(pgv_df)

        # Use only data with valid pgv data.
        pgv_df = pgv_df.loc[pgv_df.pgv.notna(), :]

        # Interpolate to a regular grid using ordinary kriging.
        krig_z, krig_sigmasq, grid_x, grid_y = util.compute_pgv_krigging(x = pgv_df.x_utm.values,
                                                                         y = pgv_df.y_utm.values,
                                                                         z = np.log10(pgv_df.pgv),
                                                                         nlags = 40,
                                                                         verbose = False,
                                                                         enable_plotting = False,
                                                                         weight = True)

        # Compute the contours.
        intensity = np.arange(0, 6.1, 0.5)
        # Add lower and upper limits to catch all the data below or 
        # above the desired intensity range.
        intensity = np.hstack([[-10], intensity, [20]])
        intensity_pgv = self.intensity_to_pgv(intensity = intensity)

        # Create and delete a figure to prevent pyplot from plotting the
        # contours.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.contourf(grid_x, grid_y, krig_z, np.log10(intensity_pgv[:, 1]),
                         vmin = -6, vmax = -2)
        contours = util.contourset_to_shapely(cs)
        fig.clear()
        plt.close(fig)
        del ax
        del fig
        del cs

        # Create a geodataframe of the contour polygons.
        data = {'geometry': [],
                'intensity': [],
                'pgv': []}

        for cur_level, cur_contour in contours.items():
            cur_intensity = self.pgv_to_intensity(pgv = [10**cur_level] * len(cur_contour))
            data['geometry'].extend(cur_contour)
            data['intensity'].extend(cur_intensity[:, 1].tolist())
            data['pgv'].extend([10**cur_level] * len(cur_contour))

        df = gpd.GeoDataFrame(data = data)

        # Convert the polygon coordinates to EPSG:4326.
        src_proj = pyproj.Proj(init = 'epsg:' + self.project.inventory.get_utm_epsg()[0][0])
        dst_proj = pyproj.Proj(init = 'epsg:4326')

        for cur_id, cur_row in df.iterrows():
            # Convert the exterior.
            cur_xy = cur_row.geometry.exterior.xy
            ext_lon, ext_lat = pyproj.transform(src_proj,
                                                dst_proj,
                                                cur_xy[0],
                                                cur_xy[1])

            # Convert the interiors.
            cur_int_list = []
            for cur_interior in cur_row.geometry.interiors:
                cur_xy = cur_interior.xy
                cur_lon, cur_lat = pyproj.transform(src_proj,
                                                    dst_proj,
                                                    cur_xy[0],
                                                    cur_xy[1])
                cur_ring = shapely.geometry.LinearRing(zip(cur_lon, cur_lat))
                cur_int_list.append(cur_ring)

            if len(cur_int_list) == 0:
                cur_int_list = None
            proj_poly = shapely.geometry.Polygon(zip(ext_lon, ext_lat),
                                                 holes = cur_int_list)

            df.loc[cur_id, 'geometry'] = proj_poly

        df.set_crs('epsg:4326')

        props = {'db_id': meta['db_id'],
                 'event_start': util.isoformat_tz(meta['start_time']),
                 'event_end': util.isoformat_tz(meta['end_time']),
                 'author_uri': self.project.author_uri,
                 'agency_uri': self.project.agency_uri}

        filepath = util.save_supplement(self.event.public_id,
                                        df,
                                        output_dir = self.supplement_dir,
                                        category = 'eventpgv',
                                        name = 'isoseismalcontours',
                                        props = props)
        self.logger.info('Saved isoseismal contours to file %s.', filepath)

        return df
