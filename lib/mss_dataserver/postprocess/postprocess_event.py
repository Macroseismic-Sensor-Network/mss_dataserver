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
''' The event postprocessor.
'''

import csv
import json
import logging
import math
import os
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pyproj
import scipy.fft
import shapely

import mss_dataserver.classify.classifyer as mssds_classifyer
import mss_dataserver.event.event_type as ev_type
import mss_dataserver.localize.localizer as mssds_localizer
import mss_dataserver.postprocess.util as util
import mss_dataserver.postprocess.voronoi as voronoi

# Ignore Geoseries.notna() warning.
# GeoSeries.notna() previously returned False for both missing (None)
# and empty geometries. Now, it only returns False for missing values.
# Since the calling GeoSeries contains empty geometries, the result has
# changed compared to previous versions of GeoPandas.
# Given a GeoSeries 's', you can use '~s.is_empty & s.notna()' to get back
# the old behaviour.
warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)


class EventPostProcessor(object):
    ''' Process a detected event.

    Parameters
    ----------
    project: :class:`mss_dataserver.core.project.Project`
        The mss_dataserver project.
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

        # The event to process. It is loaded only if a
        # database is used.
        self.event = None

        # The classification of the current event.
        # This attribute is needed in the case that no
        # database is used and therefore no event is loaded.
        self.event_classification = None

        # The available events types. They are loaded only if a
        # database is used.
        self.event_types = None

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

        # The postprocess common metadata.
        self._pp_meta = None


    @property
    def meta(self):
        ''' :obj:`dict`: The metadata supplement.
        '''
        if self._meta is None:
            # Load the event metadata from the supplement file.
            self._meta = util.get_supplement_data(self.event_public_id,
                                                  category = 'detectiondata',
                                                  name = 'metadata',
                                                  directory = self.supplement_dir)
        return self._meta['metadata']


    @property
    def pp_meta(self):
        ''' :obj:`dict`: The postprocess metadata supplement.
        '''
        ret_val = {}
        if self._pp_meta is None:
            # Load the event metadata from the supplement file.
            try:
                self._pp_meta = util.get_supplement_data(self.event_public_id,
                                                         category = 'postprocess',
                                                         name = 'metadata',
                                                         directory = self.supplement_dir)
                ret_val = self._pp_meta['metadata']
            except Exception:
                self.logger.exception("Couldn't load the metadata file.")
        else:
            ret_val = self._pp_meta

        return ret_val
        
    @pp_meta.setter
    def pp_meta(self, value):
        self._pp_meta = value
    

    def set_event(self, public_id):
        ''' Set the event to process.

        Parameters
        ----------
        public_id: str 
            The public id of the event.
        '''
        if self.project.is_connected_to_db:
            # Load the event from the database.
            self.event = self.project.load_event_by_id(public_id = public_id)
            if self.event is not None:
                msg = 'Loaded the event {} from the database.'.format(self.event.public_id)
                self.logger.info(msg)

            # Load the event types tree from the database.
            self.event_types = ev_type.EventType.load_from_db(project = self.project)
            self.logger.debug('event_types: %s',
                              [x.name for x in self.event_types])
        else:
            # TODO: Create an event instance using the supplement data.
            # Load the event types from the json file.
            filepath = os.path.join(self.data_dir,
                                    'event_types.json')
            event_types = util.load_eventtypes_from_json(filepath)
            self.event_types = event_types
            
        self.event_public_id = public_id
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
                cur_nsl = cur_row['nsl']
                #cur_snl = cur_snl.split(':')
                #cur_nsl = ':'.join([cur_snl[1], cur_snl[0], cur_snl[2]])
                station_amp[cur_nsl] = {'serial': cur_serial,
                                        'amp': cur_amp}

        return station_amp

    def add_station_amplification(self, df):
        ''' Add the station amplification values to a dataframe.

        Parameters
        ----------
        df: :class:`geopandas.GeoDataFrame`
            The dataframe to which to add the station amplification factors.
        '''
        station_amp = self.station_amp
        sorted_sa = [station_amp[row.nsl]['amp'] if row.nsl in station_amp.keys() else np.nan for index, row in df.iterrows()]
        df['sa'] = sorted_sa

        
    def compute_pgv_df(self, meta):
        ''' Create a dataframe of pgv values with station coordinates.

        Parameters
        ----------
        meta: :obj:`dict`
            The event metadata.
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
            cur_data = [cur_nsl, cur_station.x, cur_station.y, cur_station.z,
                        cur_station.x_utm, cur_station.y_utm,
                        cur_pgv, cur_trigger]
            pgv_data.append(cur_data)

        for cur_nsl in untriggered_nsl:
            cur_station = inventory.get_station(nsl_string = cur_nsl)[0]
            cur_pgv = meta['max_network_pgv'][cur_nsl]
            cur_trigger = False
            cur_data = [cur_nsl, cur_station.x, cur_station.y, cur_station.z,
                        cur_station.x_utm, cur_station.y_utm,
                        cur_pgv, cur_trigger]
            pgv_data.append(cur_data)

        for cur_nsl in nodata_nsl:
            cur_station = inventory.get_station(nsl_string = cur_nsl)[0]
            cur_pgv = None
            cur_trigger = False
            cur_data = [cur_nsl, cur_station.x, cur_station.y, cur_station.z,
                        cur_station.x_utm, cur_station.y_utm,
                        cur_pgv, cur_trigger]
            pgv_data.append(cur_data)

        x_coord = [x[1] for x in pgv_data]
        y_coord = [x[2] for x in pgv_data]
        z_coord = [x[3] for x in pgv_data]
        pgv_data = {'geom_stat': [shapely.geometry.Point([x[0], x[1]]) for x in zip(x_coord, y_coord)],
                    'geom_vor': [shapely.geometry.Polygon([])] * len(pgv_data),
                    'nsl': [x[0] for x in pgv_data],
                    'x': x_coord,
                    'y': y_coord,
                    'z': z_coord,
                    'x_utm': [x[4] for x in pgv_data],
                    'y_utm': [x[5] for x in pgv_data],
                    'pgv': [x[6] for x in pgv_data],
                    'triggered': [x[7] for x in pgv_data]}

        df = gpd.GeoDataFrame(data = pgv_data,
                              crs = 'epsg:4326',
                              geometry = 'geom_stat')

        return df
    

    def classify_event(self):
        ''' Classify the event.

        '''
        # Load the event metadata from the supplement file.
        meta = self.meta

        # Compute a PGV geodataframe using the event metadata.
        pgv_df = self.compute_pgv_df(meta)

        # Add the station amplification column to the dataframe.
        self.add_station_amplification(pgv_df)

        pub_id = self.event_public_id
        classifyer = mssds_classifyer.EventClassifyer(public_id = pub_id,
                                                      meta = self.meta,
                                                      pgv_df = pgv_df,
                                                      project = self.project,
                                                      event = self.event,
                                                      event_types = self.event_types)
        event_type = classifyer.classify()
        
        # Write the classification result to the database.
        if self.project.is_connected_to_db and self.event is not None:
            self.event.write_to_database(self.project)

        # Write the classification result to the supplement file.
        pp_meta = self.pp_meta
        if event_type is not None:
            type_dict = {'event_type': event_type.name,
                         'description': event_type.description}
        else:
            type_dict = None
           
        tmp = {'classification': type_dict}
        pp_meta.update(tmp)
        
        # Get some event properties to add to the properties.
        props = {'db_id': meta['db_id'],
                 'event_start': util.isoformat_tz(meta['start_time']),
                 'event_end': util.isoformat_tz(meta['end_time']),
                 'author_uri': self.project.author_uri,
                 'agency_uri': self.project.agency_uri}
        
        filepath = util.save_supplement(self.event_public_id,
                                        pp_meta,
                                        output_dir = self.supplement_dir,
                                        category = 'postprocess',
                                        name = 'metadata',
                                        props = props)
        self.logger.info('Added the classification to the postprocess metadata file %s.', filepath)

        # Save relevant data in the instance.
        self.pp_meta = pp_meta
        self.event_classification = event_type


    def localize_event(self):
        ''' Compute the origin of an event.

        '''
        # Check the event type before running the localization.
        event_type = self.event_classification
        types_to_localize = ['root-blast', 'root-earthquake-inside network']
        if event_type is None:
            self.logger.info('No event type set. Ignoring the localization of this event.')
            return

        if event_type.full_name not in types_to_localize:
            self.logger.info('The localization of event type "%s" not supported or needed.',
                             event_type.full_name)
            return

        # Special handling of Pfaffenberg blasts.
        if 'class_region:Steinbruch Pfaffenberg' in self.event.tags:
            self.logger.info('Automatic localization of blasts in Pfaffenberg is not yet supported.')
            return
        
        # Load the event metadata from the supplement file.
        meta = self.meta

        # Compute a PGV geodataframe using the event metadata.
        pgv_df = self.compute_pgv_df(meta)

        # Add the station amplification column to the dataframe.
        self.add_station_amplification(pgv_df)

        # Eliminate rows without a PGV data.
        no_data_mask = pgv_df['pgv'].isna()
        pgv_df = pgv_df[~no_data_mask]

        pub_id = self.event_public_id
        localizer = mssds_localizer.EventLocalizer(public_id = pub_id,
                                                   meta = self.meta,
                                                   pgv_df = pgv_df,
                                                   project = self.project,
                                                   event = self.event)
        # Run the Apollonius localization.
        localizer.loc_apollonius(dist_exp = -2.2)

        # Compute the MSS magnitude for the origins.
        origins = localizer.origins
        stat_coord = pgv_df[['x_utm', 'y_utm', 'z']].values
        pgv = pgv_df['pgv'].values
        
        # Apply the station amplification factors.
        pgv = pgv / pgv_df['sa'].values
        
        for cur_origin in origins:
            mag = cur_origin.compute_mss_magnitude(stat_coord = stat_coord,
                                                   amp = pgv)
            cur_origin.add_magnitude(mag)
            cur_origin.set_preferred_magnitude(mag)

        # Assign the regions to the origins.
        for cur_origin in origins:
            localizer.assign_region(cur_origin)

        # Write the origins to the database.
        # Convert the coordinates to lon/lat before saving them
        # to the database.
        if self.project.is_connected_to_db:
            for cur_origin in origins:
                cur_origin.convert_to_lonlat()
                cur_origin.write_to_database(project = self.project)

        # Set the preferred origin.
        pref_origin = [x for x in origins if x.method == 'apollonius_circle']
        if len(pref_origin) >= 1:
            pref_origin = pref_origin[0]
            if self.event is not None:
                self.logger.info('Setting the pref_origin.')
                self.logger.info('pref_origin.db_id: %d', pref_origin.db_id)
                self.event.set_preferred_origin(pref_origin)
                self.event.write_to_database(project = self.project)

        # Write the origins to the origins geojson supplement data.
        x_coord = [x.x for x in origins]
        y_coord = [x.y for x in origins]
        z_coord = [x.z for x in origins]
        data = {'geom_origin': [shapely.geometry.Point([x[0], x[1]]) for x in zip(x_coord, y_coord)],
                'z': z_coord,
                'method': [x.method for x in origins],
                'region': [x.region for x in origins]}
        df = gpd.GeoDataFrame(data = data,
                              crs = 'epsg:4326',
                              geometry = 'geom_origin')
        # Get some event properties to add to the properties of the feature collections.
        props = {'db_id': meta['db_id'],
                 'event_start': util.isoformat_tz(meta['start_time']),
                 'event_end': util.isoformat_tz(meta['end_time']),
                 'author_uri': self.project.author_uri,
                 'agency_uri': self.project.agency_uri}

        # Save the origins supplement.
        filepath = util.save_supplement(self.event_public_id,
                                        df,
                                        output_dir = self.supplement_dir,
                                        category = 'localize',
                                        name = 'origins',
                                        props = props)
        self.logger.info('Saved the origins to file %s.', filepath)

        # Write the origins to the postprocess metadata supplement file.
        pp_meta = self.pp_meta
        origins_list = []
        for cur_origin in origins:
            creation_time = cur_origin.creation_time
            creation_time = util.isoformat_tz(creation_time)
            tmp = {'time': cur_origin.time,
                   'x': cur_origin.x,
                   'y': cur_origin.y,
                   'z': cur_origin.z,
                   'coord_system': cur_origin.coord_system,
                   'method': cur_origin.method,
                   'comment': cur_origin.comment,
                   'agency_uri': cur_origin.agency_uri,
                   'author_uri': cur_origin.author_uri,
                   'creation_time': creation_time}
            origins_list.append(tmp)
        pp_meta['origins'] = origins_list

        filepath = util.save_supplement(self.event_public_id,
                                        pp_meta,
                                        output_dir = self.supplement_dir,
                                        category = 'postprocess',
                                        name = 'metadata')
        self.logger.info('Added the origins to the postprocess metadata file %s.', filepath)

        self.pp_meta = pp_meta

    
    def compute_detection_data_df(self, trigger_data):
        ''' Compute the detection frames for a common time.

        Parameters
        ----------
        trigger_data: :obj:`list`
            The event trigger data.
        '''
        df = None

        for cur_data in trigger_data:
            cur_coord = [(x.x, x.y) for x in cur_data['simplices_stations']]
            cur_nsl = [x.nsl_string for x in cur_data['simplices_stations']]
            cur_nsl = ','.join(cur_nsl)
            cur_simp_poly = shapely.geometry.Polygon(cur_coord)
            cur_time = [util.isoformat_tz(obspy.UTCDateTime(round(x, 2))) for x in cur_data['time']]
            cur_added_to_event = np.zeros(len(cur_time), dtype = bool)
            #tmp = np.where(cur_data['trigger'])[0]
            #if len(tmp) > 0:
            #    cur_first_trigger = int(tmp[0])
            #    cur_added_to_event[cur_first_trigger:] = True
            cur_df = gpd.GeoDataFrame({'geom_simp': [cur_simp_poly] * len(cur_data['time']),
                                       'nsl': [cur_nsl] * len(cur_data['time']),
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

        filepath = util.save_supplement(self.event_public_id,
                                        pgv_df.loc[:, ['geom_stat', 'nsl',
                                                       'pgv', 'sa', 'triggered']],
                                        output_dir = self.supplement_dir,
                                        category = 'eventpgv',
                                        name = 'pgvstation',
                                        props = props)
        self.logger.info('Saved station pgv points to file %s.', filepath)

        pgv_df = pgv_df.set_geometry('geom_vor')
        filepath = util.save_supplement(self.event_public_id,
                                        pgv_df.loc[:, ['geom_vor', 'nsl',
                                                       'pgv', 'sa', 'triggered']],
                                        output_dir = self.supplement_dir,
                                        category = 'eventpgv',
                                        name = 'pgvvoronoi',
                                        props = props)
        self.logger.info('Saved station pgv voronoi cells to file %s.',
                         filepath)


    def compute_event_specific_supplement(self):
        ''' Compute the supplement data depending on the event type.

        '''
        if self.event.event_type is not None:
            ev_type = self.event.event_type.name
            if ev_type == 'blast':
                if 'class_region:Steinbruch Dürnbach' in self.event.tags:
                    self.compute_blast_duernbach_supplement()

                
    def compute_blast_duernbach_supplement(self):
        ''' Compute the supplement data for a blast at Dürnbach.

        '''
        public_id = self.event_public_id
        meta = self.meta

        # Load the velocity seismogram data.
        vel_st = util.get_supplement_data(public_id = public_id,
                                          category = 'detectiondata',
                                          name = 'velocity',
                                          directory = self.supplement_dir)

        # Fix the miniseed header values.
        network_map = {'MS': 'MSSNet'}
        channel_map = {'Hno': 'Hnormal',
                       'Hpa': 'Hparallel'}
        for cur_trace in vel_st:
            cur_net = cur_trace.stats.network
            cur_chan = cur_trace.stats.channel
            mapped_net = cur_net
            mapped_chan = cur_chan
            
            if cur_net in network_map.keys():
                mapped_net = network_map[cur_net]
        
            if cur_chan in channel_map.keys():
                mapped_chan = channel_map[cur_chan]
        
            cur_trace.stats.network = mapped_net
            cur_trace.stats.channel = mapped_chan

        # Get the streams of stations with 3 channels.
        inv = self.project.inventory
        stations = inv.get_station()
        stations_3_chan = [x for x in stations if len(x.channels) >= 3]
        st_3d = obspy.Stream()
        for cur_stat in stations_3_chan:
            cur_st = vel_st.select(network = cur_stat.network,
                                   station = cur_stat.name,
                                   location = cur_stat.location)
            st_3d += cur_st

        # Compute the 3D PGV data.
        channel_names = ['Hnormal', 'Hparallel', 'Z']
        st_res_3d = self.compute_resultant(st_3d, channel_names)
        max_pgv_3d = [(str.join(':', (x.stats.network, x.stats.station, x.stats.location)), np.max(x.data))  for x in st_res_3d]
        max_pgv_3d = dict(max_pgv_3d)

        # Create the data used for the geojson dataframe.
        x_coord = [x.x for x in stations_3_chan]
        y_coord = [x.y for x in stations_3_chan]
        z_coord = [x.z for x in stations_3_chan]
        pgv_3d_sorted = [max_pgv_3d[x.nsl_string] for x in stations_3_chan]

        # Get some event properties to add to the supplement properties.
        props = {'db_id': meta['db_id'],
                 'event_start': util.isoformat_tz(meta['start_time']),
                 'event_end': util.isoformat_tz(meta['end_time']),
                 'author_uri': self.project.author_uri,
                 'agency_uri': self.project.agency_uri}
        
        # Save the PGV 3D supplement data to file.
        pgv3d_data = {'geom_stat': [shapely.geometry.Point([x[0], x[1]]) for x in zip(x_coord, y_coord)],
                      'nsl': [x.nsl_string for x in stations_3_chan],
                      'x': x_coord,
                      'y': y_coord,
                      'z': z_coord,
                      'pgv3d': pgv_3d_sorted}
        df = gpd.GeoDataFrame(data = pgv3d_data,
                              crs = 'epsg:4326',
                              geometry = 'geom_stat')
        
        filepath = util.save_supplement(self.event_public_id,
                                        df,
                                        output_dir = self.supplement_dir,
                                        category = 'custom',
                                        name = 'pgv3d',
                                        props = props)
        self.logger.info('Saved the PGV 3D supplement data to file %s.',
                         filepath)
        

        # Compute the PSD and the dominant frequencies.
        psd_data = {}
        for cur_trace in st_3d:
            cur_psd_data = self.compute_psd(cur_trace)
            psd_data[cur_trace.id] = cur_psd_data

        # Compute the dominant frequency.
        dom_frequ = {}
        dom_stat_frequ = {}
        for cur_station in stations_3_chan:
            cur_psd_keys = [x for x in psd_data.keys() if x.startswith(cur_station.network + '.' + cur_station.name + '.')]
        cur_df = []
        for cur_key in cur_psd_keys:
            cur_nfft = psd_data[cur_key]['n_fft']
            left_fft = int(np.ceil(cur_nfft / 2.))
            max_ind = np.argmax(psd_data[cur_key]['psd'][1:left_fft])
            dom_frequ[cur_key] = psd_data[cur_key]['frequ'][max_ind]
            cur_df.append(dom_frequ[cur_key])

        dom_stat_frequ[cur_station.nsl_string] = np.mean(cur_df)
        
        
    def compute_pgv_sequence_supplement(self):
        ''' Compute the supplement data representing the PGV sequence.
        '''
        # Load the event metadata from the supplement file.
        meta = self.meta

        # Load the PGV data stream.
        pgv_stream = util.get_supplement_data(self.event_public_id,
                                              category = 'detectiondata',
                                              name = 'pgv',
                                              directory = self.supplement_dir)
        #print(pgv_stream.__str__(extended=True))
        pgv_stream.merge()

        # Trim the stream.
        start_time = meta['start_time'] - 6
        end_time = meta['end_time'] + 6
        pgv_stream.trim(starttime = start_time,
                        endtime = end_time,
                        pad = True)
        pgv_stream.sort()
        #print(pgv_stream.__str__(extended=True))

        inventory = self.project.inventory

        station_nsl = [('MSSNet', x.stats.station, x.stats.location) for x in pgv_stream]
        station_nsl = [':'.join(x) for x in station_nsl]
        stations = [inventory.get_station(nsl_string = x)[0] for x in station_nsl]
        times = pgv_stream[0].times("utcdatetime")
        dt = pgv_stream[0].stats.delta
        ndata = pgv_stream[0].stats.npts
        times = [start_time + x * dt for x in range(ndata)]
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
        filepath = util.save_supplement(self.event_public_id,
                                        sequence_df.loc[:, ['geom_vor', 'time', 'nsl',
                                                            'pgv', 'sa', 'triggered']],
                                        output_dir = self.supplement_dir,
                                        category = 'pgvsequence',
                                        name = 'pgvvoronoi',
                                        props = props)
        self.logger.info('Saved pgv voronoi sequence to file %s.', filepath)

        sequence_df = sequence_df.set_geometry('geom_stat')
        filepath = util.save_supplement(self.event_public_id,
                                        sequence_df.loc[:, ['geom_stat', 'time', 'nsl',
                                                            'pgv', 'sa', 'triggered']],
                                        output_dir = self.supplement_dir,
                                        category = 'pgvsequence',
                                        name = 'pgvstation',
                                        props = props)
        self.logger.info('Saved pgv station marker sequence to file %s.',
                         filepath)

        
    def compute_pgv_contour_sequence_supplement(self):
        ''' Compute the supplement data representing the PGV sequence.
        '''
        # Load the event metadata from the supplement file.
        meta = self.meta

        # Load the PGV data stream.
        pgv_stream = util.get_supplement_data(self.event_public_id,
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

        detection_limits = meta['detection_limits']

        sequence_df = None
        last_pgv_df = None
        last_krig_z = None
        no_change_cnt = 0
        
        for k in range(len(times)):
            cur_time = times[k]
            self.logger.info("Computing frame {time}.".format(time = str(cur_time)))
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

            # Compute the corrected pgv values.
            cur_df['pgv_corr'] = cur_df.pgv / cur_df.sa

            # Use only the stations with a valid corrected pgv.
            cur_df = cur_df[cur_df['pgv_corr'].notna()]
            cur_df = cur_df.reset_index()

            # Update the pgv values to keep the event maximum pgv.
            # Track changes of the event maximum pgv.
            if last_pgv_df is not None:
                # Use the current PGV values only, if they are higher than
                # the last ones.
                #
                # Update the last_pgv_df with the current df. It is possible, that
                # rows are missing or new ones are available.
                # Remove the rows, that are not present in the cur_df.
                tmp_df = last_pgv_df[last_pgv_df.nsl.isin(cur_df.nsl)]
                # Add the rows, that are not present in the last_pgv_df.
                mask_df = tmp_df.append(cur_df[~cur_df.nsl.isin(last_pgv_df.nsl)],
                                        ignore_index = True)

                # Sort the two dataframes using the nsl.
                cur_df = cur_df.sort_values(by = 'nsl',
                                            ignore_index = True)
                mask_df = mask_df.sort_values(by = 'nsl',
                                              ignore_index = True)

                # Check for correct station snl.
                if (np.any(cur_df['nsl'].values != mask_df['nsl'].values)):
                    print(last_pgv_df)
                    print(cur_df)
                    print(mask_df)
                    raise RuntimeError("The station NSL codes of the two dataframes to compare are not equal.")

                # Reset the values for the stations, that already had a larger pgv value.
                mask = cur_df.pgv_corr < mask_df.pgv_corr               
                cur_df.loc[mask, 'pgv_corr'] = mask_df.loc[mask, 'pgv_corr']

                if np.all(mask):
                    no_change_cnt += 1
                else:
                    no_change_cnt = 0
                self.logger.info('no_change_cnt: ' + str(no_change_cnt))

            # Exit if the was no change of the max event pgv data for some time.
            if no_change_cnt >= 5:
                self.logger.info('No change for some time, stop computation of contours.')
                break

            # Keep the last pgv dataframe.
            # Get the rows, that are not available in cur_df and keep them.
            if last_pgv_df is not None:
                tmp_df = last_pgv_df[~last_pgv_df.nsl.isin(cur_df.nsl)]
                last_pgv_df = cur_df.copy()
                last_pgv_df = last_pgv_df.append(tmp_df.copy(),
                                                 ignore_index = True)
            else:
                last_pgv_df = cur_df.copy()
           
            # Interpolate to a regular grid using ordinary kriging.
            self.logger.info("Interpolate")
            krig_z, krig_sigmasq, grid_x, grid_y = util.compute_pgv_krigging(x = cur_df.x_utm.values,
                                                                             y = cur_df.y_utm.values,
                                                                             z = np.log10(cur_df.pgv_corr),
                                                                             nlags = 40,
                                                                             verbose = False,
                                                                             enable_plotting = False,
                                                                             weight = True)

            # Update the interpolated pgv values only if they are higher than the last ones.
            #if last_krig_z is not None:
            #    cur_mask = krig_z < last_krig_z
            #    krig_z[cur_mask] = last_krig_z[cur_mask]
            #last_krig_z = krig_z

            self.logger.info("Contours")
            # Compute the contours.
            intensity = np.arange(2, 8.1, 0.1)
            # Add lower and upper limits to catch all the data below or 
            # above the desired intensity range.
            intensity = np.hstack([[-10], intensity, [20]])
            # Use a low intensity_I_pgv value to make sure, that the lowest countour
            # level captures all PGV values.
            intensity_pgv = util.intensity_to_pgv(intensity = intensity,
                                                  intensity_I_pgv = 1e-9)

            # Create and delete a figure to prevent pyplot from plotting the
            # contours.
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cs = ax.contourf(grid_x, grid_y, krig_z, np.log10(intensity_pgv[:, 1]))
            contours = util.contourset_to_shapely(cs)
            fig.clear()
            plt.close(fig)
            del ax
            del fig
            del cs

            self.logger.info('dataframe')
            # Create a geodataframe of the contour polygons.
            cont_data = {'time': [],
                         'geometry': [],
                         'intensity': [],
                         'pgv': []}

            for cur_level, cur_poly in contours.items():
                cur_intensity = util.pgv_to_intensity(pgv = [10**cur_level] * len(cur_poly))
                cont_data['time'].extend([util.isoformat_tz(cur_time)] * len(cur_poly))
                cont_data['geometry'].extend(cur_poly)
                cont_data['intensity'].extend(cur_intensity[:, 1].tolist())
                cont_data['pgv'].extend([10**cur_level] * len(cur_poly))
            cur_cont_df = gpd.GeoDataFrame(data = cont_data)

            # Convert the polygon coordinates to EPSG:4326.
            src_proj = pyproj.Proj(init = 'epsg:' + self.project.inventory.get_utm_epsg()[0][0])
            dst_proj = pyproj.Proj(init = 'epsg:4326')
            cur_cont_df = util.reproject_polygons(df = cur_cont_df,
                                                  src_proj = src_proj,
                                                  dst_proj = dst_proj)

            # Clip to the network boundary.
            # Clipping a polygon may created multiple polygons.
            # Therefore create a new dataframe to have only one polygon per,
            # entry. Thus avoiding possible problems due to a mixture of 
            # multipolygons and polygons.
            self.logger.info('Clipping.')
            cont_data = {'time': [],
                         'geometry': [],
                         'intensity': [],
                         'pgv': []}
            for cur_id, cur_row in cur_cont_df.iterrows():
                cur_poly = cur_row.geometry
                clipped_poly = cur_poly.intersection(self.network_boundary.loc[0, 'geometry'])
                self.logger.info(type(clipped_poly))
                if isinstance(clipped_poly, shapely.geometry.multipolygon.MultiPolygon):
                    cont_data['time'].extend([cur_row.time] * len(clipped_poly))
                    cont_data['geometry'].extend([x for x in clipped_poly])
                    cont_data['intensity'].extend([cur_row.intensity] * len(clipped_poly))
                    cont_data['pgv'].extend([cur_row.pgv] * len(clipped_poly))
                else:
                    cont_data['time'].append(cur_row.time)
                    cont_data['geometry'].append(clipped_poly)
                    cont_data['intensity'].append(cur_row.intensity)
                    cont_data['pgv'].append(cur_row.pgv)
            cur_cont_df = gpd.GeoDataFrame(data = cont_data)

            # Remove rows having an empty geometry.
            self.logger.info(cur_cont_df['geometry'])
            cur_cont_df = cur_cont_df[~cur_cont_df['geometry'].is_empty]
            self.logger.info(cur_cont_df['geometry'])
            
            self.logger.info('Appending to sequence.')
            # Add the dataframe to the sequence.
            if sequence_df is None:
                sequence_df = cur_cont_df
            else:
                sequence_df = sequence_df.append(cur_cont_df)

        # Get some event properties to add to the properties of the feature collections.
        props = {'db_id': meta['db_id'],
                 'event_start': util.isoformat_tz(meta['start_time']),
                 'event_end': util.isoformat_tz(meta['end_time']),
                 'sequence_start': min(sequence_df.time),
                 'sequence_end': max(sequence_df.time),
                 'author_uri': self.project.author_uri,
                 'agency_uri': self.project.agency_uri,
                 'station_correction_applied': True}

        # Write the voronoi dataframe to a geojson file.
        filepath = util.save_supplement(self.event_public_id,
                                        sequence_df,
                                        output_dir = self.supplement_dir,
                                        category = 'pgvsequence',
                                        name = 'pgvcontour',
                                        props = props)
        self.logger.info('Saved pgv contour sequence to file %s.', filepath)


    def compute_detection_sequence_supplement(self):
        ''' Compute the supplement data representing the detection sequence triangles.
        '''
        # Load the event detection data from the supplement file.
        detection_data = util.get_supplement_data(self.event_public_id,
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

        # Set the added_to_event flag of the whole sequence.
        # From the first triggered time, the flag is set to true.
        simp_groups = sequence_df.groupby('nsl')
        for cur_name, cur_group in simp_groups:
            cur_added_to_event = np.zeros(len(cur_group), dtype = bool)
            tmp = np.where(cur_group['triggered'])[0]
            if len(tmp) > 0:
                cur_first_trigger = int(tmp[0])
                cur_added_to_event[cur_first_trigger:] = True
            sequence_df.loc[sequence_df['nsl'] == cur_name, 'added_to_event'] = cur_added_to_event

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
        filepath = util.save_supplement(self.event_public_id,
                                        sequence_df.loc[:, ['geom_simp', 'time', 'pgv',
                                                            'pgv_min', 'pgv_max', 'triggered',
                                                            'added_to_event']],
                                        output_dir = self.supplement_dir,
                                        category = 'detectionsequence',
                                        name = 'simplices',
                                        props = props)
        self.logger.info('Saved detectioin sequence simplices to file %s.',
                         filepath)




    def compute_isoseismal_supplement(self):
        ''' Compute the isoseismal contour lines using kriging.
        '''
        # Load the event metadata from the supplement file.
        meta = self.meta

        # Compute a PGV geodataframe using the event metadata.
        pgv_df = self.compute_pgv_df(meta)
        self.add_station_amplification(pgv_df)

        pgv_df['pgv_corr'] = pgv_df.pgv / pgv_df.sa

        # Use only data with valid pgv data.
        pgv_df = pgv_df.loc[pgv_df.pgv_corr.notna(), :]

        # Interpolate to a regular grid using ordinary kriging.
        krig_z, krig_sigmasq, grid_x, grid_y = util.compute_pgv_krigging(x = pgv_df.x_utm.values,
                                                                         y = pgv_df.y_utm.values,
                                                                         z = np.log10(pgv_df.pgv_corr),
                                                                         nlags = 40,
                                                                         verbose = False,
                                                                         enable_plotting = False,
                                                                         weight = True)

        # Compute the contours.
        intensity = np.arange(2, 8.1, 0.1)
        #intensity = np.arange(2, 8.1, 1)
        # Add lower and upper limits to catch all the data below or 
        # above the desired intensity range.
        intensity = np.hstack([[-10], intensity, [20]])
        # Use a low intensity_I_pgv value to make sure, that the lowest countour
        # level captures all PGV values.
        intensity_pgv = util.intensity_to_pgv(intensity = intensity,
                                              intensity_I_pgv = 1e-9)

        # Create and delete a figure to prevent pyplot from plotting the
        # contours.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.contourf(grid_x, grid_y, krig_z, np.log10(intensity_pgv[:, 1]))
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

        for cur_level, cur_poly in contours.items():
            cur_intensity = util.pgv_to_intensity(pgv = [10**cur_level] * len(cur_poly))
            data['geometry'].extend(cur_poly)
            data['intensity'].extend(cur_intensity[:, 1].tolist())
            data['pgv'].extend([10**cur_level] * len(cur_poly))
        df = gpd.GeoDataFrame(data = data)

        # Convert the polygon coordinates to EPSG:4326.
        src_proj = pyproj.Proj(init = 'epsg:' + self.project.inventory.get_utm_epsg()[0][0])
        dst_proj = pyproj.Proj(init = 'epsg:4326')
        df = util.reproject_polygons(df = df,
                                     src_proj = src_proj,
                                     dst_proj = dst_proj)

        # Clip to the network boundary.
        # Clipping a polygon may created multiple polygons.
        # Therefore create a new dataframe to have only one polygon per,
        # entry. Thus avoiding possible problems due to a mixture of 
        # multipolygons and polygons.
        data = {'geometry': [],
                'intensity': [],
                'pgv': []}
        for cur_id, cur_row in df.iterrows():
            cur_poly = cur_row.geometry
            clipped_poly = cur_poly.intersection(self.network_boundary.loc[0, 'geometry'])
            self.logger.info(type(clipped_poly))
            if isinstance(clipped_poly, shapely.geometry.multipolygon.MultiPolygon):
                data['geometry'].extend([x for x in clipped_poly])
                data['intensity'].extend([cur_row.intensity] * len(clipped_poly))
                data['pgv'].extend([cur_row.pgv] * len(clipped_poly))
            else:
                data['geometry'].append(clipped_poly)
                data['intensity'].append(cur_row.intensity)
                data['pgv'].append(cur_row.pgv)
        df = gpd.GeoDataFrame(data = data)

        props = {'db_id': meta['db_id'],
                 'event_start': util.isoformat_tz(meta['start_time']),
                 'event_end': util.isoformat_tz(meta['end_time']),
                 'author_uri': self.project.author_uri,
                 'agency_uri': self.project.agency_uri,
                 'station_correction_applied': True}

        filepath = util.save_supplement(self.event_public_id,
                                        df,
                                        output_dir = self.supplement_dir,
                                        category = 'eventpgv',
                                        name = 'isoseismalfilledcontour',
                                        props = props)
        self.logger.info('Saved isoseismal contours to file %s.', filepath)

        return df

    
    def compute_resultant(self, st, channel_names):
        ''' Compute the resultant of the peak-ground-velocity.
        '''
        res_st = obspy.core.Stream()
        used_streams = []
        for cur_channel in channel_names:
            cur_stream = st.select(channel = cur_channel).merge()
            if len(cur_stream) == 0:
                self.logger.error("No data found in stream %s for channel %s.",
                                  st,
                                  cur_channel)
                return res_st
            used_streams.append(cur_stream)

        for cur_traces in zip(*[x.traces for x in used_streams]):
            cur_data = [x.data for x in cur_traces]

            if len(set([len(x) for x in cur_data])) > 1:
                self.logger.error("The lenght of the data of the individual traces dont't match. Can't compute the res. PGV for these traces: %s.", [str(x) for x in cur_traces])
                continue

            cur_data = np.array(cur_data)
            cur_res = np.sqrt(np.sum(cur_data**2, axis = 0))

            cur_stats = {'network': cur_traces[0].stats['network'],
                         'station': cur_traces[0].stats['station'],
                         'location': cur_traces[0].stats['location'],
                         'channel': 'res_{0:d}d'.format(len(cur_traces)),
                         'sampling_rate': cur_traces[0].stats['sampling_rate'],
                         'starttime': cur_traces[0].stats['starttime']}
            res_trace = obspy.core.Trace(data = cur_res, header = cur_stats)
            res_st.append(res_trace)

        res_st.split()

        return res_st

    
    def compute_psd(self, trace):
        ''' Compute the power spectral density of a trace.
        '''

        # Compute the power amplitude density spectrum.
        # As defined by Havskov and Alguacil (page 164), the power density
        # spectrum can be written as
        #   P = 2* 1/T * deltaT^2 * abs(F_dft)^2
        # This is valid for the left-sided fft.
        #
        n_fft = len(trace.data)
        delta_t = 1 / trace.stats.sampling_rate
        T = (len(trace.data) - 1) * delta_t
        Y = scipy.fft.fft(trace.data, n_fft)
        psd = 2 * delta_t**2 / T * np.abs(Y)**2
        psd = 10 * np.log10(psd)
        frequ = trace.stats.sampling_rate * np.arange(0, n_fft) / float(n_fft)
        psd_data = {}
        psd_data['n_fft'] = n_fft
        psd_data['psd'] = psd
        psd_data['frequ'] = frequ

        return psd_data
