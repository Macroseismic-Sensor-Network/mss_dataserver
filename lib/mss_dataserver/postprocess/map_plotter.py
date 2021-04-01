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

import dateutil
import logging
import os

import cartopy.crs as ccrs
import ffmpeg
import geopandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import obspy
import rasterio
import rasterio.plot

import mss_dataserver.postprocess.util as util


class MapPlotter(object):
    ''' Create map images and movies using mssds geojson data.
    '''

    def __init__(self, supplement_dir, map_dir, output_dir,
                 basemap = 'mss_basemap_desaturate.tif',
                 boundary = 'mss_network_hull.geojson'):
        ''' Initialize the instance.
        '''
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # The directory containing the event supplement data.
        self.supplement_dir = supplement_dir

        # The directory containing the map data.
        self.map_dir = map_dir

        # The directory where to save the map images.
        self.output_dir = output_dir

        # The filename of the background map.
        self.basemap_filename = basemap

        # The filename of the network boundary.
        self.boundary_filename = boundary

        # The network boundary.
        self.mss_boundary = None

        # The public id of the event.
        self.event_public_id = None

        # The map figure.
        self.fig = None

        # The map axes.
        self.ax = None

        # The colorbar axes.
        self.cb = None

        # The data plot artists.
        self.artists = []


    def set_event(self, public_id):
        ''' Set the event to process.
        '''
        self.event_public_id = public_id
        self.event_dir = util.event_dir_from_publicid(public_id)
        # Load the event metadata from the supplement file.
        self.meta = util.get_supplement_data(self.event_public_id,
                                             category = 'detectiondata',
                                             name = 'metadata',
                                             directory = self.supplement_dir)

        
    def init_map(self, utm_zone=33):
        ''' Initialize the map plot.
        '''
        # Set the projection.
        self.projection = ccrs.UTM(zone=utm_zone)

        # Load the basemap georeferenced image.
        filepath = os.path.join(self.map_dir,
                                self.basemap_filename)
        basemap = rasterio.open(filepath)

        # Load the network boundary.
        filepath = os.path.join(self.map_dir,
                                self.boundary_filename)
        self.mss_boundary = geopandas.read_file(filepath)
        self.mss_boundary = self.mss_boundary.to_crs(self.projection.proj4_init)

        # Configure the colormap.
        self.cmap = plt.get_cmap('plasma')
        upper_limit = 1e-2
        max_pgv = np.max(list(self.meta['metadata']['max_event_pgv'].values()))
        if max_pgv > upper_limit:
            upper_limit = max_pgv
        pgv_limits = np.array((1e-6, upper_limit))
        pgv_limits_log = np.log10(pgv_limits)
        self.norm = matplotlib.colors.Normalize(vmin = pgv_limits_log[0],
                                                vmax = pgv_limits_log[1])

        # Create the figure and plot the base map.
        self.fig = plt.figure(figsize = (125/25.4, 100/25.4),
                              dpi = 300)
        self.ax = plt.axes(projection = self.projection)
        rasterio.plot.show(basemap,
                           origin = 'upper',
                           interpolation = None,
                           ax = self.ax,
                           zorder = 1)

        self.draw_pgv_colorbar()

        # Set the map axes to the figure limits.
        self.ax.set_position([0, 0, 1, 1])

        self.artists = []


    def clear_map(self):
        ''' Remove all data artists from the map.
        '''
        for cur_artist in self.artists:
            cur_artist.remove()
            del cur_artist

        self.artists = []

        
    def draw_pgv_colorbar(self):
        ''' Draw the PGV colorbar.
        '''
        # Create the inset axes.
        ax_inset = self.ax.inset_axes(bounds = [0.58, 0.05, 0.4, 0.05])

        ticks = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        ticks_log = np.log10(ticks)
        cb = matplotlib.colorbar.ColorbarBase(ax_inset,
                                              cmap = self.cmap,
                                              norm = self.norm,
                                              ticks = ticks_log,
                                              orientation = 'horizontal',
                                              extend = 'both')
        cb.ax.tick_params(labelsize = 8)
        cb.ax.set_xlabel('PGV [mm/s]',
                         loc = 'right',
                         fontsize = 8)
        cb.ax.xaxis.tick_top()
        ticks_mm = ticks * 1000
        tick_labels = ["{0:.3f}".format(x) for x in ticks_mm]
        cb.set_ticklabels(tick_labels)
        cb.ax.set_xticklabels(tick_labels, rotation = 90)
        self.cb = cb


    def draw_voronoi_cells(self, df, use_sa = False):
        ''' Draw PGV Voronoi cells.
        '''
        cmap = self.cmap
        norm = self.norm
        
        # Draw the cells that didn't trigger
        cur_df = df[df.triggered == False]
        if use_sa:
            color_list = [cmap(norm(x)) for x in cur_df['pgv_corr_log']]
        else:
            color_list = [cmap(norm(x)) for x in cur_df['pgv_log']]
    
        artists = []
        # Draw the polygon faces.
        cur_artist = self.ax.add_geometries(cur_df['geometry'],
                                            crs = self.projection,
                                            facecolor = color_list,
                                            edgecolor = None,
                                            alpha = 0.3,
                                            zorder = 3)
        artists.append(cur_artist)
        
        # Draw the polygon edges
        cur_artist = self.ax.add_geometries(cur_df['geometry'],
                                            crs = self.projection,
                                            facecolor = (1, 1, 1, 0),
                                            edgecolor = 'darkgray',
                                            linewidth = 0.1,
                                            zorder = 4)
        artists.append(cur_artist)

        # Draw the simplices that have an active trigger.
        cur_df = df[(df.triggered == True)]
        if use_sa:
            color_list = [cmap(norm(x)) for x in cur_df['pgv_corr_log']]
        else:
            color_list = [cmap(norm(x)) for x in cur_df['pgv_log']]
            
        # Draw the polygon faces.
        cur_artist = self.ax.add_geometries(cur_df['geometry'],
                                            crs = self.projection,
                                            facecolor = color_list,
                                            edgecolor = None,
                                            alpha = 0.8,
                                            zorder = 3)
        artists.append(cur_artist)
        # Draw the polygon edges
        cur_artist = self.ax.add_geometries(cur_df['geometry'],
                                            crs = self.projection,
                                            facecolor = (1, 1, 1, 0),
                                            edgecolor = 'k',
                                            linewidth = 0.2,
                                            zorder = 6)
        artists.append(cur_artist)
    
        self.artists.extend(artists)
        

    def draw_station_pgv(self, df, use_sa = False):
        ''' Draw the max pgv values of the stations.
        '''
        cmap = self.cmap
        norm = self.norm
        
        artists = []
    
        # Draw the stations that have data and have been triggered.
        cur_df = df[(df.pgv.notna() & (df.triggered == True))]
        if use_sa:
            color_list = [cmap(norm(x)) for x in cur_df['pgv_corr_log']]
        else:
            color_list = [cmap(norm(x)) for x in cur_df['pgv_log']]

        cur_artist = self.ax.scatter(x = cur_df.geometry.x,
                                     y = cur_df.geometry.y,
                                     transform = self.projection,
                                     s = 15,
                                     color = color_list,
                                     edgecolor = 'k',
                                     linewidth = 0.2,
                                     zorder = 10)
        artists.append(cur_artist)
    
        # Draw the stations that have data and have not been triggered.
        cur_df = df[(df.pgv.notna() & (df.triggered == False))]
        if use_sa:
            color_list = [cmap(norm(x)) for x in cur_df['pgv_corr_log']]
        else:
            color_list = [cmap(norm(x)) for x in cur_df['pgv_log']]
            cur_artist = self.ax.scatter(x = cur_df.geometry.x,
                                         y = cur_df.geometry.y,
                                         transform = self.projection,
                                         s = 15,
                                         color = color_list,
                                         edgecolor = 'k',
                                         linewidth = 0.2,
                                         alpha = 0.5,
                                         zorder = 10)
            artists.append(cur_artist)
    
            # Draw the stations that have no data.
            cur_df = df[df.pgv.isna()]
            if use_sa:
                color_list = [cmap(norm(x)) for x in cur_df['pgv_corr_log']]
            else:
                color_list = [cmap(norm(x)) for x in cur_df['pgv_log']]
                cur_artist = self.ax.scatter(x = cur_df.geometry.x,
                                             y = cur_df.geometry.y,
                                             transform = self.projection,
                                             s = 5,
                                             color = 'darkgray',
                                             edgecolor = 'k',
                                             linewidth = 0.2,
                                             zorder = 9)
                artists.append(cur_artist)
    
        self.artists.extend(artists)


    def draw_time_marker(self, time = None,
                         duration = None, note = None):
        ''' Draw the detection frame time marker.
        '''
        artists = []
        marker_text = self.event_public_id
    
        if time is not None:
            marker_text += '\n' + time.isoformat()
        
        if duration is not None:
            marker_text += "\nduration: {duration:.0f} s".format(duration = np.ceil(duration))
        
        if note is not None:
            marker_text += '\n' + note

        # Add the time marker.
        cur_artist = self.ax.text(x = 0.99,
                                  y = 0.99,
                                  s = marker_text,
                                  ha = 'right',
                                  va = 'top',
                                  fontsize = 6,
                                  transform = self.ax.transAxes)
        artists.append(cur_artist)
    
        self.artists.extend(artists)

    
    def draw_pgv_level(self, df, use_sa = False):
        ''' The the maximum pgv marker in the colorbar axes.
        '''
        ax = self.cb.ax
        artists = []
        if use_sa:
            pgv = df.pgv_corr[(df.triggered == True)]
        else:
            pgv = df.pgv[(df.triggered == True)]
    
        if len(pgv) > 0:
            pgv = np.nanmax(pgv)
            pgv_log = np.log10(pgv)
            cur_artist = ax.axvline(x = pgv_log,
                                    color = 'k',
                                    zorder = 10)
            artists.append(cur_artist)
        
            # Add value text.
            pgv_mm = pgv * 1000
            marker_text = "max: {pgv:.3f} mm/s ".format(pgv = pgv_mm)
            if pgv_mm >= 0.1:
                ha = 'right'
            else:
                ha = 'left'
            y_lim = ax.get_ylim()
            center = (y_lim[0] + y_lim[1]) / 2
            cur_artist = ax.text(x = pgv_log,
                                 y = center,
                                 s = marker_text,
                                 ha = ha,
                                 va = 'center',
                                 fontsize = 6)
            artists.append(cur_artist)
    
        self.artists.extend(artists)

    def draw_detection_pgv_level(self, df,
                                 max_event_pgv = None,
                                 add_annotation = True):
        ''' The the maximum pgv markers in the colorbar axes.
        '''
        ax = self.cb.ax
        artists = []
        
        pgv = df.pgv_max_log[(df.triggered == True)]
    
        if len(pgv) > 0:
            max_pgv = np.nanmax(pgv)
            if max_event_pgv is None:
                max_event_pgv = max_pgv
            elif max_pgv > max_event_pgv:
                max_event_pgv = max_pgv

            cur_artist = ax.axvline(x = max_pgv,
                                    color = 'gray',
                                    zorder = 9)
            artists.append(cur_artist)

        if max_event_pgv is not None:
            cur_artist = ax.axvline(x = max_event_pgv,
                                    color = 'k',
                                    zorder = 10)
            artists.append(cur_artist)

            if add_annotation:
                # Add value text.
                pgv_mm = 10**max_event_pgv * 1000
                marker_text = " max: {pgv:.3f} mm/s ".format(pgv = pgv_mm)
                if pgv_mm >= 0.1:
                    ha = 'right'
                else:
                    ha = 'left'
                y_lim = ax.get_ylim()
                center = (y_lim[0] + y_lim[1]) / 2
                cur_artist = ax.text(x = max_event_pgv,
                                     y = center,
                                     s = marker_text,
                                     ha = ha,
                                     va = 'center',
                                     fontsize = 6)
                artists.append(cur_artist)

        self.artists.extend(artists)
        return max_event_pgv


    def draw_simplices(self, df):
        ''' Draw the detection simplices. 
        '''
        cmap = self.cmap
        norm = self.norm
        
        # Draw the simplices that didn't trigger
        cur_df = df[(df.triggered == False) & (df.added_to_event == False)]
        color_list = [cmap(norm(x)) for x in cur_df['pgv_min_log']]
    
        artists = []
        # Draw the polygon faces.
        cur_artist = self.ax.add_geometries(cur_df['geometry'],
                                            crs = self.projection,
                                            facecolor = color_list,
                                            edgecolor = None,
                                            alpha = 0.3,
                                            zorder = 3)
        artists.append(cur_artist)
    
        # Draw the polygon edges
        cur_artist = self.ax.add_geometries(cur_df['geometry'],
                                            crs = self.projection,
                                            facecolor = (1, 1, 1, 0),
                                            edgecolor = 'darkgray',
                                            linewidth = 0.2,
                                            zorder = 4)
        artists.append(cur_artist)

        # Draw the simplices that have an active trigger.
        cur_df = df[(df.triggered == True)]
        color_list = [cmap(norm(x)) for x in cur_df['pgv_min_log']]
        # Draw the polygon faces.
        cur_artist = self.ax.add_geometries(cur_df['geometry'],
                                            crs = self.projection,
                                            facecolor = color_list,
                                            edgecolor = None,
                                            alpha = 0.8,
                                            zorder = 3)
        artists.append(cur_artist)
        # Draw the polygon edges
        cur_artist = self.ax.add_geometries(cur_df['geometry'],
                                            crs = self.projection,
                                            facecolor = (1, 1, 1, 0),
                                            edgecolor = 'maroon',
                                            linewidth = 0.4,
                                            zorder = 6)
        artists.append(cur_artist)


        # Draw the simplices that have no trigger, but have been added to the event.
        cur_df = df[(df.triggered == False) & (df.added_to_event == True)]
        color_list = [cmap(norm(x)) for x in cur_df['pgv_min_log']]
        # Draw the polygon faces.
        cur_artist = self.ax.add_geometries(cur_df['geometry'],
                                            crs = self.projection,
                                            facecolor = color_list,
                                            edgecolor = None,
                                            alpha = 0.8,
                                            zorder = 3)
        artists.append(cur_artist)
        # Draw the polygon edges
        cur_artist = self.ax.add_geometries(cur_df['geometry'],
                                            crs = self.projection,
                                            facecolor = (1, 1, 1, 0),
                                            edgecolor = 'darkolivegreen',
                                            linewidth = 0.6,
                                            zorder = 5)
        artists.append(cur_artist)
    
        self.artists.extend(artists)

    def draw_detection_stations(self, df):
        ''' Draw the detecion stations.
        '''
        cmap = self.cmap
        norm = self.norm
        artists = []
    
        # Get the max pgv of stations with a pgv value.
        with_data_df = df[df['pgv'].notna()]
        colorlist = [cmap(norm(x)) for x in with_data_df.pgv_log]
    
        # Plot the stations used for detection.
        x_coord = [x.geometry.x for x in with_data_df.itertuples()]
        y_coord = [x.geometry.y for x in with_data_df.itertuples()]
        cur_artist = self.ax.scatter(x_coord, y_coord,
                                     transform = self.projection,
                                     s = 15,
                                     color = colorlist,
                                     edgecolor = 'k',
                                     linewidth = 0.2,
                                     zorder = 9)
        artists.append(cur_artist)

        # Plot the stations not used for detection.
        no_data_df = df[df['pgv'].isna()]
        colorlist = [cmap(norm(x)) for x in no_data_df.pgv_log]
        x_coord = [x.geometry.x for x in no_data_df.itertuples()]
        y_coord = [x.geometry.y for x in no_data_df.itertuples()]
        cur_artist = self.ax.scatter(x_coord, y_coord,
                                     transform = self.projection,
                                     s = 5,
                                     color = 'darkgray',
                                     edgecolor = 'k',
                                     linewidth = 0.2,
                                     zorder = 10)
        artists.append(cur_artist)
        
        self.artists.extend(artists)


    def draw_boundary(self):
        ''' Draw the MSS network boundary.
        '''
        artists = []
        cur_artist = self.ax.add_geometries(self.mss_boundary['geometry'],
                                            crs = self.projection,
                                            facecolor = (1, 1, 1, 0),
                                            edgecolor = 'turquoise',
                                            linestyle = 'dashed',
                                            zorder = 2)
        artists.append(cur_artist)
        self.artists.extend(artists)

        
    def create_movie(self, image_dir, output_dir,
                     name, file_ext = 'png'):
        ''' Create a movie using ffmpeg.
        '''
        img_filepath = os.path.join(image_dir,
                                    '{public_id}_{name}_*.{ext}'.format(public_id = self.event_public_id,
                                                                        name = name,
                                                                        ext = file_ext))
        movie_filepath = os.path.join(output_dir,
                                      '{public_id}_detectionsequence.mp4'.format(public_id = self.event_public_id))
                                                                             
        stream = ffmpeg.input(img_filepath,
                              pattern_type = 'glob',
                              framerate = 2)
        stream = stream.filter('scale',
                               size='hd1080',
                               force_original_aspect_ratio='increase')
        stream = stream.output(filename = movie_filepath)
        stream = stream.overwrite_output()
        stream = stream.run()
        

    def create_event_pgv_map(self):
        ''' Create the event PGV map.
        '''
        # Initialize the map.
        if self.fig is None:
            self.init_map(utm_zone = 33)
        else:
            self.clear_map()

        # Create the output directory.
        output_dir = os.path.join(self.output_dir,
                                  self.event_dir,
                                  'eventpgv',
                                  'images')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        

        # Read the event pgv voronoi data.
        event_pgv_df = util.get_supplement_data(public_id = self.event_public_id,
                                                category = 'eventpgv',
                                                name = 'pgvvoronoi',
                                                directory = self.supplement_dir)
        event_pgv_props = event_pgv_df.attrs
        print(event_pgv_props)
    
        # Convert the pandas dataframe to cartopy projection.
        event_pgv_df = event_pgv_df.to_crs(self.projection.proj4_init)
        # Add the logarithmic pgv values.
        event_pgv_df.insert(3, "pgv_corr", event_pgv_df.pgv / event_pgv_df.sa)
        event_pgv_df.insert(4, "pgv_log", np.log10(event_pgv_df.pgv))
        event_pgv_df.insert(5, "pgv_corr_log", np.log10(event_pgv_df.pgv_corr))


        # Read the station pgv data.
        station_pgv_df = util.get_supplement_data(public_id = self.event_public_id,
                                                  category = 'eventpgv',
                                                  name = 'pgvstation',
                                                  directory = self.supplement_dir)
        station_pgv_props = station_pgv_df.attrs
        # Convert the pandas dataframe to cartopy projection.
        station_pgv_df = station_pgv_df.to_crs(self.projection.proj4_init)
        # Compute the corrected and the logarithmic pgv data.
        station_pgv_df.insert(4, "pgv_corr", station_pgv_df.pgv / station_pgv_df.sa)
        station_pgv_df.insert(5, 'pgv_log', np.log10(station_pgv_df.pgv))
        station_pgv_df.insert(6, 'pgv_corr_log', np.log10(station_pgv_df.pgv_corr))
        

        # Draw the voronoi cells.
        self.draw_voronoi_cells(df = event_pgv_df)

        # Plot the station max pgv markers.
        self.draw_station_pgv(df = station_pgv_df)

        # Draw the public id and the time marker.
        from_zone = dateutil.tz.gettz('UTC')
        to_zone = dateutil.tz.gettz('CET')
        event_start = obspy.UTCDateTime(station_pgv_props['event_start'])
        event_end = obspy.UTCDateTime(station_pgv_props['event_end'])
        event_start_local = event_start.datetime.replace(tzinfo = from_zone).astimezone(to_zone)
        self.draw_time_marker(duration = event_end - event_start,
                              time = event_start_local)

        # Draw the max. pgv level indicator in the colorbar axes.
        self.draw_pgv_level(df = station_pgv_df)

        # Save the map image.
        filename = self.event_public_id + '_pgvvoronoi.png'
        filepath = os.path.join(output_dir,
                                filename)
        self.fig.savefig(filepath,
                         dpi = 300,
                         pil_kwargs = {'quality': 85},
                         bbox_inches = 'tight',
                         pad_inches = 0,)

        
        # Plot the same data with the station correction applied.
        self.clear_map()
        
        # Draw the voronoi cells.
        self.draw_voronoi_cells(df = event_pgv_df,
                                use_sa = True)

        # Plot the station max pgv markers.
        self.draw_station_pgv(df = station_pgv_df,
                              use_sa = True)

        # Draw the max. pgv level indicator in the colorbar axes.
        self.draw_pgv_level(df = station_pgv_df,
                            use_sa = True)
        
        # Draw the public id and the time marker.
        self.draw_time_marker(duration = event_end - event_start,
                              time = event_start_local,
                              note = 'station correction applied')

        # Save the map image.
        filename = self.event_public_id + '_pgvvoronoi_corr.png'
        filepath = os.path.join(output_dir,
                                filename)
        self.fig.savefig(filepath,
                         dpi = 300,
                         pil_kwargs = {'quality': 85},
                         bbox_inches = 'tight',
                         pad_inches = 0,)


    def create_detection_sequence_movie(self):
        ''' Create the movie of the detection sequence.
        '''
        # Initialize the map.
        if self.fig is None:
            self.init_map(utm_zone = 33)
        else:
            self.clear_map()
            
        # Create the output directory.
        img_output_dir = os.path.join(self.output_dir,
                                      self.event_dir,
                                      'detectionsequence',
                                      'images')
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)
            
        movie_output_dir = os.path.join(self.output_dir,
                                        self.event_dir,
                                        'detectionsequence',
                                        'movie')
        if not os.path.exists(movie_output_dir):
            os.makedirs(movie_output_dir)

        # Load the detection sequence data from the geojson file.
        sequ_df = util.get_supplement_data(public_id = self.event_public_id,
                                           category = 'detectionsequence',
                                           name = 'simplices',
                                           directory = self.supplement_dir)
        sequ_df_props = sequ_df.attrs

        # Convert the geopandas dataframe to cartopy projection.
        sequ_df = sequ_df.to_crs(self.projection.proj4_init)
        
        # Add the logarithmic pgv values.
        sequ_df.insert(5, "pgv_min_log", np.log10(sequ_df.pgv_min))
        sequ_df.insert(6, "pgv_max_log", np.log10(sequ_df.pgv_max))

        # Load the station pgv sequence data from the geojson file.
        stat_df = util.get_supplement_data(public_id = self.event_public_id,
                                           category = 'pgvsequence',
                                           name = 'pgvstation',
                                           directory = self.supplement_dir)
        stat_df_props = stat_df.attrs

        # convert the geopandas dataframe to cartopy projection.
        stat_df = stat_df.to_crs(self.projection.proj4_init)

        stat_df.insert(3, "pgv_log", np.log10(stat_df.pgv))

        # Initialize the maximum event PGV value used to plot the PGV level.
        max_event_pgv = None

        # Set the time zones for conversion.
        from_zone = dateutil.tz.gettz('UTC')
        to_zone = dateutil.tz.gettz('CET')
        
        # Iterate through the time groups.
        time_groups = sequ_df.groupby('time')
        stat_time_groups = stat_df.groupby('time')
        
        for cur_name, cur_group in enumerate(time_groups):
            cur_time = obspy.UTCDateTime(cur_name)

            # Get the related pgvstation frame.
            cur_stat_df = stat_time_groups.get_group(cur_name)

            # Convert to local time.
            cur_time_local = cur_time.datetime.replace(tzinfo = from_zone).astimezone(to_zone)

            # Draw the network boundary.
            self.draw_boundary()

            # Draw the detection triangles.
            self.draw_simplices(df = cur_group)

            # Draw the station markers.
            self.draw_detection_stations(df = cur_stat_df)

            # Draw the time information.
            self.draw_time_marker(time = cur_time_local)

            # Draw the PGV level.
            max_event_pgv = self.draw_detection_pgv_level(df = cur_group,
                                                          max_event_pgv = max_event_pgv)
                
            cur_date_string = cur_time.isoformat().replace(':', '').replace('.', '')
            cur_filename = self.event_public_id + '_detectionframe_' + cur_date_string + '.png'
            cur_filepath = os.path.join(img_output_dir,
                                        cur_filename)
            self.fig.savefig(cur_filepath,
                             dpi = 300,
                             pil_kwargs = {'quality': 80},
                             bbox_inches = 'tight',
                             pad_inches = 0,)
            self.clear_map()

        self.create_movie(image_dir = img_output_dir,
                          output_dir = movie_output_dir,
                          name = 'detectionframe')