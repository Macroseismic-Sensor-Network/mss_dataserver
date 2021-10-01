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
''' Utilities to plot maps.
'''

import dateutil
import logging
import os

import cartopy.crs as ccrs
import ffmpeg
import geopandas
import matplotlib
import matplotlib.colors
import matplotlib.collections as mpl_col
import matplotlib.pyplot as plt
import numpy as np
import obspy
import rasterio
import rasterio.plot

import mss_dataserver.postprocess.util as util

class MapPlotter(object):
    ''' Create map images and movies using mssds geojson data.

    Parameters
    ----------
    supplement_dir: str 
        The directory where the supplement data is saved.

    map_dir: str 
        The directory where the map data is saved.

    output_dir: str 
        The directory where to save the computed map images.

    basemap: str 
        The filename of the background map in geotiff file format.

    boundary: str 
        The filename of the network boundary in geojson file format.
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

        # The intensity labels.
        self.intensity_labels = {1: 'I',
                                 2: 'II',
                                 3: 'III',
                                 4: 'IV',
                                 5: 'V',
                                 6: 'VI',
                                 7: 'VII',
                                 8: 'VIII',
                                 9: 'IX',
                                 10: 'X',
                                 11: 'XI',
                                 12: 'XII'}

        # The plot mode ['pgv', 'intensity']
        self.mode = 'pgv'


    def set_event(self, public_id):
        ''' Set the event to process.

        Parameters
        ----------
        public_id: str 
            The public id of the event.
        '''
        self.event_public_id = public_id
        self.event_dir = util.event_dir_from_publicid(public_id)
        # Load the event metadata from the supplement file.
        self.meta = util.get_supplement_data(self.event_public_id,
                                             category = 'detectiondata',
                                             name = 'metadata',
                                             directory = self.supplement_dir)

        
    def init_map(self, utm_zone=33, mode = 'pgv', colorbar_pgv_labels = False,
                 draw_basemap = True, draw_colorbar = True, draw_logo = True,
                 draw_attribution_note = True):
        ''' Initialize the map plot.

        Parameters
        ----------
        utm_zone: int 
            The UTM zone number.
        '''
        # Set the mode.
        self.mode = mode
        
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


        # Create the qualitative colormap for intensities.
        intensity_list = np.array([1, 2, 3, 4, 5, 6, 7])
        #intensity_limits = util.pgv_to_intensity(pgv_limits)[:, 1]
        #intensity_list = np.arange(1, intensity_limits[1] + 1, 1)
        intensity_pgv = util.intensity_to_pgv(intensity_list)
        color_list = self.cmap(self.norm(np.log10(intensity_pgv[:, 1])))
        self.cmap_qualitative_intensity = matplotlib.colors.ListedColormap(color_list)
        self.norm_qualitative_intensity = matplotlib.colors.BoundaryNorm(np.log10(intensity_pgv[:, 1]),
                                                                         ncolors = len(intensity_list))

        # Create the figure and plot the base map.
        self.fig = plt.figure(figsize = (120 / 25.4, 100 / 25.4),
                              dpi = 300)
        self.ax = plt.axes(projection = self.projection)

        # Set the map axes to the figure limits.
        self.ax.set_position([0, 0, 1, 1])

        # Plot the background map.
        if draw_basemap:
            rasterio.plot.show(basemap,
                               origin = 'upper',
                               interpolation = None,
                               ax = self.ax,
                               zorder = 1)
        else:
            rasterio.plot.show(basemap,
                               origin = 'upper',
                               interpolation = None,
                               ax = self.ax,
                               alpha = 0,
                               zorder = 1)

        # Add the colorbar.
        if draw_colorbar:
            if self.mode == 'pgv':
                self.draw_pgv_colorbar()
            elif self.mode == 'pgv_categorized':
                self.draw_pgv_colorbar_categorized(pgv_labels = colorbar_pgv_labels)
            elif self.mode == 'intensity':
                self.draw_intensity_colorbar(pgv_labels = colorbar_pgv_labels)
            

        # Add the MSS logo.
        if draw_logo:
            logo_filepath = os.path.join(self.map_dir, 'mss_logo.png')
            logo = plt.imread(logo_filepath)
            self.fig.figimage(logo,
                              origin = 'upper',
                              yo = 1100,
                              xo = 15)

        # Add the attribution.
        if draw_attribution_note:
            self.draw_attribution_note()

        self.ax.set_axis_off()

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
        cb_bounds = [0.58, 0.08, 0.4, 0.05]
        ax_inset = self.ax.inset_axes(bounds = cb_bounds)

        intensity_to_plot = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        intensity_pgv = util.intensity_to_pgv(intensity_to_plot)

        ticks = intensity_pgv[:, 1]
        ticks[0] = 0.001e-3
        ticks_log = np.log10(ticks)
        cb = matplotlib.colorbar.ColorbarBase(ax_inset,
                                              cmap = self.cmap,
                                              norm = self.norm,
                                              ticks = ticks_log,
                                              orientation = 'horizontal',
                                              extend = 'both')
        
        cb.ax.tick_params(labelsize = 8)
        #cb.ax.set_xlabel('PGV [mm/s]',
        #                 loc = 'right',
        #                 fontsize = 8)
        cb.ax.xaxis.tick_top()
        ticks_mm = ticks * 1000
        tick_labels = ["{0:.3f}".format(x) for x in ticks_mm]
        cb.set_ticklabels(tick_labels)
        cb.ax.set_xticklabels(tick_labels, rotation = 90)
        self.cb = cb      

        height = 0.03
        intensity_bounds = [cb_bounds[0],
                            cb_bounds[1] - height,
                            cb_bounds[2],
                            height]
        ax_inset = self.ax.inset_axes(bounds = intensity_bounds,
                                      xlim = cb.ax.get_xlim())
        
        

        # Add the intensity label:
        xlim = ax_inset.get_xlim()
        ax_inset.text(x = xlim[0] + 0.1,
                      y = 0.45,
                      s = 'intensity: ',
                      ha = 'left',
                      va = 'center',
                      fontsize = 6)
        ax_inset.set_xlabel('PGV [mm/s]',
                            loc = 'right',
                            fontsize = 8)
       
        for k, cur_intensity_pgv in enumerate(intensity_pgv):
            if k == len(intensity_pgv) - 1:
                break

            if np.log10(cur_intensity_pgv[1]) >= xlim[1]:
                break
           
            if k > 0:
                ax_inset.axvline(np.log10(cur_intensity_pgv[1]),
                                 color = 'black',
                                 linewidth = 0.5)
                cur_x = (np.log10(intensity_pgv[k + 1][1]) + np.log10(cur_intensity_pgv[1])) / 2
   
            if k == 0:
                cur_x = (np.log10(intensity_pgv[k + 1][1]) + np.log10(0.01e-3)) / 2
            elif np.log10(intensity_pgv[k +1][1]) >= xlim[1]:
                cur_x = (np.log10(cur_intensity_pgv[1]) + xlim[1]) / 2

            #ax_inset.axvline(cur_x, color = 'gray')
            #cur_label = '{intensity:.0f}'.format(intensity = cur_intensity_pgv[0])
            cur_label = self.intensity_labels[int(cur_intensity_pgv[0])]
            ax_inset.text(cur_x,
                          y = 0.45,
                          s = cur_label,
                          ha = 'center',
                          va = 'center',
                          fontsize = 6)
            #ax_inset.set_axis_off()
            #ax_inset.set_frame_on(True)
            #ax_inset.get_xaxis().set_visible(False)
            ax_inset.set_xticks([])
            ax_inset.get_yaxis().set_visible(False)
            ax_inset.set_facecolor((1, 1, 1, 0.4))


    def draw_intensity_colorbar(self, pgv_labels = False):
        ''' Draw the intensity colorbar.
        '''
        # Create the inset axes.
        cb_bounds = [0.58, 0.08, 0.4, 0.05]
        ax_inset = self.ax.inset_axes(bounds = cb_bounds)

        intensity_to_plot = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        intensity_pgv = util.intensity_to_pgv(intensity_to_plot)

        ticks = intensity_pgv[:, 1]
        ticks[0] = 0.001e-3
        ticks_log = np.log10(ticks)
        cb = matplotlib.colorbar.ColorbarBase(ax_inset,
                                              cmap = self.cmap_qualitative_intensity,
                                              norm = self.norm_qualitative_intensity,
                                              ticks = ticks_log,
                                              orientation = 'horizontal',
                                              extend = 'both',
                                              spacing = 'proportional')

        for cur_x in ticks_log[1:]:
            cb.ax.axvline(x = cur_x,
                          color = 'k',
                          linewidth = 0.5)
        
        cb.ax.tick_params(labelsize = 8)
        #cb.ax.set_xlabel('PGV [mm/s]',
        #                 loc = 'right',
        #                 fontsize = 8)
        cb.ax.xaxis.tick_top()
        ax_ticks = cb.ax.get_xticks()
        if pgv_labels:
            ticks_mm = 10**ax_ticks * 1000
            tick_labels = ["{0:.2f}".format(x) for x in ticks_mm]
            cb.ax.set_xticklabels(tick_labels, rotation = 90)
        else:
            tick_labels = []
            cb.ax.set_xticklabels(tick_labels)
            cb.set_ticks([], update_ticks = True)
        self.cb = cb

        height = 0.03
        intensity_bounds = [cb_bounds[0],
                            cb_bounds[1] - height,
                            cb_bounds[2],
                            height]
        ax_inset = self.ax.inset_axes(bounds = intensity_bounds,
                                      xlim = cb.ax.get_xlim())
        
        

        # Add the intensity label:
        xlim = ax_inset.get_xlim()
        #ax_inset.text(x = xlim[0] - 0.01,
        #              y = 0.45,
        #              s = 'Intensität: ',
        #              ha = 'right',
        #              va = 'center',
        #              fontsize = 6)
        ax_inset.set_xlabel('Intensität',
                            loc = 'right',
                            fontsize = 8)
       
        for k, cur_intensity_pgv in enumerate(intensity_pgv):
            if np.log10(cur_intensity_pgv[1]) >= xlim[1]:
                break
           
            if k > 0:
                ax_inset.axvline(np.log10(cur_intensity_pgv[1]),
                                 color = 'black',
                                 linewidth = 0.5)
                cur_x = (np.log10(intensity_pgv[k + 1][1]) + np.log10(cur_intensity_pgv[1])) / 2

            # Don't plot the label of the extending intensity.
            if k == len(ax_ticks) - 1:
                break
            
            if k == 0:
                cur_x = (np.log10(intensity_pgv[k + 1][1]) + xlim[0]) / 2
            elif np.log10(intensity_pgv[k +1][1]) >= xlim[1]:
                cur_x = (np.log10(cur_intensity_pgv[1]) + xlim[1]) / 2

            #ax_inset.axvline(cur_x, color = 'gray')
            #cur_label = '{intensity:.0f}'.format(intensity = cur_intensity_pgv[0])
            cur_label = self.intensity_labels[int(cur_intensity_pgv[0])]
            ax_inset.text(cur_x,
                          y = 0.45,
                          s = cur_label,
                          ha = 'center',
                          va = 'center',
                          fontsize = 6)
            #ax_inset.set_axis_off()
            #ax_inset.set_frame_on(True)
            #ax_inset.get_xaxis().set_visible(False)
            ax_inset.set_xticks([])
            ax_inset.get_yaxis().set_visible(False)
            ax_inset.set_facecolor((1, 1, 1, 0.4))
            

    def draw_voronoi_cells(self, df, use_sa = False):
        ''' Draw PGV Voronoi cells.

        Parameters
        ----------
        df: :class:`geopandas.GeoDataFrame`
            The dataframe used to compute the voronoi cells.

        use_sa: bool
            True: used the station correction factors.
            False: don't use the station correction factors.
        '''
        cmap = self.cmap
        norm = self.norm

        # Use only the cells with valid data. When using
        # station corrections, some cells might have no data.
        if use_sa:
            df_nodata = df[df.pgv_corr_log.isna()]
            df = df[df.pgv_corr_log.notna()]
        else:
            df_nodata = df[df.pgv.isna()]
            df = df[df.pgv.notna()] 

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
                                            edgecolor = 'k',
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


        # Plot the polygons that have no valid data.
        # Draw the polygon faces. 
        cur_artist = self.ax.add_geometries(df_nodata['geometry'],
                                            crs = self.projection,
                                            facecolor = 'darkgray',
                                            edgecolor = None,
                                            alpha = 0.3,
                                            zorder = 3)
        artists.append(cur_artist)
        # Draw the polygon edges
        cur_artist = self.ax.add_geometries(df_nodata['geometry'],
                                            crs = self.projection,
                                            facecolor = (1, 1, 1, 0),
                                            edgecolor = 'k',
                                            linewidth = 0.2,
                                            zorder = 6)
        artists.append(cur_artist)
    
        self.artists.extend(artists)
        

    def draw_station_pgv(self, df, use_sa = False,
                         max_dia = 3):
        ''' Draw the max pgv values of the stations.

        Parameters
        ----------
        df: :class:`geopandas.GeoDataFrame`
            The dataframe used to compute the station pgv.

        use_sa: bool
            True: used the station correction factors.
            False: don't use the station correction factors.

        max_dia: float 
            The maximum marker diameter [mm].
        '''
        cmap = self.cmap
        norm = self.norm
        
        artists = []

        if use_sa:
            data_col = 'pgv_corr_log'
        else:
            data_col = 'pgv_log'

        # TODO:
        # Use the EllipseCollection for a better constraint of the
        # scatter circle radius.
        # Use the pandas dataseries map() function to map the
        # PGV to a radius range.
        
        # Convert mm to inches.
        max_dia = max_dia / 25.4
    
        # Draw the stations that have data and have been triggered.
        cur_df = df[(df[data_col].notna() & (df.triggered == True))]
        if use_sa:
            color_list = [cmap(norm(x)) for x in cur_df[data_col]]
            size_list = [max_dia * norm(x) for x in cur_df[data_col]]
        else:
            color_list = [cmap(norm(x)) for x in cur_df[data_col]]
            size_list = [max_dia * norm(x) for x in cur_df[data_col]]

        cur_artist = self.ax.scatter(x = cur_df.geometry.x,
                                     y = cur_df.geometry.y,
                                     transform = self.projection,
                                     s = 15,
                                     color = color_list,
                                     edgecolor = 'k',
                                     linewidth = 0.2,
                                     zorder = 10)
    
        #offsets = list(zip(cur_df.geometry.x,
        #                   cur_df.geometry.y))
        #cur_ell = mpl_col.EllipseCollection(widths = size_list,
        #                                    heights = size_list,
        #                                    angles = 0,
        #                                    units = 'inches',
        #                                    edgecolor = 'k',
        #                                    facecolors = color_list,
        #                                    linewidth = 0.2,
        #                                    offsets = offsets,
        #                                    transOffset = self.ax.transData,
        #                                    zorder = 10)
        #self.ax.add_collection(cur_ell)
        artists.append(cur_artist)

        # Draw the stations that have data and have not been triggered.
        cur_df = df[(df[data_col].notna() & (df.triggered == False))]
        if use_sa:
            color_list = [cmap(norm(x)) for x in cur_df[data_col]]
        else:
            color_list = [cmap(norm(x)) for x in cur_df[data_col]]
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
        cur_df = df[df[data_col].isna()]
        if use_sa:
            color_list = [cmap(norm(x)) for x in cur_df[data_col]]
        else:
            color_list = [cmap(norm(x)) for x in cur_df[data_col]]
        cur_artist = self.ax.scatter(x = cur_df.geometry.x,
                                     y = cur_df.geometry.y,
                                     transform = self.projection,
                                     s = 5,
                                     color = 'darkgray',
                                     edgecolor = 'k',
                                     linewidth = 0.2,
                                     zorder = 11)
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

    
    def draw_pgv_level(self, df, use_sa = False,
                       max_event_pgv = None,
                       show_max_level = False,
                       add_annotation = True):
        ''' The the maximum pgv marker in the colorbar axes.

        Parameters
        ----------
        df: :class:`geopandas.GeoDataFrame`
            The dataframe used to compute the pgv level.

        use_sa: bool
            True: used the station correction factors.
            False: don't use the station correction factors.

        max_event_pgv: float 
            The maximum PGV of the event.

        show_max_level: bool
            If True, the maximum level is shown.

        add_annotation: bool
            If True, the annotation is added.
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

            if max_event_pgv is None:
                max_event_pgv = pgv
            elif pgv > max_event_pgv:
                max_event_pgv = pgv

            if show_max_level:
                line_color = 'gray'
            else:
                line_color = 'k'
            cur_artist = ax.axvline(x = pgv_log,
                                    color = line_color,
                                    zorder = 2)
            artists.append(cur_artist)
        
            # Add value text.
            if add_annotation and not show_max_level:
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
                                     fontsize = 6,
                                     zorder = 4)
                artists.append(cur_artist)

        if show_max_level and max_event_pgv is not None:
            cur_artist = ax.axvline(x = np.log10(max_event_pgv),
                                    color = 'k',
                                    zorder = 3)
            artists.append(cur_artist)

            if add_annotation:
                # Add value text.
                pgv_mm = max_event_pgv * 1000
                marker_text = " max: {pgv:.3f} mm/s ".format(pgv = pgv_mm)
                if pgv_mm >= 0.1:
                    ha = 'right'
                else:
                    ha = 'left'
                y_lim = ax.get_ylim()
                center = (y_lim[0] + y_lim[1]) / 2
                cur_artist = ax.text(x = np.log10(max_event_pgv),
                                     y = center,
                                     s = marker_text,
                                     ha = ha,
                                     va = 'center',
                                     fontsize = 6,
                                     zorder = 4)
                artists.append(cur_artist)
    
        self.artists.extend(artists)

        return max_event_pgv

        
    def draw_detection_pgv_level(self, df,
                                 max_event_pgv = None,
                                 add_annotation = True,
                                 data_col = 'pgv_min_log'):
        ''' The the maximum pgv markers in the colorbar axes.

        Parameters
        ----------
        df: :class:`geopandas.GeoDataFrame`
            The dataframe used to compute the detection pgv level.

        max_event_pgv: float 
            The maximum PGV of the event.

        add_annotation: bool
            If True, the annotation is added.

        data_col: str 
            The data column of the df used for the pgv data.
        '''
        ax = self.cb.ax
        artists = []
        
        pgv = df[data_col][(df.triggered == True)]
    
        if len(pgv) > 0:
            max_pgv = np.nanmax(pgv)
            if max_event_pgv is None:
                max_event_pgv = max_pgv
            elif max_pgv > max_event_pgv:
                max_event_pgv = max_pgv

            cur_artist = ax.axvline(x = max_pgv,
                                    color = 'gray',
                                    zorder = 2)
            artists.append(cur_artist)

        if max_event_pgv is not None:
            cur_artist = ax.axvline(x = max_event_pgv,
                                    color = 'k',
                                    zorder = 4)
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
                                     fontsize = 6,
                                     zorder = 3)
                artists.append(cur_artist)

        self.artists.extend(artists)
        return max_event_pgv


    def draw_pgv_colorbar_categorized(self, pgv_labels = False):
        ''' Draw the pgv colorbar using categorized colors.
        '''
        # Create the inset axes.
        cb_bounds = [0.58, 0.08, 0.4, 0.05]
        ax_inset = self.ax.inset_axes(bounds = cb_bounds)

        intensity_to_plot = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        intensity_pgv = util.intensity_to_pgv(intensity_to_plot)

        ticks = intensity_pgv[:, 1]
        ticks[0] = 0.001e-3
        ticks_log = np.log10(ticks)
        cb = matplotlib.colorbar.ColorbarBase(ax_inset,
                                              cmap = self.cmap_qualitative_intensity,
                                              norm = self.norm_qualitative_intensity,
                                              ticks = ticks_log,
                                              orientation = 'horizontal',
                                              extend = 'both',
                                              spacing = 'proportional')

        for cur_x in ticks_log[1:]:
            cb.ax.axvline(x = cur_x,
                          color = 'k',
                          linewidth = 0.5)
        
        cb.ax.tick_params(labelsize = 8)
        #cb.ax.set_xlabel('PGV [mm/s]',
        #                 loc = 'right',
        #                 fontsize = 8)
        cb.ax.xaxis.tick_top()
        ax_ticks = cb.ax.get_xticks()
        if pgv_labels:
            print(10**ax_ticks)
            ticks_mm = 10**ax_ticks * 1000
            tick_labels = ["{0:.3f}".format(x) if x <= 1e-3 else "{0:.2f}".format(x) for x in ticks_mm]
            cb.ax.set_xticklabels(tick_labels, rotation = 90)
        else:
            tick_labels = []
            cb.ax.set_xticklabels(tick_labels)
        self.cb = cb

        height = 0.03
        intensity_bounds = [cb_bounds[0],
                            cb_bounds[1] - height,
                            cb_bounds[2],
                            height]
        ax_inset = self.ax.inset_axes(bounds = intensity_bounds,
                                      xlim = cb.ax.get_xlim())
        
        

        # Add the intensity label:
        xlim = ax_inset.get_xlim()
        ax_inset.text(x = xlim[0] - 0.01,
                      y = 0.45,
                      s = 'Intensität: ',
                      ha = 'right',
                      va = 'center',
                      fontsize = 6)
        ax_inset.set_xlabel('PGV [mm/s]',
                            loc = 'right',
                            fontsize = 8)
       
        for k, cur_intensity_pgv in enumerate(intensity_pgv):
            if np.log10(cur_intensity_pgv[1]) >= xlim[1]:
                break
           
            if k > 0:
                ax_inset.axvline(np.log10(cur_intensity_pgv[1]),
                                 color = 'black',
                                 linewidth = 0.5)
                cur_x = (np.log10(intensity_pgv[k + 1][1]) + np.log10(cur_intensity_pgv[1])) / 2

            # Don't plot the label of the extending intensity.
            if k == len(ax_ticks) - 1:
                break
                
            if k == 0:
                cur_x = (np.log10(intensity_pgv[k + 1][1]) + xlim[0]) / 2
            elif np.log10(intensity_pgv[k + 1][1]) >= xlim[1]:
                cur_x = (np.log10(cur_intensity_pgv[1]) + xlim[1]) / 2

            #ax_inset.axvline(cur_x, color = 'gray')
            #cur_label = '{intensity:.0f}'.format(intensity = cur_intensity_pgv[0])
            cur_label = self.intensity_labels[int(cur_intensity_pgv[0])]
            ax_inset.text(cur_x,
                          y = 0.45,
                          s = cur_label,
                          ha = 'center',
                          va = 'center',
                          fontsize = 6)
            #ax_inset.set_axis_off()
            #ax_inset.set_frame_on(True)
            #ax_inset.get_xaxis().set_visible(False)
            ax_inset.set_xticks([])
            ax_inset.get_yaxis().set_visible(False)
            ax_inset.set_facecolor((1, 1, 1, 0.4))

    
    def draw_contours(self, df, draw_labels = False,
                      ignore_not_felt = True):
        ''' Draw the PGV contours.

        Parameters
        ----------
        df: :class:`geopandas.GeoDataFrame`
            The dataframe used to compute the contours.

        draw_labels: bool
            If True, the contour labels are plotted.
        '''
        cmap = self.cmap
        norm = self.norm
        artists = []

        # Ignore the contours below the felt threshold.
        if ignore_not_felt:
            felt = df['pgv'] >= 0.1e-6
            df = df[felt]

        # Remove the rows having no geometry.
        df = df[~df['geometry'].is_empty & df['geometry'].notna()]
      
        if len(df) == 0:
            return

        color_list = [cmap(norm(x)) for x in df['pgv_log']]
        pgv_intensity = util.intensity_to_pgv(np.arange(1, 9))
        
        # Draw the contour faces.
        cur_artist = self.ax.add_geometries(df['geometry'],
                                            crs = self.projection,
                                            facecolor = color_list,
                                            edgecolor = 'None',
                                            linewidth = 0,
                                            alpha = 1.0,
                                            zorder = 3)
        artists.append(cur_artist)

        import shapely
        mss_boundary = self.mss_boundary.geometry[0][0]
        mss_boundary_shrink = mss_boundary.buffer(-100)
        mss_boundary_split = mss_boundary.buffer(-120)
        #geometries = geometries[1:3]
        contour_groups = df.groupby('pgv')
        group_keys = list(contour_groups.groups.keys())

        # Initialize the first contour linewidth.
        mask = [np.any(np.isclose(x, pgv_intensity[:,1])) for x in group_keys]
        if np.any(mask):
            first_ind = np.array(mask).nonzero()[0][0]
            if first_ind % 2 == 0:
                contour_width = 'fat'
            else:
                contour_width = 'thin'   
        else:
            contour_width = 'fat'
                
        for cnt_group, (cur_name, cur_group) in enumerate(contour_groups):
            geometries = cur_group.geometry

            if contour_width == 'fat':
                linewidth = 0.4
                contour_width = 'thin'
            else:
                linewidth = 0.2
                contour_width = 'fat'

            if np.any(np.isclose(cur_name, pgv_intensity[:, 1])):
                linestyle = 'dashed'
                linewidth = 0.6
            else:
                linestyle = 'solid'
            
            for k, cur_geom in enumerate(geometries):
                if not cur_geom.within(mss_boundary):
                    cur_split = shapely.ops.split(cur_geom.boundary, mss_boundary_split.boundary)
                    cur_split = [x for x in cur_split if x.within(mss_boundary_shrink)]
                    edgecolor = 'k'
                else:
                    cur_split = [cur_geom.exterior]
                    edgecolor = 'k'

                if cnt_group < len(contour_groups) - 1:
                    next_group = contour_groups.get_group(group_keys[cnt_group + 1])
                    next_level_geom = next_group.geometry
                    for cur_next_level in next_level_geom:
                        cur_split = [x for x in cur_split if (not x.overlaps(cur_next_level.boundary)) and (not x.within(cur_next_level.buffer(50)))]

                cur_artist = self.ax.add_geometries(cur_split,
                                                    crs = self.projection,
                                                    edgecolor = edgecolor,
                                                    facecolor = (1, 1, 1, 0),
                                                    linewidth = linewidth,
                                                    linestyle = linestyle,
                                                    zorder = 4)
                artists.append(cur_artist)

                if draw_labels:
                    # Add the contour line annotations.
                    cur_mask = np.isclose(cur_name, pgv_intensity[:, 1])
                    if np.any(cur_mask) and (len(cur_split) > 0):
                        cur_intensity = int(pgv_intensity[cur_mask, 0][0])
                        if self.mode == 'pgv' or self.mode == 'pgv_categorized':
                            cur_label = '{pgv} ({intensity})'.format(pgv = np.round(cur_name * 1000, 3),
                                                                     intensity = self.intensity_labels[cur_intensity])
                        elif self.mode == 'intensity':
                            cur_label = '{intensity}'.format(intensity = self.intensity_labels[cur_intensity])
                            
                        longest_ind = np.argmax([x.length for x in cur_split])
                            
                        cur_geom = cur_split[longest_ind]
                        cur_artist = self.ax.annotate(cur_label,
                                                      xy = cur_geom.coords[0],
                                                      fontsize = 4,
                                                      ha = 'center',
                                                      va = 'center',
                                                      zorder = 20)
                        artists.append(cur_artist)    
        
        self.artists.extend(artists)


    def draw_contours_working(self, df, draw_labels = False,
                      draw_fat_contours = False):
        ''' Deprecated. Draw the PGV contours.
        '''
        cmap = self.cmap
        norm = self.norm
        artists = [];

        # Ignore the contours below the felt threshold.
        felt = df['pgv'] >= 0.1e-6
        df = df[felt]

        # Remove the rows having no geometry.
        df = df[df['geometry'].notna()]
      
        if len(df) == 0:
            return

        color_list = [cmap(norm(x)) for x in df['pgv_log']]
        linewidth_list = np.array([0.2] * len(df['pgv_log']))
        pgv_intensity = util.intensity_to_pgv(np.arange(1, 9))
        pgv_annotate = pgv_intensity[:, 1]
        
        if draw_fat_contours:    
            for cur_annotate in pgv_annotate:
                linewidth_list[np.isclose(df['pgv'], cur_annotate)] = 0.4

        # Plot the contours below the felt threshold.
        cur_artist = self.ax.add_geometries(df['geometry'],
                                            crs = self.projection,
                                            facecolor = color_list,
                                            edgecolor = 'black',
                                            linewidth = linewidth_list,
                                            alpha = 0.8,
                                            zorder = 3)
        artists.append(cur_artist)

        if draw_labels:
            # Add the contour line annotations.
            for cur_annotate in pgv_annotate:
                cur_mask = np.isclose(df['pgv'], cur_annotate)
                cur_df = df[cur_mask]
                for cur_id, cur_row in cur_df.iterrows():
                    cur_label = np.round(cur_row['pgv'] * 1000, 3)
                    cur_geom = cur_row['geometry']
        
                    cur_artist = self.ax.annotate(cur_label,
                                                  xy = cur_geom.exterior.coords[0],
                                                  fontsize = 4,
                                                  ha = 'center',
                                                  va = 'center',
                                                  zorder = 10)
                    artists.append(cur_artist)
        
        self.artists.extend(artists)

    def draw_simplices(self, df, data_col = 'pgv_min_log'):
        ''' Draw the detection simplices. 

        Parameters
        ----------
        df: :class:`geopandas.GeoDataFrame`
            The dataframe used to compute the pgv level.

        data_col: str 
            The data columne in df used for the PGV data.
        '''
        cmap = self.cmap
        norm = self.norm
        
        # Draw the simplices that didn't trigger
        cur_df = df[(df.triggered == False) & (df.added_to_event == False)]
        color_list = [cmap(norm(x)) for x in cur_df[data_col]]
    
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
        color_list = [cmap(norm(x)) for x in cur_df[data_col]]
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
        color_list = [cmap(norm(x)) for x in cur_df[data_col]]
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

    def draw_detection_stations(self, df, use_sa = False):
        ''' Draw the detecion stations.
        '''
        cmap = self.cmap
        norm = self.norm

        #cmap = self.cmap_qualitative_intensity
        #norm = self.norm_qualitative_intensity

        
        artists = []
   
        # Get the max pgv of stations with a pgv value.
        if use_sa:
            with_data_df = df[df['pgv_corr'].notna()]
            colorlist = [cmap(norm(x)) for x in with_data_df.pgv_corr_log]
        else:
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
        if use_sa:
            no_data_df = df[df['pgv_corr'].isna()]
        else:
            no_data_df = df[df['pgv'].isna()]
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

        
    def draw_attribution_note(self):
        ''' Draw the map data contribution note.
        '''
        artists = []
        cont_string = 'Map based on data from OE3D and OpenStreetMap. Generated with QGis and Python. CC BY-SA 4.0.'     
        cur_artist = self.ax.text(x = 0.005,
                                  y = 0.002,
                                  s = cont_string,
                                  ha = 'left',
                                  va = 'bottom',
                                  fontsize = 3,
                                  transform = self.ax.transAxes)
        artists.append(cur_artist)
    
        text_extent = cur_artist.get_window_extent(renderer = self.fig.canvas.get_renderer()).transformed(self.ax.transAxes.inverted())

        width = text_extent.width + 0.03
        height = text_extent.height + 0.003
        background_rect = matplotlib.patches.Rectangle((0, 0),
                                                       width = width,
                                                       height = height,
                                                       facecolor = 'lightgray',
                                                       edgecolor = 'None',
                                                       alpha = 0.7,
                                                       transform = self.ax.transAxes,
                                                       zorder = 2)
        cur_artist = self.ax.add_patch(background_rect)
        artists.append(cur_artist)

        self.artists.extend(artists)

        
    def create_movie(self, image_dir, output_dir,
                     img_name, video_name, file_ext = 'png'):
        ''' Create a movie using ffmpeg.

        Parameters
        ----------
        image_dir: str 
            The directory where the input images used to create the movie are saved.

        output_dir: str 
            The directory where to save the movie.

        img_name: str 
            The base name of the image files. The complete search string is built 
            using '{public_id}_{name}_*.{ext}'.

        video_name: str
            The base name of the video output file.

        file_ext: str 
            The file extension of the input images.
        '''
        img_filepath = os.path.join(image_dir,
                                    '{public_id}_{name}_*.{ext}'.format(public_id = self.event_public_id,
                                                                        name = img_name,
                                                                        ext = file_ext))
        movie_filepath = os.path.join(output_dir,
                                      '{public_id}_{vid_name}.mp4'.format(public_id = self.event_public_id,
                                                                          vid_name = video_name))
                                                                             
        stream = ffmpeg.input(img_filepath,
                              pattern_type = 'glob',
                              framerate = 2)
        stream = stream.filter('scale',
                               width = 'trunc(iw/2)*2',
                               height = 'trunc(ih/2)*2')
        stream = stream.filter('scale',
                               size = 'hd1080',
                               force_original_aspect_ratio = 'decrease')
        #stream = stream.filter('pad',
        #                       width = 1920,
        #                       height = 1080,
        #                       x = '(ow - iw) / 2',
        #                       y = '(oh - ih) / 2',
        #                       color = 'red')
        stream = stream.output(filename = movie_filepath,
                               format = 'mp4',
                               pix_fmt = 'yuv420p')
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
        

        # Draw the voronoi cells using the station corrections.
        # The voronoi cells represent interpolated data, therefore
        # the station corrections are applied.
        self.draw_voronoi_cells(df = event_pgv_df,
                                use_sa = True)

        # Plot the station max pgv markers without station corrections.
        # The stations mark individual data markers, therefore, the
        # station correction is not applied.
        self.draw_station_pgv(df = station_pgv_df,
                              use_sa = False)

        # Draw the network boundary.
        self.draw_boundary()

        # Draw the public id and the time marker.
        from_zone = dateutil.tz.gettz('UTC')
        to_zone = dateutil.tz.gettz('CET')
        event_start = obspy.UTCDateTime(station_pgv_props['event_start'])
        event_end = obspy.UTCDateTime(station_pgv_props['event_end'])
        event_start_local = event_start.datetime.replace(tzinfo = from_zone).astimezone(to_zone)
        self.draw_time_marker(duration = event_end - event_start,
                              time = event_start_local)

        # Draw the max. pgv level indicator in the colorbar axes.
        # Use the station corrections to represent the interpolated,
        # regional data.
        self.draw_pgv_level(df = station_pgv_df,
                            use_sa = True)

        # Save the map image.
        filename = self.event_public_id + '_pgvvoronoi.png'
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
        
        for cur_name, cur_group in time_groups:
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
                          img_name = 'detectionframe',
                          video_name = 'detection_sequence')


    def create_pgv_contour_sequence_movie(self):
        ''' Create the movie of the pgv contour sequence.
        '''
        # Initialize the map.
        if self.fig is None:
            self.init_map(utm_zone = 33)
        else:
            self.clear_map()
            
        # Create the output directory.
        img_output_dir = os.path.join(self.output_dir,
                                      self.event_dir,
                                      'pgvcontoursequence',
                                      'images')
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)
            
        movie_output_dir = os.path.join(self.output_dir,
                                        self.event_dir,
                                        'pgvcontoursequence',
                                        'movie')
        if not os.path.exists(movie_output_dir):
            os.makedirs(movie_output_dir)

        # Load the detection sequence data from the geojson file.
        self.logger.info('Loading the contour sequence data.')
        sequ_df = util.get_supplement_data(public_id = self.event_public_id,
                                           category = 'pgvsequence',
                                           name = 'pgvcontour',
                                           directory = self.supplement_dir)
        sequ_df_props = sequ_df.attrs

        # Convert the geopandas dataframe to cartopy projection.
        sequ_df = sequ_df.to_crs(self.projection.proj4_init)
        
        # Add the logarithmic pgv values.
        sequ_df.insert(4, "pgv_log", np.log10(sequ_df.pgv))

        # Load the station pgv sequence data from the geojson file.
        stat_df = util.get_supplement_data(public_id = self.event_public_id,
                                           category = 'pgvsequence',
                                           name = 'pgvstation',
                                           directory = self.supplement_dir)
        stat_df_props = stat_df.attrs

        # convert the geopandas dataframe to cartopy projection.
        stat_df = stat_df.to_crs(self.projection.proj4_init)

        stat_df.insert(4, "pgv_corr", stat_df.pgv / stat_df.sa)
        stat_df.insert(5, "pgv_log", np.log10(stat_df.pgv))
        stat_df.insert(6, "pgv_corr_log", np.log10(stat_df.pgv_corr))
        
        # Set the time zones for conversion.
        from_zone = dateutil.tz.gettz('UTC')
        to_zone = dateutil.tz.gettz('CET')
        
        # Iterate through the time groups.
        time_groups = sequ_df.groupby('time')
        stat_time_groups = stat_df.groupby('time')

        max_event_pgv = None
        for cur_name, cur_group in time_groups:
            cur_time = obspy.UTCDateTime(cur_name)
            self.logger.info('Processing time frame: %s.', cur_time)

            # Get the related pgvstation frame.
            cur_stat_df = stat_time_groups.get_group(cur_name)
            
            # Convert to local time.
            cur_time_local = cur_time.datetime.replace(tzinfo = from_zone).astimezone(to_zone)

            # Draw the pgv contour polygons.
            self.draw_contours(df = cur_group,
                               draw_labels = True)

            # Draw the station markers.
            # The stations represent individual data points, therefore
            # the station correction is not applied.
            self.draw_detection_stations(df = cur_stat_df,
                                         use_sa = False)

            # Draw the time information.
            self.draw_time_marker(time = cur_time_local)

            # Draw the network boundary.
            self.draw_boundary()

            # Draw the PGV level.
            max_event_pgv = self.draw_pgv_level(df = cur_stat_df,
                                                max_event_pgv = max_event_pgv,
                                                use_sa = True,
                                                show_max_level = True)
            
            cur_date_string = cur_time.isoformat().replace(':', '').replace('.', '')
            cur_filename = self.event_public_id + '_pgvcontourframe_' + cur_date_string + '.png'
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
                          img_name = 'pgvcontourframe',
                          video_name = 'pgvcontoursequence')

        
    def create_pgv_contour_map(self, draw_contour_labels = True,
                               draw_pgv_level = True, draw_event_info = True,
                               ignore_not_felt = True):
        ''' Create a map of the event pgv contours.
        '''
        # Initialize the map.
        if self.fig is None:
            self.init_map(utm_zone = 33)
        else:
            self.clear_map()
            
        # Create the output directory.
        img_output_dir = os.path.join(self.output_dir,
                                      self.event_dir,
                                      'eventpgv',
                                      'images')
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)

        # Read the event pgv contour data.
        cont_df = util.get_supplement_data(public_id = self.event_public_id,
                                           category = 'eventpgv',
                                           name = 'isoseismalfilledcontour',
                                           directory = self.supplement_dir)
        cont_props = cont_df.attrs
        # Convert the geopandas dataframe to cartopy projection.
        cont_df = cont_df.to_crs(self.projection.proj4_init)
        # Add the logarithmic pgv values.
        cont_df["pgv_log"] = np.log10(cont_df.pgv)

        # Read the station pgv data.
        station_pgv_df = util.get_supplement_data(public_id = self.event_public_id,
                                                  category = 'eventpgv',
                                                  name = 'pgvstation',
                                                  directory = self.supplement_dir)
        # Convert the pandas dataframe to cartopy projection.
        station_pgv_df = station_pgv_df.to_crs(self.projection.proj4_init)
        # Compute the corrected and the logarithmic pgv data.
        station_pgv_df.insert(4, "pgv_corr", station_pgv_df.pgv / station_pgv_df.sa)
        station_pgv_df.insert(5, 'pgv_log', np.log10(station_pgv_df.pgv))
        station_pgv_df.insert(6, 'pgv_corr_log', np.log10(station_pgv_df.pgv_corr))

        # Draw the pgv contour polygons.
        self.draw_contours(df = cont_df,
                           draw_labels = draw_contour_labels,
                           ignore_not_felt = ignore_not_felt)

        # Plot the station max pgv markers.
        self.draw_detection_stations(df = station_pgv_df,
                                     use_sa = False)

        # Draw the max. pgv level indicator in the colorbar axes.
        if draw_pgv_level:
            self.draw_pgv_level(df = station_pgv_df,
                                use_sa = True)

        # Draw the time information.
        # Draw the public id and the time marker.
        if draw_event_info:
            from_zone = dateutil.tz.gettz('UTC')
            to_zone = dateutil.tz.gettz('CET')
            event_start = obspy.UTCDateTime(cont_props['event_start'])
            event_end = obspy.UTCDateTime(cont_props['event_end'])
            event_start_local = event_start.datetime.replace(tzinfo = from_zone).astimezone(to_zone)
            self.draw_time_marker(duration = event_end - event_start,
                                  time = event_start_local,)

        # Draw the network boundary.
        self.draw_boundary()

        cur_filename = self.event_public_id + '_pgvcontour.png'
        cur_filepath = os.path.join(img_output_dir,
                                    cur_filename)
        self.fig.savefig(cur_filepath,
                         dpi = 300,
                         pil_kwargs = {'quality': 80},
                         bbox_inches = 'tight',
                         pad_inches = 0,)
        self.clear_map()


    def create_voronoi_cell_sequence_movie(self, file_format = 'jpg'):
        ''' Create the movie of the voronoi cell sequence.
        '''
        # Initialize the map.
        if self.fig is None:
            self.init_map(utm_zone = 33)
        else:
            self.clear_map()
            
        # Create the output directory.
        img_output_dir = os.path.join(self.output_dir,
                                      self.event_dir,
                                      'pgvvoronoicellsequence',
                                      'images')
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)
            
        movie_output_dir = os.path.join(self.output_dir,
                                        self.event_dir,
                                        'pgvvoronoicellsequence',
                                        'movie')
        if not os.path.exists(movie_output_dir):
            os.makedirs(movie_output_dir)

        # Load the pgv voronoi cell sequence data from the geojson file.
        self.logger.info('Loading the pgv voronoi cell sequence data.')
        sequ_df = util.get_supplement_data(public_id = self.event_public_id,
                                           category = 'pgvsequence',
                                           name = 'pgvvoronoi',
                                           directory = self.supplement_dir)

        # Convert the geopandas dataframe to cartopy projection.
        sequ_df = sequ_df.to_crs(self.projection.proj4_init)

        # Add the logarithmic pgv values.
        sequ_df.insert(3, "pgv_corr", sequ_df.pgv / sequ_df.sa)
        sequ_df.insert(4, "pgv_log", np.log10(sequ_df.pgv))
        sequ_df.insert(5, "pgv_corr_log", np.log10(sequ_df.pgv_corr))

        # Load the station pgv sequence data from the geojson file.
        stat_df = util.get_supplement_data(public_id = self.event_public_id,
                                           category = 'pgvsequence',
                                           name = 'pgvstation',
                                           directory = self.supplement_dir)

        # convert the geopandas dataframe to cartopy projection.
        stat_df = stat_df.to_crs(self.projection.proj4_init)

        stat_df.insert(4, "pgv_corr", stat_df.pgv / stat_df.sa)
        stat_df.insert(5, "pgv_log", np.log10(stat_df.pgv))
        stat_df.insert(6, "pgv_corr_log", np.log10(stat_df.pgv_corr))
        
        # Set the time zones for conversion.
        from_zone = dateutil.tz.gettz('UTC')
        to_zone = dateutil.tz.gettz('CET')
        
        # Iterate through the time groups.
        time_groups = sequ_df.groupby('time')
        stat_time_groups = stat_df.groupby('time')

        max_event_pgv = None
        for cur_name, cur_group in time_groups:
            cur_time = obspy.UTCDateTime(cur_name)
            self.logger.info('Processing time frame: %s.', cur_time)

            # Get the related pgvstation frame.
            cur_stat_df = stat_time_groups.get_group(cur_name)
            
            # Convert to local time.
            cur_time_local = cur_time.datetime.replace(tzinfo = from_zone).astimezone(to_zone)

            # Draw the pgv contour polygons.
            self.draw_voronoi_cells(df = cur_group,
                                    use_sa = True)

            # Draw the station markers.
            # The stations represent individual data points, therefore
            # the station correction is not applied.
            self.draw_detection_stations(df = cur_stat_df,
                                         use_sa = False)

            # Draw the time information.
            self.draw_time_marker(time = cur_time_local)

            # Draw the network boundary.
            self.draw_boundary()

            # Draw the PGV level.
            max_event_pgv = self.draw_pgv_level(df = cur_stat_df,
                                                max_event_pgv = max_event_pgv,
                                                use_sa = False,
                                                show_max_level = True)
            
            cur_date_string = cur_time.isoformat().replace(':', '').replace('.', '')

            cur_ext = '.' + file_format
            cur_filename = self.event_public_id + '_pgvvoronoicellframe_' + cur_date_string + cur_ext
            cur_filepath = os.path.join(img_output_dir,
                                        cur_filename)
            self.fig.savefig(cur_filepath,
                             dpi = 300,
                             pil_kwargs = {'quality': 90},
                             bbox_inches = 'tight',
                             pad_inches = 0,)
            self.clear_map()
            
        self.create_movie(image_dir = img_output_dir,
                          output_dir = movie_output_dir,
                          img_name = 'pgvvoronoicellframe',
                          video_name = 'pgvvoronoicellsequence',
                          file_ext = file_format)
