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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyproj

import mss_dataserver.geometry as geom
import mss_dataserver.postprocess.util as util


class SeismogramPlotter(object):
    ''' Create images of the seismograms.
    '''

    def __init__(self, supplement_dir, output_dir):
        ''' Initialize the instance.
        '''

        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # The directory containing the event supplement data.
        self.supplement_dir = supplement_dir
       
        # The directory where to save the map images.
        self.output_dir = output_dir

        # The event supplement sub-directory.
        self.event_dir = None
       
        # The public id of the event.
        self.event_public_id = None

        # The event metadata.
        self.event_meta = None

        # The station geometry.
        self.geom_inventory = None
        
        # The hypocenter of the event.
        self.event_hypocenter = None


    def set_event(self, public_id, hypocenter = None):
        ''' Set the event to process.
        '''
        self.event_public_id = public_id
        self.event_dir = util.event_dir_from_publicid(public_id)
        self.event_hypocenter = hypocenter
        # Load the event metadata from the supplement file.
        self.event_meta = util.get_supplement_data(self.event_public_id,
                                                   category = 'detectiondata',
                                                   name = 'metadata',
                                                   directory = self.supplement_dir)
        geom_dict = util.get_supplement_data(self.event_public_id,
                                             category = 'detectiondata',
                                             name = 'geometryinventory',
                                             directory = self.supplement_dir)
        self.geom_inventory = geom.inventory.Inventory.from_dict(geom_dict['inventory'])
        self.geom_inventory.compute_utm_coordinates()

        if self.event_hypocenter:
            self.compute_hypodistance()

        
    def compute_hypodistance(self):
        ''' Compute the hypodistance of the stations.
        '''
        # Convert the hypocenter to utm coordinates.
        code = self.geom_inventory.get_utm_epsg()
        proj = pyproj.Proj(init = 'epsg:' + code[0][0])
        hypo_x, hypo_y = proj(self.event_hypocenter[0],
                              self.event_hypocenter[1])
        hypo_z = self.event_hypocenter[2]
        
        for cur_station in self.geom_inventory.get_station():
            stat_x = cur_station.x_utm
            stat_y = cur_station.y_utm
            stat_z = cur_station.z
            cur_station.hypodist = np.sqrt((stat_x - hypo_x)**2 + (stat_y - hypo_y)**2 + (stat_z - hypo_z)**2)


    def create_figure(self, width, height):
        ''' Create a matplotlib figure.
        '''
        mm_per_inch = 25.4
        fig = plt.figure(figsize = (width / mm_per_inch,
                                    height / mm_per_inch),
                         dpi = 300)
        return fig

    
    def plot_seismogram(self, width = 120,
                        trace_height = 10,
                        stations_per_panel = 8,
                        start = 10,
                        length = 60):
        ''' Plot the seismogram data.
        '''
        # Load the velocity seismogram data.
        vel_st = util.get_supplement_data(public_id = self.event_public_id,
                                          category = 'detectiondata',
                                          name = 'velocity',
                                          directory = self.supplement_dir)
        
        if start:
            min_start = np.min([x.stats.starttime for x in vel_st])
            vel_st = vel_st.trim(starttime = min_start + start)
        if length:
            min_start = np.min([x.stats.starttime for x in vel_st])
            vel_st = vel_st.trim(endtime = min_start + length)

        print(vel_st)

        # Determine the stations to plot.
        stations_to_plot = sorted(list(set([x.stats.station for x in vel_st])))
        channels_to_plot = ['Hno', 'Hpa']

        channel_colors = ['black', 'grey']

        # Get the station instances from the inventory.
        stations_to_plot = [self.geom_inventory.get_station(name = x)[0] for x in stations_to_plot]
        
        # If a hypocenter is available sort the stations by
        # hypodistance.
        if self.event_hypocenter:
            stations_to_plot = sorted(stations_to_plot,
                                      key = lambda x: x.hypodist)
            
        max_amp = []

        if stations_per_panel:
            n_panels = int(np.ceil(len(stations_to_plot) / stations_per_panel))
        else:
            n_panels = 1
            stations_per_plot = len(stations_to_plot)

        for cur_panel_num in range(n_panels):
            start_ind = cur_panel_num * stations_per_panel
            end_ind = start_ind + stations_per_panel
            cur_stations_to_plot = stations_to_plot[start_ind:end_ind]
            
            # Create the figure.
            height = trace_height * len(cur_stations_to_plot) * len(channels_to_plot)
            fig = self.create_figure(width = width,
                                     height = height)
            gs = gridspec.GridSpec(len(cur_stations_to_plot), 1)
            gs.update(hspace = 0.1)
            
            for k, cur_station in enumerate(cur_stations_to_plot):
                cur_stat_st = vel_st.select(station = cur_station.name)
                cur_stat_gs = gs[k].subgridspec(len(channels_to_plot), 1,
                                                wspace = 0,
                                                hspace = 0)

                for m, cur_channel_name in enumerate(channels_to_plot):
                    cur_chan_st = cur_stat_st.select(channel = cur_channel_name)
                    cur_trace = cur_chan_st[0]
                    cur_trace.detrend('constant')
                    cur_ax = fig.add_subplot(cur_stat_gs[m])

                    cur_max_amp = np.max(np.abs(cur_trace.data))
                    cur_ax.plot(cur_trace.times(type = 'relative'),
                                cur_trace.data,
                                linewidth = 0.5,
                                color = channel_colors[m])

                    if (k == 0) and (m == 0):
                        cur_ax.tick_params(axis = 'x',
                                           direction = 'in',
                                           top = True,
                                           bottom = False,
                                           labeltop = False,
                                           labelbottom = False)
                    elif (k == stations_per_panel - 1) and (m == len(channels_to_plot) - 1):
                        cur_ax.tick_params(axis = 'x',
                                           direction = 'in',
                                           top = False,
                                           bottom = True,
                                           labelsize = 6)
                    else:
                        cur_ax.tick_params(axis = 'x',
                                           direction = 'in',
                                           top = False,
                                           bottom = False,
                                           labeltop = False,
                                           labelbottom = False)

                    cur_ax.tick_params(axis = 'y',
                                       direction = 'in',
                                       left = False,
                                       right = False,
                                       labelleft = False,
                                       labelright = False)

                    cur_ax.set_xlim(0, cur_trace.times(type = 'relative')[-1])
                    max_amp.append(cur_max_amp)
                    cur_ax.set_ylim(-cur_max_amp, cur_max_amp)

                    props = dict(boxstyle = 'round',
                                 edgecolor = 'black',
                                 facecolor='white')

                    stat_string = '{stat}:{chan}'.format(stat = cur_station.name,
                                                         chan = cur_channel_name)
                    cur_ax.text(0.02, 0.9, stat_string,
                                transform = cur_ax.transAxes,
                                fontsize = 6,
                                va = 'top',
                                ha = 'left',
                                bbox = props)

                    amp_string = "{amp:.3f} mm/s".format(amp = cur_max_amp * 1000)
                    cur_xlim = cur_ax.get_xlim()
                    cur_ax.text(cur_xlim[1], cur_max_amp, amp_string,
                                transform = cur_ax.transData,
                                fontsize = 6,
                                va = 'top',
                                ha = 'right')
                    cur_ax.text(cur_xlim[1], 0, '0',
                                transform = cur_ax.transData,
                                fontsize = 6,
                                va = 'top',
                                ha = 'right')

            plt.tight_layout()
            
            # Create the output directory.
            img_output_dir = os.path.join(self.output_dir,
                                          self.event_dir,
                                          'seismogram',
                                          'images')
            if not os.path.exists(img_output_dir):
                os.makedirs(img_output_dir)
            filename = '{pub_id}_seismogram_panel_{panel:02d}.png'.format(pub_id = self.event_public_id,
                                                                          panel = cur_panel_num)
            filepath = os.path.join(img_output_dir, filename)
            plt.savefig(filepath,
                        dpi = 300,
                        bbox_inches = 'tight',
                        pad_inches = 0)
