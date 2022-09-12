# -*- coding: utf-8 -*-
##############################################################################
# LICENSE
#
# This file is part of mss_dataserver.

# If you use mss_dataserver in any program or publication, please inform and
# acknowledge its authors

# mss_dataserver is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# mss_dataserver is distributed in the hope that it will be useful
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mss_dataserver. If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2022 Stefan Mertl
##############################################################################


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class LocApolloniusCircle(object):

    def __init__(self, stations, amplitude, dist_exp = -2.2):
        ''' Initialize the instance.
        '''

        # The station xyz coordinates in a cartesian
        # coordinate system.
        self.stations = stations

        # The amplitudes related to the stations.
        self.amp = amplitude

        # The exponent of the distance relationship.
        self.dist_exp = dist_exp

        # Sort the pgv in descending order.
        max_amp_filter = np.argsort(self.amp)
        max_amp_filter = max_amp_filter[::-1]
        self.stations = self.stations[max_amp_filter, :]
        self.amp = self.amp[max_amp_filter]

        # The mask of the valid data.
        self.mask_valid = np.ones(len(self.amp)).astype(bool)

        # The mask of the main stations used for the circle computation.
        self.mask_main = None

        # The threshold of the outliers in the valid data computation.
        self.outlier_threshold = None

        # The preliminary hypocenter used for the valid data.
        self.hypo_prelim = None

        # The source amplitude of the preliminary hypocenter.
        self.prelim_A0 = None

        # The reference distance for A0 of the preliminary hypocenter.
        self.prelim_ref_dist = None
        
        # The Apolonius circle center points.
        self.center = []
        
        # The Apolonius circle radii.
        self.radius = []

        # The cell hit grid.
        self.cell_hit = None

        # The computed hypocenter.
        self.hypo = None

        # The computed grid indices of the hypocenter.
        self.hypo_ind = None


    @property
    def amp_valid(self):
        ''' The valid amplitude data.
        '''
        return self.amp[self.mask_valid]


    @property
    def stations_valid(self):
        ''' The valid stations.
        '''
        return self.stations[self.mask_valid, :]


    @property
    def stations_main(self):
        ''' The main stations used for the circle computation.
        '''
        ret = None
        if self.mask_main is not None:
            ret = self.stations_valid[self.mask_main, :]

        return ret


    @property
    def amp_main(self):
        ''' The amplitudes of the main stations used for the circle computation.
        '''
        ret = None
        if self.mask_main is not None:
            ret = self.amp_valid[self.mask_main]

        return ret

    
    def compute_prelim_hypo(self, percentile = 90):
        ''' Compute the preliminary hypocenter.
        '''
        # Compute a preliminary hypocenter.
        p_hypo = np.percentile(self.amp, percentile)
        mask_p_hypo = self.amp >= p_hypo
        p_hypo_stat_coord = self.stations[mask_p_hypo, :]

        # TODO: Find stations that are too far away from the others.
        # Try to compute statistics of the inter-station distances.

        hypo_prelim = np.median(p_hypo_stat_coord,
                               axis = 0)
        # TODO: Handle the depth of the preliminary hypocenter.
        
        p_hypo_pgv = self.amp[mask_p_hypo]
        prelim_A0 = np.mean(p_hypo_pgv)

        prelim_hypodist = np.sqrt(np.sum((hypo_prelim - self.stations)**2,
                                        axis = 1))
        prelim_ref_dist = np.mean(prelim_hypodist[mask_p_hypo])

        self.hypo_prelim = hypo_prelim
        self.prelim_A0 = prelim_A0
        self.prelim_ref_dist = prelim_ref_dist
        


    def compute_valid_data(self, percentile = 90, thr_tolerance = 3.162277,
                           thr_noise = 1e-6):
        ''' Compute the valid data based on the outlier analysis.
        '''
        # Compute the amplitude decay for the preliminary hypocenter.
        hypo_prelim = self.hypo_prelim
        prelim_A0 = self.prelim_A0
        prelim_ref_dist = self.prelim_ref_dist
        prelim_hypodist = np.sqrt(np.sum((hypo_prelim - self.stations)**2,
                                        axis = 1))

        # Compute the outlier upper threshold.
        outlier_threshold = prelim_A0 * thr_tolerance * (prelim_hypodist / prelim_ref_dist)**self.dist_exp

        # Limit floor low amplitudes of the outlier threshold.
        # mask_noise = outlier_threshold < thr_noise * thr_tolerance**2
        mask_noise = np.log10(outlier_threshold) < np.log10(thr_noise) + 2 * np.log10(thr_tolerance)
        outlier_threshold[mask_noise] = thr_noise

        # Create the mask for the valid data.
        mask_outlier = self.amp < outlier_threshold
        mask_noise_floor = self.amp > thr_noise

        self.outlier_threshold = outlier_threshold
        self.mask_valid = mask_outlier & mask_noise_floor

        
    def compute_circles(self, percentile = 90):
        ''' Compute the Apolonius circles.
        '''
        # Use the valid data for the computation.
        amp_valid = self.amp_valid
        stat_valid = self.stations_valid

        # Compute the main stations.
        p_main = np.percentile(amp_valid, percentile)
        mask_p_main = amp_valid >= p_main
        stat_main = stat_valid[mask_p_main, :]
        amp_main = amp_valid[mask_p_main]

        for k, cur_stat_main in enumerate(stat_main):
            # The main station pgv.
            cur_amp_main = amp_main[k]
            
            # Get the reference stations and pgv values.
            ref_ind = k + 1
            cur_stat_ref = stat_valid[ref_ind:, :]
            cur_amp_ref = amp_valid[ref_ind:]
            
            # Compute the pgv ratio.
            ratio = (cur_amp_main / cur_amp_ref)**(1 / self.dist_exp) 
            ratio = ratio[:, np.newaxis]
            
            # Compute the inline source locations
            p1 = (cur_stat_main + ratio * cur_stat_ref) / (1 + ratio)
            p2 = (cur_stat_main - ratio * cur_stat_ref) / (1 - ratio)
            
            # Compute the Apolonius circle midpoint and Radius
            cur_c_ap = (p1 + p2) / 2
            cur_r_ap = np.sqrt(np.sum((p2 - p1)**2, axis = 1)) / 2
            
            # Add the circle parameters to the lists.
            self.center.append(cur_c_ap)
            self.radius.append(cur_r_ap)

        self.mask_main = mask_p_main

            
    def compute_grid(self, width = 40000, height = 40000, depth = 20000,
                     step_x = 200, step_y = 200, step_z = 1000):
        ''' Compute the cell hit grid.
        '''
        # Compute the grid coordinates.
        center = self.hypo_prelim
        d_grid_x = width / 2
        d_grid_y = height / 2
        x_coord = np.arange(center[0] - d_grid_x,
                            center[0] + d_grid_x + step_x,
                            step_x)
        y_coord = np.arange(center[1] - d_grid_y,
                            center[1] + d_grid_y + step_y,
                            step_y)
        z_coord = np.arange(-depth,
                            0 + step_z,
                            step_z)
        # Flip the z_coord to set the lowest depth (highest height) to index 0.
        z_coord = z_coord[::-1]
        
        # Compute the grid arrays.
        x_grid, y_grid, z_grid = np.meshgrid(x_coord, y_coord, z_coord)

        # Save the grids in the instance.
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.z_coord = z_coord
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = z_grid
        

    def compute_cell_hit(self, sigma = 1000):
        ''' Compute the cell hits of the circles.
        '''
        x_grid = self.x_grid
        y_grid = self.y_grid
        z_grid = self.z_grid
        cell_hit = np.zeros_like(x_grid)
        for n in range(len(self.center)):
            cur_center_list = self.center[n]
            cur_radius_list = self.radius[n]
            
            for k in np.arange(len(cur_center_list)):
                center = cur_center_list[k, :]
                radius = cur_radius_list[k]
                cell_dist = np.sqrt((x_grid - center[0])**2 + (y_grid - center[1])**2 + (z_grid - center[2])**2)
                cell_weight = np.exp(-((radius - cell_dist)**2) / (2 * sigma**2))
                cell_hit += cell_weight

        self.cell_hit = cell_hit

                
    def compute_hypo(self):
        ''' Compute the hypocenter.
        '''
        max_ind = np.argmax(self.cell_hit)
        max_ind = np.unravel_index(max_ind,
                                   self.cell_hit.shape)
        self.hypo_ind = max_ind
        self.hypo = [float(self.x_grid[max_ind]),
                     float(self.y_grid[max_ind]),
                     float(self.z_grid[max_ind])]

        
    def plot_valid_data_selection(self):
        ''' Create the plot showing the valid data thresholds.
        '''
        prelim_hypodist = np.sqrt(np.sum((self.hypo_prelim - self.stations)**2,
                                         axis = 1))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(prelim_hypodist,
                  self.outlier_threshold,
                  marker = '+',
                  linestyle = '')
        ax.loglog(prelim_hypodist,
                  self.amp,
                  marker = 'x',
                  linestyle = '')
        ax.loglog(prelim_hypodist[self.mask_valid],
                  self.amp_valid,
                  marker = 'o',
                  markersize = 3,
                  linestyle = '')

        return fig

    
    def plot_cellhit_hypo_solution(self):
        ''' Plot the cell hit grids of the optimum solution.
        '''
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Plot the cell hit grid.
        pcm = ax.pcolormesh(self.x_coord,
                            self.y_coord,
                            self.cell_hit[:, :, self.hypo_ind[2]])

        # Get the stations inside the grid.
        stat_valid = self.stations_valid
        grid_lim_x = [np.min(self.x_coord),
                      np.max(self.x_coord)]
        grid_lim_y = [np.min(self.y_coord),
                      np.max(self.y_coord)]

        mask_grid_x = ((stat_valid[:, 0] >= grid_lim_x[0]) &
                       (stat_valid[:, 0] <= grid_lim_x[1]))
        mask_grid_y = ((stat_valid[:, 1] >= grid_lim_y[0]) &
                       (stat_valid[:, 1] <= grid_lim_y[1]))
        mask_inside = mask_grid_x & mask_grid_y
        stat_inside = stat_valid[mask_inside, :]

        # Add the station markers.
        ax.plot(stat_inside[:, 0],
                stat_inside[:, 1],
                marker = '^',
                color = 'k',
                alpha = 0.3,
                markersize = 4,
                linestyle = '')

        # Add the hypocenter marker.
        ax.plot(self.hypo[0],
                self.hypo[1],
                '+',
                color = 'r')
        
        fig.colorbar(pcm,
                     ax = ax)
        #ax.set_aspect('equal')
        ax.axis('equal')

        return fig

    
    def plot_circles(self):
        ''' Plot the Apollonius circles.
        '''
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        # Create the circle patches
        c_ap = self.center
        r_ap = self.radius
        for n in range(len(c_ap)):
            cur_c_ap = c_ap[n]
            cur_r_ap = r_ap[n]
            
            circle_patches = []
            for k in np.arange(len(cur_c_ap)):
                center = cur_c_ap[k, :]
                center = center[:2]
                radius = cur_r_ap[k]
                patch = plt.Circle(center,
                                   radius = radius,
                                   color = 'k')
                circle_patches.append(patch)

            # Add the circle patches.
            circle_collection = matplotlib.collections.PatchCollection(circle_patches)
            circle_collection.set(facecolor = 'none',
                                  edgecolor = 'k',
                                  linewidth = 0.5)
            ax.add_collection(circle_collection)

        vmin = np.log10(np.min(self.amp_valid))
        vmax = np.log10(np.max(self.amp_valid))
        # Plot the main stations.
        ax.scatter(self.stations_main[:, 0],
                   self.stations_main[:, 1],
                   marker = '^',
                   s = 20,
                   c = np.log10(self.amp_main),
                   vmin = vmin,
                   vmax = vmax)

        n_main = len(self.stations_main)
        stat_ref = self.stations_valid[n_main:, :]
        ax.scatter(stat_ref[:, 0], stat_ref[:, 1],
                   marker = 'o',
                   s = 20,
                   c = np.log10(self.amp_valid[n_main:]),
                   vmin = vmin,
                   vmax = vmax)

        # Add the hypocenter marker.
        ax.plot(self.hypo[0],
                self.hypo[1],
                '+',
                color = 'r')

        # Set the axes limits to the extent of the stations.
        border = 10000
        stat_valid = self.stations_valid
        xlim = [np.min(stat_valid[:, 0]) - border,
                np.max(stat_valid[:, 0]) + border]

        ylim = [np.min(stat_valid[:, 1]) - border,
                np.max(stat_valid[:, 1]) + border]
        
        ax.axis('equal')
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

        return fig
