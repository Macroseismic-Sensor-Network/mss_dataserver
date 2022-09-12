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
''' The event localizer.
'''

import logging

import obspy

import mss_dataserver.localize.apollonius_circle as loc_ap
import mss_dataserver.localize.origin as mssds_origin


class EventLocalizer(object):
    ''' Localize an event.

    '''

    def __init__(self, public_id, meta, pgv_df, project,
                 event = None):
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

        # The event instance to localize.
        self.event = event

        # The computed event origins.
        self.origins = []

        # The localizer used for the latest localization.
        # It can be used for further processing (e.g. image creation).
        self.localizer = None


    def loc_apollonius(self, dist_exp = -2.2):
        ''' Localize the event using the Apollonius circle method.

        Parameters
        ----------
        dist_exp: float
            The amplitude-distance relationship exponent.

        '''
        msg = 'Localizing event {} using the Apollonius method.'.format(self.public_id)
        self.logger.info(msg)

        # Remove no data values.
        no_data_mask = self.pgv_df['pgv'].isna()
        pgv_df = self.pgv_df[~no_data_mask]

        # Extract the station coordinates and the pgv data.
        stat_coord = pgv_df[['x_utm', 'y_utm', 'z']].values
        pgv = pgv_df['pgv'].values

        # Create the specific localizer instance.
        localizer = loc_ap.LocApolloniusCircle(stations = stat_coord,
                                               amplitude = pgv,
                                               dist_exp = dist_exp)

        # Compute the preliminary hypocenter.
        localizer.compute_prelim_hypo(percentile = 90)

        # Compute the cell hit grid.
        localizer.compute_grid(width = 40000,
                               height = 40000,
                               depth = 20000,
                               step_x = 200,
                               step_y = 200,
                               step_z = 1000)

        # Raise the pgv limit by a constant factor.
        # In Ewald's code log10(thr) = 0.5 was used.
        thr_tolerance = 10**(0.5)

        # The noise threshold. All values below the threshold
        # will be set to a constant noise floor value.
        # Ewald's value was 0.001 mm/s
        thr_noise = 1e-6

        # Compute the valid data.
        localizer.compute_valid_data(percentile = 90,
                                     thr_tolerance = thr_tolerance,
                                     thr_noise = thr_noise)

        # Compute the apollonius circles.
        localizer.compute_circles(percentile = 90)

        # Compute the cell hit grid.
        localizer.compute_cell_hit(sigma = 1000)

        # Compute the hypocenter.
        localizer.compute_hypo()

        # Create the origin.
        # TODO: Compute the origin time.
        origin_time = None
        author_uri = self.project.author_uri
        agency_uri = self.project.agency_uri
        creation_time = obspy.UTCDateTime()

        code = self.inventory.get_utm_epsg()
        epsg_code = 'epsg:' + code[0][0]
        hypo = localizer.hypo
        origin = mssds_origin.Origin(time = origin_time,
                                     x = hypo[0],
                                     y = hypo[1],
                                     z = hypo[2],
                                     coord_system = epsg_code,
                                     method = 'apollonius_circle',
                                     comment = 'Test comment',
                                     agency_uri = agency_uri,
                                     author_uri = author_uri,
                                     creation_time = creation_time)

        # Compute the longitude and latitude of the hypocenter.
        origin.convert_to_lonlat()

        # Add the origin to the event.
        if self.event is not None:
            self.event.add_origin(origin)

        # Add the computed origin to the origins list.
        self.origins.append(origin)

        # Store the used localizer.
        self.localizer = localizer

