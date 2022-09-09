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
# Copyright 2022 Stefan Mertl
##############################################################################
''' 

'''
import obspy

import mss_dataserver.localize.magnitude as mssds_mag

class Origin(object):
    ''' Anevent origin.

    '''
    def __init__(self, time, x, y, z, coord_system, method,
                 parent = None, comment = None, db_id = None,
                 agency_uri = None, author_uri = None, creation_time = None):
        ''' Initialize the instance.
        '''
        # The parent object holding the event. Most likely this is
        # an event.
        self.parent = parent

        # The unique database id.
        self.db_id = db_id

        # The origin time.
        self.time = time

        # The x coordinate of the origin.
        self.x = x

        # The y coordinate of the origin.
        self.y = y

        # The z coordinate of the origin.
        self.z = z

        # The coordinate system used.
        self.coord_system = coord_system

        # The method used for the creation of the origin.
        self.method = method
        
        # The comment for the origin.
        self.comment = comment

        # The agency_uri of the creator.
        self.agency_uri = agency_uri

        # The author_uri of the creator.
        self.author_uri = author_uri

        # The time of creation of this event.
        if creation_time is None:
            creation_time = obspy.UTCDateTime()
        self.creation_time = obspy.UTCDateTime(creation_time)

        # The magnitudes related to the origin.
        self.magnitudes = []

        # The preferred magnitude of the origin.
        self.pref_magnitude = None


    @property
    def rid(self):
        ''' str: The resource ID of the event type.
        '''
        return '/origin/' + self.db_id


    def add_magnitude(self, mag):
        ''' Add a magnitude to the origin.

        Parameters
        ----------
        mag: :class:`mss_dataserver.localize.magnitude.Magnitude` or :obj:`list` of :class:`mss_dataserver.localize.magnitude.Magnitude`
            The magnitude or a list of magnitudes to add.
        '''
        if type(mag) is list:
            self.magnitudes.extend(mag)
            for cur_mag in mag:
                cur_mag.parent = self
        else:
            self.magnitudes.append(mag)
            mag.parent = self

            
    def set_preferred_magnitude(self, mag):
        ''' Set the preferred magnitude.
        
        Parameters
        ----------
        mag: :class:`mss_dataserver.localize.magnitude.Magnitude`
            The preferred magnitude.
        '''
        if mag not in self.magnitudes:
            self.logger.error("The preferred magnitude is not available in the origin magnitudes.")
            return

        self.pref_magnitude = mag

    
    def write_to_database(self, project, db_session = None,
                          close_session = True):
        ''' Write the origin to the database.

        Parameters
        ----------
        project: :class:`mss_dataserver.core.project.Project`
            The project to use to access the database.
        '''
        if self.db_id is None:
            # If the db_id is None, insert a new origin.
            if self.creation_time is not None:
                creation_time = self.creation_time.isoformat()
            else:
                creation_time = None

            if self.parent is not None:
                parent_id = self.parent.db_id
            else:
                parent_id = None

            if db_session is None:
                db_session = project.get_db_session()
            try:
                db_orm = project.db_tables['origin']
                db_origin = db_orm(event_id = parent_id,
                                   time = self.time.timestamp,
                                   x = self.x,
                                   y = self.y,
                                   z = self.z,
                                   coord_system = self.coord_system,
                                   method = self.method,
                                   comment = self.comment,
                                   author_uri = self.author_uri,
                                   agency_uri = self.agency_uri,
                                   creation_time = creation_time)

                # Commit the origin to the database to get an id.
                db_session.add(db_origin)
                db_session.commit()
                self.db_id = db_origin.id

                # Add the magnitudes to the database.
                # This updates the db_id of the magnitudes.
                if len(self.magnitudes) > 0:
                    for cur_mag in self.magnitudes:
                        cur_mag.write_to_database(project = project,
                                                  db_session = db_session,
                                                  close_session = False)

                # Update the preferred magnitude id of the origin.
                if self.pref_magnitude:
                    pref_mag_id = self.pref_magnitude.db_id
                    db_origin.pref_magnitude_id = pref_mag_id

                db_session.commit()
            finally:
                if close_session:
                    db_session.close()


    @classmethod
    def from_orm(cls, db_origin):
        ''' Convert a database orm origin to an Origin instance.

        Parameters
        ----------
        db_event_type: :class:`Origin`
            An instance of the sqlalchemy Table Mapper classe defined in
            :meth:`mss_dataserver.localize.databaseFactory`

        Returns
        -------
        :class:`mss_dataserver.localize.origin.Origin`
            The origin created from the database ORM instance.
        '''
        origin = cls(db_id = db_origin.id,
                     time = obspy.UTCDateTime(db_origin.time),
                     x = db_origin.x,
                     y = db_origin.y,
                     z = db_origin.z,
                     coord_system = db_origin.coord_system,
                     method = db_origin.method,
                     comment = db_origin.comment,
                     agency_uri = db_origin.agency_uri,
                     author_uri = db_origin.author_uri,
                     creation_time = db_origin.creation_time)

        # Add the magnitudes to the origin.
        assigned_mags = [mssds_mag.Magnitude.from_orm(x) for x in db_origin.magnitudes]
        origin.add_magnitude(mag = assigned_mags)

        # Set the preferred magnitude.
        if db_origin.pref_magnitude_id is not None:
            pmid = db_origin.pref_magnitude_id
            pref_mag = [x for x in assigned_mags if x.db_id == pmid]
            if len(pref_mag) == 1:
                pref_mag = pref_mag[0]
                origin.set_preferred_magnitude(mag = pref_mag)
            elif len(pref_mag) > 1:
                cls.logger.error("Multiple magnitudes returned for id %d.",
                                 pmid)

        return origin
