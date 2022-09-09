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
''' Handling of event types.

'''

import obspy


class Origin(object):
    ''' A type of an event.

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


    @property
    def rid(self):
        ''' str: The resource ID of the event type.
        '''
        return '/origin/' + self.name

    
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

        return origin   
