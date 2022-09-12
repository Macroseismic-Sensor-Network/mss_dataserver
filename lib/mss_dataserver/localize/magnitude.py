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


class Magnitude(object):
    ''' An event magnitude.

    '''
    def __init__(self, mag, mag_type, parent = None,
                 comment = None, db_id = None, agency_uri = None,
                 author_uri = None, creation_time = None):
        ''' Initialize the instance.
        '''
        # The parent object holding the event. Most likely this is
        # an origin.
        self.parent = parent

        # The unique database id.
        self.db_id = db_id

        # The magnitude value.
        self.mag = mag

        # The magnitude type.
        self.mag_type = mag_type
        
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
        mag_slug = str(self).replace(' ', '_')
        return '/magnitude/' + mag_slug

    
    def __str__(self):
        return '{mag_type} {mag:.2f}'.format(mag_type = self.mag_type,
                                             mag = self.mag)

    
    def write_to_database(self, project, db_session = None,
                          close_session = True):
        ''' Write the magnitude to the database.

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
                db_orm = project.db_tables['magnitude']
                db_mag = db_orm(origin_id = parent_id,
                                mag = self.mag,
                                mag_type = self.mag_type,
                                comment = self.comment,
                                author_uri = self.author_uri,
                                agency_uri = self.agency_uri,
                                creation_time = creation_time)

                # Commit the origin to the database to get an id.
                db_session.add(db_mag)
                db_session.commit()
                self.db_id = db_mag.id
            finally:
                if close_session:
                    db_session.close()


    @classmethod
    def from_orm(cls, db_mag):
        ''' Convert a database orm magnitude to a Magnitude instance.

        Parameters
        ----------
        db_mag: :class:`Magnitude`
            An instance of the sqlalchemy Table Mapper classe defined in
            :meth:`mss_dataserver.localize.databaseFactory`

        Returns
        -------
        :class:`mss_dataserver.localize.magnitude.Magnitude`
            The magnitude created from the database ORM instance.
        '''
        origin = cls(db_id = db_mag.id,
                     mag = db_mag.mag,
                     mag_type = db_mag.mag_type,
                     comment = db_mag.comment,
                     agency_uri = db_mag.agency_uri,
                     author_uri = db_mag.author_uri,
                     creation_time = db_mag.creation_time)

        return origin