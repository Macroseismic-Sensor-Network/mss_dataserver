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


class EventType(object):
    ''' A type of an event.

    '''
    def __init__(self, name, description = None, parent = None,
                 db_id = None, agency_uri = None, author_uri = None,
                 creation_time = None):
        ''' Initialize the instance.
        '''
        # The parent object holding the event. Most likely this is
        # another EventType instance.
        self.parent = parent

        # The children event types.
        self.event_types = []

        # The unique database id.
        self.db_id = db_id

        # The name of the event type.
        self.name = name

        # The description of the event type.
        self.description = description

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
        return '/eventtype/' + self.name

    
    def add_child(self, event_type):
        ''' Add a sub event type.

        Parameters
        ----------
        event_type: :class:`mss_dataserver.event.event_type.EventType`
            The event type to assign.
        '''
        if not isinstance(event_type, EventType):
            self.RuntimeError("Wrong event_type class.")

        if event_type not in self.event_types:
            event_type.parent = self
            self.event_types.append(event_type)


    def write_to_database(self, project, db_session = None,
                          close_session = True):
        ''' Write the event type to the database.

        Parameters
        ----------
        project: :class:`mss_dataserver.core.project.Project`
            The project to use to access the database.
        '''
        if self.db_id is None:
            # If the db_id is None, insert a new event type.
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
                db_orm = project.db_tables['event_type']
                db_event_type = db_orm(name = self.name,
                                       description = self.description,
                                       author_uri = self.author_uri,
                                       agency_uri = self.agency_uri,
                                       creation_time = creation_time,
                                       parent_id = parent_id)

                # Add the event type children.
                #for cur_child in self.event_types:
                #    db_child = db_orm(name = cur_child.name,
                #                      description = cur_child.description,
                #                      author_uri = cur_child.author_uri,
                #                      agency_uri = cur_child.agency_uri,
                #                      creation_time = cur_child.creation_time)
                #    db_event_type.children.append(db_child)

                # Commit the event type to the database to get an id.
                db_session.add(db_event_type)
                db_session.commit()
                self.db_id = db_event_type.id
                
                # Add all children to the database.
                for cur_child in self.event_types:
                    cur_child.write_to_database(project,
                                                db_session = db_session,
                                                close_session = False)
            finally:
                if close_session:
                    db_session.close()


    @classmethod
    def from_orm(cls, db_event_type):
        ''' Convert a database orm event type to an EventType instance.

        Parameters
        ----------
        db_event_type: :class:`EventTypeDb`
            An instance of the sqlalchemy Table Mapper classe defined in
            :meth:`mss_dataserver.event.databaseFactory`

        Returns
        -------
        :class:`mss_dataserver.event.event_type.EventType`
            The event type created from the database ORM instance.
        '''
        event_type = cls(name = db_event_type.name,
                         description = db_event_type.description,
                         db_id = db_event_type.id,
                         agency_uri = db_event_type.agency_uri,
                         author_uri = db_event_type.author_uri,
                         creation_time = db_event_type.creation_time)

        # Add the children:
        for cur_db_child in db_event_type.children:
            cur_child = EventType.from_orm(cur_db_child)
            event_type.add_child(cur_child)
        return event_type
    
    
    @classmethod
    def load_from_db(cls, project):
        ''' Load the event type tree from the database.

        Parameters
        ----------
        project: :class:`mss_dataserver.core.project.Project`
            The project to use to access the database.

        Returns
        -------
        :class:`EventTypeDb`
            An instance of the sqlalchemy Table Mapper classe defined in
            :meth:`mss_dataserver.event.databaseFactory`
        '''
        roots = []
        db_session = project.get_db_session()
        try:
            et_orm = project.db_tables['event_type']
            query = db_session.query(et_orm)
            query = query.filter(et_orm.parent_id.is_(None))
            
            for cur_orm in query:
                cur_event_type = EventType.from_orm(db_event_type = cur_orm)
                roots.append(cur_event_type)
        finally:
            db_session.close()

        return roots
        
