# LICENSE
#
# This file is part of pSysmon.
#
# If you use pSysmon in any program or publication, please inform and
# acknowledge its author Stefan Mertl (stefan@mertl-research.at).
#
# pSysmon is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from builtins import str
from builtins import zip
from past.builtins import basestring
from builtins import object
import itertools
import logging
import warnings

import obspy.core.utcdatetime as utcdatetime

import psysmon
import psysmon.packages.event.detect as detect

#from profilehooks import profile

class Event(object):

    def __init__(self, start_time, end_time, db_id = None, public_id = None, event_type = None,
            event_type_certainty = None, description = None, comment = None,
            tags = [], agency_uri = None, author_uri = None, creation_time = None,
            parent = None, changed = True, detections = None):
        ''' Instance initialization

        '''
        # Check for correct input arguments.
        # Check for None values in the event limits.
        if start_time is None or end_time is None:
            raise ValueError("None values are not allowed for the event time limits.")

        # Check the event limits.
        if end_time < start_time:
            raise ValueError("The end_time %s is smaller than the start_time %s.", end_time, start_time)
        elif end_time == start_time:
            raise ValueError("The end_time %s is equal to the start_time %s.", end_time, start_time)

        # The parent object holding this event. Most likely this is a event
        # Catalog instance.
        self.parent = parent

        # The unique database id.
        self.db_id = db_id

        # The unique public id.
        self.public_id = public_id

        # The start time of the event.
        self.start_time = utcdatetime.UTCDateTime(start_time)

        # The end time of the event.
        self.end_time = utcdatetime.UTCDateTime(end_time)

        # The event type.
        self.event_type = event_type

        # The certainty of the event_type.
        self.event_type_certainty = event_type_certainty

        # The description of the event.
        self.description = description

        # The comment added to the event.
        self.comment = comment

        # The tags of the event.
        self.tags = tags

        # The detections associated with the event.
        if detections is None:
            self.detections = []
        else:
            self.detections = detections

        # The agency_uri of the creator.
        self.agency_uri = agency_uri

        # The author_uri of the creator.
        self.author_uri = author_uri

        # The time of creation of this event.
        if creation_time is None:
            creation_time = utcdatetime.UTCDateTime()
        self.creation_time = utcdatetime.UTCDateTime(creation_time)

        # Flag to indicate a change of the event attributes.
        self.changed = changed


    @property
    def rid(self):
        ''' The resource ID of the event.
        '''
        return '/event/' + str(self.db_id)


    @property
    def start_time_string(self):
        ''' The string representation of the start time.
        '''
        return self.start_time.isoformat()


    @property
    def end_time_string(self):
        ''' The string representation of the end time.
        '''
        return self.end_time.isoformat()


    @property
    def length(self):
        ''' The length of the event in seconds.
        '''
        return self.end_time - self.start_time


    def assign_channel_to_detections(self, inventory):
        ''' Set the channels according to the rec_stream_ids.
        '''
        # Get the unique stream ids.
        id_list = [x.rec_stream_id for x in self.detections]
        id_list = list(set(id_list))
        # Get the channels for the ids.
        channels = [inventory.get_channel_from_stream(id = x) for x in id_list]
        channels = [x[0] if len(x) == 1 else None for x in channels]
        channels = dict(list(zip(id_list, channels)))

        for cur_detection in self.detections:
            cur_detection.channel = channels[cur_detection.rec_stream_id]


    #@profile(immediate=True)
    def write_to_database(self, project):
        ''' Write the event to the pSysmon database.
        '''
        if self.db_id is None:
            # If the db_id is None, insert a new event.
            if self.creation_time is not None:
                creation_time = self.creation_time.isoformat()
            else:
                creation_time = None

            if self.parent is not None:
                catalog_id = self.parent.db_id
            else:
                catalog_id = None

            db_session = project.getDbSession()
            db_event_orm = project.dbTables['event']
            db_event = db_event_orm(ev_catalog_id = catalog_id,
                                    start_time = self.start_time.timestamp,
                                    end_time = self.end_time.timestamp,
                                    public_id = self.public_id,
                                    pref_origin_id = None,
                                    pref_magnitude_id = None,
                                    pref_focmec_id = None,
                                    ev_type_id = None,
                                    ev_type_certainty = self.event_type_certainty,
                                    description = self.description,
                                    agency_uri = self.agency_uri,
                                    author_uri = self.author_uri,
                                    creation_time = creation_time)

            # Commit the event to the database to get an id.
            db_session.add(db_event)
            db_session.commit()
            self.db_id = db_event.id

            # Add the detections to the event. Do this after the event got an
            # id.
            if len(self.detections) > 0 :
                # Load the detection_orms from the database.
                detection_table = project.dbTables['detection']
                d2e_orm_class = project.dbTables['detection_to_event']
                query = db_session.query(detection_table).\
                        filter(detection_table.id.in_([x.db_id for x in self.detections]))
                for cur_detection_orm in query:
                    d2e_orm = d2e_orm_class(ev_id = self.db_id,
                                            det_id = cur_detection_orm.id)
                    db_event.detections.append(d2e_orm)
            db_session.commit()

            db_session.close()
            self.changed = False
        else:
            # If the db_id is not None, update the existing event.
            db_session = project.getDbSession()
            db_event_orm = project.dbTables['event']
            query = db_session.query(db_event_orm).filter(db_event_orm.id == self.db_id)
            if db_session.query(query.exists()):
                db_event = query.scalar()
                if self.parent is not None:
                    db_event.ev_catalog_id = self.parent.db_id
                else:
                    db_event.ev_catalog_id = None
                db_event.start_time = self.start_time.timestamp
                db_event.end_time = self.end_time.timestamp
                db_event.public_id = self.public_id
                #db_event.pref_origin_id = self.pref_origin_id
                #db_event.pref_magnitude_id = self.pref_magnitude_id
                #db_event.pref_focmec_id = self.pref_focmec_id
                db_event.ev_type = self.event_type
                db_event.ev_type_certainty = self.event_type_certainty
                db_event.tags = ','.join(self.tags)
                db_event.agency_uri = self.agency_uri
                db_event.author_uri = self.author_uri
                if self.creation_time is not None:
                    db_event.creation_time = self.creation_time.isoformat()
                else:
                    db_event.creation_time = None

                # TODO: Add the handling of changed detections assigned to this
                # event.

                db_session.commit()
                db_session.close()
                self.changed = False
            else:
                raise RuntimeError("The event with ID=%d was not found in the database.", self.db_id)

    def get_db_orm(self, project):
        ''' Get an orm representation to use it for bulk insertion into
        the database.
        '''
        db_event_orm_class = project.dbTables['event']
        d2e_orm_class = project.dbTables['detection_to_event']

        if self.creation_time is not None:
            cur_creation_time = self.creation_time.isoformat()
        else:
            cur_creation_time = None

        if self.parent is not None:
            catalog_id = self.parent.db_id
        else:
            catalog_id = None

        labels = ['ev_catalog_id', 'start_time', 'end_time',
                  'public_id', 'description', 'comment', 'tags',
                  'ev_type_id', 'ev_type_certainty', 'pref_origin_id',
                  'pref_magnitude_id', 'pref_focmec_id', 'agency_uri',
                  'author_uri', 'creation_time']
        db_dict = dict(list(zip(labels,
                           (catalog_id,
                            self.start_time.timestamp,
                            self.end_time.timestamp,
                            self.public_id,
                            self.description,
                            self.comment,
                            ','.join(self.tags),
                            self.event_type,
                            self.event_type_certainty,
                            None,
                            None,
                            None,
                            self.agency_uri,
                            self.author_uri,
                            cur_creation_time))))
        db_event = db_event_orm_class(**db_dict)

        for cur_detection in self.detections:
            cur_d2e_orm = d2e_orm_class(ev_id = None,
                                        det_id = cur_detection.db_id)
            #cur_d2e_orm.detection = cur_detection.get_db_orm(project)
            db_event.detections.append(cur_d2e_orm)

        return db_event

    @classmethod
    def from_db_event(cls, db_event):
        ''' Convert a database orm mapper event to a event.

        Parameters
        ----------
        db_event : SQLAlchemy ORM
            The ORM of the events database table.
        '''
        if db_event.tags:
            event_tags = db_event.tags.split(',')
        else:
            event_tags = []
        event = cls(start_time = db_event.start_time,
                    end_time = db_event.end_time,
                    db_id = db_event.id,
                    public_id = db_event.public_id,
                    event_type = db_event.event_type,
                    event_type_certainty = db_event.ev_type_certainty,
                    description = db_event.description,
                    tags = event_tags,
                    agency_uri = db_event.agency_uri,
                    author_uri = db_event.author_uri,
                    creation_time = db_event.creation_time,
                    detections = [detect.Detection.from_db_detection(x.detection) for x in db_event.detections],
                    changed = False
                    )
        return event




class Catalog(object):

    def __init__(self, name, db_id = None, description = None, agency_uri = None,
            author_uri = None, creation_time = None, events = None):
        ''' Instance initialization.
        '''
        # The logging logger instance.
        logger_prefix = psysmon.logConfig['package_prefix']
        loggerName = logger_prefix + "." + __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(loggerName)

        # The unique database ID.
        self.db_id = db_id

        # The name of the catalog.
        self.name = name

        # The description of the catalog.
        self.description = description

        # The agency_uri of the creator.
        self.agency_uri = agency_uri

        # The author_uri of the creator.
        self.author_uri = author_uri

        # The time of creation of this event.
        if creation_time is None:
            self.creation_time = utcdatetime.UTCDateTime();
        else:
            self.creation_time = utcdatetime.UTCDateTime(creation_time);

        # The events of the catalog.
        if events is None:
            self.events = []
        else:
            self.events = events


    def add_events(self, events):
        ''' Add one or more events to the events.

        Parameters
        ----------
        events : list of :class:`Event`
            The events to add to the catalog.
        '''
        for cur_event in events:
            cur_event.parent = self
        self.events.extend(events)


    def get_events(self, start_time = None, end_time = None, **kwargs):
        ''' Get events using search criteria passed as keywords.

        Parameters
        ----------
        start_time : :class:`~obspy.core.utcdatetime.UTCDateTime`
            The minimum starttime of the detections.

        end_time : :class:`~obspy.core.utcdatetime.UTCDateTime`
            The maximum end_time of the detections.

        scnl : tuple of Strings
            The scnl code of the channel (e.g. ('GILA, 'HHZ', 'XX', '00')).
        '''
        ret_events = self.events

        valid_keys = ['db_id', 'public_id', 'event_type', 'changed']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_events = [x for x in ret_events if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        if start_time is not None:
            ret_events = [x for x in ret_events if (x.end_time is None) or (x.end_time > start_time)]

        if end_time is not None:
            ret_events = [x for x in ret_events if x.start_time < end_time]

        return ret_events


    #@profile(immediate=True)
    def write_to_database(self, project,
                          only_changed_events = True,
                          bulk_insert = False):
        ''' Write the catalog to the database.

        '''
        if self.db_id is None:
            # If the db_id is None, insert a new catalog.
            if self.creation_time is not None:
                creation_time = self.creation_time.isoformat()
            else:
                creation_time = None

            db_session = project.getDbSession()
            db_catalog_orm = project.dbTables['event_catalog']
            db_catalog = db_catalog_orm(name = self.name,
                                    description = self.description,
                                    agency_uri = self.agency_uri,
                                    author_uri = self.author_uri,
                                    creation_time = creation_time
                                   )
            db_session.add(db_catalog)
            db_session.commit()
            self.db_id = db_catalog.id
            db_session.close()

        else:
            # If the db_id is not None, update the existing event.
            db_session = project.getDbSession()
            db_catalog_orm = project.dbTables['event_catalog']
            query = db_session.query(db_catalog_orm).filter(db_catalog_orm.id == self.db_id)
            if db_session.query(query.exists()):
                db_catalog = query.scalar()

                db_catalog.name = self.name
                db_catalog.description = self.description
                db_catalog.agency_uri = self.agency_uri
                db_catalog.author_uri = self.author_uri
                if self.creation_time is not None:
                    db_catalog.creation_time = self.creation_time.isoformat()
                else:
                    db_catalog.creation_time = None

                db_session.commit()
                db_session.close()
            else:
                raise RuntimeError("The event catalog with ID=%d was not found in the database.", self.db_id)


        # Write or update all events of the catalog to the database.
        if bulk_insert:
            db_data = self.get_events_db_data(project = project)
            db_session = project.getDbSession()
            try:
                assigned_detections = [x.detections for x in db_data]

                for cur_db_data in db_data:
                    cur_db_data.detections = []

                db_session.add_all(db_data)
                db_session.flush()

                for k, cur_db_data in enumerate(db_data):
                    for cur_detection in assigned_detections[k]:
                        cur_detection.ev_id = cur_db_data.id
                db_session.add_all(itertools.chain.from_iterable(assigned_detections))
                db_session.commit()
            finally:
                db_session.close()
        else:
            for cur_event in [x for x in self.events if x.changed is True]:
                cur_event.write_to_database(project)

    def get_events_db_data(self, project):
        ''' Get a dictionary to bulk insert into the dabase.
        '''
        db_data = [x.get_db_orm(project) for x in self.events]
        return db_data


    def load_events(self, project, start_time = None, end_time = None, event_id = None,
            min_event_length = None, event_types = None, event_tags = None):
        ''' Load events from the database.

        The query can be limited using the allowed keyword arguments.

        Parameters
        ----------
        start_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The begin of the time-span to load.

        end_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The end of the time-span to load.
        '''
        if project is None:
            raise RuntimeError("The project is None. Can't query the database without a project.")

        db_session = project.getDbSession()
        try:
            events_table = project.dbTables['event']
            query = db_session.query(events_table).\
                    filter(events_table.ev_catalog_id == self.db_id)

            if start_time:
                query = query.filter(events_table.start_time >= start_time.timestamp)

            if end_time:
                query = query.filter(events_table.start_time <= end_time.timestamp)

            if event_id:
                query = query.filter(events_table.id in event_id)

            if min_event_length:
                query = query.filter(events_table.end_time - events_table.start_time >= min_event_length)

            if event_tags:
                for cur_tag in event_tags:
                    query = query.filter(events_table.tags.like('%' + cur_tag + '%'))

            events_to_add = []
            for cur_orm in query:
                try:
                    cur_event = Event.from_db_event(cur_orm)
                    events_to_add.append(cur_event)
                except:
                    self.logger.exception("Error when creating an event object from database values for event %d. Skipping this event.", cur_orm.id)
            self.add_events(events_to_add)

        finally:
            db_session.close()


    def clear_events(self):
        ''' Clear the events list.
        '''
        self.events = []


    def write_to_csv(self, filepath):
        ''' Write the events in the catalog to CSV file.
        '''
        pass


    @classmethod
    def from_db_catalog(cls, db_catalog, load_events = False):
        ''' Convert a database orm mapper catalog to a catalog.

        Parameters
        ----------
        db_catalog : SQLAlchemy ORM
            The ORM of the events catalog database table.

        load_events : Boolean
            If true all events contained in the catalog are loaded
            from the database.
        '''
        catalog = cls(name = db_catalog.name,
                      db_id = db_catalog.id,
                      description = db_catalog.description,
                      agency_uri = db_catalog.agency_uri,
                      author_uri = db_catalog.author_uri,
                      creation_time = db_catalog.creation_time
                      )

        # Add the events to the catalog.
        if load_events is True:
            for cur_db_event in db_catalog.events:
                cur_event = Event.from_db_event(cur_db_event)
                catalog.add_events([cur_event,])
        return catalog




class Library(object):
    ''' Manage a set of event catalogs.
    '''

    def __init__(self, name):
        ''' Initialize the instance.
        '''

        # The name of the library.
        self.name = name

        # The catalogs of the library.
        self.catalogs = {}


    def add_catalog(self, catalog):
        ''' Add one or more catalogs to the library.

        Parameters
        ----------
        catalog : :class:`Catalog` or list of :class:`Catalog`
            The catalog(s) to add to the library.
        '''

        if isinstance(catalog, list):
            for cur_catalog in catalog:
                self.add_catalog(cur_catalog)
        else:
            self.catalogs[catalog.name] = catalog


    def remove_catalog(self, name):
        ''' Remove a catalog from the library.

        Parameters
        ----------
        name : String
            The name of the catalog to remove.

        Returns
        -------
        removed_catalog : :class:`Catalog`
            The removed catalog. None if no catalog was removed.
        '''
        if name in iter(self.catalogs.keys()):
            return self.catalogs.pop(name)
        else:
            return None

    def clear(self):
        ''' Remove all catalogs.
        '''
        self.catalogs = {}


    def get_catalogs_in_db(self, project):
        ''' Query the available catalogs in the database.

        Parameters
        ----------
        project : :class:`psysmon.core.project.Project`
            The project managing the database.

        Returns
        -------
        catalog_names : List of Strings
            The available catalog names in the database.
        '''
        catalog_names = []
        db_session = project.getDbSession()
        try:
            db_catalog_orm = project.dbTables['event_catalog']
            query = db_session.query(db_catalog_orm)
            if db_session.query(query.exists()):
                catalog_names = [x.name for x in query.order_by(db_catalog_orm.name)]
        finally:
            db_session.close()

        return catalog_names


    def load_catalog_from_db(self, project, name, load_events = False):
        ''' Load catalogs from the database.

        Parameters
        ----------
        project : :class:`psysmon.core.project.Project`
            The project managing the database.

        name : String or list of Strings
            The name of the catalog to load from the database.
        '''
        if isinstance(name, basestring):
            name = [name, ]

        db_session = project.getDbSession()
        try:
            db_catalog_orm = project.dbTables['event_catalog']
            query = db_session.query(db_catalog_orm).filter(db_catalog_orm.name.in_(name))
            if db_session.query(query.exists()):
                for cur_db_catalog in query:
                    cur_catalog = Catalog.from_db_catalog(cur_db_catalog, load_events)
                    self.add_catalog(cur_catalog)
        finally:
            db_session.close()



