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
 # Copyright 2019 Stefan Mertl
##############################################################################
''' Event handling with library, catalogs and events.
'''

from builtins import str
from builtins import zip
from builtins import object
import itertools
import logging
import warnings

import obspy
import obspy.core.utcdatetime as utcdatetime
import mss_dataserver.event.detection as detection
import mss_dataserver.event.event_type as ev_type
import mss_dataserver.localize.origin as mssds_origin

#from profilehooks import profile

class Event(object):
    ''' A seismic event.

    Parameters
    ----------
    start_time: str or :class:`obspy.UTCDateTime`
        The start time of the event. A string that can be parsed
        by :class:`obspy.UTCDateTime` or a :class:`obspy.UTCDateTime` instance.

    end_time: str or :class:`obspy.UTCDateTime`
        The end time of the event. A string that can be parsed
        by :class:`obspy.UTCDateTime` or a :class:`obspy.UTCDateTime` instance.

    db_id: int
        The database id of the event.

    public_id: str
        The public id of the event.

    event_type: str
        The type of the event.

    event_type_certainty: str
        The certainty of the event type.

    description: str
        The description of the event.

    comment: str
        The comment to the event.

    tags: list of str
        The tags of the event.

    agency_uri: str
        The Uniform Resource Identifier of the author agency.

    author_uri: str
        The Uniform Resource Identifier of the author.

    creation_time: str or :class:`obspy.UTCDateTime`
        The creation time of the event. A string that can be parsed
        by :class:`obspy.UTCDateTime` or a :class:`obspy.UTCDateTime` instance.

    parent: object
        The parent object containing the event.

    changed: bool
        Flag indicating if one of the event attributes has changes.

    detection: list of :class:`mss_dataserver.detection.Detection`
        The detections associated to the event.


    Attributes
    ----------
    detection_state: str
        The state of the event detection [new, updated, closed].

    pgv_stream: :class:`obspy.Stream`
        The PGV stream of the event.

    detection_data: dict
        A dictionary holding the detection data related to the event.
        The detection data is created during the detection process
        using :class:`mss_dataserver.event.delaunay_detection.DelaunayDetector`.
    '''

    def __init__(self, start_time, end_time, db_id = None, db_cat_id = None,
                 public_id = None, event_type = None,
                 event_type_certainty = None, description = None, comment = None,
                 tags = [], agency_uri = None, author_uri = None, creation_time = None,
                 parent = None, changed = True, detections = None):
        ''' Instance initialization

        '''
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        
        # Check for correct input arguments.
        # Check for None values in the event limits.
        if start_time is None or end_time is None:
            raise ValueError("None values are not allowed for the event time limits.")

        # Check the event limits.
        if end_time < start_time:
            raise ValueError("The end_time %s is smaller than the start_time %s.", end_time, start_time)

        # The parent object holding this event. Most likely this is a event
        # Catalog instance.
        self.parent = parent

        # The unique database id.
        self.db_id = db_id

        # The database id of the related catalog.
        self.db_cat_id = db_cat_id

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

        # State of the event during detection (new, updated, closed).
        self.detection_state = None

        # The PGV data stream.
        self.pgv_stream = obspy.Stream()

        # The event detection data.
        self.detection_data = {}

        # The unique public id.
        self._public_id = public_id

        # The origins of the event.
        self.origins = []

        # The preferred origin of the event.
        self.pref_origin = None


    @property
    def rid(self):
        ''' str: The resource ID of the event.
        '''
        return '/event/' + str(self.db_id)

    @property
    def public_id(self):
        ''' str: The public ID of the event.
        '''
        if self._public_id is None:
            prefix = ''
            if self.agency_uri is not None:
                prefix += self.agency_uri + '_'

            if self.author_uri is not None:
                prefix += self.author_uri + '_'

            strftime_format = '%Y-%m-%dT%H%M%S'
            start_time_string = self.start_time.strftime(strftime_format)
            start_time_string += '{0:06d}'.format(self.start_time.microsecond)
            if len(prefix) > 0:
                public_id = prefix + start_time_string
            else:
                public_id = start_time_string

            return public_id
        else:
            return self._public_id

    @property
    def start_time_string(self):
        ''' str: The string representation of the start time.
        '''
        return self.start_time.isoformat()


    @property
    def end_time_string(self):
        ''' str: The string representation of the end time.
        '''
        return self.end_time.isoformat()


    @property
    def length(self):
        ''' float: The length of the event in seconds.
        '''
        return self.end_time - self.start_time

    @property
    def max_pgv(self):
        ''' float: The maximum PGV of all detections.
        '''
        if len(self.detections) > 0:
            return max([x.absolute_max_pgv for x in self.detections])
        else:
            return None

    @property
    def triggered_stations(self):
        ''' list of tuple of str: The nsl codes of the stations that contributed to a detection.
        '''
        nsl = []
        for cur_detection in self.detections:
            for cur_key in cur_detection.max_pgv.keys():
                if cur_key not in nsl:
                    nsl.append(cur_key)

        return sorted(nsl)

    def get_max_pgv_per_station(self):
        ''' Compute the maximum PGV of the individual stations that have
            been associated to a detection.

        Returns
        -------
        max_pgv: dict of float
            The maximum PGV values of the stations.
        '''
        max_pgv = {}
        for cur_detection in self.detections:
            for cur_key, cur_pgv in cur_detection.max_pgv.items():
                if cur_key not in max_pgv:
                    max_pgv[cur_key] = float(cur_pgv)
                elif cur_pgv > max_pgv[cur_key]:
                    max_pgv[cur_key] = float(cur_pgv)

        return max_pgv

    def get_detection_limits_per_station(self):
        ''' Compute the detection start and end times for each station.

        Returns
        -------
        detection_limits: dict of :class:`obspy.UTCDateTime`
            The detection limits of the stations.
        '''
        detection_limits = {}
        for cur_detection in self.detections:
            cur_start = cur_detection.start_time
            cur_end = cur_detection.end_time
            for cur_station in cur_detection.stations:
                if cur_station.nsl_string not in detection_limits.keys():
                    detection_limits[cur_station.nsl_string] = [cur_start, cur_end]
                elif cur_start < detection_limits[cur_station.nsl_string][0]:
                    detection_limits[cur_station.nsl_string][0] = cur_start
                elif cur_end > detection_limits[cur_station.nsl_string][1]:
                    detection_limits[cur_station.nsl_string][1] = cur_end

        return detection_limits


    def station_has_triggered(self, station):
        ''' Check if a detection has been triggered at a station.

        Parameters
        ----------
        station: :class:`mss_dataserver.geometry.inventory.Station`
            The station to check.

        Returns
        -------
        has_triggered: bool
            True, if the station has triggered, False otherwise.
        '''
        found_detections = []
        has_triggered = False
        for cur_detection in self.detections:
            if station in cur_detection.stations:
                found_detections.append(cur_detection)

        if len(found_detections) > 0:
            has_triggered = True

        return has_triggered


    def add_detection(self, detection):
        ''' Add a detection to the event.

        Parameters
        ----------
        detection: :class:`mss_dataserver.event.detection.Detection` or :obj:`list` of :class:`mss_dataserver.event.detection.Detection`
            The detection(s) to add.
        '''
        if type(detection) is list:
            self.detections.extend(detection)
        else:
            self.detections.append(detection)


    def get_detection(self, stations):
        ''' Get a detection with the provided stations.

        Parameters
        ----------
        stations: :obj:`list` of :class:`mss_dataserver.geometry.Inventory.Station`
            The stations for which to get the detections.

        Returns
        -------
        found_detections: :obj:`list` of :class:`mss_dataserver.event.detection.Detection`
            The matching detections.
        '''
        unique_stations = []
        for cur_station in stations:
            if cur_station not in unique_stations:
                unique_stations.append(cur_station)

        found_detections = []
        for cur_detection in self.detections:
            found = [x for x in unique_stations if x in cur_detection.stations]
            if len(found) == len(cur_detection.stations):
                found_detections.append(cur_detection)

        return found_detections

    
    def has_detection(self, stations):
        ''' Check if a detection is available for a set of stations.

        Parameters
        ----------
        stations: :obj:`list` of :class:`mss_dataserver.geometry.Inventory.Station`
            The stations for which to get the detections.

        Returns
        -------
        bool
            True if a detection is available for the stations. False otherwise.
        '''
        detections = self.get_detection(stations)
        if len(detections) > 0:
            return True
        else:
            return False

        
    def add_origin(self, origin):
        ''' Add an origin to the event.

        Parameters
        ----------
        origin: :class:`mss_dataserver.localize.origin.Origin` or :obj:`list` of :class:`mss_dataserver.localize.origin.Origin`
            The origin or a list of origins to add.
        '''
        if type(origin) is list:
            self.origins.extend(origin)
            for cur_origin in origin:
                cur_origin.parent = self
        else:
            self.origins.append(origin)
            origin.parent = self


    def set_preferred_origin(self, origin):
        ''' Set the preferred origin.

        Parameters
        ----------
        origin: :class:`mss_dataserver.localize.origin.Origin`
            The preferred origin of the event.

        '''
        if origin not in self.origins:
            self.logger.Error("The preferred origin is not available in the event origins.")
            return

        self.pref_origin = origin


    def set_event_type(self, event_type):
        ''' Set the event type.
       
        Parameters
        ----------
        event_type: :class:`mss_dataserver.event.event_type.EventType`
            The event type to assign.
        '''
        if not isinstance(event_type, ev_type.EventType):
            self.RuntimeError("Wrong event_type class.")
        else:
            self.event_type = event_type
        

    def assign_channel_to_detections(self, inventory):
        ''' Set the channels according to the rec_stream_ids.

        Parameters
        ----------
        inventory: :class:`mss_dataserver.geometry.inventory.Inventory`
            The inventory used to find the matching channel.
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
        ''' Write the event to the database.

        Parameters
        ----------
        project: :class:`mss_dataserver.core.project.Project`
            The project to use to access the database.
        '''
        if self.parent is not None:
            catalog_id = self.parent.db_id
        else:
            catalog_id = self.db_cat_id

        if self.event_type is not None:
            event_type_id = self.event_type.db_id
        else:
            event_type_id = None

        if self.creation_time is not None:
            creation_time = self.creation_time.isoformat()
        else:
            creation_time = None
                
        if self.db_id is None:
            # If the db_id is None, insert a new event.
            db_session = project.get_db_session()
            try:
                db_event_orm = project.db_tables['event']
                db_event = db_event_orm(ev_catalog_id = catalog_id,
                                        start_time = self.start_time.timestamp,
                                        end_time = self.end_time.timestamp,
                                        public_id = self.public_id,
                                        pref_origin_id = None,
                                        pref_magnitude_id = None,
                                        pref_focmec_id = None,
                                        ev_type_id = event_type_id,
                                        ev_type_certainty = self.event_type_certainty,
                                        description = self.description,
                                        tags = ','.join(self.tags),
                                        agency_uri = self.agency_uri,
                                        author_uri = self.author_uri,
                                        creation_time = creation_time)

                # Commit the event to the database to get an id.
                db_session.add(db_event)
                db_session.commit()
                self.db_id = db_event.id

                # Add the detections to the event. Do this after the event
                # got an id.
                if len(self.detections) > 0:
                    # Load the detection_orms from the database.
                    detection_table = project.db_tables['detection']
                    d2e_orm_class = project.db_tables['detection_to_event']
                    id_filter = [x.db_id for x in self.detections]
                    query = db_session.query(detection_table).\
                        filter(detection_table.id.in_(id_filter))
                    for cur_detection_orm in query:
                        d2e_orm = d2e_orm_class(ev_id = self.db_id,
                                                det_id = cur_detection_orm.id)
                        db_event.detections.append(d2e_orm)

                # Add the origins of the event to the database.
                # This updated the db_id of the origins.
                if len(self.origins) > 0:
                    for cur_origin in self.origins:
                        cur_origin.write_to_database(project = project,
                                                     db_session = db_session,
                                                     close_session = False)

                # Update the preferred origin id of the event.
                if self.pref_origin is not None:
                    pref_origin_id = self.pref_origin.db_id
                    db_event.pref_origin_id = pref_origin_id

                db_session.commit()
                self.changed = False
            finally:
                db_session.close()
        else:
            # If the db_id is not None, update the existing event.
            db_session = project.get_db_session()
            try:
                db_event_orm = project.db_tables['event']
                query = db_session.query(db_event_orm).filter(db_event_orm.id == self.db_id)
                if db_session.query(query.exists()):
                    self.logger.debug('event_type_id: %s', event_type_id)
                    db_event = query.scalar()
                    db_event.ev_catalog_id = catalog_id
                    db_event.start_time = self.start_time.timestamp
                    db_event.end_time = self.end_time.timestamp
                    db_event.public_id = self.public_id
                    #db_event.pref_origin_id = self.pref_origin_id
                    #db_event.pref_magnitude_id = self.pref_magnitude_id
                    #db_event.pref_focmec_id = self.pref_focmec_id
                    db_event.ev_type_id = event_type_id
                    db_event.ev_type_certainty = self.event_type_certainty
                    db_event.tags = ','.join(self.tags)
                    db_event.agency_uri = self.agency_uri
                    db_event.author_uri = self.author_uri
                    db_event.creation_time = creation_time

                    # Update the preferred origin id of the event.
                    if self.pref_origin is not None:
                        pref_origin_id = self.pref_origin.db_id
                        db_event.pref_origin_id = pref_origin_id

                    # TODO: Add the handling of changed detections assigned to
                    # the event.

                    # TODO: Add the handling of changed origins assigned to
                    # the event.

                    db_session.commit()
                    self.changed = False
                else:
                    raise RuntimeError("The event with ID=%d was not found in the database.", self.db_id)
            finally:
                db_session.close()

    def get_db_orm(self, project):
        ''' Get an orm representation to use it for bulk insertion into
        the database.

        Parameters
        ----------
        project: :class:`mss_dataserver.core.project.Project`
            The project to use to access the database.

        Returns
        -------
        :class:`EventDb`
            An instance of the sqlalchemy Table Mapper class defined in 
            :meth:`mss_dataserver.event.databaseFactory`.
        '''
        db_event_orm_class = project.db_tables['event']
        d2e_orm_class = project.db_tables['detection_to_event']

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
    def from_orm(cls, db_event, inventory, ev_type_tree = None):
        ''' Convert a database orm mapper event to a event.

        Parameters
        ----------
        db_event : :class:`EventDb`
            An instance of the sqlalchemy Table Mapper class defined in 
            :meth:`mss_dataserver.event.databaseFactory`.

        inventory: :class:`mss_dataserver.geometry.inventory.Inventory`
            The inventory used to map geometry information to the event.

        Returns
        -------
        :class:`mss_dataserver.event.core.Event`
            The event created from the database ORM instance.
        '''
        if db_event.tags:
            event_tags = db_event.tags.split(',')
        else:
            event_tags = []

        if db_event.event_type is not None and ev_type_tree is not None:
            event_type = ev_type_tree[0].get_child_by_id(id = db_event.ev_type_id)
        else:
            event_type = None

        assigned_detections = [detection.Detection.from_orm(x.detection, inventory) for x in db_event.detections]
        event = cls(start_time = db_event.start_time,
                    end_time = db_event.end_time,
                    db_id = db_event.id,
                    db_cat_id = db_event.ev_catalog_id,
                    public_id = db_event.public_id,
                    event_type = event_type,
                    event_type_certainty = db_event.ev_type_certainty,
                    description = db_event.description,
                    tags = event_tags,
                    agency_uri = db_event.agency_uri,
                    author_uri = db_event.author_uri,
                    creation_time = db_event.creation_time,
                    detections = assigned_detections,
                    changed = False)

        # Add the origins to the event.
        assigned_origins = [mssds_origin.Origin.from_orm(x) for x in db_event.origins]
        event.add_origin(assigned_origins)

        # Set the preferred origin.
        if db_event.pref_origin_id is not None:
            poid = db_event.pref_origin_id
            pref_origin = [x for x in assigned_origins if x.db_id == poid]
            if len(pref_origin) == 1:
                pref_origin = pref_origin[0]
                event.set_preferred_origin(pref_origin)
            elif len(pref_origin) > 1:
                cls.logger.error("Multiple event origins returned for id %d.",
                                 poid)

        return event


class Catalog(object):
    ''' A catalog holding seismic events.

    Parameters
    ----------
    name: str
        The name of the catalog.

    db_id: int
        The database id of the catalog.

    description: str
        The description of the catalog.

    agency_uri: str
        The uniform resource identifier of the author agency.

    author_uri: str
        The uniform resource identifier of the author.

    creation_time: :obj:`str` or :class:`obspy.UTCDateTime`
        The creation time of the event. A string that can be parsed
        by :class:`obspy.UTCDateTime` or a :class:`obspy.UTCDateTime` instance.

    events: :obj:`list` of :class:`~mss_dataserver.event.core.Event`
        The events of the catalog.

    
    Attributes
    ----------
    logger: logging.Logger
        The logger of the instance.
    '''

    def __init__(self, name, db_id = None, description = None, agency_uri = None,
            author_uri = None, creation_time = None, events = None):
        ''' Instance initialization.
        '''
        # The logging logger instance.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

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
            self.creation_time = utcdatetime.UTCDateTime()
        else:
            self.creation_time = utcdatetime.UTCDateTime(creation_time)

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


    def remove_event(self, event):
        ''' Remove an event from the catalog.
        
        Parameters
        ----------
        event : :class: `Event`
            The event to remove from the catalog.
        '''
        if event in self.events:
            self.events.remove(event)
            event.parent = None


    def get_events(self, start_time = None, end_time = None, **kwargs):
        ''' Get events using search criteria passed as keywords.

        Only events already loaded from the database are searched.

        Parameters
        ----------
        start_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The minimum starttime of the detections.

        end_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The maximum end_time of the detections.


        Keyword Arguments
        -----------------
        db_id: int
            The database id of the event.

        public_id: str
            The public_id of the event.

        event_type: str
            The event type (not yet implemented).

        changed: bool
            True is an event has changed, False otherwise.

        Returns
        -------
        :obj:`list` of :class:`Event`
            The events matching the search criteria.
        '''
        ret_events = self.events

        valid_keys = ['db_id', 'public_id', 'event_type', 'changed']

        for cur_key, cur_value in kwargs.items():
            if cur_value is None:
                continue
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

        Attributes
        ----------
        project: :class:`~mss_dataserver.core.project.Project`
            The project used to access the database.

        only_changed_events: bool
            Write only events that have changed to the database.

        bulk_insert: bool
            If True, insert all events in one database transaction,
            otherwise each event is written to the database individually.

        '''
        if self.db_id is None:
            # If the db_id is None, insert a new catalog.
            if self.creation_time is not None:
                creation_time = self.creation_time.isoformat()
            else:
                creation_time = None

            db_session = project.get_db_session()
            db_catalog_orm = project.db_tables['event_catalog']
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
            db_session = project.get_db_session()
            db_catalog_orm = project.db_tables['event_catalog']
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
            db_session = project.get_db_session()
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
        ''' Get a list of mapper class instances for bulk insert.

        Attributes
        ----------
        project: :class:`~mss_dataserver.core.project.Project`
            The project used to access the database.

        Returns
        -------
        :obj:`list` of :class:`mss_dataserver.event.databaseFactory.EventDb`
            A list of EventDb mapper class instances.

        See Also
        --------
        :meth:`mss_dataserver.event.databaseFactory`
        '''
        db_data = [x.get_db_orm(project) for x in self.events]
        return db_data


    def load_events(self, project, start_time = None, end_time = None, event_id = None,
            min_event_length = None, event_types = None, event_tags = None):
        ''' Load events from the database.

        The query can be limited using the allowed keyword arguments.

        Parameters
        ----------
        project: :class:`~mss_dataserver.core.project.Project`
            The project used to access the database.

        start_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The begin of the time-span to load.

        end_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The end of the time-span to load.

        event_id: list of int
            The database ids of the events to load.

        min_event_length: float
            The minimum length of the events to load.

        event_types: not yet implemented
            Not yet implemented.

        event_tags: list of str
            The tags of an event.
        
        '''
        if project is None:
            raise RuntimeError("The project is None. Can't query the database without a project.")

        db_session = project.get_db_session()
        try:
            events_table = project.db_tables['event']
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
                    cur_event = Event.from_orm(cur_orm)
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

        Not yet implemented.
        '''
        pass


    @classmethod
    def from_orm(cls, db_catalog, inventory, ev_type_tree = None, load_events = False):
        ''' Convert a database orm mapper catalog to a catalog.

        Parameters
        ----------
        db_catalog : :class:`mss_dataserver.event.databaseFactory.EventCatalogDb`
            The mapper class instance of the event catalog database table.

        inventory: :class:`mss_dataserver.geometry.inventory.Inventory`
            The inventory used to map geometry information to the event.

        load_events : bool
            If true all events contained in the catalog are loaded
            from the database.

        Returns
        -------
        :class:`~mss_dataserver.event.core.Catalog`
            The event catalog instance.
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
                cur_event = Event.from_orm(db_event = cur_db_event,
                                           inventory = inventory,
                                           ev_type_tree = ev_type_tree)
                catalog.add_events([cur_event])
        return catalog




class Library(object):
    ''' Manage a set of event catalogs.

    Parameters
    ----------
    name: str
        The name of the library.

    Attributes
    ----------
    catalogs: dict
        A dictionary of event catalogs (:class:`~mss_dataserver.event.core.Catalog`) with
        the name of the catalog as the dictionary key.
    '''

    def __init__(self, name):
        ''' Initialize the instance.
        '''
        # The logging logger instance.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

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
        name : str
            The name of the catalog to remove.

        Returns
        -------
        :class:`Catalog`
            The removed catalog. None if no catalog was removed.
        '''
        if name in iter(self.catalogs.keys()):
            return self.catalogs.pop(name)
        else:
            return None

        
    def get_catalog_by_id(self, cat_id):
        ''' Get a catalog by the database id.
        '''
        ret_cat = [x for x in self.catalogs.values() if x.db_id == cat_id]
        if len(ret_cat) == 0:
            ret_cat = None
        elif len(ret_cat) == 1:
            ret_cat = ret_cat[0]
        else:
            cat_names = [x.name for x in ret_cat]
            msg = 'Multiple events returned with the same id {:d}: {}'.format(cat_id,
                                                                              cat_names)
            self.logger.error(msg)
        return ret_cat
    

    def clear(self):
        ''' Remove all catalogs.
        '''
        self.catalogs = {}


    def get_catalogs_in_db(self, project):
        ''' Query the available catalogs in the database.

        Parameters
        ----------
        project : :class:`~mss_dataserver.core.project.Project`
            The project used to access the database.

        Returns
        -------
        :obj:`list` of :obj:`str`
            The available catalog names in the database.
        '''
        catalog_names = []
        db_session = project.get_db_session()
        try:
            db_catalog_orm = project.db_tables['event_catalog']
            query = db_session.query(db_catalog_orm)
            if db_session.query(query.exists()):
                catalog_names = [x.name for x in query.order_by(db_catalog_orm.name)]
        finally:
            db_session.close()

        return catalog_names


    def load_catalog_from_db(self, project, name = None, cat_id = None,
                             load_events = False):
        ''' Load catalogs from the database.

        Parameters
        ----------
        project : :class:`~mss_dataserver.core.project.Project`
            The project used to access the database.

        name : :obj:`str` of :obj:`list` of :obj:`str`
            The name of the catalog to load from the database.

        cat_id: int
            The database id of the catalog to load.

        load_events: bool
            Load the events from the database.
        '''
        if isinstance(name, str):
            name = [name, ]

        # Load the event types tree from the database.
        ev_type_tree = ev_type.EventType.load_from_db(project = project)

        db_session = project.get_db_session()
        try:
            db_catalog_orm = project.db_tables['event_catalog']
            query = db_session.query(db_catalog_orm)
            if name is not None:
                query = query.filter(db_catalog_orm.name.in_(name))

            if cat_id is not None:
                query = query.filter(db_catalog_orm.id == cat_id)
                
            if db_session.query(query.exists()):
                for cur_db_catalog in query:
                    cur_catalog = Catalog.from_orm(db_catalog = cur_db_catalog,
                                                   load_events = load_events,
                                                   inventory = project.inventory,
                                                   ev_type_tree = ev_type_tree)
                    self.add_catalog(cur_catalog)
        finally:
            db_session.close()


    def load_event_from_db(self, project, ev_id = None, public_id = None):
        ''' Load an event from the database by database id or the
        public id.

        Parameters
        ----------
        project : :class:`~mss_dataserver.core.project.Project`
            The project used to access the database.

        ev_id : int
            The unique database id of the event.

        public_id : str
            The public id of the event.

        Returns
        -------
        :obj:`list` of :class:`Event`
            The events found in the database matching the search criteria.
        '''
        if ev_id is None and public_id is None:
            raise RuntimeError(("You have to specify at least one of the two "
                                "parameters ev_id and public_id."))

        found_events = []
        db_session = project.get_db_session()

        # Load the event types tree from the database.
        ev_type_tree = ev_type.EventType.load_from_db(project = project)
        
        try:
            events_table = project.db_tables['event']
            query = db_session.query(events_table)

            if ev_id is not None:
                query = query.filter(events_table.id == ev_id)

            if public_id is not None:
                query = query.filter(events_table.public_id.like(public_id))

            for cur_orm in query:
                try:
                    cur_event = Event.from_orm(db_event = cur_orm,
                                               inventory = project.db_inventory,
                                               ev_type_tree = ev_type_tree)
                    # Get the parent catalog of the event.
                    cat_id = cur_orm.ev_catalog_id

                    # TODO: Better handling of the event catalog.
                    #if cat_id is not None:
                    #    # Load the required catalog from the database to the library.
                    #    self.load_catalog_from_db(cat_id = cat_id,
                    #                              project = project)
                    #    # Get the catalog from the library.
                    #    parent_cat = self.get_catalog_by_id(cat_id = cat_id)
                    #    # Set the parent catalog of the event.
                    #    cur_event.parent = parent_cat
                        
                    found_events.append(cur_event)
                except Exception:
                    self.logger.exception("Error when creating an event object from database values for event %d. Skipping this event.", cur_orm.id)
        finally:
            db_session.close()

        return found_events


    def get_events(self, catalog_names = None, start_time = None, end_time = None, **kwargs):
        ''' Get events from the library using from search criteria passed as keywords.

        Only events already loaded from the database are processed.

        Parameters
        ----------
        start_time : :class:`~bspy.core.utcdatetime.UTCDateTime`
            The minimum starttime of the detections.

        end_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The maximum end_time of the detections.


        Keyword Arguments
        -----------------
        kwargs:
            Keyword arguments passed to :meth:`Catalog.get_events`.

        Returns
        -------
        :obj:`list` of :class:`Event`
            The events matching the search criteria.
        '''
        ret_events = []

        if catalog_names is None:
            catalog_names = list(self.catalogs.keys())

        # Filter out catalog names, that are not available.
        catalog_names = [x for x in catalog_names if x in self.catalogs.keys()]

        for cur_catalog_name in catalog_names:
            cur_catalog = self.catalogs[cur_catalog_name]
            ret_events.extend(cur_catalog.get_events(start_time = start_time,
                                                     end_time = end_time,
                                                     **kwargs))
        return ret_events


    def load_event_by_id(self, project, ev_id = None, public_id = None):
        ''' Get an event by the database id or the public id.

        Parameters
        ----------
        project: :class:`mss_dataserver.core.project.Project`
            The project to use to access the database.

        ev_id : int
            The unique database id of the event.

        public_id : str
            The unique public id of the event.

        Returns
        -------
        :class:`Event`
            The event matching the search criteria.

        '''
        if ev_id is None and public_id is None:
            raise RuntimeError(("You have to specify at least one of the two "
                                "parameters ev_id and public_id."))

        event = None
        # Check if the event is available in the existing catalogs.
        event_list = self.get_events(db_id = ev_id,
                                     public_id = public_id)

        if len(event_list) == 0:
            # Load the event directly from the database.
            event_list = self.load_event_from_db(project = project,
                                                 ev_id = ev_id,
                                                 public_id = public_id)

        if len(event_list) == 1:
            event = event_list[0]
        elif len(event_list) > 1:
            raise RuntimeError(("More than one events found, "
                                "this shouldn't happen for unique ids."))

        return event
