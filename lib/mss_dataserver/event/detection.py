# -*- coding: utf-8 -*-

import mss_dataserver.geometry.db_inventory as db_inventory
import obspy.core.utcdatetime as utcdatetime

class Detection(object):
    ''' A MSS Delaunay detection.
    '''
    def __init__(self, start_time, end_time, stations, max_pgv,
                 db_id = None, catalog_id = None,
                 agency_uri = None, author_uri = None, creation_time = None,
                 parent = None, changed = True):
        ''' Initialize the instance.
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

        # The parent object holding this event. Most likely this is a detection
        # Catalog instance or an event instance.
        self.parent = parent

        # The unique database id.
        self.db_id = db_id

        # The channel matching the rec_stream_id. This is loaded only if a
        self.channel = None

        # The catalog id to which the detection belongs.
        self.catalog_id = catalog_id

        # The start time of the event.
        self.start_time = utcdatetime.UTCDateTime(start_time)

        # The end time of the event.
        self.end_time = utcdatetime.UTCDateTime(end_time)

        # The stations of the Delaunay triangle.
        self.stations = stations

        # The max. PGV values of the detection timespan.
        self.max_pgv = max_pgv

        # The agency_uri of the creator.
        self.agency_uri = agency_uri

        # The author_uri of the creator.
        self.author_uri = author_uri

        # The time of creation of this event.
        if creation_time is None:
            creation_time = utcdatetime.UTCDateTime()
        self.creation_time = utcdatetime.UTCDateTime(creation_time)

        # Flag to indicate a change of the detection attributes.
        self.changed = changed

    @property
    def rid(self):
        ''' The resource ID of the detection.
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
        ''' The length of the detection in seconds.
        '''
        return self.end_time - self.start_time

    @property
    def scnl(self):
        ''' The SCNL code of the related channel.
        '''
        if self.channel is None:
            return None
        else:
            return self.channel.scnl


    @property
    def snl(self):
        ''' The SCNL code of the related channel.
        '''
        if self.channel is None:
            return None
        else:
            return (self.channel.scnl[0], self.channel.scnl[2], self.channel.scnl[3])


    def update(self, start_time = None, end_time = None,
               max_pgv = None):
        ''' Update the attributes of the detection.
        '''
        if start_time is not None:
            self.start_time = utcdatetime.UTCDateTime(start_time)

        if end_time is not None:
            self.end_time = utcdatetime.UTCDateTime(end_time)

        if max_pgv is not None:
            for cur_key, cur_value in max_pgv.items():
                if cur_value > self.max_pgv[cur_key]:
                    self.max_pgv[cur_key] = cur_value



    def set_channel_from_inventory(self, inventory):
        ''' Set the channel matching the recorder stream.
        '''
        self.channel = inventory.get_channel_from_stream(start_time = self.start_time,
                                                         end_time = self.end_time)


    def write_to_database(self, project):
        ''' Write the detection to the pSysmon database.
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

            db_session = project.get_db_session()
            db_detection_orm = project.db_tables['detection']
            db_detection = db_detection_orm(catalog_id = catalog_id,
                                            start_time = self.start_time.timestamp,
                                            end_time = self.end_time.timestamp,
                                            stat1_id = self.stations[0].id,
                                            stat2_id = self.stations[1].id,
                                            stat3_id = self.stations[2].id,
                                            max_pgv1 = self.max_pgv[self.stations[0].snl_string],
                                            max_pgv2 = self.max_pgv[self.stations[1].snl_string],
                                            max_pgv3 = self.max_pgv[self.stations[2].snl_string],
                                            agency_uri = self.agency_uri,
                                            author_uri = self.author_uri,
                                            creation_time = creation_time)
            db_session.add(db_detection)
            db_session.commit()
            self.db_id = db_detection.id
            db_session.close()

        else:
            # If the db_id is not None, update the existing event.
            db_session = project.get_db_session()
            db_detection_orm = project.db_tables['detection']
            query = db_session.query(db_detection_orm).filter(db_detection_orm.id == self.db_id)
            if db_session.query(query.exists()):
                db_detection = query.scalar()
                if self.parent is not None:
                    db_detection.catalog_id = self.parent.db_id
                else:
                    db_detection.catalog_id = None
                db_detection.rec_stream_id = self.rec_stream_id
                db_detection.start_time = self.start_time.timestamp
                db_detection.end_time = self.end_time.timestamp
                db_detection.method = self.method
                db_detection.agency_uri = self.agency_uri
                db_detection.author_uri = self.author_uri
                if self.creation_time is not None:
                    db_detection.creation_time = self.creation_time.isoformat()
                else:
                    db_detection.creation_time = None
                db_session.commit()
                db_session.close()
            else:
                raise RuntimeError("The detection with ID=%d was not found in the database.", self.db_id)

    def get_db_orm(self, project):
        ''' Get an orm representation to use it for bulk insertion into
        the database.
        '''
        db_detection_orm = project.db_tables['detection']

        if self.creation_time is not None:
            creation_time = self.creation_time.isoformat()
        else:
            creation_time = None

        if self.parent is not None:
            catalog_id = self.parent.db_id
        else:
            catalog_id = None

        labels = ['catalog_id', 'rec_stream_id',
                  'start_time', 'end_time',
                  'method', 'agency_uri',
                  'author_uri', 'creation_time']
        db_dict = dict(list(zip(labels,
                           (catalog_id,
                            self.rec_stream_id,
                            self.start_time.timestamp,
                            self.end_time.timestamp,
                            self.method,
                            self.agency_uri,
                            self.author_uri,
                            creation_time))))
        db_detection = db_detection_orm(**db_dict)
        db_detection.id = self.db_id
        return db_detection

    @classmethod
    def from_orm(cls, detection_orm, inventory):
        ''' Convert a database orm mapper detection to a detection.

        Parameters
        ----------
        detection_orm : SQLAlchemy ORM
            The ORM of the detection_orm database table.
        '''
        stat1 = inventory.get_station(id = detection_orm.stat1_id)[0]
        stat2 = inventory.get_station(id = detection_orm.stat2_id)[0]
        stat3 = inventory.get_station(id = detection_orm.stat3_id)[0]
        detection = cls(start_time = detection_orm.start_time,
                        end_time = detection_orm.end_time,
                        db_id = detection_orm.id,
                        catalog_id = detection_orm.catalog_id,
                        stations = [stat1, stat2, stat3],
                        max_pgv = {stat1.snl_string: detection_orm.max_pgv1,
                                   stat2.snl_string: detection_orm.max_pgv2,
                                   stat3.snl_string: detection_orm.max_pgv3},
                        agency_uri = detection_orm.agency_uri,
                        author_uri = detection_orm.author_uri,
                        creation_time = detection_orm.creation_time)
        return detection



class Catalog(object):
    ''' A detection catalog.
    '''

    def __init__(self, name, db_id = None, description = None, agency_uri = None,
            author_uri = None, creation_time = None, detections = None):
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

        # The detections of the catalog.
        if detections is None:
            self.detections = []
        else:
            self.events = detections


    def add_detections(self, detections):
        ''' Add one or more detections to the catalog.

        Parameters
        ----------
        detections : list of :class:`Detection`
            The detections to add to the catalog.
        '''
        # Check for potential duplicates.
        # TODO: add a compare method for the detection class.
        db_ids = [x.db_id for x in self.detections]
        detections = [x for x in detections if x.db_id is None or x.db_id not in db_ids]

        for cur_detection in detections:
            cur_detection.parent = self
        self.detections.extend(detections)


    def remove_detections(self, detections):
        ''' Remove the detections from the catalog.

        Parameters
        ----------
        detections : list of :class:`Detection`
            The detections to add to the catalog.
        '''
        for cur_detection in detections:
            if cur_detection in self.detections:
                self.detections.remove(cur_detection)


    def get_detections(self, start_time = None, end_time = None,
                       start_inside = False, end_inside = False, **kwargs):
        ''' Get detections using search criteria passed as keywords.

        Parameters
        ----------
        start_time : class:`~obspy.core.utcdatetime.UTCDateTime`
            The minimum starttime of the detections.

        end_time : class:`~obspy.core.utcdatetime.UTCDateTime`
            The maximum end_time of the detections.

        start_inside : Boolean
            If True, select only those detection with a start time
            inside the search window.

        end_inside : Boolean
            If True, select only those detection with an end time
            inside the search window.

        scnl : tuple of Strings
            The scnl code of the channel (e.g. ('GILA, 'HHZ', 'XX', '00')).
        '''
        ret_detections = self.detections

        valid_keys = ['scnl']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_detections = [x for x in ret_detections if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        if start_time is not None:
            if start_inside:
                ret_detections = [x for x in ret_detections if (x.end_time is None) or (x.start_time >= start_time)]
            else:
                ret_detections = [x for x in ret_detections if (x.end_time is None) or (x.end_time > start_time)]

        if end_time is not None:
            if end_inside:
                ret_detections = [x for x in ret_detections if x.end_time <= end_time]
            else:
                ret_detections = [x for x in ret_detections if x.start_time < end_time]

        return ret_detections


    def assign_channel(self, inventory):
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
    def load_detections(self, project, start_time = None, end_time = None,
                        min_detection_length = None):
        ''' Load detections from the database.

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

        db_session = project.get_db_session()
        try:
            detection_table = project.db_tables['detection']
            query = db_session.query(detection_table).\
                    filter(detection_table.catalog_id == self.db_id).\
                    filter(detection_table.end_time > detection_table.start_time)

            if start_time:
                query = query.filter(detection_table.start_time >= start_time.timestamp)

            if end_time:
                query = query.filter(detection_table.start_time <= end_time.timestamp)

            if min_detection_length:
                query = query.filter(detection_table.end_time - detection_table.start_time >= min_detection_length)

            detections_to_add = []
            for cur_orm in query:
                try:
                    cur_detection = Detection.from_orm(cur_orm)
                    detections_to_add.append(cur_detection)
                except:
                    self.logger.exception("Error when creating a detection object from database values for detection id %d. Skipping this detection.", cur_orm.id)
            self.add_detections(detections_to_add)

        finally:
            db_session.close()


    def clear_detections(self):
        ''' Clear the detections list.
        '''
        self.detections = []


    def write_to_database(self, project):
        ''' Write the catalog to the database.

        '''
        if self.db_id is None:
            # If the db_id is None, insert a new catalog.
            if self.creation_time is not None:
                creation_time = self.creation_time.isoformat()
            else:
                creation_time = None

            db_session = project.get_db_session()
            db_catalog_orm = project.db_tables['detection_catalog']
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
            # If the db_id is not None, update the existing catalog.
            db_session = project.get_db_session()
            db_catalog_orm = project.db_tables['detection_catalog']
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
                raise RuntimeError("The detection catalog with ID=%d was not found in the database.", self.db_id)


        # Write or update all detections of the catalog to the database.
        for cur_detection in [x for x in self.detections if x.changed is True]:
            cur_detection.write_to_database(project)



    @classmethod
    def from_orm(cls, db_catalog, load_detections = False):
        ''' Convert a database orm mapper catalog to a catalog.

        Parameters
        ----------
        db_catalog : SQLAlchemy ORM
            The ORM of the events catalog database table.

        load_detections : Boolean
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

        # Add the detections to the catalog.
        if load_detections is True:
            for cur_detection_orm in db_catalog.detections:
                cur_detection = Detection.from_orm(cur_detection_orm)
                catalog.add_detections([cur_detection,])
        return catalog



class Library(object):
    ''' Manage detection catalogs.
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
        db_session = project.get_db_session()
        try:
            db_catalog_orm = project.db_tables['detection_catalog']
            query = db_session.query(db_catalog_orm)
            if db_session.query(query.exists()):
                catalog_names = [x.name for x in query.order_by(db_catalog_orm.name)]
        finally:
            db_session.close()

        return catalog_names


    def load_catalog_from_db(self, project, name, load_detections = False):
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

        db_session = project.get_db_session()
        try:
            db_catalog_orm = project.db_tables['detection_catalog']
            query = db_session.query(db_catalog_orm).filter(db_catalog_orm.name.in_(name))
            if db_session.query(query.exists()):
                for cur_db_catalog in query:
                    cur_catalog = Catalog.from_orm(cur_db_catalog, load_detections)
                    self.add_catalog(cur_catalog)
        finally:
            db_session.close()


