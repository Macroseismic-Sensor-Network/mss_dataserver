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

''' MSS Dataserver Project.
'''

import configparser
import copy
import json
import logging
import os

import sqlalchemy
import sqlalchemy.orm

import mss_dataserver.event as event
import mss_dataserver.event.core
import mss_dataserver.geometry as geometry
import mss_dataserver.localize as localize
import mss_dataserver.core.database_util as db_util
import mss_dataserver.geometry.inventory_parser as inventory_parser
import mss_dataserver.geometry.db_inventory as database_inventory


class Project(object):
    ''' A project holds global configuration and settings.


    Parameters
    ----------
    kwargs: dict
        The dictionary created from the configuration file.


    Attributes
    ----------
    logger: logging.Logger
        The logger instance.

    project_config: dict
        The *project* configuration section.

    author_uri: String
        The Uniform Resource Identifier of the author.
  
    agency_uri: String
        The Uniform Resource Identifier of the author agency.

    config: dict
        The complete configuration dictionary (kwargs).

    process_config: dict
        The *process* configuration section.

    db_host: String
        The URL or IP of the database host.

    db_username: String
        The database user name.

    db_pwd: String
        The database password.

    db_dialect: String
        The dialect of the database.

    db_driver: String
        The driver of the database.

    db_database_name: String
        The name of the database.

    db_tables: list
        The tables loaded from the database.

    db_inventory: :class:`mss_dataserver.geometry.DbInventory`
        The geometry inventory of the project.

    inventory: :class:`mss_dataserer.geometry.DbInventory`
        A dynamic property returning db_inventory.

    event_library: :class:`mss_dataserver.event.core.Library`
        The event library of the project.

    detection_library: :class:`mss_dataserver.detection.Library`
        The detection library of the project.

    
    See Also
    --------
    :meth:`mss_dataserver.core.util.load_configuration`

    '''

    def __init__(self, **kwargs):
        ''' Initialize the instance.
        '''
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        self.project_config = kwargs['project']
        self.author_uri = self.project_config['author_uri']
        self.agency_uri = self.project_config['agency_uri']

        # The complete configuration content.
        self.config = kwargs

        # The processing configuration.
        self.process_config = kwargs['process']

        # The database configuration.
        db_config = kwargs['database']
        self.db_host = db_config['host']
        self.db_username = db_config['username']
        self.db_pwd = db_config['password']
        self.db_dialect = db_config['dialect']
        self.db_driver = db_config['driver']
        self.db_database_name = db_config['database_name']

        # Check and create the output directories.
        output_dirs = [self.config['output']['data_dir'],
                       self.config['output']['event_dir']]
        for cur_dir in output_dirs:
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)

        # The database connection state.
        self.db_engine = None
        self.db_metadata = None
        self.db_base = None
        self.db_session_class = None

        # A dictionary of the project database tables.
        self.db_tables = {}

        # The geometry inventory.
        self.db_inventory = None

        # The geometry inventory of third party stations.
        self.tp_inventory = None

        # The events library.
        self.event_library = event.core.Library(name = 'mss events')

        # The detections library.
        self.detection_library = event.detection.Library(name = 'mss detections')


    @property
    def inventory(self):
        ''' The geometry inventory.
        '''
        return self.db_inventory

    @property
    def is_connected_to_db(self):
        ''' Flag indicating if a valid database connection exists.
        '''
        is_connected = False
        if self.db_base is not None:
            is_connected = True

        return is_connected


    def connect_to_db(self):
        ''' Connect to the database.

        Connect to the database using the parameters specified in 
        the configuration file.
        '''
        try:
            if self.db_driver is not None:
                dialect_string = self.db_dialect + "+" + self.db_driver
            else:
                dialect_string = self.db_dialect

            if self.db_pwd is not None:
                engine_string = dialect_string + "://" + self.db_username + ":" + self.db_pwd + "@" + self.db_host + "/" + self.db_database_name
            else:
                engine_string = dialect_string + "://" + self.db_username + "@" + self.db_host + "/" + self.db_database_name

            engine_string = engine_string + "?charset=utf8"

            self.db_engine = sqlalchemy.create_engine(engine_string)
            self.db_engine.echo = False
            self.db_metadata = sqlalchemy.MetaData(self.db_engine)
            #self.db_base = sqlalchemy.ext.declarative.declarative_base(metadata = self.db_metadata)
            self.db_base = sqlalchemy.orm.declarative_base(metadata = self.db_metadata)
            self.db_session_class = sqlalchemy.orm.sessionmaker(bind = self.db_engine)
        except Exception:
            logging.exception("Can't connect to the database.")

        if self.db_base is not None:
            self.load_database_table_structure()
        else:
            self.logger.error("The db_metadata is empty. There seems to be no connection to the database.")


    def load_database_table_structure(self):
        ''' Load the required database tables from the modules.
        '''
        geom_tables = geometry.databaseFactory(self.db_base)
        for cur_table in geom_tables:
            cur_name = cur_table.__table__.name
            self.db_tables[cur_name] = cur_table

        event_tables = event.databaseFactory(self.db_base)
        for cur_table in event_tables:
            cur_name = cur_table.__table__.name
            self.db_tables[cur_name] = cur_table

        loc_tables = localize.databaseFactory(self.db_base)
        for cur_table in loc_tables:
            cur_name = cur_table.__table__.name
            self.db_tables[cur_name] = cur_table



    def create_database_tables(self):
        ''' Create the database tables needed for the project.
        '''
        for cur_key, cur_table in self.db_tables.items():
            self.logger.info("Creating table %s.", cur_key)
            db_util.db_table_migration(table = cur_table,
                                       engine = self.db_engine,
                                       prefix = '')
        try:
            if self.db_metadata is not None:
                self.db_metadata.create_all()
            else:
                self.logger.error("The db_metadata is empty. There seems to be no connection to the database.")
        except Exception:
            self.logger.exception("Error creating the database tables.")


    def get_db_session(self):
        ''' Create a sqlAlchemy database session.

        Returns
        -------
        session : :class:`orm.session.Session`
            A sqlAlchemy database session.
        '''
        return self.db_session_class()


    def split_inventory(self):
        ''' Create the third party inventory.

        '''
        # TODO: Make this an entry in the config file.
        ignore_stations = ['MSSNet:DUBAM:00']

        # Create the third party inventory.
        self.tp_inventory = geometry.inventory.Inventory(name = 'third party')

        for cur_stat_nsl in ignore_stations:
            cur_stat_list = self.inventory.get_station(nsl_string = cur_stat_nsl)

            for cur_stat in cur_stat_list:
                cur_net_name = cur_stat.network
                cur_net = self.tp_inventory.get_network(name = cur_net_name)
                if len(cur_net) == 1:
                    cur_net = cur_net[0]
                else:
                    cur_net = geometry.inventory.Network(name = cur_net_name)
                    self.tp_inventory.add_network(cur_net)
                cur_stat.parent_network.remove_station_by_instance(cur_stat)
                cur_net.add_station(cur_stat)
            
        
    def load_inventory(self, update_from_xml = False):
        ''' Load the geometry inventory.

        Load the geometry inventory from the database. 
        If specified, read the inventory XML file specified in the 
        project configuration and update the database with the loaded
        inventory.

        Parameters
        ----------
        update_from_xml: bool
            If True, the database is updated with the data loaded from 
            the inventory XML file.
        '''
        # Load the existing inventory from the database.
        try:
            self.db_inventory = database_inventory.DbInventory(project = self)
            self.db_inventory.load()
        except Exception:
            self.logger.exception("Error while loading the database inventory.")

        if update_from_xml:
            # Read the inventory from the XML file.
            inventory_file = self.project_config['inventory_file']
            if not os.path.exists(inventory_file):
                self.logger.error("Can't find the inventory file %s.",
                                  inventory_file)
                return None

            parser = inventory_parser.InventoryXmlParser()
            try:
                xml_inventory = parser.parse(inventory_file)
            except Exception:
                self.logger.exception("Couldn't load the inventory from file %s.",
                                      inventory_file)

            # Update the database inventory with the loaded XML inventory.
            self.db_inventory.merge(xml_inventory)
            self.db_inventory.commit()

            # Reload to get all ORM objects.
            # TODO: Check how to get the ORM objects without reloading the
            # inventory.
            self.db_inventory = database_inventory.DbInventory(project = self)
            self.db_inventory.load()

            self.logger.info("Updated the database inventory with data read from %s.", inventory_file)

        self.split_inventory()
        

    def load_inventory_from_xml(self):
        ''' Load the inventory directly from the XML file ignoring the database.
        
        This function can be used when a database connction is not available,
        but the inventory information is needed, e.g. when postprocessing 
        events.
        '''
        # Read the inventory from the XML file.
        inventory_file = self.project_config['inventory_file']
        if not os.path.exists(inventory_file):
            self.logger.error("Can't find the inventory file %s.",
                              inventory_file)
            return None

        parser = inventory_parser.InventoryXmlParser()
        try:
            xml_inventory = parser.parse(inventory_file)
        except Exception:
            self.logger.exception("Couldn't load the inventory from file %s.",
                                  inventory_file)

        self.db_inventory = xml_inventory
        self.split_inventory()


    def get_event_catalog(self, name):
        ''' Get an event catalog.

        If the catalog doesn't exist, a new catalog is created.

        Parameters
        ----------
        name: String
            The name of the catalog to get.

        Returns
        -------
        cur_cat: :class:`mss_dataserver.event.core.Catalog`
            The catalog with name *name*.
        '''
        if name in self.event_library.catalogs.keys():
            self.logger.info("Using an already existing catalog in the library.")
            cur_cat = self.event_library.catalogs[name]
        else:
            ev_catalogs = self.event_library.get_catalogs_in_db(project = self)
            if name not in ev_catalogs:
                self.logger.info("Creating a new event catalog.")
                cur_cat = self.create_event_catalog(name = name)
            else:
                self.logger.info("Loading the event catalog from database.")
                cur_cat = self.load_event_catalog(name = name,
                                                  load_events = True)
        return cur_cat

    
    def get_event_catalog_names(self):
        ''' Get the event catalog names available in the database.

        Returns
        -------
        cat_names: list of Strings
            The names of the event catalogs available in the database.
        '''
        return self.event_library.get_catalogs_in_db(project = self)

    
    def create_event_catalog(self, name, description = ''):
        ''' Create an event catalog in the database.

        Parameters
        ----------
        name: String
            The name of the catalog.

        description: String
            The description of the catalog.

        Returns
        -------
        cat: :class:`mss_dataserver.event.core.Catalog`
            The created catalog.
        '''
        cat = event.core.Catalog(name = name,
                                 description = description,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri)
        cat.write_to_database(self)
        self.event_library.add_catalog(cat)
        return cat

    
    def load_event_catalog(self, name, load_events = False):
        ''' Load an event catalog from the database.

        Parameters
        ----------
        name: String
            The name of the catalog to load.

        load_events: bool
            Load the events from the database.

        Returns
        -------
        cat: :class:`mss_dataserver.event.core.Catalog`
            The loaded catalog.
        '''
        self.event_library.load_catalog_from_db(project = self,
                                                name = name,
                                                load_events = load_events)
        cat = None
        if name in self.event_library.catalogs.keys():
            cat = self.event_library.catalogs[name]

        return cat


    def load_event_by_id(self, ev_id = None, public_id = None):
        ''' Get an event by event id or public id.

        Parameters
        ----------
        ev_id: Integer
            The database id of the event to load.

        public_id: String
            The public id of the event to load.

        Returns
        -------
        event: :class:`mss_dataserver.event.core.Event`
            The loaded event.
        '''
        event = self.event_library.load_event_by_id(project = self,
                                                    ev_id = ev_id,
                                                    public_id = public_id)
        return event


    def get_events(self, catalog_names = None,
                   start_time = None, end_time = None, **kwargs):
        ''' Get events using search criteria passed as keywords.

        Parameters
        ----------
        catalog_names: List of String
            The catalog names to load.

        start_time : :class:`~obspy.core.utcdatetime.UTCDateTime`
            The minimum starttime of the detections.

        end_time : :class:`~obspy.core.utcdatetime.UTCDateTime`
            The maximum end_time of the detections.

        nslc : tuple of Strings
            The NSLC (network, station, location, channel)code
            of the channel (e.g. ('XX', 'GILA', '00', 'HHZ')).

        kwargs: Keyword arguments
            Additional keyword arguments passed to :meth:`mss_dataserver.event.core.Library.get_events`.

        Returns
        -------
        ret_events: List of :class:`mss_dataserver.event.core.Event`
            The events found in the library matching the search criterias.
        '''
        ret_events = self.event_library.get_events(catalog_names = catalog_names,
                                                   start_time = start_time,
                                                   end_time = end_time,
                                                   **kwargs)
        return ret_events

    
    def get_detection_catalog_names(self):
        ''' Get the detection catalog names available in the database.

        Returns
        -------
        cat_names: List of String
            The names of the detection catalogs available in the database.
        '''
        return self.detection_library.get_catalogs_in_db(project = self)

    
    def create_detection_catalog(self, name, description = ''):
        ''' Create an detection catalog in the database.

        Parameters
        ----------
        name: String
            The name of the catalog.

        description: String
            The description of the catalog.

        Returns
        -------
        cat: :class:`mss_dataserver.detection.Catalog`
            The created catalog.     
        '''
        cat = event.detection.Catalog(name = name,
                                      description = description,
                                      agency_uri = self.agency_uri,
                                      author_uri = self.author_uri)
        cat.write_to_database(self)
        return cat

    
    def load_detection_catalog(self, name, load_detections = False):
        ''' Load a detection catalog from the database.

        Parameters
        ----------
        name: String
            The name of the detection catalog.

        load_detections: bool
            Load the detections from the database.

        Returns
        -------
        cat: :class:`mss_dataserver.detection.Library.Catalog`
            The loaded detection catalog.
        '''
        self.detection_library.load_catalog_from_db(project = self,
                                                    name = name,
                                                    load_detections = load_detections)
        cat = None
        if name in self.detection_library.catalogs.keys():
            cat = self.detection_library.catalogs[name]

        return cat
