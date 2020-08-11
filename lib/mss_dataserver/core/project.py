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

import configparser
import json
import logging
import os

import sqlalchemy
import sqlalchemy.ext.declarative
import sqlalchemy.orm

import mss_dataserver.event as event
import mss_dataserver.geometry as geometry
import mss_dataserver.core.database_util as db_util
import mss_dataserver.geometry.inventory_parser as inventory_parser
import mss_dataserver.geometry.db_inventory as database_inventory


class Project(object):
    ''' A project holds global configuration and settings.
    '''

    def __init__(self, **kwargs):
        ''' Initialize the instance.
        '''
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        self.project_config = kwargs['project']
        self.author_uri = self.project_config['author_uri']
        self.agency_uri = self.project_config['agency_uri']

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

        # The database connection state.
        self.db_engine = None
        self.db_metadata = None
        self.db_base = None
        self.db_session_class = None

        # A dictionary of the project database tables.
        self.db_tables = {}

        # The geometry inventory.
        self.db_inventory = None


    @property
    def inventory(self):
        ''' The geometry inventory.
        '''
        return self.db_inventory

    @classmethod
    def load_configuration(cls, filename):
        ''' Load the configuration from a file.
        '''
        parser = configparser.ConfigParser()
        parser.read(filename)

        config = {}
        config['websocket'] = {}
        config['websocket']['host'] = parser.get('websocket', 'host').strip()
        config['websocket']['port'] = int(parser.get('websocket', 'port'))
        config['seedlink'] = {}
        config['seedlink']['host'] = parser.get('seedlink', 'host').strip()
        config['seedlink']['port'] = int(parser.get('seedlink', 'port'))
        config['output'] = {}
        config['output']['data_dir'] = parser.get('output', 'data_dir').strip()
        config['log'] = {}
        config['log']['loglevel'] = parser.get('log', 'loglevel').strip()
        config['project'] = {}
        config['project']['author_uri'] = parser.get('project', 'author_uri').strip()
        config['project']['agency_uri'] = parser.get('project', 'agency_uri').strip()
        config['project']['inventory_file'] = parser.get('project', 'inventory_file').strip()
        config['database'] = {}
        config['database']['host'] = parser.get('database', 'host').strip()
        config['database']['username'] = parser.get('database', 'username').strip()
        config['database']['password'] = parser.get('database', 'password').strip()
        config['database']['dialect'] = parser.get('database', 'dialect').strip()
        config['database']['driver'] = parser.get('database', 'driver').strip()
        config['database']['database_name'] = parser.get('database', 'database_name').strip()
        config['process'] = {}
        config['process']['stations'] = json.loads(parser.get('process', 'stations'))
        config['process']['interval'] = int(parser.get('process', 'interval'))
        config['process']['pgv_sps'] = int(parser.get('process', 'pgv_sps'))
        config['process']['trigger_threshold'] = float(parser.get('process', 'trigger_threshold'))
        config['process']['warn_threshold'] = float(parser.get('process', 'warn_threshold'))
        config['process']['valid_event_threshold'] = float(parser.get('process', 'valid_event_threshold'))
        config['process']['pgv_archive_time'] = int(parser.get('process', 'pgv_archive_time'))
        config['process']['event_archive_size'] = int(parser.get('process', 'event_archive_size'))

        return config


    def connect_to_db(self):
        ''' Connect to the database.
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
            self.db_base = sqlalchemy.ext.declarative.declarative_base(metadata = self.db_metadata)
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



    def create_database_tables(self):
        ''' Create the database tables needed for the project.
        '''
        for cur_key, cur_table in self.db_tables.items():
            self.logger.info("Creating table %s.", cur_key)
            db_util.db_table_migration(table = cur_table,
                                       engine = self.db_engine,
                                       prefix = 'dataserver_')
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


    def load_inventory(self, update_from_xml = False):
        ''' Load the geometry inventory from a XML file.
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

            self.logger.info("Updated the database inventory with data read from %s.", inventory_file)
