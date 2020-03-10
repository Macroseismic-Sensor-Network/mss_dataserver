
import logging
import os

import sqlalchemy
import sqlalchemy.ext.declarative
import sqlalchemy.orm

import mss_dataserver.event as event
import mss_dataserver.core.database_util as db_util
import mss_dataserver.geometry.inventory_parser as inventory_parser


class Project(object):
    ''' A project holds global configuration and settings.
    '''

    def __init__(self, **kwargs):
        ''' Initialize the instance.
        '''
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        project_config = kwargs['project']
        self.author_uri = project_config['author_uri']
        self.agency_uri = project_config['agency_uri']

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
        self.inventory = None

        # Load the database table definitions.
        try:
            self.connect_to_db()
        except Exception:
            logging.exception("Can't connect to the database.")

        if self.db_base is not None:
            self.load_database_table_structure()
        else:
            self.logger.error("The db_metadata is empty. There seems to be no connection to the database.")


    def connect_to_db(self):
        ''' Connect to the database.
        '''
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


    def load_database_table_structure(self):
        ''' Load the required database tables from the modules.
        '''
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


    def load_inventory(self):
        ''' Load the geometry inventory from a XML file.
        '''
        inventory_file = self.process_config['inventory_file']
        if not os.path.exists(inventory_file):
            self.logger.error("Can't find the inventory file %s.",
                              inventory_file)
            return None

        parser = inventory_parser.InventoryXmlParser()
        try:
            self.inventory = parser.parse(inventory_file)
        except Exception:
            self.logger.exception("Couldn't load the inventory from file %s.",
                                  inventory_file)
