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

import os

import mss_dataserver
import mss_dataserver.core.project


def clear_database_tables(db_dialect, db_driver, db_user,
                          db_pwd, db_host, db_name):
    from sqlalchemy import create_engine, MetaData

    if db_driver is not None:
        dialect_string = db_dialect + "+" + db_driver
    else:
        dialect_string = db_dialect

    if db_pwd is not None:
        engine_string = dialect_string + "://" + db_user + ":" + db_pwd + "@" + db_host + "/" + db_name
    else:
        engine_string = dialect_string + "://" + db_user + "@" + db_host + "/" + db_name

    db_engine = create_engine(engine_string)
    db_engine.echo = True
    db_metadata = MetaData(db_engine)

    db_metadata.reflect(db_engine)
    tables_to_clear = [table for table in reversed(db_metadata.sorted_tables)]
    for cur_table in tables_to_clear:
        db_engine.execute(cur_table.delete())


def drop_database_tables(db_dialect, db_driver, db_user,
                         db_pwd, db_host, db_name):
    from sqlalchemy import create_engine, MetaData

    if db_driver is not None:
        dialect_string = db_dialect + "+" + db_driver
    else:
        dialect_string = db_dialect

    if db_pwd is not None:
        engine_string = dialect_string + "://" + db_user + ":" + db_pwd + "@" + db_host + "/" + db_name
    else:
        engine_string = dialect_string + "://" + db_user + "@" + db_host + "/" + db_name

    db_engine = create_engine(engine_string)
    db_engine.echo = True
    db_metadata = MetaData(db_engine)

    db_metadata.reflect(db_engine)
    tables_to_drop = [table for table in reversed(db_metadata.sorted_tables)]
    db_metadata.drop_all(tables = tables_to_drop)


def drop_project_database_tables(project):
    project.connect_to_db()
    project.db_metadata.reflect(project.db_engine)
    tables_to_remove = [table for key, table in list(project.db_metadata.tables.items())]
    project.db_metadata.drop_all(tables = tables_to_remove)


def clear_project_database_tables(project, tables = None):
    project.connect_to_db()
    project.db_metadata.reflect(project.db_engine)
    tables_to_clear = [table for table in reversed(project.db_metadata.sorted_tables)]

    if tables is not None:
        tables_to_clear = [table for table in tables_to_clear if str(table) in tables]
    for cur_table in tables_to_clear:
        project.db_engine.execute(cur_table.delete())
        project.db_engine.execute('alter table {0:s} AUTO_INCREMENT = 1'.format(str(cur_table)))


def create_db_test_project():
    ''' Create a project with a database connection to the unit_test database.
    '''
    base_dir = os.path.dirname(os.path.abspath(mss_dataserver.__file__))
    config_file = os.path.join(base_dir, 'test', 'config', 'mss_dataserver_unittest.ini')
    config = mss_dataserver.core.project.Project.load_configuration(config_file)
    # Update the configuration filepaths.
    config['project']['inventory_file'] = os.path.join(base_dir, 'test', 'data', config['project']['inventory_file'])
    config['output']['data_dir'] = os.path.join(base_dir, 'test', 'output', config['output']['data_dir'])
    project = mss_dataserver.core.project.Project(**config)
    return project
