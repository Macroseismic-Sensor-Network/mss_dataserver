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


def create_db_test_project():
    ''' Create a project with a database connection to the unit_test database.
    '''
    base_dir = os.path.dirname(os.path.abspath(mss_dataserver.__file__))
    config_file = os.path.join(base_dir, 'test', 'mss_dataserver_unittest.ini')
    config = mss_dataserver.core.project.Project.load_configuration(config_file)
    project = mss_dataserver.core.project.Project(**config)
