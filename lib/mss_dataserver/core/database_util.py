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

'''
Utility functions to interact with the database.

:copyright:
    Stefan Mertl

:license:
    GNU General Public License, Version 3 
    (http://www.gnu.org/licenses/gpl-3.0.html)
'''

import sqlalchemy as sqa
import logging
import re

logger_name = __name__
logger = logging.getLogger(logger_name)

def db_table_migration(engine, table, prefix):
    ''' Check if a database table migration is needed and apply the changes.
    '''
    logger.info('Checking if database table %s needs an update.', table.__table__.name)
    migrate_success = False
    cur_metadata = sqa.MetaData(engine)
    cur_metadata.reflect(engine)
    if table.__table__.name in iter(cur_metadata.tables.keys()):
        # Check for changes between the existing and the new table.
        table_updated = update_db_table(engine = engine,
                                        table = table,
                                        metadata = cur_metadata,
                                        prefix = prefix)
        if not table_updated:
            logger.info('Everything is up-to-date. No update needed.')

        migrate_success = True
    else:
        # The table is missing in the schema, create it.
        logger.info('The table %s is not existing, create it.', table.__table__.name)
        try:
            table.__table__.create()
            migrate_success = True
        except:
            logger.exception('Error creating the table %s.', table.__table__.name)

    return migrate_success


def update_db_table(engine, table, metadata, prefix):
    ''' Update the table structure to the new schema.
    '''
    table_updated = False
    # Check for added columns.
    new_table = table.__table__
    exist_table = metadata.tables[new_table.name]
    columns_to_add = set(new_table.columns.keys()).difference(set(exist_table.columns.keys()))
    if columns_to_add:
        if not table_updated:
            logger.info('A database table migration is needed.')

        for cur_col in columns_to_add:
            logger.info('Adding column %s to table %s.', cur_col, table.__table__.name)
            add_column(engine = engine,
                       table = table,
                       column = new_table.columns[cur_col],
                       prefix = prefix)

        table_updated = True

    # Check for columns to remove.
    columns_to_remove = set(exist_table.columns.keys()).difference(set(new_table.columns.keys()))
    if columns_to_remove:
        if not table_updated:
            logger.info('A database table migration is needed.')

        for cur_col in columns_to_remove:
            logger.info('Removing column %s from table %s.', cur_col, table.__table__.name)
            remove_column(engine = engine,
                          table = table,
                          column = exist_table.columns[cur_col])

        table_updated = True

    # Check for changed column specifications.
    for cur_name, cur_col in list(new_table.columns.items()):
        if cur_name not in iter(exist_table.columns.keys()):
            # The column is not existing in the database. 
            # Might have been deleted. Ignore it.
            continue
        exist_col = exist_table.columns[cur_name]
        # Check for the column type.
        new_type = cur_col.type.compile(engine.dialect)
        exist_type = exist_col.type.compile(engine.dialect)
        if not compare_column_type(new_type, exist_type):
            if not table_updated:
                logger.info('A database table migration is needed.')
            logger.info('Changing the type of column %s from %s to %s.', cur_name, exist_type, new_type)
            change_column_type(engine = engine,
                               table = table,
                               column = cur_col)
            table_updated = True

        # Drop all existing foreign keys. This has to be done in case that unique
        # keys have to be removed which might be needed by foreign keys.
        for cur_key in exist_col.foreign_keys:
            logger.info("Removing the foreign key %s from table %s.", cur_key.name, table.__table__.name)
            remove_foreign_key(engine = engine,
                               table = table,
                               fk_symbol = cur_key.name)


    # Remove all unique contraints.
    insp = sqa.inspect(engine)
    unique_const = insp.get_unique_constraints(exist_table.name)
    for cur_const in unique_const:
        logger.info('Removing the unique constraint %s.', cur_const['name'])
        remove_unique_constraint(engine, table, cur_const['name'])


    # Add the new constraints to the table.
    const_to_add = new_table.constraints.difference(exist_table.constraints)
    unique_const = [x for x in const_to_add if isinstance(x, sqa.schema.UniqueConstraint)]
    foreign_const = [x for x in const_to_add if isinstance(x, sqa.schema.ForeignKeyConstraint)]
    # TODO: Handle changes of the primary key.
    primary_const = [x for x in const_to_add if isinstance(x, sqa.schema.PrimaryKeyConstraint)]

    for cur_const in unique_const:
        logger.info('Adding the unique constraint %s.', cur_const.name)
        add_unique_constraint(engine, table, cur_const)
        table_updated = True

    for cur_const in foreign_const:
        #logger.warning("Changing foreign key constraint is not yet implemented (%s).", cur_const)
        target_columns = [x.target_fullname for x in cur_const.elements]
        logger.info('Adding foreign key (%s) referring (%s).', ','.join(cur_const.column_keys),
                    ','.join(target_columns))
        try:
            # Get the target table using the ForeignKey elements. Using the
            # referred_table was not working for all constraints because the
            # referred_table is a dynamic attribute.
            target_name = [x.target_fullname.split('.')[0] for x in cur_const.elements]
            target_name = list(set(target_name))
            if len(target_name) != 1:
                logger.error("Too many target tables for the foreign constraint: %s.", target_name)
                continue
            else:
                target_name = prefix + target_name[0]

            add_foreign_key(engine = engine,
                            table = table,
                            columns = cur_const.column_keys,
                            target_table = target_name,
                            target_columns = [x.target_fullname.split('.')[1] for x in cur_const.elements],
                            on_update = cur_const.onupdate,
                            on_delete = cur_const.ondelete)
        except:
            logger.exception("Error creating the foreign key.")


    return table_updated


def add_column(engine, table, column, prefix):
    ''' Add a column to a database table.
    '''
    table_name = table.__table__.name
    column_type = column.type.compile(engine.dialect)
    if column.nullable:
        engine.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table_name, column.name, column_type))
    else:
        engine.execute('ALTER TABLE %s ADD COLUMN %s %s NOT NULL' % (table_name, column.name, column_type))

    for cur_key in column.foreign_keys:
        add_foreign_key(engine = engine,
                        table = table,
                        column = column,
                        target = prefix + cur_key.target_fullname)

def remove_column(engine, table, column):
    ''' Remove a column from the database table.
    '''
    table_name = table.__table__.name
    for cur_key in column.foreign_keys:
        remove_foreign_key(engine, table, cur_key.name)
    engine.execute('ALTER TABLE %s DROP COLUMN %s' % (table_name, column.name))


def change_column_type(engine, table, column):
    ''' Change the type of a database table column.
    '''
    table_name = table.__table__.name
    column_type = column.type.compile(dialect = engine.dialect)
    engine.execute('ALTER TABLE %s MODIFY %s %s' % (table_name, column.name, column_type))


def add_foreign_key(engine, table, columns, target_table, target_columns, on_update, on_delete):
    ''' Add a foreign key constraint.
    '''
    table_name = table.__table__.name
    if on_update is None:
        on_update = 'RESTRICT'

    if on_delete is None:
        on_delete = 'RESTRICT'

    engine.execute('ALTER TABLE %s ADD FOREIGN KEY (%s) REFERENCES %s(%s) ON UPDATE %s ON DELETE %s' % (table_name,
                                                                              ','.join(columns),
                                                                              target_table,
                                                                              ','.join(target_columns),
                                                                              on_update,
                                                                              on_delete))


def add_foreign_key_old(engine, table, column, target, on_update, on_delete):
    ''' Add a foreign key to the column.
    '''
    table_name = table.__table__.name
    tmp = target.split('.')
    target_table = tmp[0]
    target_column = tmp[1]
    engine.execute('ALTER TABLE %s ADD FOREIGN KEY (%s) REFERENCES %s(%s) ON_DELETE %s ON_UPDATE %s' % (table_name, column.name, target_table, target_column, on_update, on_delete))


def remove_foreign_key(engine, table, fk_symbol):
    ''' Remove a foreign key.
    '''
    table_name = table.__table__.name
    engine.execute('ALTER TABLE %s DROP FOREIGN KEY %s' % (table_name, fk_symbol))


def add_unique_constraint(engine, table, constraint):
    ''' Add a unique constraint to the table.
    '''
    table_name = table.__table__.name
    col_names = [x.name for x in constraint.columns]
    const_name = constraint.name

    if const_name:
        sql_cmd = 'ALTER TABLE %s ADD CONSTRAINT %s UNIQUE (%s)' % (table_name, const_name, ','.join(col_names))
    else:
        sql_cmd = 'ALTER TABLE %s ADD CONSTRAINT UNIQUE (%s)' % (table_name, ','.join(col_names))

    engine.execute(sql_cmd)


def remove_unique_constraint(engine, table, name):
    ''' Remove a unique constraint from the table using the constraint name.
    '''
    table_name = table.__table__.name
    engine.execute('ALTER TABLE %s DROP INDEX %s' % (table_name, name))


def compare_column_type(col1, col2):
    ''' Compare the type of two columns.
    '''
    is_equal = False
    tmp = re.split('[()]', col1)
    if tmp:
        if len(tmp) >= 1:
            col1_type = tmp[0]

        if len(tmp) >= 2:
            col1_len = tmp[1]
        else:
            col1_len = None

    tmp = re.split('[()]', col2)
    if tmp:
        if len(tmp) >= 1:
            col2_type = tmp[0]

        if len(tmp) >= 2:
            col2_len = tmp[1]
        else:
            col2_len = None

    if col1_type == col2_type:
        if col1_len is not None:
            if col1_len == col2_len:
                is_equal = True
        else:
            is_equal = True

    return is_equal


