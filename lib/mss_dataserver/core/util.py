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
import logging.handlers
import os


def load_configuration(filename):
    ''' Load the configuration from a file.
    '''
    if not os.path.exists(filename):
        raise RuntimeError("The configuration filename {filename} doesn't exist.".format(filename = filename))
    parser = configparser.ConfigParser()
    parser.read(filename)

    config = {}
    config['config_filepath'] = filename
    config['websocket'] = {}
    config['websocket']['host'] = parser.get('websocket', 'host').strip()
    config['websocket']['port'] = int(parser.get('websocket', 'port'))
    config['seedlink'] = {}
    config['seedlink']['host'] = parser.get('seedlink', 'host').strip()
    config['seedlink']['port'] = int(parser.get('seedlink', 'port'))
    config['output'] = {}
    config['output']['data_dir'] = parser.get('output', 'data_dir').strip()
    config['output']['event_dir'] = parser.get('output', 'event_dir').strip()
    config['log'] = {}
    config['log']['log_dir'] = parser.get('log', 'log_dir').strip()
    config['log']['loglevel'] = parser.get('log', 'loglevel').strip()
    config['log']['max_bytes'] = int(parser.get('log', 'max_bytes'))
    config['log']['backup_count'] = int(parser.get('log', 'backup_count'))
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
    config['process']['event_archive_timespan'] = int(parser.get('process', 'event_archive_timespan'))
    config['postprocess'] = {}
    config['postprocess']['data_dir'] = parser.get('postprocess', 'data_dir').strip()
    config['postprocess']['map_dir'] = parser.get('postprocess', 'map_dir').strip()
    config['postprocess']['boundary_filename'] = parser.get('postprocess', 'boundary_filename').strip()
    config['postprocess']['station_amplification_filename'] = parser.get('postprocess', 'station_amplification_filename').strip()

    return config


def get_logger_rotating_file_handler(filename = None,
                                     log_level = 'INFO',
                                     max_bytes = 1000,
                                     backup_count = 3):
    if not filename:
        return

    ch = logging.handlers.RotatingFileHandler(filename = filename,
                                              maxBytes = max_bytes,
                                              backupCount = backup_count)
    ch.setLevel(log_level)
    formatter = logging.Formatter("#LOG# - %(asctime)s - %(process)d - %(levelname)s - %(name)s: %(message)s")
    ch.setFormatter(formatter)
    return ch


class Version(object):
    ''' A version String representation.
    '''

    def __init__(self, version = '0.0.1'):
        ''' Initialize the instance.

        Parameters
        ----------
        version:String
            The version as a point-seperated string.

        '''
        self.version = self.string_to_tuple(version)


    def __str__(self):
        ''' The string representation.
        '''
        return '.'.join([str(x) for x in self.version])


    def __eq__(self, c):
        ''' Test for equality.
        '''
        for k, cur_n in enumerate(self.version):
            if cur_n != c.version[k]:
                return False

        return True

    def __ne__(self, c):
        ''' Test for inequality.
        '''
        return not self.__eq__(c)


    def __gt__(self, c):
        ''' Test for greater than.
        '''
        for k, cur_n in enumerate(self.version):
            if cur_n > c.version[k]:
                return True
            elif cur_n != c.version[k]:
                return False

        return False


    def __lt__(self, c):
        ''' Test for less than.
        '''
        for k, cur_n in enumerate(self.version):
            if cur_n < c.version[k]:
                return True
            elif cur_n != c.version[k]:
                return False

        return False


    def __ge__(self, c):
        ''' Test for greater or equal.
        '''
        return self.__eq__(c) or self.__gt__(c)

    def __le__(self, c):
        ''' Test for less or equal.
        '''
        return self.__eq__(c) or self.__lt__(c)




    def string_to_tuple(self, vs):
        ''' Convert a version string to a tuple.
        '''
        nn = vs.split('.')
        for k,x in enumerate(nn):
            if x.isdigit():
                nn[k] = int(x)
            else:
                tmp = re.split('[A-Za-z]', x)
                tmp = [x for x in tmp if x.isdigit()]
                if len(tmp) > 0:
                    nn[k] = int(tmp[0])
                else:
                    nn[k] = 0

        return tuple(nn)

