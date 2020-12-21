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


def load_configuration(filename):
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
    config['output']['event_dir'] = parser.get('output', 'event_dir').strip()
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
