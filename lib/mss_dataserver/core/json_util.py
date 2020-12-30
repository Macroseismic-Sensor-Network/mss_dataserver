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

import json
import logging

import obspy

import mss_dataserver.core.util as util


def object_to_dict(obj, attr):
    ''' Copy selceted attributes of object to a dictionary.
    '''
    def hint_tuples(item):
        if isinstance(item, tuple):
            return {'__tuple__': True, 'items': item}
        if isinstance(item, list):
            return [hint_tuples(e) for e in item]
        else:
            return item

    d = {}
    for cur_attr in attr:
        d[cur_attr] = hint_tuples(getattr(obj, cur_attr))

    return d


class FileContainer(object):

    def __init__(self, data = {}):
        self.data = data


class SupplementDetectionDataEncoder(json.JSONEncoder):
    ''' A JSON encoder for the event supplement detection data.
    '''

    version = util.Version('1.0.0')

    def __init__(self, **kwarg):
        json.JSONEncoder.__init__(self, **kwarg)

        # The logger.
        loggerName = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(loggerName)

        # File format settings.
        self.indent = 4
        self.sort_keys = True


    def default(self, obj):
        ''' Convert the detection data instances to dictionaries.
        '''
        obj_class = obj.__class__.__name__
        base_class = [x.__name__ for x in obj.__class__.__bases__]
        #print 'Converting %s' % obj_class

        if obj_class == 'FileContainer':
            d = self.convert_filecontainer(obj)
        elif obj_class == 'UTCDateTime':
            d = self.convert_utcdatetime(obj)
        elif obj_class == 'Version':
            d = self.convert_version(obj)
        elif obj_class == 'ndarray':
            d = self.convert_numpy_array(obj)
        elif 'Station' in base_class:
            d = self.convert_station(obj)
        else:
            d = {'ERROR': 'MISSING CONVERTER for obj_class {obj_class} with base_class {base_class}'.format(obj_class = str(obj_class),
                                                                                                            base_class = str(base_class))}

        # Add the class and module information to the dictionary.
        if obj_class != 'FileContainer':
            try:
                module = obj.__module__
            except Exception:
                module = obj.__class__.__module__
            tmp = {'__baseclass__': base_class,
                   '__class__': obj.__class__.__name__,
                   '__module__': module}
            d.update(tmp)

        return d

    def convert_filecontainer(self, obj):
        d = obj.data
        file_meta = {'file_version': self.version,
                     'save_date': obspy.UTCDateTime()}
        d['file_meta'] = file_meta
        return d

    def convert_utcdatetime(self, obj):
        return {'utcdatetime': obj.isoformat()}

    def convert_version(self, obj):
        return {'version': str(obj)}

    def convert_numpy_array(self, obj):
        return {'data': obj.tolist()}

    def convert_station(self, obj):
        attr = ['name', 'location', 'network']
        d = object_to_dict(obj, attr)
        return d
