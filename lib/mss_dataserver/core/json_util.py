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

import numpy as np
import obspy

import mss_dataserver.core.util as util
import mss_dataserver.geometry as geom


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

    def __init__(self, data, agency_uri = None, author_uri = None):
        self.data = data
        self.agency_uri = agency_uri
        self.author_uri = author_uri


class GeneralFileEncoder(json.JSONEncoder):
    ''' A JSON encoder for the serialization of general data.
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
        ''' Convert pSysmon project objects to a dictionary.
        '''
        obj_class = obj.__class__.__name__
        base_class = [x.__name__ for x in obj.__class__.__bases__]

        #self.logger.debug('obj_class: %s.', obj_class)
        if obj_class == 'FileContainer':
            d = self.convert_filecontainer(obj)
        elif obj_class == 'Version':
            d = self.convert_version(obj)
        elif obj_class == 'UTCDateTime':
            d = self.convert_utcdatetime(obj)
        elif obj_class == 'ndarray':
            d = self.convert_np_ndarray(obj)
        elif obj_class == 'type':
            d = {}
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

        self.logger.debug('d: %s', d)
        return d


    def convert_filecontainer(self, obj):
        d = obj.data
        file_meta = {'file_version': self.version,
                     'save_date': obspy.UTCDateTime(),
                     'agency_uri': obj.agency_uri,
                     'author_uri': obj.author_uri}
        d['file_meta'] = file_meta
        return d


    def convert_utcdatetime(self, obj):
        return {'utcdatetime': obj.isoformat()}


    def convert_version(self, obj):
        return {'version': str(obj)}


    def convert_np_ndarray(self, obj):
        return {'data': obj.tolist()}



class GeneralFileDecoder(json.JSONDecoder):
    version = util.Version('1.0.0')

    def __init__(self, **kwarg):
        json.JSONDecoder.__init__(self, object_hook = self.convert_dict)


    def convert_dict(self, d):

        if '__class__' in d:
            class_name = d.pop('__class__')
            module_name = d.pop('__module__')
            base_class = d.pop('__baseclass__')

            if class_name == 'Version':
                inst = self.convert_version(d)
            elif class_name == 'UTCDateTime':
                inst = self.convert_utcdatetime(d)
            elif class_name == 'ndarray':
                inst = self.convert_np_array(d)
            else:
                inst = {'ERROR': 'MISSING CONVERTER'}

        else:
            inst = d

        return inst


    def decode_hinted_tuple(self, item):
        if isinstance(item, dict):
            if '__tuple__' in item:
                return tuple(item['items'])
        elif isinstance(item, list):
                return [self.decode_hinted_tuple(x) for x in item]
        else:
            return item


    def convert_version(self, d):
        inst = util.Version(d['version'])
        return inst


    def convert_utcdatetime(self, d):
        inst = obspy.UTCDateTime(d['utcdatetime'])
        return inst


    def convert_np_array(self, d):
        inst = np.array(d['data'])
        return inst



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
            d = self.convert_np_ndarray(obj)
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
                     'save_date': obspy.UTCDateTime(),
                     'agency_uri': obj.agency_uri,
                     'author_uri': obj.author_uri}
        d['file_meta'] = file_meta
        return d

    def convert_utcdatetime(self, obj):
        return {'utcdatetime': obj.isoformat()}

    def convert_version(self, obj):
        return {'version': str(obj)}

    def convert_np_ndarray(self, obj):
        return {'data': obj.tolist()}

    def convert_station(self, obj):
        attr = ['name', 'location', 'network',
                'x', 'y', 'z', 'coord_system',
                'description', 'author_uri', 'agency_uri',
                'creation_time']
        d = object_to_dict(obj, attr)
        return d


class SupplementDetectionDataDecoder(json.JSONDecoder):
    version = util.Version('1.0.0')

    def __init__(self, **kwarg):
        json.JSONDecoder.__init__(self, object_hook = self.convert_dict)


    def convert_dict(self, d):
        #print "Converting dict: %s." % str(d)

        if '__class__' in d:
            class_name = d.pop('__class__')
            module_name = d.pop('__module__')
            base_class = d.pop('__baseclass__')

            if class_name == 'Version':
                inst = self.convert_version(d)
            elif class_name == 'UTCDateTime':
                inst = self.convert_utcdatetime(d)
            elif class_name == 'ndarray':
                inst = self.convert_np_array(d)
            elif 'Station' in base_class:
                inst = self.convert_station(d)
            else:
                inst = {'ERROR': 'MISSING CONVERTER'}

        else:
            inst = d

        return inst


    def decode_hinted_tuple(self, item):
        if isinstance(item, dict):
            if '__tuple__' in item:
                return tuple(item['items'])
        elif isinstance(item, list):
                return [self.decode_hinted_tuple(x) for x in item]
        else:
            return item


    def convert_version(self, d):
        inst = util.Version(d['version'])
        return inst


    def convert_utcdatetime(self, d):
        inst = obspy.UTCDateTime(d['utcdatetime'])
        return inst


    def convert_np_array(self, d):
        inst = np.array(d['data'])
        return inst


    def convert_station(self, d):
        inst = geom.inventory.Station(name = d['name'],
                                      location = d['location'],
                                      x = d['x'],
                                      y = d['y'],
                                      z = d['z'],
                                      coord_system = d['coord_system'],
                                      description = d['description'],
                                      author_uri = d['author_uri'],
                                      agency_uri = d['agency_uri'],
                                      creation_time = d['creation_time'])
        return inst



