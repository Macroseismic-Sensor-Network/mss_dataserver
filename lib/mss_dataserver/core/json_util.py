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

''' Utilities for handling JSON import and export.

'''

import json
import logging

import numpy as np
import obspy

import mss_dataserver.core.util as util
import mss_dataserver.geometry as geom
import mss_dataserver.geometry.inventory


def object_to_dict(obj, attr):
    ''' Copy selected attributes of object to a dictionary.

    Parameters
    ----------
    obj: :class:`object`
        An instance of a python class.

    attr: list of String
        The attributes to copy to the dictionary.

    Returns
    -------
    d: dict
        A dictionary with the selected attributes.
    '''

    def hint_tuples(item):
        ''' Convert a tuple.

        JSON doesn't support tuples. Use a custom dictionary
        to encode a tuple.

        Parameters
        ----------
        item: object
            The instance to convert.

        Returns
        -------
        item: dict
            The dictionary representation of the instance item.
        '''
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
    ''' A container json data.

    Along with the data, the container holds information about
    the file creation.

    Parameters
    ----------
    data: dict
        The data to write to a JSON file.

    agency_uri: String
        The agency uniform resource identifier.

    author_uri: String
        The author uniform resource identifier
    '''

    def __init__(self, data, agency_uri = None, author_uri = None):
        self.data = data
        self.agency_uri = agency_uri
        self.author_uri = author_uri


class GeneralFileEncoder(json.JSONEncoder):
    ''' A JSON encoder for the serialization of general data.

    Parameters
    ----------
    **kwargs: keyword argument
        Keyword arguments passed to :class:`json.encoder.JSONEncoder`.

    Attributes
    ----------
    version: :class:`mss_dataserver.core.util`
        The version of the file encoder.

    logger: logging.Logger
        The logging instance.
    '''
    version = util.Version('1.0.0')

    def __init__(self, **kwarg):
        ''' Initialization of the instance.
        '''
        json.JSONEncoder.__init__(self, **kwarg)

        # The logger.
        loggerName = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(loggerName)

        # File format settings.
        self.indent = 4
        self.sort_keys = True


    def default(self, obj):
        ''' Convert objects to a dictionary.

        The instance class, module and base_class relations are stored in 
        the __class__, __module__ and __base_class__ keys. These are used 
        by the related file decoder to restore the correct class instances.

        Parameters
        ----------
        obj: object
            The instance to convert to a dictionary.

        Returns
        -------
        d: dict
           The dictionary representation of the instance obj.
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
        ''' Convert a filecontainer instance.

        Parameters
        ----------
        obj: FileContainer
            The filecontainer to convert.

        Returns
        -------
        d: dict
           The dictionary representation of the instance obj.
        '''
        d = obj.data
        file_meta = {'file_version': self.version,
                     'save_date': obspy.UTCDateTime(),
                     'agency_uri': obj.agency_uri,
                     'author_uri': obj.author_uri}
        d['file_meta'] = file_meta
        return d


    def convert_utcdatetime(self, obj):
        ''' Convert a UTCDateTime instance.

        Parameters
        ----------
        obj: obspy.utcdatetime.UTCDateTime
            The UTCDateTime instance to convert.

        Returns
        -------
        d: dict
            The dictionary representation of obj.
        '''
        return {'utcdatetime': obj.isoformat()}


    def convert_version(self, obj):
        ''' Convert a UTCDateTime instance.

        Parameters
        ----------
        obj: obspy.utcdatetime.UTCDateTime
            The UTCDateTime instance to convert.

        Returns
        -------
        d: dict
            The dictionary representation of obj.
        '''        ''' Convert a Version instance.

        Parameters
        ----------
        obj: mss_dataserver.core.util.Version
            The instance to convert.

        Returns
        -------
        d: dict
            The dictionary representation of obj.
        '''
        return {'version': str(obj)}


    def convert_np_ndarray(self, obj):
        ''' Convert a numpy array instance.

        Parameters
        ----------
        obj: numpy.ndarray
            The instance to convert.

        Returns
        -------
        d: dict
            The dictionary representation of obj.
        '''
        return {'data': obj.tolist()}



class GeneralFileDecoder(json.JSONDecoder):
    ''' A JSON decoder for the deserialization of general data.

    Parameters
    ----------
    **kwargs: keyword argument
        Keyword arguments passed to :class:`json.encoder.JSONDecoder`.

    Attributes
    ----------
    version: :class:`mss_dataserver.core.util`
        The version of the file decoder.
    '''

    version = util.Version('1.0.0')

    def __init__(self, **kwarg):
        json.JSONDecoder.__init__(self, object_hook = self.convert_dict)


    def convert_dict(self, d):
        ''' Convert a dictionary to objects.

        The dictionary to convert should have been with an mss_dataserver
        JSON file encoder class. In this case, the dictionaries contains
        hints of the original class and module name in the __class__, 
        __module__ and __base_class__ keys. These are used to convert 
        the dictionary to instances of the given classes.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        inst: object
            The object representation of dict d.
        '''
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
        ''' Decode a tuple.

        JSON doesn't support tuples. Use a custom dictionary
        to decode. If the dictionary contains the __tuple__
        attribute, the dictionary is converted to a tuple.

        Parameters
        ----------
        item: dict
            The dictionary to decode.

        Returns
        ------
        item: object
            The object representation of the item dictionary.
        '''
        if isinstance(item, dict):
            if '__tuple__' in item:
                return tuple(item['items'])
        elif isinstance(item, list):
                return [self.decode_hinted_tuple(x) for x in item]
        else:
            return item


    def convert_version(self, d):
        ''' Convert a Version dictionary.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        inst: :class:`mss_dataserver.core.util.Version`
            The instance of the converted dictionary.
        '''
        inst = util.Version(d['version'])
        return inst


    def convert_utcdatetime(self, d):
        ''' Convert a UTCDateTime dictionary.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        d: obspy.UTCDateTime
            The instance of the converted dictionary.
        '''
        inst = obspy.UTCDateTime(d['utcdatetime'])
        return inst


    def convert_np_array(self, d):
        ''' Convert a numpy ndarray dictionary.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        inst: numpy.ndarray
            The instance of the converted dictionary.
        '''
        inst = np.array(d['data'])
        return inst



class SupplementDetectionDataEncoder(json.JSONEncoder):
    ''' A JSON encoder for the event supplement detection data.

    Parameters
    ----------
    **kwargs: keyword argument
        Keyword arguments passed to :class:`json.encoder.JSONEncoder`.

    Attributes
    ----------
    version: :class:`mss_dataserver.core.util`
        The version of the file encoder.
    '''

    version = util.Version('1.0.0')

    def __init__(self, **kwarg):
        ''' Initialization of the instance.
        '''
        json.JSONEncoder.__init__(self, **kwarg)

        # The logger.
        loggerName = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(loggerName)

        # File format settings.
        self.indent = 4
        self.sort_keys = True


    def default(self, obj):
        ''' Convert objects to a dictionary.

        The instance class, module and base_class relations are stored in 
        the __class__, __module__ and __base_class__ keys. These are used 
        by the related file decoder to restore the correct class instances.

        Parameters
        ----------
        obj: object
            The instance to convert to a dictionary.

        Returns
        -------
        d: dict
           The dictionary representation of the instance obj.
        '''        ''' Convert the detection data instances to dictionaries.
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
        ''' Convert a filecontainer instance.

        Parameters
        ----------
        obj: FileContainer
            The filecontainer to convert.

        Returns
        -------
        d: dict
           The dictionary representation of the instance obj.
        '''
        d = obj.data
        file_meta = {'file_version': self.version,
                     'save_date': obspy.UTCDateTime(),
                     'agency_uri': obj.agency_uri,
                     'author_uri': obj.author_uri}
        d['file_meta'] = file_meta
        return d

    def convert_utcdatetime(self, obj):
        ''' Convert a UTCDateTime instance.

        Parameters
        ----------
        obj: obspy.utcdatetime.UTCDateTime
            The UTCDateTime instance to convert.

        Returns
        -------
        d: dict
            The dictionary representation of obj.
        '''
        return {'utcdatetime': obj.isoformat()}

    def convert_version(self, obj):
        ''' Convert a Version dictionary.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        inst: :class:`mss_dataserver.core.util.Version`
            The instance of the converted dictionary..
        '''
        return {'version': str(obj)}

    def convert_np_ndarray(self, obj):
        ''' Convert a numpy ndarray dictionary.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        inst: numpy.ndarray
            The instance of the converted dictionary..
        '''
        return {'data': obj.tolist()}

    def convert_station(self, obj):
        ''' Convert a Station dictionary.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        inst: :class:`mss_dataserver.geometry.inventory.Station`
            The instance of the converted dictionary.
        '''
        attr = ['name', 'location', 'network',
                'x', 'y', 'z', 'coord_system',
                'description', 'author_uri', 'agency_uri',
                'creation_time']
        d = object_to_dict(obj, attr)
        return d


class SupplementDetectionDataDecoder(json.JSONDecoder):
    ''' A JSON decoder for the deserialization of detection supplement data.

    Parameters
    ----------
    **kwargs: keyword argument
        Keyword arguments passed to :class:`json.encoder.JSONDecoder`.

    Attributes
    ----------
    version: :class:`mss_dataserver.core.util`
        The version of the file decoder.
    '''
    version = util.Version('1.0.0')

    def __init__(self, **kwarg):
        ''' Initialize the instance.
        '''
        json.JSONDecoder.__init__(self, object_hook = self.convert_dict)

        self.inventory = geom.inventory.Inventory(name = 'detection_data_import')


    def convert_dict(self, d):
        ''' Convert a dictionary to objects.

        The dictionary to convert should have been with an mss_dataserver
        JSON file encoder class. In this case, the dictionaries contains
        hints of the original class and module name in the __class__, 
        __module__ and __base_class__ keys. These are used to convert 
        the dictionary to instances of the given classes.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        inst: object
            The object representation of dict d.
        '''
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
        ''' Decode a tuple.

        JSON doesn't support tuples. Use a custom dictionary
        to decode. If the dictionary contains the __tuple__
        attribute, the dictionary is converted to a tuple.

        Parameters
        ----------
        item: dict
            The dictionary to decode.

        Returns
        ------
        item: object
            The object representation of the item dictionary.
        '''
        if isinstance(item, dict):
            if '__tuple__' in item:
                return tuple(item['items'])
        elif isinstance(item, list):
                return [self.decode_hinted_tuple(x) for x in item]
        else:
            return item


    def convert_version(self, d):
        ''' Convert a Version dictionary.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        inst: :class:`mss_dataserver.core.util.Version`
            The instance of the converted dictionary.
        '''
        inst = util.Version(d['version'])
        return inst


    def convert_utcdatetime(self, d):
        ''' Convert a UTCDateTime dictionary.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        d: obspy.UTCDateTime
            The instance of the converted dictionary.
        '''
        inst = obspy.UTCDateTime(d['utcdatetime'])
        return inst


    def convert_np_array(self, d):
        ''' Convert a numpy ndarray dictionary.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        inst: numpy.ndarray
            The instance of the converted dictionary.
        '''
        inst = np.array(d['data'])
        return inst


    def convert_station(self, d):
        ''' Convert a Station dictionary.

        Parameters
        ----------
        d: dict
            The dictionary to convert.

        Returns
        -------
        inst: mss_dataserver.geometry.inventory.Station
            The instance of the converted dictionary.
        '''
        cur_station = self.inventory.get_station(name = d['name'],
                                                 location = d['location'],
                                                 network = d['network'])
        if len(cur_station) == 1:
            # Use the found station.
            inst = cur_station[0]
        else:
            # Create a new station and add it to the inventory.
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
            network_name = d['network']
            cur_net = self.inventory.get_network(name = network_name)
            if len(cur_net) == 1:
                cur_net = cur_net[0]
            else:
                cur_net = geom.inventory.Network(name = network_name,
                                                 author_uri = d['author_uri'],
                                                 agency_uri = d['agency_uri'])
            self.inventory.add_network(cur_net)
            self.inventory.add_station(network_name = d['network'],
                                       station_to_add = inst)
        return inst
