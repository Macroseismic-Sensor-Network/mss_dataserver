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

'''
The inventory module.

This module contains the classed needed to build a pSysmon geometry 
inventory.
'''

import itertools
from obspy.core.utcdatetime import UTCDateTime
import mss_dataserver.geometry.util as geom_util
try:
    import pyproj
except:
    from mpl_toolkits.basemap import pyproj
import numpy as np
import warnings
import logging
from operator import attrgetter

class Inventory(object):

    def __init__(self, name, type = None):
        ''' Initialize the instance.

        Parameters
        ----------
        name : String
            The name of the inventory.

        type : String
            The type of the inventory.
        '''

        # The logger.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        ## The name of the inventory.
        self.name = name

        ## The type of the inventory.
        #
        # Based on the source the inventory can be of the following types:
        # - xml
        # - db
        # - manual
        self.type = type

        ## The recorders contained in the inventory.
        self.recorders = []

        ## The sensors contained in the inventory.
        self.sensors = []

        ## The networks contained in the inventory.
        self.networks = []

        ## The arrays contained in the inventory.
        self.arrays = []


    def __str__(self):
        ''' Print the string representation of the inventory.
        '''
        out = "Inventory %s of type %s\n" % (self.name, self.type) 

        # Print the networks.
        out =  out + str(len(self.networks)) + " network(s) in the inventory:\n"
        out = out + "\n".join([net.__str__() for net in self.networks])

        # Print the recorders.
        out = out + '\n\n'
        out =  out + str(len(self.recorders)) + " recorder(s) in the inventory:\n"
        out = out + "\n".join([rec.__str__() for rec in self.recorders])

        return out



    def __eq__(self, other):
        if type(self) is type(other):
            compare_attributes = ['name', 'type', 'recorders', 'networks', 'arrays']
            for cur_attribute in compare_attributes:
                if getattr(self, cur_attribute) != getattr(other, cur_attribute):
                    return False

            return True
        else:
            return False


    def clear(self):
        ''' Clear all elements in the inventory.
        '''
        self.networks = []
        self.arrays = []
        self.recorders = []
        self.sensors = []

    def as_dict(self, style = None):
        ''' Convert the inventory to a dictionary.
        '''
        export_attributes = ['name', 'type']
        d = {}
        for cur_attr in export_attributes:
            d[cur_attr] = getattr(self, cur_attr)
        d['networks'] = [x.as_dict(style = style) for x in self.networks]
        return d



    def add_recorder(self, recorder):
        ''' Add a recorder to the inventory.

        Parameters
        ----------
        recorder : :class:`Recorder`
            The recorder to add to the inventory.
        '''
        added_recorder = None

        if not self.get_recorder(serial = recorder.serial):
            self.recorders.append(recorder)
            recorder.parent_inventory = self
            added_recorder = recorder
        else:
            self.logger.warning('The recorder with serial %s already exists in the inventory.',
                    recorder.serial)

        return added_recorder


    def remove_recorder_by_instance(self, recorder):
        ''' Remove a recorder from the inventory.
        '''
        if recorder in self.recorders:
            self.recorders.remove(recorder)



    def add_station(self, network_name, station_to_add):
        ''' Add a station to the inventory.

        Parameters
        ----------
        network_name : String
            The name of the network to which to add the station.

        station_to_add : :class:`Station`
            The station to add to the inventory.
        '''
        added_station = None

        # If the network is found in the inventory, add it to the network.
        cur_net = self.get_network(name = network_name)
        if len(cur_net) == 1:
            cur_net = cur_net[0]
            added_station = cur_net.add_station(station_to_add)
        elif len(cur_net) > 1:
            self.logger.error("Multiple networks found with the same name. Don't know how to proceed.")
        else:
            self.logger.error("The network %s of station %s doesn't exist in the inventory.\n", network_name, station_to_add.name)

        return added_station



    def remove_station(self, nsl):
        ''' Remove a station from the inventory.

        Parameters
        ----------
        nsl : tuple (String, String, String)
            The NSL (network, station, location) code of the station to remove
            from the inventory.
        '''
        removed_station = None

        cur_net = self.get_network(name = nsl[0])

        if cur_net:
            if len(cur_net) == 1:
                cur_net = cur_net[0]
                removed_station = cur_net.remove_station(name = nsl[1], location = nsl[2])
            else:
                self.logger.error('More than one networks with name %s where found in the inventory.', nsl[0])
        return removed_station



    def add_sensor(self, sensor_to_add):
        ''' Add a sensor to the inventory.

        Parameters
        ----------
        sensor_to_add : :class:`Sensor`
            The sensor to add to the inventory.
        '''
        added_sensor = None
        if not self.get_sensor(serial = sensor_to_add.serial):
            self.sensors.append(sensor_to_add)
            sensor_to_add.parent_inventory = self
            added_sensor = sensor_to_add
        else:
            self.logger.warning('The sensor with serial %s already exists in the inventory.',
                    sensor_to_add.serial)

        return added_sensor


    def remove_sensor_by_instance(self, sensor_to_remove):
        ''' Remove a sensor from the inventory.

        Parameters
        ----------
        sensor_to_remove : :class:`Sensor`
            The sensor to remove from the inventory.
        '''
        if sensor_to_remove in self.sensors:
            self.sensors.remove(sensor_to_remove)


    def add_network(self, network):
        ''' Add a new network to the database inventory.

        Parameters
        ----------
        network : :class:`Network`
            The network to add to the database inventory.
        '''
        added_network = None

        if not self.get_network(name = network.name):
            self.networks.append(network)
            network.parent_inventory = self
            added_network = network
        else:
            self.logger.warning('The network %s already exists in the inventory.', network.name)

        return added_network

    def remove_network_by_instance(self, network_to_remove):
        ''' Remove a network instance from the inventory.
        '''
        if network_to_remove in self.networks:
            self.networks.remove(network_to_remove)


    def remove_network(self, name):
        ''' Remove a network from the inventory.

        Parameters
        ----------
        name : String
            The name of the network to remove.
        '''
        removed_network = None

        net_2_remove = [x for x in self.networks if x.name == name]

        if len(net_2_remove) == 1:
            self.networks.remove(net_2_remove[0])
            removed_network = net_2_remove[0]
        else:
            # This shouldn't happen.
            self.logger.error('Found more than one network with the name %s.', name)

        return removed_network


    def add_array(self, array):
        ''' Add a new array to the inventory.
        '''
        added_array = None

        if not self.get_array(name = array.name):
            self.arrays.append(array)
            array.parent_inventory = self
            added_array = array
        else:
            self.logger.error('The array %s already exists in the inventory.', array.name)

        return added_array


    def has_changed(self):
        ''' Check if any element in the inventory has been changed.
        '''
        for cur_sensor in self.sensors:
            if cur_sensor.has_changed is True:
                self.logger.debug('Sensor changed')
                return True

        for cur_recorder in self.recorders:
            if cur_recorder.has_changed is True:
                self.logger.debug('Recorder changed')
                return True

        for cur_network in self.networks:
            if cur_network.has_changed is True:
                self.logger.debug('Network changed.')
                return True

        return False


    def merge(self, merge_inventory):
        ''' Merge two inventories.
        '''
        # Merge the sensors.
        for cur_sensor in merge_inventory.sensors:
            self.logger.debug('Checking sensor %s.', cur_sensor.serial)
            exist_sensor = self.get_sensor(serial = cur_sensor.serial,
                                   model = cur_sensor.model,
                                   producer = cur_sensor.producer)
            if not exist_sensor:
                self.logger.debug('Adding the sensor to the inventory.')
                self.add_sensor(cur_sensor)
            else:
                exist_sensor = exist_sensor[0]
                self.logger.debug('Merging the sensor with existing sensor %s.', exist_sensor.serial)
                exist_sensor.merge(cur_sensor)

        # Merge the recorders.
        for cur_recorder in merge_inventory.recorders:
            self.logger.debug('Checking recorder %s.', cur_recorder.serial)
            exist_recorder = self.get_recorder(serial = cur_recorder.serial,
                                               model = cur_recorder.model,
                                               producer = cur_recorder.producer)
            if not exist_recorder:
                self.logger.debug('Adding the recorder to the inventory.')
                self.add_recorder(cur_recorder)
            else:
                exist_recorder = exist_recorder[0]
                self.logger.debug('Merging the recorder with existing recorder %s.', exist_recorder.serial)
                exist_recorder.merge(cur_recorder)

        # Merge the networks.
        for cur_network in merge_inventory.networks:
            self.logger.debug('Checking network %s.', cur_network.name)

            exist_network = self.get_network(name = cur_network.name)

            if not exist_network:
                self.logger.debug('Adding the network to the inventory.')
                self.add_network(cur_network)
            else:
                exist_network = exist_network[0]
                self.logger.debug('Merging the network with existing network %s.', exist_network.name)
                exist_network.merge(cur_network)


        # Merge the arrays.
        for cur_array in merge_inventory.arrays:
            self.logger.debug('Checking array %s.', cur_array.name)

            exist_array = self.get_array(name = cur_array.name)

            if not exist_array:
                self.logger.debug('Adding the array to the inventory.')
                self.add_array(cur_array)
            else:
                exist_array = exist_array[0]
                self.logger.debug('Merging the array with existing array %s.', exist_array.name)
                exist_array.merge(cur_array)


    ## Refresh the inventory networks.
    def update_networks(self):
        for cur_network in self.networks:
            cur_network.refresh_stations(self.stations)


    ## Refresh the inventory recorders.
    def refresh_recorders(self):
        for cur_recorder in self.recorders:
            cur_recorder.refresh_sensors()

        for cur_sensor in self.sensors:
            self.add_sensor(cur_sensor)


    ## Read the inventory from an XML file.
    def import_from_xml(self, filename):
        inventory_parser = InventoryXmlParser(self, filename)
        try:
            inventory_parser.parse()
        except Exception as e:
            raise e

        self.type = 'xml'


    def get_recorder(self, **kwargs):
        ''' Get a recorder from the inventory.

        Parameters
        ----------
        serial : String
            The serial number of the recorder.

        model : String
            The recorder model.

        producer : String
            The recorder producer.

        Returns
        -------
        recorder : List of :class:'~Recorder'
            The recorder(s) in the inventory matching the search criteria.
        '''
        ret_recorder = self.recorders

        valid_keys = ['serial', 'model', 'producer']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_recorder = [x for x in ret_recorder if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_recorder


    def get_stream(self, **kwargs):
        ''' Get a stream of a recorder from the inventory.

        Parameters
        ----------
        serial : String
            The serial number of the recorder containing the stream.

        model : String
            The model of the recorder containing the stream.

        producer : String
            The producer of the recorder containing the stream.

        name : String
            The name of the component.
        '''
        ret_stream = list(itertools.chain.from_iterable([x.streams for x in self.recorders]))

        valid_keys = ['name', 'serial', 'model', 'producer']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_stream = [x for x in ret_stream if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_stream


    def get_sensor(self, **kwargs):
        ''' Get a sensor from the inventory.

        Parameters
        ----------
        serial : String
            The serial number of the sensor.

        model : String
            The model of the sensor.

        producer : String
            The producer of the sensor.

        label : String
            The label of the sensor
        '''
        ret_sensor = self.sensors

        valid_keys = ['serial', 'model', 'producer']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_sensor = [x for x in ret_sensor if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_sensor



    def get_component(self, **kwargs):
        ''' Get a component of a sensor from the inventory.

        Parameters
        ----------
        serial : String
            The serial number of the sensor containing the component.

        model : String
            The model of the sensor containing the component.

        producer : String
            The producer of the sensor containing the component.

        name : String
            The name of the component.
        '''
        ret_component = list(itertools.chain.from_iterable([x.components for x in self.sensors]))


        valid_keys = ['name', 'serial', 'model', 'producer']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_component = [x for x in ret_component if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_component


    def get_station(self, **kwargs):
        ''' Get a station from the inventory.

        Parameters
        ----------
        name : String
            The name (code) of the station.

        network : String
            The name of the network of the station.

        location : String
            The location code of the station.
        '''
        ret_station = list(itertools.chain.from_iterable([x.stations for x in self.networks]))

        valid_keys = ['id', 'name', 'network', 'location', 'nsl', 'nsl_string']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_station = [x for x in ret_station if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_station


    def get_channel(self, **kwargs):
        ''' Get a chennel from the inventory.

        Paramters
        ---------
        network : String
            The name of the network.

        station : String
            The name (code) of the station.

        station : String
            The location identifier.

        name : String
            The name of the channel.
        '''

        search_dict = {}
        if 'network' in kwargs.keys():
            search_dict['network'] = kwargs['network']
            kwargs.pop('network')

        if 'station' in kwargs.keys():
            search_dict['name'] = kwargs['station']
            kwargs.pop('station')

        if 'location' in kwargs.keys():
            search_dict['location'] = kwargs['location']
            kwargs.pop('location')

        stations = self.get_station(**search_dict)

        ret_channel = list(itertools.chain.from_iterable([x.channels for x in stations]))

        valid_keys = ['name',]

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_channel = [x for x in ret_channel if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)
        return ret_channel


    def get_channel_from_stream(self, start_time = None, end_time = None, **kwargs):
        ''' Get the channels to which a stream is assigned to.
        '''
        ret_channel = list(itertools.chain.from_iterable(x.channels for x in self.get_station()))

        ret_channel = [x for x in ret_channel if x.get_stream(start_time = start_time,
                                                              end_time = end_time,
                                                              **kwargs)]
        return ret_channel


    def get_network(self, **kwargs):
        ''' Get a network from the inventory.

        Parameters
        ----------
        name : String
            The name of the network.

        type : String
            The type of the network.
        '''
        ret_network = self.networks

        valid_keys = ['name', 'type']
        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_network = [x for x in ret_network if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_network


    def get_array(self, **kwargs):
        ''' Get an array from the inventory.

        Parameters
        ----------
        name : String
            The name of the array.
        '''
        ret_array = self.arrays

        valid_keys = ['name']
        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_array = [x for x in ret_array if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_array


    def get_utm_epsg(self):
        ''' Compute a the epsg code of the best fitting UTM coordinate system.
        '''
        # Get the lon/lat limits of the inventory.
        lonLat = []
        for curNet in self.networks:
            lonLat.extend([stat.get_lon_lat() for stat in curNet.stations])

        if len(lonLat) == 0:
            self.logger.error("Length of lonLat is zero. No stations found in the inventory. Can't compute the UTM zone.")
            return

        lonLatMin = np.min(lonLat, 0)
        lonLatMax = np.max(lonLat, 0)
        utm_zone = geom_util.lon2UtmZone(np.mean([lonLatMin[0], lonLatMax[0]]))
        if np.mean([lonLatMin[1], lonLatMax[1]]) >= 0:
            hemisphere = 'north'
        else:
            hemisphere = 'south'

        # Get the epsg code of the UTM projection.
        search_dict = {'projection': 'utm',
                       'ellps': 'WGS84',
                       'zone': utm_zone,
                       'no_defs': True,
                       'units': 'm'}
        if hemisphere == 'south':
            search_dict['south'] = True

        epsg_dict = geom_util.get_epsg_dict()
        code = [(c, x) for c, x in epsg_dict.items() if  x == search_dict]
        return code

    def compute_utm_coordinates(self):
        code = self.get_utm_epsg()
        proj = pyproj.Proj(init = 'epsg:' + code[0][0])

        for cur_station in self.get_station():
            x, y = proj(cur_station.get_lon_lat()[0],
                        cur_station.get_lon_lat()[1])
            cur_station.x_utm = x
            cur_station.y_utm = y


    @classmethod
    def from_db_inventory(cls, db_inventory):
        pass



class Recorder(object):
    ''' A seismic data recorder.
    '''

    def __init__(self, serial, model, producer, description = None, id=None, parent_inventory=None,
            author_uri = None, agency_uri = None, creation_time = None):
        ''' Initialize the instance.

        '''
        # The logging logger instance.
        logger_prefix = 'mss_data_server'
        loggerName = logger_prefix + "." + __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(loggerName)

        ## The recorder database id.
        self.id = id

        ## The recorder serial number.
        self.serial = str(serial)

        # The model name or number.
        self.model = model

        # The producer of the sensor.
        self.producer = producer

        # The description of the recorder.
        self.description = description

        # Indicates if the attributes have been changed.
        self.has_changed = False

        # A list of Stream instances related to the recorder.
        self.streams = [];

        ## The parent inventory.
        self.parent_inventory = parent_inventory

        # The author.
        self.author_uri = author_uri

        # The agency of the author.
        self.agency_uri = agency_uri

        # The datetime of the creation.
        if creation_time == None:
            self.creation_time = UTCDateTime();
        else:
            self.creation_time = UTCDateTime(creation_time);


    def __str__(self):
        ''' Returns a readable representation of the Recorder instance.
        '''
        out = 'id:\t%s\nserial:\t%s\nmodel:\t%s\n%d sensor(s):\n' % (str(self.id), self.serial, self.model, len(self.sensors))
        return out


    def __setitem__(self, name, value):
        self.__dict__[name] = value
        self.has_changed = True 


    def __eq__(self, other):
        if type(self) is type(other):
            compare_attributes = ['id', 'serial', 'model', 'producer', 'description', 'has_changed',
                                  'streams']
            for cur_attribute in compare_attributes:
                if getattr(self, cur_attribute) != getattr(other, cur_attribute):
                    return False

            return True
        else:
            return False



    def add_stream(self, cur_stream):
        ''' Add a stream to the recorder.

        Parameters
        ----------
        stream : :class:`Stream`
            The stream to add to the recorder.
        '''
        added_stream = None
        if cur_stream not in self.streams:
            self.streams.append(cur_stream)
            cur_stream.parent_recorder = self
            added_stream = cur_stream

        return added_stream


    def pop_stream_by_instance(self, stream):
        ''' Remove a component from the sensor using the component instance.
        '''
        removed_stream = None
        if not stream.assigned_channels:
            # If the stream is not assigned to a channel, remove it.
            if stream in self.streams:
                self.streams.remove(stream)
                removed_stream = stream

        return removed_stream


    def pop_stream(self, **kwargs):
        ''' Remove a stream from the recorder.

        Parameters
        ----------
        name : String
            The name of the stream.

        label : String
            The label of the stream.

        agency_uri : String
            The agency_uri of the stream.

        author_uri : string
            The author_uri of the stream.

        Returns
        -------
        streams_popped : List of :class:`Stream`
            The removed streams.
        '''
        streams_popped = []
        streams_to_pop = self.get_stream(**kwargs)

        for cur_stream in streams_to_pop:
            cur_stream.parent_recorder = None
            streams_popped.append(self.streams.pop(self.streams.index(cur_stream)))

        return streams_popped


    def get_stream(self, **kwargs):
        ''' Get a stream from the recorder.

        Parameters
        ----------
        name : String
            The name of the stream.

        label : String
            The label of the stream.

        agency_uri : String
            The agency_uri of the stream.

        author_uri : string
            The author_uri of the stream.
        '''
        ret_stream = self.streams

        valid_keys = ['name', 'label', 'agency_uri', 'author_uri']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_stream = [x for x in ret_stream if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_stream


    def merge(self, merge_recorder):
        ''' Merge a recorder with the existing.
        '''
        # Update the attributes.
        self.description = merge_recorder.description

        # Merge the streams.
        for cur_stream in merge_recorder.streams:
            exist_stream = self.get_stream(name = cur_stream.name)
            if not exist_stream:
                self.add_stream(cur_stream)
            else:
                exist_stream = exist_stream[0]
                self.logger.debug('Merging db stream %s (id: %d) with %s.', '-'.join([exist_stream.model, exist_stream.name]), exist_stream.id, '-'.join([cur_stream.model, cur_stream.name]))
                exist_stream.merge(cur_stream)





class RecorderStream(object):
    ''' A digital stream of a data recorder.
    '''

    def __init__(self, name, label,
                 agency_uri = None, author_uri = None,
                 creation_time = None, parent_recorder = None):
        ''' Initialization of the instance.
        '''
        # The logging logger instance.
        loggerName = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(loggerName)

        # The name of the stream.
        self.name = name

        # The label of the stream.
        self.label = label

        # The author.
        self.author_uri = author_uri

        # The agency of the author.
        self.agency_uri = agency_uri

        # The datetime of the creation.
        if creation_time == None:
            self.creation_time = UTCDateTime();
        else:
            self.creation_time = UTCDateTime(creation_time);

        # The parent recorder holding the stream.
        self.parent_recorder = parent_recorder

        # Indicates if the attributes have been changed.
        self.has_changed = False

        # A list of :class: `TimeBox` instances holding the assigned
        # components.
        self.components = []

        # A list of :class: `RecorderStreamParameter` instances.
        self.parameters = []


    @property
    def parent_inventory(self):
        if self.parent_recorder is not None:
            return self.parent_recorder.parent_inventory
        else:
            return None


    @property
    def serial(self):
        if self.parent_recorder is not None:
            return self.parent_recorder.serial
        else:
            return None


    @property
    def model(self):
        if self.parent_recorder is not None:
            return self.parent_recorder.model
        else:
            return None

    @property
    def producer(self):
        if self.parent_recorder is not None:
            return self.parent_recorder.producer
        else:
            return None


    @property
    def assigned_channels(self):
        # The channels to which the stream is assigned to.
        assigned_channels = []
        station_list = self.parent_inventory.get_station()
        for cur_station in station_list:
            for cur_channel in cur_station.channels:
                if cur_channel.get_stream(serial = self.serial, name = self.name):
                    assigned_channels.append(cur_channel)
        return assigned_channels



    def __setitem__(self, name, value):
        self.__dict__[name] = value
        self.has_changed = True
        if self.parent_recorder is not None:
            self.parent_recorder.has_changed =  True

        # Send an inventory update event.
        # TODO: The sending of the pub messages should be done somewhere else.
        # A GUI related method should be used for that.
        #msgTopic = 'inventory.update.stream'
        #msg = (self, name, value)
        #pub.sendMessage(msgTopic, msg)


    def __eq__(self, other):
        if type(self) is type(other):
            compare_attributes = ['name', 'label', 'gain',
                    'bitweight', 'bitweight_units', 'components', 'has_changed']
            for cur_attribute in compare_attributes:
                if getattr(self, cur_attribute) != getattr(other, cur_attribute):
                    return False

            return True
        else:
            return False

    def as_dict(self, style = None):
        export_attributes = ['name', 'label', 'serial', 'model', 'producer',
                             'author_uri', 'agency_uri', 'creation_time']

        d = {}
        for cur_attr in export_attributes:
            d[cur_attr] = getattr(self, cur_attr)
        return d


    def add_component(self, serial, model, producer, name, start_time, end_time):
        ''' Add a sensor component to the stream.

        The component with specified serial and name is searched
        in the parent inventory and if available, the sensor is added to
        the stream for the specified time-span.

        Parameters
        ----------
        serial : String
            The serial number of the sensor which holds the component.

        model : String
            The model of the sensor which holds the component.

        producer : String
            The producer of the sensor which holds the component.

        name : String
            The name of the component.

        start_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time from which on the sensor has been operating at the station.

        end_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time up to which the sensor has been operating at the station. "None" if the station is still running.
        '''
        if self.parent_inventory is None:
            raise RuntimeError('The stream needs to be part of an inventory before a sensor can be added.')

        added_component = None
        cur_component = self.parent_inventory.get_component(serial = serial,
                                                            model = model,
                                                            producer = producer,
                                                            name = name)
        if not cur_component:
            msg = 'The specified component (serial = %s, name = %s) was not found in the inventory.' % (serial, name)
            raise RuntimeError(msg)
        elif len(cur_component) == 1:
            cur_component = cur_component[0]

            try:
                start_time = UTCDateTime(start_time)
            except:
                start_time = None

            try:
                end_time = UTCDateTime(end_time)
            except:
                end_time = None

            if self.get_component(start_time = start_time,
                                  end_time = end_time):
                # A sensor is already assigned to the stream for this timespan.
                if start_time is not None:
                    start_string = start_time.isoformat
                else:
                    start_string = 'big bang'

                if end_time is not None:
                    end_string = end_time.isoformat
                else:
                    end_string = 'running'

                msg = 'The component (serial: %s,  name: %s) is already deployed during the specified timespan from %s to %s.' % (serial, name, start_string, end_string)
                raise RuntimeError(msg)
            else:
                self.components.append(TimeBox(item = cur_component,
                                               start_time = start_time,
                                               end_time = end_time,
                                               parent = self))
                self.has_changed = True
                added_component = cur_component
        else:
            msg = "Got more than one component with serial=%s and name = %s. Only one component with a serial-component combination should be in the inventory. Don't know how to proceed." % (serial, name)
            raise RuntimeError(msg)

        return added_component


    def remove_component_by_instance(self, timebox):
        ''' Remove a component from the stream.

        '''
        if timebox in self.components:
            self.components.remove(timebox)


    def remove_parameter_by_instance(self, parameter):
        ''' Remove a parameter from the stream.
        '''
        if parameter in self.parameters:
            self.parameters.remove(parameter)


    def get_component(self, start_time = None, end_time = None, **kwargs):
        ''' Get a sensor from the stream.

        Parameters
        ----------
        name : String
            The name of the component.

        serial : String
            The serial of the sensor containing the component.

        start_time : :class:`~obspy.core.utcdatetime.UTCDateTime`
            The start time of the timespan to return.

        end_time : :class:`~obspy.core.utcdatetime.UTCDateTime`
            The end time of the timespan to return.

        '''
        ret_component = self.components

        valid_keys = ['serial', 'name']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_component = [x for x in ret_component if getattr(x.item, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        if start_time is not None:
            ret_component = [x for x in ret_component if (x.end_time is None) or (x.end_time > start_time)]

        if end_time is not None:
            ret_component = [x for x in ret_component if x.start_time < end_time]

        return ret_component


    def add_parameter(self, parameter_to_add):
        ''' Add a paramter to the recorder stream.

        Parameters
        ----------
        parameter_to_add : :class:`RecorderStreamParameter`
            The recorder stream parameter to add to the stream.
        '''
        added_parameter = None
        if not self.get_parameter(start_time = parameter_to_add.start_time,
                                  end_time = parameter_to_add.end_time):
            self.parameters.append(parameter_to_add)
            self.parameters = sorted(self.parameters, key = attrgetter('start_time'))
            parameter_to_add.parent_recorder_stream = self
            added_parameter = parameter_to_add
        else:
            raise RuntimeError('A parameter already exists for the given timespan.')

        return added_parameter


    def get_parameter(self, start_time = None, end_time = None):
        ''' Get parameter for a given timespan.

        '''
        ret_parameter = self.parameters

        if start_time is not None:
            start_time = UTCDateTime(start_time)
            ret_parameter = [x for x in ret_parameter if x.end_time is None or x.end_time > start_time]

        if end_time is not None:
            end_time = UTCDateTime(end_time)
            ret_parameter = [x for x in ret_parameter if x.start_time is None or x.start_time < end_time]

        return ret_parameter


    def get_free_parameter_slot(self, pos = 'both'):
        ''' Get a free time slot for a parameter.

        Parameters
        ----------
        pos : String
            The postion of the list of the next free time slot ('front', 'back', 'both').
        '''
        if self.parameters:
            last_parameter = sorted(self.parameters, key = attrgetter('start_time'))[-1]
            first_parameter = sorted(self.parameters, key = attrgetter('start_time'))[0]

            if pos == 'back':
                if last_parameter.end_time is None:
                    return None
                else:
                    return (last_parameter.end_time + 1, None)

            elif pos == 'front':
                if first_parameter.start_time is None:
                    return None
                else:
                    return (None, first_parameter.start_time - 1)
            elif pos == 'both':
                if last_parameter.end_time is None:
                    if first_parameter.start_time is None:
                        return None
                    else:
                        return (None, first_parameter.start_time - 1)
                else:
                    return (last_parameter.end_time + 1, None)
            else:
                raise ValueError('Use either back, front or both for the pos argument.')
        else:
            return (None, None)


    def get_free_component_slot(self, pos = 'both'):
        ''' Get a free time slot for a component.

        Parameters
        ----------
        pos : String
            The postion of the list of the next free time slot ('front', 'back', 'both').
        '''
        if self.components:
            last_component = sorted(self.components, key = attrgetter('start_time'))[-1]
            first_component = sorted(self.components, key = attrgetter('start_time'))[0]

            if pos == 'back':
                if last_component.end_time is None:
                    return None
                else:
                    return (last_component.end_time + 1, None)

            elif pos == 'front':
                if first_component.start_time is None:
                    return None
                else:
                    return (None, first_component.start_time - 1)
            elif pos == 'both':
                if last_component.end_time is None:
                    if first_component.start_time is None:
                        return None
                    else:
                        return (None, first_component.start_time - 1)
                else:
                    return (last_component.end_time + 1, None)
            else:
                raise ValueError('Use either back, front or both for the pos argument.')
        else:
            return (None, None)


    def merge(self, merge_stream):
        ''' Merge a stream.
        '''
        # Update the attributes.
        self.label = merge_stream.label

        # Replace existing parameters with the new ones.
        #for cur_parameter in [x for x in self.parameters]:
        #    self.logger.info('Removing parameter %d.', cur_parameter.id)
        #    self.remove_parameter_by_instance(cur_parameter)

        updated_parameters = []
        parameters_to_add = []
        for cur_parameter in merge_stream.parameters:
            cur_exist_parameter = self.get_parameter(start_time = cur_parameter.start_time,
                                                     end_time = cur_parameter.end_time)
            if len(cur_exist_parameter) == 1:
                cur_exist_parameter = cur_exist_parameter[0]
                cur_key = (cur_exist_parameter.rec_stream_id,
                           cur_exist_parameter.start_time,
                           cur_exist_parameter.end_time)
                self.logger.info('Updating existing parameter %s.', cur_key)
                cur_exist_parameter.gain = cur_parameter.gain
                cur_exist_parameter.bitweight = cur_parameter.bitweight
                cur_exist_parameter.author_uri = cur_parameter.author_uri
                cur_exist_parameter.agency_uri = cur_parameter.agency_uri
                updated_parameters.append(cur_exist_parameter)
            elif len(cur_exist_parameter) == 0:
                parameters_to_add.append(cur_parameter)
            else:
                self.logger.error('More than one parameter returned.')

        # Remove the parameters, that have not been updated.
        parameters_to_remove = [x for x in self.parameters if x not in updated_parameters]
        for cur_parameter in parameters_to_remove:
            cur_key = (cur_parameter.rec_stream_id,
                       cur_parameter.start_time,
                       cur_parameter.end_time)
            self.logger.info('Removing parameter %s.', cur_key)
            self.remove_parameter_by_instance(cur_parameter)

        # Add all new parameters.
        for cur_parameter in parameters_to_add:
            cur_key = (cur_parameter.rec_stream_id,
                       cur_parameter.start_time,
                       cur_parameter.end_time)
            self.logger.info('Adding new parameter %s.', cur_key)
            self.add_parameter(cur_parameter)

        # Replace existing components with the new ones.
        for cur_component in [x for x in self.components]:
            self.remove_component_by_instance(cur_component)

        for cur_component in merge_stream.components:
            self.add_component(serial = cur_component.serial,
                               model = cur_component.model,
                               producer = cur_component.producer,
                               name = cur_component.name,
                               start_time = cur_component.start_time,
                               end_time = cur_component.end_time)


class RecorderStreamParameter(object):
    ''' Parameters of a recorder stream.
    '''

    def __init__(self, start_time, end_time = None,
                 gain = None, bitweight = None,
                 agency_uri = None, author_uri = None, creation_time = None,
                 parent_recorder_stream = None):
        ''' Initialize the instance.
        '''
        # The logger instance.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # The gain of the stream.
        try:
            self.gain = float(gain)
        except:
            self.gain = None

        # The bitweight of the stream.
        try:
            self.bitweight = float(bitweight)
        except:
            self.bitweight = None

        # The start time from which on the parameters were valid.
        try:
            self.start_time = UTCDateTime(start_time)
        except:
            self.start_time = None

        # The end time until which the parameters were valid.
        try:
            self.end_time = UTCDateTime(end_time)
        except:
            self.end_time = None

        # The recorder stream for which the parameters were set.
        self.parent_recorder_stream = parent_recorder_stream

        # The author.
        self.author_uri = author_uri

        # The agency of the author.
        self.agency_uri = agency_uri

        # The datetime of the creation.
        if creation_time == None:
            self.creation_time = UTCDateTime();
        else:
            self.creation_time = UTCDateTime(creation_time);

        # Indicates if the attributes have been changed.
        self.has_changed = False


    @property
    def parent_inventory(self):
        if self.parent_recorder_stream is not None:
            return self.parent_recorder_stream.parent_inventory
        else:
            return None

    @property
    def rec_stream_id(self):
        try:
            return self.parent_recorder_stream.id
        except Exception:
            return None


    @property
    def start_time_string(self):
        if self.start_time is None:
            return 'big bang'
        else:
            return self.start_time.isoformat()


    @property
    def end_time_string(self):
        if self.end_time is None:
            return 'running'
        else:
            return self.end_time.isoformat()





class Sensor(object):
    ''' A seismic sensor.

    '''

    def __init__(self, serial, model, producer, description = None,
                 author_uri = None, agency_uri = None,
                 creation_time = None, parent_inventory = None):
        ''' Initialize the instance

        '''
        # The logger instance.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # The serial number of the sensor.
        self.serial = str(serial)

        # The model name or number.
        self.model = model

        # The producer of the sensor.
        self.producer = producer

        # A description of the sensor.
        self.description = description

        # The components of the sensor.
        self.components = []

        # The inventory containing this sensor.
        self.parent_inventory = parent_inventory

        # The author.
        self.author_uri = author_uri

        # The agency of the author.
        self.agency_uri = agency_uri

        # The datetime of the creation.
        if creation_time == None:
            self.creation_time = UTCDateTime();
        else:
            self.creation_time = UTCDateTime(creation_time);

        # Indicates if the attributes have been changed.
        self.has_changed = False


    def __setitem__(self, name, value):
        self.__dict__[name] = value
        self.has_changed = True


    def add_component(self, component_to_add):
        ''' Add a component to the sensor.

        Parameters
        ----------
        component_to_add : :class:`SensorComponent`
            The component to add to the sensor.
        '''
        added_component = None
        if component_to_add not in self.components:
            self.components.append(component_to_add)
            component_to_add.parent_sensor = self
            added_component = component_to_add

        return added_component


    def get_component(self, **kwargs):
        ''' Get a component from the sensor.

        Parameters
        ----------
        name : String
            The name of the component.

        agency_uri : String
            The agency_uri of the component.

        author_uri : string
            The author_uri of the component.
        '''
        ret_component = self.components

        valid_keys = ['name', 'agency_uri', 'author_uri']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_component = [x for x in ret_component if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_component


    def pop_component_by_instance(self, component):
        ''' Remove a component from the sensor using the component instance.
        '''
        removed_component = None
        if not component.assigned_streams:
            # If the component is not assigned to a stream, remove it.
            if component in self.components:
                self.components.remove(component)
                removed_component = component

        return removed_component


    def pop_component(self, **kwargs):
        ''' Remove a component from the sensor.

        Parameters
        ----------
        name : String
            The name of the component.

        agency_uri : String
            The agency_uri of the component.

        author_uri : string
            The author_uri of the component.

        Returns
        -------
        components_popped : List of :class:`SensorComponent`
            The removed components.
        '''
        components_popped = []
        components_to_pop = self.get_component(**kwargs)

        for cur_component in components_to_pop:
            cur_component.parent_recorder = None
            components_popped.append(self.components.pop(self.components.index(cur_component)))

        return components_popped


    def merge(self, merge_sensor):
        ''' Merge a sensor to the existing.
        '''
        # Update the attributes.
        self.description = merge_sensor.description

        # Update the components.
        for cur_component in merge_sensor.components:
            self.logger.debug('Checking component %s.', cur_component.name)
            exist_component = self.get_component(name = cur_component.name)
            if not exist_component:
                self.logger.debug('Adding component %s.', cur_component.name)
                self.add_component(cur_component)
            else:
                exist_component = exist_component[0]
                self.logger.debug('Merging component %s.', cur_component.name)
                exist_component.merge(cur_component)





class SensorComponent(object):
    ''' A component of a seismic sensor.
    '''

    def __init__(self, name, description = None,
                 input_unit = None, output_unit = None, deliver_unit = None,
                 author_uri = None, agency_uri = None, creation_time = None,
                 parent_sensor = None):
        ''' Initialize the instance.

        '''
        # The logger instance.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # The name of the component.
        self.name = name

        # The description.
        self.description = description

        # The unit of the parameter measured by the sensor (e.g. m).
        self.input_unit = input_unit

        # The unit to which the input unit is transformed by the sensor (e.g.
        # m/s).
        self.output_unit = output_unit

        # The unit of the measureable signal which is proportional to the
        # output unit (e.g. V).
        self.deliver_unit = deliver_unit

        ## The component parameters.
        self.parameters = []

        # The inventory containing this sensor.
        self.parent_sensor = parent_sensor

        # Indicates if the attributes have been changed.
        self.has_changed = False

        # The author.
        self.author_uri = author_uri

        # The agency of the author.
        self.agency_uri = agency_uri

        # The datetime of the creation.
        if creation_time == None:
            self.creation_time = UTCDateTime();
        else:
            self.creation_time = UTCDateTime(creation_time);

    @property
    def parent_inventory(self):
        if self.parent_sensor is not None:
            return self.parent_sensor.parent_inventory
        else:
            return None

    @property
    def serial(self):
        if self.parent_sensor is not None:
            return self.parent_sensor.serial
        else:
            return None

    @property
    def model(self):
        if self.parent_sensor is not None:
            return self.parent_sensor.model
        else:
            return None

    @property
    def producer(self):
        if self.parent_sensor is not None:
            return self.parent_sensor.producer
        else:
            return None



    @property
    def assigned_streams(self):
        # Check if the component is assigned to a recorder stream.
        assigned_streams = []
        recorder_list = self.parent_inventory.get_recorder()
        for cur_recorder in recorder_list:
            stream_list = cur_recorder.get_stream()
            for cur_stream in stream_list:
                if cur_stream.get_component(serial = self.serial,
                                            name = self.name):
                    assigned_streams.append(cur_stream)
        return assigned_streams


    def __setitem__(self, name, value):
        self.__dict__[name] = value
        self.has_changed = True
        self.logger.debug('Changing attribute %s of sensor %d', name, self.id)

        # Send an inventory update event.
        #msgTopic = 'inventory.update.sensor'
        #msg = (self, name, value)
        #pub.sendMessage(msgTopic, msg)


    def __eq__(self, other):
        if type(self) is type(other):
            compare_attributes = ['name', 'description',
                                  'has_changed', 'parameters']
            for cur_attribute in compare_attributes:
                if getattr(self, cur_attribute) != getattr(other, cur_attribute):
                    return False

            return True
        else:
            return False


    def add_parameter(self, parameter_to_add):
        ''' Add a sensor paramter instance to the sensor.

        Parameters
        ----------
        parameter_to_add : :class:`SensorParameter`
            The sensor parameter instance to be added.
        '''
        added_parameter = None
        if not self.get_parameter(start_time = parameter_to_add.start_time,
                                  end_time = parameter_to_add.end_time):
            self.parameters.append(parameter_to_add)
            parameter_to_add.parent_component = self
            added_parameter = parameter_to_add
        else:
            raise RuntimeError('A parameter already exists for the given timespan.')

        return added_parameter


    def remove_parameter(self, parameter_to_remove):
        ''' Remove a parameter from the component.
        '''
        self.parameters.remove(parameter_to_remove)



    def get_parameter(self, start_time = None, end_time = None):
        ''' Get a sensor from the recorder.

        Parameters
        ----------

        '''
        parameter = self.parameters

        if start_time is not None:
            start_time = UTCDateTime(start_time)
            parameter = [x for x in parameter if x.end_time is None or x.end_time > start_time]

        if end_time is not None:
            end_time = UTCDateTime(end_time)
            parameter = [x for x in parameter if x.start_time is None or x.start_time < end_time]

        return parameter



    def change_parameter_start_time(self, position, start_time):
        msg = ''
        cur_row = self.parameters[position]

        if not isinstance(start_time, UTCDateTime):
            try:
                start_time = UTCDateTime(start_time)
            except:
                start_time = cur_row[1]
                msg = "The entered value is not a valid time."


        if not cur_row[2] or start_time < cur_row[2]:
            self.parameters[position] = (cur_row[0], start_time, cur_row[2])
            cur_row[0]['start_time'] = start_time
        else:
            start_time = cur_row[1]
            msg = "The start-time has to be smaller than the begin time."

        return (start_time, msg)


    def change_parameter_end_time(self, position, end_time):
        msg = ''
        cur_row = self.parameters[position]

        if end_time == 'running':
            self.parameters[position] = (cur_row[0], cur_row[1], None)
            cur_row[0]['end_time'] = None
            return(end_time, msg)

        if not isinstance(end_time, UTCDateTime):
            try:
                end_time = UTCDateTime(end_time)
            except:
                end_time = cur_row[2]
                msg = "The entered value is not a valid time."


        if end_time:
            if not cur_row[1] or end_time > cur_row[1]:
                self.parameters[position] = (cur_row[0], cur_row[1], end_time)
                cur_row[0]['end_time'] = end_time
                # Send an inventory update event.
                #msgTopic = 'inventory.update.sensorParameterTime'
                #msg = (cur_row[0], 'time', (self, cur_row[0], cur_row[1], end_time))
                #pub.sendMessage(msgTopic, msg)
            else:
                end_time = cur_row[2]
                msg = "The end-time has to be larger than the begin time."

        return (end_time, msg)


    def merge(self, merge_component):
        ''' Merge the components.
        '''
        # Update the attributes.
        self.description = merge_component.description
        self.input_unit = merge_component.input_unit
        self.output_unit = merge_component.output_unit
        self.deliver_unit = merge_component.deliver_unit

        # Replace the existing parameters with the new ones.
        for cur_parameter in [x for x in self.parameters]:
            self.remove_parameter(cur_parameter)

        for cur_parameter in merge_component.parameters:
            self.add_parameter(cur_parameter)







## The sensor parameter class.
#
class SensorComponentParameter(object):
    ## The constructor.
    #
    # @param self The object pointer.
    def __init__(self, sensitivity,
                 start_time, end_time, tf_type=None,
                 tf_units=None, tf_normalization_factor=None,
                 tf_normalization_frequency=None, tf_poles = None, tf_zeros = None,
                 parent_component = None, author_uri = None,
                 agency_uri = None, creation_time = None):

        # The logger instance.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        ## The sensor sensitivity.
        try:
            self.sensitivity = float(sensitivity)
        except:
            self.sensitivity = None


        # TODO: The parameters tf_type and tf_units seem to make no sense
        # anylonger. They have been replaced by the units of the sensor itself.
        # Think about these values and eventually remove them. Removing will
        # the values will also affect the database. So it's no trivial change.

        ## The transfer function type.
        # - displacement
        # - velocity
        # - acceleration
        self.tf_type = tf_type

        ## The transfer function units.
        self.tf_units = tf_units

        ## The transfer function normalization factor.
        try:
            self.tf_normalization_factor = float(tf_normalization_factor)
        except:
            self.tf_normalization_factor = None

        ## The transfer function normalization factor frequency.
        try:
            self.tf_normalization_frequency = float(tf_normalization_frequency)
        except:
            self.tf_normalization_frequency = None

        ## The transfer function as PAZ.
        if tf_poles is None:
            tf_poles = []

        if tf_zeros is None:
            tf_zeros = []

        self.tf_poles = tf_poles
        self.tf_zeros = tf_zeros

        # The start_time from which the parameters are valid.
        try:
            self.start_time = UTCDateTime(start_time)
        except:
            self.start_time = None

        # The end time up to which the parameters are valid.
        try:
            self.end_time = UTCDateTime(end_time)
        except:
            self.end_time = None

        # The parent sensor holding the parameter.
        self.parent_component = parent_component

        # Indicates if the attributes have been changed.
        self.has_changed = False

        # The author.
        self.author_uri = author_uri

        # The agency of the author.
        self.agency_uri = agency_uri

        # The datetime of the creation.
        if creation_time == None:
            self.creation_time = UTCDateTime();
        else:
            self.creation_time = UTCDateTime(creation_time);

    @property
    def parent_inventory(self):
        if self.parent_component is not None:
            return self.parent_component.parent_inventory
        else:
            return None


    @property
    def start_time_string(self):
        if self.start_time is None:
            return 'big bang'
        else:
            return self.start_time.isoformat()


    @property
    def end_time_string(self):
        if self.end_time is None:
            return 'running'
        else:
            return self.end_time.isoformat()


    @property
    def zeros_string(self):
        zero_str = ''
        if self.tf_zeros:
            for cur_zero in self.tf_zeros:
                if zero_str == '':
                    zero_str = cur_zero.__str__()
                else:
                    zero_str = zero_str + ',' + cur_zero.__str__()

        return zero_str

    @property
    def poles_string(self):
        pole_str = ''
        if self.tf_poles:
            for cur_pole in self.tf_poles:
                if pole_str == '':
                    pole_str = cur_pole.__str__()
                else:
                    pole_str = pole_str + ',' + cur_pole.__str__()

        return pole_str


    def __eq__(self, other):
        if type(self) is type(other):
            compare_attributes = ['sensitivity', 'tf_type',
                                  'tf_units', 'tf_normalization_factor', 'tf_normalization_frequency',
                                  'id', 'tf_poles', 'tf_zeros', 'start_time', 'end_time',
                                  'has_changed']
            for cur_attribute in compare_attributes:
                if getattr(self, cur_attribute) != getattr(other, cur_attribute):
                    return False

            return True
        else:
            return False


    def set_transfer_function(self, tf_type, tf_units, tf_normalization_factor, 
                            tf_normalization_frequency):
        ''' Set the transfer function parameters.

        '''
        self.tf_type = tf_type
        self.tf_units = tf_units
        self.tf_normalization_factor = tf_normalization_factor
        self.tf_normalization_frequency = tf_normalization_frequency


    def tf_add_complex_zero(self, zero):
        ''' Add a complex zero to the transfer function PAZ.

        '''
        self.logger.debug('Adding zero %s to parameter %s.', zero, self)
        self.logger.debug('len(self.tf_zeros): %s', len(self.tf_zeros))
        self.tf_zeros.append(zero)
        self.logger.debug('len(self.tf_zeros): %s', len(self.tf_zeros))

    def tf_add_complex_pole(self, pole):
        ''' Add a complec pole to the transfer function PAZ.

        '''
        self.tf_poles.append(pole)



## The station class.
#
class Station(object):

    ## The constructor.
    #
    # @param self The object pointer.
    def __init__(self, name, location, x, y, z,
            parent_network=None, coord_system=None, description=None, id=None,
            author_uri = None, agency_uri = None, creation_time = None):

        # The logger instance.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        ## The station id.
        self.id = id

        ## The station name.
        self.name = name

        ## The station location.
        if location:
            self.location = str(location)
        else:
            self.location = '--'

        ## The station description.
        #
        # The extended name of the station.
        self.description = description

        ## The x coordinate of the station location.
        #
        # The coordinate system used is a right handed coordinate system with 
        # x pointing in the East direction, y pointing in the North direction and 
        # z pointing upwards.@n 
        # Depending on the coordinate system used x and y can also represent 
        # longitude and latitude.
        if x is None:
            raise ValueError("The x coordinate can't be None.")
        self.x = float(x)

        ## The y coordinate of the station location.
        #
        # The coordinate system used is a right handed coordinate system with 
        # x pointing in the East direction, y pointing in the North direction and 
        # z pointing upwards.@n 
        # Depending on the coordinate system used x and y can also represent 
        # longitude and latitude.
        if y is None:
            raise ValueError("The y coordinate can't be None.")
        self.y = float(y)

        ## The z coordinate of the station location.
        #
        # The coordinate system used is a right handed coordinate system with 
        # x pointing in the East direction, y pointing in the North direction and 
        # z pointing upwards.@n 
        # Depending on the coordinate system used x and y can also represent 
        # longitude and latitude.
        if z is None:
            raise ValueError("The z coordinate can't be None.")
        self.z = float(z)

        ## The coordinate system in which the x/y coordinates are given.
        # 
        # The coord_system string should be a valid EPSG code.@n 
        # See http://www.epsg-registry.org/ to find your EPSG code.
        self.coord_system = coord_system

        # A list of tuples of channels assigned to the station.
        self.channels = []

        # The network containing this station.
        self.parent_network = parent_network

        # Indicates if the attributes have been changed.
        self.has_changed = False

        # The author.
        self.author_uri = author_uri

        # The agency of the author.
        self.agency_uri = agency_uri

        # The datetime of the creation.
        if creation_time == None:
            self.creation_time = UTCDateTime();
        else:
            self.creation_time = UTCDateTime(creation_time);

    @property
    def network(self):
        if self.parent_network is not None:
            return self.parent_network.name
        else:
            return None

    @property
    def nsl(self):
        return (self.network, self.name, self.location)

    @property
    def nsl_string(self):
        return str.join(':', self.nsl)

    @property
    def parent_inventory(self):
        if self.parent_network is not None:
            return self.parent_network.parent_inventory
        else:
            return None

    @property
    def location_string(self):
        if self.location is None:
            return '--'
        else:
            return self.location

    @property
    def available_channels_string(self):
        if self.channels:
            return str.join(',', sorted([x.name for x in self.channels]))
        else:
            return ''

    @property
    def assigned_recorders(self):
        ''' The unique recorders assigned to the station.
        '''
        recorders = []
        for cur_channel in self.channels:
            recorders.extend(cur_channel.assigned_recorders)

        recorders = list(set(recorders))

        return recorders

    @property
    def assigned_recorders_string(self):
        recorders = []
        for cur_channel in self.channels:
            recorders.extend(cur_channel.assigned_recorders)

        recorders = list(set(recorders))

        if recorders:
            return str.join(',', sorted(recorders))
        else:
            return ''

    @property
    def assigned_sensors_string(self):
        sensor_components = []
        for cur_channel in self.channels:
            for cur_stream in cur_channel.streams:
                for cur_component in cur_stream.components:
                    sensor_components.append(cur_component.item.serial)

        sensor_components = list(set(sensor_components))

        if sensor_components:
            return ','.join(sorted(sensor_components))
        else:
            return ''


    def __setitem__(self, name, value):
        self.logger.debug("Setting the %s attribute to %s.", name, value)
        self.__dict__[name] = value
        self.has_changed = True


    def __eq__(self, other):
        if type(self) is type(other):
            compare_attributes = ['id', 'name', 'location', 'description', 'x', 'y', 'z',
                                  'coord_system', 'channels', 'has_changed']
            for cur_attribute in compare_attributes:
                if getattr(self, cur_attribute) != getattr(other, cur_attribute):
                    self.logger.debug('Attribute %s not matching %s != %s.', cur_attribute, str(getattr(self, cur_attribute)), str(getattr(other, cur_attribute)))
                    return False

            return True
        else:
            return False

    def as_dict(self, style = None):
        export_attributes = ['name', 'location', 'description',
                             'x', 'y', 'z', 'coord_system',
                             'author_uri', 'agency_uri', 'creation_time']

        d = {}
        if style == 'seed':
            for cur_attr in export_attributes:
                d[cur_attr] = getattr(self, cur_attr)
            d['channels'] = []
            for cur_channel in self.channels:
                d['channels'].extend(cur_channel.as_dict(style = style))
        else:
            for cur_attr in export_attributes:
                d[cur_attr] = getattr(self, cur_attr)
            d['channels'] = [x.as_dict(style = style) for x in self.channels]
        return d


    def get_nslc(self):
        nslc = []
        for cur_sensor, start_time, end_time in self.sensors:
            cur_nslc = (self.network, self.station,
                        self.location, cur_sensor.channel_name)
            if cur_nslc not in nslc:
                nslc.append(cur_nslc)

        return nslc


    def get_lon_lat(self):
        '''
        Return the coordinate system as WGS84 longitude latitude tuples.
        '''
        # TODO: Add a check for valid epsg string.

        dest_sys = "epsg:4326"

        if self.coord_system == dest_sys:
            return(self.x, self.y)

        src_proj = pyproj.Proj(init=self.coord_system)
        dst_proj = pyproj.Proj(init=dest_sys) 


        lon, lat = pyproj.transform(src_proj, dst_proj, self.x, self.y)
        self.logger.debug('Converting from "%s" to "%s"', src_proj.srs, dst_proj.srs)
        return (lon, lat)


    def add_channel(self, cur_channel):
        ''' Add a channel to the station

        Parameters
        ----------
        cur_channel : :class:`Channel`
            The channel to add to the station.
        '''
        added_channel = None
        if not self.get_channel(name = cur_channel.name):
            cur_channel.parent_station = self
            self.channels.append(cur_channel)
            self.has_changed = True
            added_channel = cur_channel

        return added_channel


    def remove_channel_by_instance(self, channel):
        ''' Remove a channel instance from the station.
        '''
        if channel in self.channels:
            self.channels.remove(channel)



    def get_channel(self, **kwargs):
        ''' Get a channel from the stream.

        Parameters
        ----------
        name : String
            The name of the channel.
        '''
        ret_channel = self.channels

        valid_keys = ['name']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_channel = [x for x in ret_channel if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_channel


    def get_unique_channel_names(self):
        channel_names = []

        for cur_channel, start, end in self.channels:
            if cur_channel.name not in channel_names:
                channel_names.append(cur_channel.name)

        return channel_names


    def merge(self, merge_station):
        ''' Merge a station into the existing one.
        '''
        # Update the attributes.
        self.description = merge_station.description
        self.x = merge_station.x
        self.y = merge_station.y
        self.z = merge_station.z
        self.coord_system = merge_station.coord_system

        # Merge the channels.
        for cur_channel in merge_station.channels:
            exist_channel = self.get_channel(name = cur_channel.name)
            if not exist_channel:
                self.add_channel(cur_channel)
            else:
                exist_channel = exist_channel[0]
                exist_channel.merge(cur_channel)




class Channel(object):
    ''' A channel of a station.
    '''
    def __init__(self, name, description = None, id = None,
            agency_uri = None, author_uri = None, creation_time = None,
            parent_station = None):
        ''' Initialize the instance

        Parameters
        ----------
        name : String
            The name of the channel.

        streams : List of tuples.
            The streams assigned to the channel.

        '''
        # The logger instance.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # The database id of the channel.
        self.id = id

        # The name of the channel.
        self.name = name

        # The description of the channel.
        self.description = description

        # The streams assigned to the channel.
        self.streams = []

        # The station holding the channel.
        self.parent_station = parent_station

        # Indicates if the attributes have been changed.
        self.has_changed = False

        # The author.
        self.author_uri = author_uri

        # The agency of the author.
        self.agency_uri = agency_uri

        # The datetime of the creation.
        if creation_time == None:
            self.creation_time = UTCDateTime();
        else:
            self.creation_time = UTCDateTime(creation_time);

    @property
    def parent_inventory(self):
        if self.parent_station is not None:
            return self.parent_station.parent_inventory
        else:
            return None

    @property
    def nslc(self):
        if self.parent_station is not None:
            return (self.parent_station.network,
                    self.parent_station.name,
                    self.parent_station.location,
                    self.name)
        else:
            return None

    @property
    def nslc_string(self):
        return str.join(':', self.nslc)

    @property
    def assigned_recorders(self):
        return list(set([x.item.serial for x in self.streams]))


    def as_dict(self, style = None):
        export_attributes = ['name', 'description',
                             'author_uri', 'agency_uri', 'creation_time']


        if style == 'seed':
            d = []
            for cur_tb in self.streams:
                cur_stream = cur_tb.item
                cur_stream_start = cur_tb.start_time
                cur_stream_end = cur_tb.end_time
                for cur_comp_tb in cur_stream.components:
                    cur_comp = cur_comp_tb.item
                    cur_d = {}
                    cur_comp_start = cur_comp_tb.start_time
                    cur_comp_end = cur_comp_tb.end_time

                    if cur_comp_start is None:
                        cur_chan_start = cur_stream_start
                    elif cur_comp_start < cur_stream_start:
                        cur_chan_start = cur_stream_start
                    else:
                        cur_chan_start = cur_comp_start

                    if cur_comp_end is None:
                        cur_chan_end = cur_stream_end
                    elif cur_comp_end > cur_stream_end:
                        cur_chan_end = cur_stream_end
                    else:
                        cur_chan_end = cur_comp_end

                    cur_d['start_time'] = cur_chan_start
                    cur_d['end_time'] = cur_chan_end
                    cur_d['name'] = self.name
                    cur_d['description'] = self.description
                    cur_d['stream_name'] = cur_stream.name
                    cur_d['stream_label'] = cur_stream.label
                    cur_d['recorder_serial'] = cur_stream.serial
                    cur_d['recorder_model'] = cur_stream.model
                    cur_d['recorder_producer'] = cur_stream.producer
                    cur_d['sensor_serial'] = cur_comp.serial
                    cur_d['sensor_model'] = cur_comp.model
                    cur_d['sensor_producer'] = cur_comp.producer
                    d.append(cur_d)
        else:
            d = {}
            for cur_attr in export_attributes:
                d[cur_attr] = getattr(self, cur_attr)
            d['streams'] = []
            for cur_stream in self.streams:
                cur_d = {}
                cur_d['start_time'] = cur_stream.start_time
                cur_d['end_time'] = cur_stream.end_time
                cur_d.update(cur_stream.item.as_dict(style = style))
                d['streams'].append(cur_d)
        return d



    def add_stream(self, serial, model, producer, name, start_time, end_time):
        ''' Add a stream to the channel.

        Parameters
        ----------
        serial : String
            The serial number of the recorder containing the stream.

        model : String
            The model of the recorder containing the stream.

        producer : String
            The producer of the recorder containing the stream.

        name : String
            The name of the stream.

        start_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time from which on the stream has been operating at the channel.

        end_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time up to which the stream has been operating at the channel. "None" if the channel is still running.
        '''
        if self.parent_inventory is None:
            raise RuntimeError('The channel needs to be part of an inventory before a sensor can be added.')

        added_stream = None
        cur_stream = self.parent_inventory.get_stream(serial = serial,
                                                      model = model,
                                                      producer = producer,
                                                      name = name)

        if not cur_stream:
            self.logger.error('The specified stream (serial = %s, model = %s, producer = %s, name = %s) was not found in the inventory.',
                              serial, model, producer, name)
        elif len(cur_stream) == 1:
            cur_stream = cur_stream[0]

            try:
                start_time = UTCDateTime(start_time)
            except:
                start_time = None

            try:
                end_time = UTCDateTime(end_time)
            except:
                end_time = None

            if self.get_stream(serial = serial,
                               model = model,
                               producer = producer,
                               name = name,
                               start_time = start_time,
                               end_time = end_time):
                # The stream is already assigned to the station for this
                # time-span.
                if start_time is not None:
                    start_string = start_time.isoformat
                else:
                    start_string = 'big bang'

                if end_time is not None:
                    end_string = end_time.isoformat
                else:
                    end_string = 'running'

                self.logger.error('The stream (serial: %s,  name: %s) is already deployed during the specified timespan from %s to %s.', serial, name, start_string, end_string)
            else:
                self.streams.append(TimeBox(item = cur_stream,
                                            start_time = start_time,
                                            end_time = end_time,
                                            parent = self))
                self.has_changed = True
                added_stream = cur_stream

        return added_stream


    def remove_stream_by_instance(self, stream_timebox):
        ''' Remove a stream timebox.
        '''
        self.streams.remove(stream_timebox)


    def remove_stream(self, start_time = None, end_time = None, **kwargs):
        ''' Remove a stream from the channel.

        Parameters
        ----------
        '''
        stream_tb_to_remove = self.get_stream(start_time = start_time,
                                            end_time = end_time,
                                            **kwargs)

        for cur_stream_tb in stream_tb_to_remove:
            self.streams.remove(cur_stream_tb)


    def get_stream(self, start_time = None, end_time = None, **kwargs):
        ''' Get a stream from the channel.

        Parameters
        ----------
        serial : String
            The serial number of the recorder containing the stream.

        model : String
            The model of the recorder containing the stream.

        producer : String
            The producer of the recorder containing the stream.

        name : String
            The name of the stream.
        '''
        ret_stream = self.streams

        valid_keys = ['serial', 'model', 'producer', 'name', 'id']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_stream = [x for x in ret_stream if hasattr(x.item, cur_key) and getattr(x.item, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        if start_time is not None:
            ret_stream = [x for x in ret_stream if (x.end_time is None) or (x.end_time >= start_time)]

        if end_time is not None:
            ret_stream = [x for x in ret_stream if x.start_time <= end_time]

        return ret_stream


    def merge(self, merge_channel):
        ''' Merge a channel with the existing one.
        '''
        # Update the attributes.
        self.description = merge_channel.description

        # Replace existing streams with the new ones.
        for cur_stream in [x for x in self.streams]:
            self.remove_stream_by_instance(cur_stream)

        for cur_stream in merge_channel.streams:
            self.add_stream(serial = cur_stream.serial,
                            model = cur_stream.model,
                            producer = cur_stream.producer,
                            name = cur_stream.name,
                            start_time = cur_stream.start_time,
                            end_time = cur_stream.end_time)

    # TODO: Implement the methods to change the stream start- and end-time.
    # This will be analog to the sensors in the streams. It would be great to
    # have these methods in the TimeBox class.
    # Keep in mind, that in the DbInventory, the ORM mapper values of the 
    # time-spans have to be changed as well.




class Network(object):

    def __init__(self, name, description=None, type=None, author_uri = None,
            agency_uri = None, creation_time = None, parent_inventory=None):
        ''' Initialize the instance.
        '''
        # The logger instance.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        ## The parent inventory.
        self.parent_inventory = parent_inventory

        ## The network name (code).
        self.name = name

        ## The network description.
        self.description = description

        ## The network type.
        self.type = type

        ## The stations contained in the network.
        self.stations = []

        # Indicates if the attributes have been changed.
        self.has_changed = False

        # The author of the network.
        self.author_uri = author_uri

        # The agency of the author.
        self.agency_uri = agency_uri

        # The datetime of the creation.
        if creation_time == None:
            self.creation_time = UTCDateTime();
        else:
            self.creation_time = UTCDateTime(creation_time);


    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        self.__dict__[attr] = value

        self.__dict__['has_changed'] = True


    def __eq__(self, other):
        if type(self) is type(other):
            compare_attributes = ['name', 'type', 'description', 'has_changed', 'stations'] 
            for cur_attribute in compare_attributes:
                if getattr(self, cur_attribute) != getattr(other, cur_attribute):
                    return False

            return True
        else:
            return False

    def as_dict(self, style = None):
        export_attributes = ['name', 'description', 'type',
                             'author_uri', 'agency_uri', 'creation_time']
        d = {}
        for cur_attr in export_attributes:
            d[cur_attr] = getattr(self, cur_attr)
        d['stations'] = [x.as_dict(style = style) for x in self.stations]
        return d




    def add_station(self, station):
        ''' Add a station to the network.

        Parameters
        ----------
        station : :class:`Station`
            The station instance to add to the network.
        '''
        available_sl = [(x.name, x.location) for x in self.stations]
        if((station.name, station.location) not in available_sl):
            station.parent_network = self
            self.stations.append(station)
            return station
        else:
            self.logger.error("The station with SL code %s is already in the network.", station.name + ':' + station.location)
            return None


    def remove_station_by_instance(self, station_to_remove):
        ''' Remove a station instance from the network.
        '''
        if station_to_remove in self.stations:
            self.stations.remove(station_to_remove)


    def remove_station(self, name, location):
        ''' Remove a station from the network.

        Parameters
        ----------
        name : String
            The name of the station to remove.

        location : String
            The location of the station to remove.
        '''
        station_2_remove = [x for x in self.stations if x.name == name and x.location == location]

        removed_station = None
        if len(station_2_remove) == 0:
            removed_station = None
        elif len(station_2_remove) == 1:
            station_2_remove = station_2_remove[0]
            self.stations.remove(station_2_remove)
            station_2_remove.network = None
            station_2_remove.parent_network = None
            removed_station = station_2_remove
        else:
            # This shouldn't happen.
            self.logger.error('Found more than one network with the name %s.', name)
            return None

        return removed_station


    def get_station(self, **kwargs):
        ''' Get a station from the network.

        Parameters
        ----------
        name : String
            The name (code) of the station.

        location : String
            The location code of the station.

        id : Integer
            The database id of the station.

        nsl : Tuple (network, station, location)
            The NSL tuple of the station.

        nsl_string : String
            The NSL string in the format 'network:station:location'.
        '''
        ret_station = self.stations

        valid_keys = ['name', 'network', 'location', 'id', 'nsl', 'nsl_string']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_station = [x for x in ret_station if getattr(x, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        return ret_station


    def merge(self, merge_network):
        ''' Merge a network with the existing.
        '''
        # Update the attributes.
        self.description = merge_network.description
        self.type = merge_network.type

        # Merge the stations.
        for cur_station in merge_network.stations:
            exist_station = self.get_station(name = cur_station.name,
                                             location = cur_station.location)
            if not exist_station:
                self.add_station(cur_station)
            else:
                exist_station = exist_station[0]
                exist_station.merge(cur_station)



class Array(object):
    ''' An array holding multiple stations.
    '''

    def __init__(self, name, description = None, author_uri = None,
                 agency_uri = None, creation_time = None, parent_inventory = None):
        ''' Initialization of the instance.
        '''
        # The logging logger instance.
        logger_prefix = 'mss_data_server'
        loggerName = logger_prefix + "." + __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(loggerName)

        # The unique name of the array.
        self.name = name

        # The description of the array.
        self.description = description

        ## The stations contained in the array.
        self.stations = []

        # Indicates if the attributes have been changed.
        self.has_changed = False

        # The author.
        self.author_uri = author_uri

        # The agency of the author.
        self.agency_uri = agency_uri

        # The datetime of the creation.
        if creation_time == None:
            self.creation_time = UTCDateTime();
        else:
            self.creation_time = UTCDateTime(creation_time);

        # The parent recorder holding the stream.
        self.parent_inventory = parent_inventory


    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        self.__dict__[attr] = value

        self.__dict__['has_changed'] = True


    def __eq__(self, other):
        if type(self) is type(other):
            compare_attributes = ['name', 'description', 'has_changed', 'stations']
            for cur_attribute in compare_attributes:
                if getattr(self, cur_attribute) != getattr(other, cur_attribute):
                    return False

            return True
        else:
            return False


    def add_station(self, station, start_time, end_time):
        ''' Add a station to the array.

        Parameters
        ----------
        station : :class:`Station`
            The station instance to add to the network.

        start_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time from which on the stream has been operating at the channel.

        end_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time up to which the stream has been operating at the channel. "None" if the channel is still running.
        '''
        if self.parent_inventory != station.parent_inventory:
            raise RuntimeError('The station and the array have to be in the same inventory.')

        added_station = None
        try:
            start_time = UTCDateTime(start_time)
        except:
            start_time = None

        try:
            end_time = UTCDateTime(end_time)
        except:
            end_time = None

        if self.get_station(nsl = station.nsl):
                # The station is already assigned to the array for this
                # time-span.
                if start_time is not None:
                    start_string = start_time.isoformat
                else:
                    start_string = 'big bang'

                if end_time is not None:
                    end_string = end_time.isoformat
                else:
                    end_string = 'running'

                self.logger.error('The station %s is already deployed during the specified timespan from %s to %s.', station.nsl_string, start_string, end_string)
        else:
            self.stations.append(TimeBox(item = station,
                                         start_time = start_time,
                                         end_time = end_time,
                                         parent = self))
            self.has_changed = True
            added_station = station

        return added_station


    def remove_station_by_instance(self, station_timebox):
        ''' Remove a station timebox instance.
        '''
        self.stations.remove(station_timebox)


    def remove_station(self, start_time = None, end_time = None, **kwargs):
        ''' Remove a station from the array.

        Parameters
        ----------
        start_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time from which on the stream has been operating at the channel.

        end_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time up to which the stream has been operating at the channel. "None" if the channel is still running.
        '''
        stat_to_remove = self.get_station(start_time = start_time,
                                             end_time = end_time,
                                             **kwargs)
        for cur_stat in stat_to_remove:
            self.stations.remove(cur_stat)

        return stat_to_remove



    def get_station(self, start_time = None, end_time = None, **kwargs):
        ''' Get a station from the network.

        Parameters
        ----------
        start_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time from which on the stream has been operating at the channel.

        end_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time up to which the stream has been operating at the channel. "None" if the channel is still running.

        name : String
            The name (code) of the station.

        location : String
            The location code of the station.

        id : Integer
            The database id of the station.

        nsl : Tuple (network, station, location)
            The NSL tuple of the station.

        nsl_string : String
            The NSL string in the format 'station:network:location'.
        '''
        ret_station = self.stations

        valid_keys = ['name', 'network', 'location', 'id', 'nsl', 'nsl_string']

        for cur_key, cur_value in kwargs.items():
            if cur_key in valid_keys:
                ret_station = [x for x in ret_station if hasattr(x.item, cur_key) and getattr(x.item, cur_key) == cur_value]
            else:
                warnings.warn('Search attribute %s is not existing.' % cur_key, RuntimeWarning)

        if start_time is not None:
            ret_station = [x for x in ret_station if (x.end_time is None) or (x.end_time >= start_time)]

        if end_time is not None:
            ret_station = [x for x in ret_station if x.start_time <= end_time]


        return ret_station


    def merge(self, merge_station):
        ''' Merge an array with the existing one.
        '''
        # Update the attributes.
        self.description = merge_station.description

        # Replace the assigned stations with the new ones.
        for cur_station in [x for x in self.stations]:
            self.remove_station_by_instance(cur_station)

        for cur_station in merge_station.stations:
            self.add_station(station = cur_station.item,
                             start_time = cur_station.start_time,
                             end_time = cur_station.end_time)




class TimeBox(object):
    ''' A container to hold an instance with an assigned time-span.

    '''

    def __init__(self, item, start_time, end_time = None, parent = None):
        ''' Initialize the instance.

        Parameters
        ----------

        '''
        # The instance holding the time-box.
        self.parent = parent

        # The item contained in the box.
        self.item = item

        # The start_time of the time-span.
        self.start_time = start_time

        # The end_time of the time-span.
        self.end_time = end_time


    def __eq__(self, other):
        is_equal = False
        try:
            if self.item is other.item:
                if self.start_time == other.start_time:
                    if self.end_time == other.end_time:
                        is_equal = True
        except:
            pass

        return is_equal


    def __getattr__(self, attr_name):
        return getattr(self.item, attr_name)


    @property
    def start_time_string(self):
        if self.start_time is None:
            return 'big bang'
        else:
            return self.start_time.isoformat()

    @property
    def end_time_string(self):
        if self.end_time is None:
            return 'running'
        else:
            return self.end_time.isoformat()

    def as_dict(self, style = None):
        export_attributes = ['start_time',
                             'end_time']

        d = {}
        for cur_attr in export_attributes:
            d[cur_attr] = getattr(self, cur_attr)
        d['item'] = self.item.as_dict(style = style)
        return d
