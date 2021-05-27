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
The database inventory module. 

This module contains the extension to the Inventory classes used to handle an 
inventory in a mariaDB database. All classes inherit from the related inventory 
classes and add functionality to write and retrieve data from the database.
'''

from obspy.core.utcdatetime import UTCDateTime

from mss_dataserver.geometry.inventory import Inventory
from mss_dataserver.geometry.inventory import Network
from mss_dataserver.geometry.inventory import Array
from mss_dataserver.geometry.inventory import Station
from mss_dataserver.geometry.inventory import Channel
from mss_dataserver.geometry.inventory import Recorder
from mss_dataserver.geometry.inventory import Sensor
from mss_dataserver.geometry.inventory import SensorComponent
from mss_dataserver.geometry.inventory import SensorComponentParameter
from mss_dataserver.geometry.inventory import RecorderStream
from mss_dataserver.geometry.inventory import RecorderStreamParameter


class DbInventory(Inventory):
    ''' The database inventory.

    Parameters
    ----------
    project: :class:`mss_dataserver.core.project.Project`
        The project used to get the database connection and database
        session.

    name: str 
        The name of the inventory.

    Arguments
    ---------
    db_session: :class:`sqlalchemy.orm.Session`
        The session used to communicate with the database.
    '''

    def __init__(self, project, name = 'db_inventory'):
        ''' Initialize the instance.
        '''
        Inventory.__init__(self, name = name, type = 'db')

        # The pSysmon project containing the inventory.
        self.project = project

        # The pSysmon database session.
        self.db_session = self.project.get_db_session()



    def __del__(self):
        ''' Clean up the database connection.
        '''
        self.db_session.close()


    def close(self):
        ''' Close the inventory database connection.
        '''
        self.db_session.close()


    def add_recorder(self, recorder):
        ''' Add a recorder to the inventory.

        Parameters
        ----------
        recorder: :class:`mss_dataserver.geometry.inventory.Recorder` or :class:`DbRecorder`
            The recorder to add to the inventory.

        
        Returns
        -------
        :class:`DbRecorder`
            The recorder added.
        '''
        if recorder.__class__ is Recorder:
            recorder = DbRecorder.from_inventory_instance(self, recorder)

        added_recorder = Inventory.add_recorder(self, recorder)
        if added_recorder is not None:
            self.db_session.add(added_recorder.orm)

        return added_recorder


    def add_sensor(self, sensor):
        ''' Add a sensor to the inventory.

        Parameters
        ----------
        sensor: :class:`mss_dataserver.geometry.inventory.Sensor` or :class:`DbSensor`
            The sensor to add to the inventory.


        Returns
        -------
        :class:`DbSensor`
            The sensor added.
        '''
        if sensor.__class__ is Sensor:
            sensor = DbSensor.from_inventory_instance(self, sensor)

        added_sensor = Inventory.add_sensor(self, sensor)
        if added_sensor is not None:
            self.db_session.add(added_sensor.orm)

        return added_sensor


    def add_network(self, network):
        ''' Add a new network to the database inventory.

        Parameters
        ----------
        network: :class:`mss_dataserver.geometry.inventory.Network` or :class:`DbNetwork`
            The network to add to the database inventory.


        Returns
        -------
        :class:`DbNetwork`
            The network added.
        '''
        if network.__class__ is Network:
            network = DbNetwork.from_inventory_instance(self, network)

        added_network =  Inventory.add_network(self, network)
        if added_network is not None:
            self.db_session.add(added_network.orm)

        return added_network


    def remove_network(self, name):
        ''' Remove a network from the database inventory.

        Parameters
        ----------
        name: str
            The name of the network to remove.

        Returns
        -------
        :class:`DbNetwork`
            The removed network.
        '''
        removed_network = Inventory.remove_network(self, name)
        if removed_network is not None:
            self.db_session.expunge(removed_network.orm)

        return removed_network


    def add_array(self, array):
        ''' Add a new array to the database inventory.

        Parameters
        ----------
        array: :class:`mss_dataserver.geometry.inventory.Array` of :class:`DbArray`
            The array to add to the database inventory.

        Returns
        -------
        :class:`DbArray`
            The added array.
        '''
        if array.__class__ is Array:
            array = DbArray.from_inventory_instance(self, array)

        added_array =  Inventory.add_array(self, array)
        if added_array is not None:
            self.db_session.add(added_array.orm)

        return added_array


    def load_networks(self):
        ''' Load the networks from the database.
        '''
        network_table = self.project.db_tables['geom_network']
        for cur_network_orm in self.db_session.query(network_table).order_by(network_table.name):
            network = DbNetwork.from_sqlalchemy_orm(self, cur_network_orm)

            #    for cur_geom_sensor in cur_geom_station.sensors:
            #        db_sensor = self.get_sensor(id = cur_geom_sensor.sensor_id)
            #        if len(db_sensor) == 1:
            #            if cur_geom_sensor.start_time is not None:
            #                start_time = UTCDateTime(cur_geom_sensor.start_time)
            #            else:
            #                start_time = None

            #            if cur_geom_sensor.end_time is not None:
            #                end_time = UTCDateTime(cur_geom_sensor.end_time)
            #            else:
            #                end_time = None
            #            db_station.sensors.append((db_sensor[0], start_time, end_time))

            self.networks.append(network)


    def load_sensors(self):
        ''' Load the sensors from the database.

        '''
        sensor_table = self.project.db_tables['geom_sensor']
        for cur_sensor_orm in self.db_session.query(sensor_table).order_by(sensor_table.serial):
            sensor = DbSensor.from_sqlalchemy_orm(self, cur_sensor_orm)
            self.sensors.append(sensor)


    def load_recorders(self):
        ''' Load the recorders from the database.
        '''
        geom_recorder_orm = self.project.db_tables['geom_recorder']
        for cur_geom_recorder in self.db_session.query(geom_recorder_orm).order_by(geom_recorder_orm.serial):
            db_recorder = DbRecorder.from_sqlalchemy_orm(self, cur_geom_recorder)

            self.recorders.append(db_recorder)


    def load_arrays(self):
        ''' Load the arrays from the database.
        '''
        array_table = self.project.db_tables['geom_array']
        for cur_geom_array in self.db_session.query(array_table).order_by(array_table.name):
            db_array = DbArray.from_sqlalchemy_orm(self, cur_geom_array)
            self.arrays.append(db_array)


    def load(self):
        ''' Load the inventory from the database.
        '''
        try:
            self.load_sensors()
            self.load_recorders()
            self.load_networks()
            self.load_arrays()
        except Exception:
            self.logger.exception("Error loading the geometry from the database.")

    def commit(self):
        ''' Commit the database changes.
        '''
        self.db_session.commit()


    @classmethod
    def from_inventory_instance(cls, name, project, inventory):
        ''' Create a :class:`DbInventory` instance from a :class:`mss_data_server.geometry.inventory.Inventory` instance.

        Parameters
        ----------
        name: str
            The name of the inventory.

        project: :class:`mss_data_server.core.project.Project`
            The project used to communicate with the database.

        inventory: :class:`mss_dataserver.geometry.inventory.Inventory`
            The inventory to convert to the :class:`DbInventory` instance.


        Returns
        -------
        :class:`DbInventory`
            The created instance.
        '''
        db_inventory = cls(name = name, project = project)
        for cur_sensor in inventory.sensors:
            db_inventory.add_sensor(cur_sensor)

        for cur_recorder in inventory.recorders:
            db_inventory.add_recorder(cur_recorder)

        for cur_network in inventory.networks:
            db_inventory.add_network(cur_network)

        for cur_array in inventory.arrays:
            db_inventory.add_array(cur_array)

        return db_inventory


    @classmethod
    def load_inventory(cls, project):
        ''' Load an inventory from the database.
        
        Parameters
        ----------
        project: :class:`mss_data_server.core.project.Project`
            The project used to communicate with the database.
        '''
        db_inventory = cls(name = 'db_inventory', project = project)
        db_inventory.load_sensors()
        db_inventory.load_recorders()
        db_inventory.load_networks()
        db_inventory.load_arrays()
        db_inventory.close()

        return db_inventory




class DbNetwork(Network):
    ''' The database network.

    Parameters
    ----------
    orm: :class:`mss_data_server.geometry.databaseFactory.GeomNetwork`
        The sqlalchemy table mapper class for the network.

    
    Keyword Arguments
    -----------------
    **kwargs
        The keyword arguments passed to the __init__ method of the parent class :class:`mss_dataserver.geoemetry.inventory.Network`.
    '''

    def __init__(self, orm = None, **kwargs):
        Network.__init__(self, **kwargs)

        if orm is None:
            # Create a new database network instance.
            orm_class = self.parent_inventory.project.db_tables['geom_network']
            self.orm = orm_class(name = self.name,
                                 description = self.description,
                                 type = self.type,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri,
                                 creation_time = self.creation_time)
        else:
            self.orm = orm

    @property
    def id(self):
        ''' int: The database id.
        '''
        if self.orm is not None:
            return self.orm.id
        else:
            return None


    @classmethod
    def from_sqlalchemy_orm(cls, parent_inventory, orm):
        ''' Create a :class:`DbNetwork` instance from a :class:`mss_data_server.geometry.databaseFactory.GeomNetwork` table mapper class.

        Parameters
        ----------
        parent_inventory: :class:`mss_data_server.geometry.inventory.Inventory`
            The parent inventory to which the instance is related.

        orm: :class:`mss_data_server.geometry.databaseFactory.GeomNetwork`
            The sqlalchemy table mapper class for the network.
        
        Returns
        -------
        :class:`DbNetwork`
            The created instance.
        '''
        network = cls(parent_inventory = parent_inventory,
                      name = orm.name,
                      description = orm.description,
                      type = orm.type,
                      author_uri = orm.author_uri,
                      agency_uri = orm.agency_uri,
                      creation_time = orm.creation_time,
                      orm = orm)

        for cur_station in orm.stations:
            network.add_station(DbStation.from_sqlalchemy_orm(network, cur_station))

        return network

    @classmethod
    def from_inventory_instance(cls, parent_inventory, instance):
        ''' Create a :class:`DbNetwork` instance from a :class:`mss_data_server.geometry.inventory.Network` instance.

        Parameters
        ----------
        parent_inventory: :class:`mss_data_server.geometry.inventory.Inventory`
            The parent inventory to which the DbNetwork is related.

        instance: :class:`mss_data_server.geometry.inventory.Network`
            The instance used to create the :class:`DbNetwork` instance.
        
        Returns
        -------
        :class:`DbNetwork`
            The created instance.
        '''
        cur_network =  cls(parent_inventory = parent_inventory,
                           name = instance.name,
                           description = instance.description,
                           type = instance.type,
                           author_uri = instance.author_uri,
                           agency_uri = instance.agency_uri,
                           creation_time = instance.creation_time)

        for cur_station in instance.stations:
            cur_network.add_station(cur_station)

        return cur_network



    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        Network.__setattr__(self, attr, value)
        attr_map = {};
        attr_map['name'] = 'name'
        attr_map['description'] = 'description'
        attr_map['type'] = 'type'
        attr_map['author_uri'] = 'author_uri'
        attr_map['agency_uri'] = 'agency_uri'
        attr_map['creation_time'] = 'creation_time'

        if attr in iter(attr_map.keys()):
            if 'orm' in self.__dict__:
                setattr(self.orm, attr_map[attr], value)



    def add_station(self, station):
        ''' Add a station to the network.

        Parameters
        ----------
        station: :class:`DbStation`
            The station instance to add to the network.

        Returns
        -------
        :class:`DbStation`
            The added station.
        '''
        if station.__class__ is Station:
            station = DbStation.from_inventory_instance(self, station)

        added_station = Network.add_station(self, station)
        if added_station is not None:
            if station.orm not in self.orm.stations:
                self.orm.stations.append(station.orm)

        return added_station



    def remove_station(self, name, location):
        ''' Remove a station from the network.

        Parameters
        ----------
        name: str 
            The name of the station to remove.

        location: str
            The location of the station to remove.


        Returns
        -------
        :class:`DbStation`
            The removed station.
        '''
        removed_station = Network.remove_station(self, name = name, location = location)

        if removed_station is not None:
            self.orm.stations.remove(removed_station.orm)
            self.parent_inventory.db_session.expunge(removed_station.orm)

        return removed_station




class DbArray(Array):
    ''' The database Array.
    
    Parameters
    ----------
    orm: :class:`mss_data_server.geometry.databaseFactory.GeomArray
        The sqlalchemy table mapper class for the array.


    Keyword Arguments
    -----------------
    **kwargs
        The keyword arguments passed to the __init__ method of the parent class :class:`mss_dataserver.geoemetry.inventory.Array`.
    '''

    def __init__(self, orm = None, **kwargs):
        ''' Initialize the instance.
        '''
        Array.__init__(self, **kwargs)

        if orm is None:
            # Create a new database array instance.
            orm_class = self.parent_inventory.project.db_tables['geom_array']
            self.orm = orm_class(name = self.name,
                                 description = self.description,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri,
                                 creation_time = self.creation_time)
        else:
            self.orm = orm


    @classmethod
    def from_sqlalchemy_orm(cls, parent_inventory, orm):
        ''' Create a :class:`DbArray` instance from a :class:`mss_data_server.geometry.databaseFactory.GeomArray` table mapper class.

        Parameters
        ----------
        parent_inventory: :class:`mss_data_server.geometry.inventory.Inventory`
            The parent inventory to which the instance is related.

        orm: :class:`mss_data_server.geometry.databaseFactory.GeomArray`
            The sqlalchemy table mapper class for the array.

        Returns
        -------
        :class:`DbArray`
            The created instance.

        '''
        array = cls(parent_inventory = parent_inventory,
                    name = orm.name,
                    description = orm.description,
                    author_uri = orm.author_uri,
                    agency_uri = orm.agency_uri,
                    creation_time = orm.creation_time,
                    orm = orm)

        for cur_stat_to_array in orm.stations:
            cur_station = cur_stat_to_array.station
            db_station = parent_inventory.get_station(network = cur_station.network,
                                                      name = cur_station.name,
                                                      location = cur_station.location)[0]
            array.add_station(station = db_station,
                              start_time = cur_stat_to_array.start_time,
                              end_time = cur_stat_to_array.end_time,
                              ignore_orm = True)

        return array

    @classmethod
    def from_inventory_instance(cls, parent_inventory, instance):
        ''' Create a :class:`DbArray` instance from a :class:`mss_data_server.geometry.inventory.Array` instance.
        Parameters
        ----------
        parent_inventory: :class:`mss_data_server.geometry.inventory.Inventory`
            The parent inventory to which the instance is related.

         instance: :class:`mss_data_server.geometry.inventory.Array`
            The instance used to create the :class:`DbArray` instance.

        Returns
        -------
        :class:`DbArray`
            The created array.
        '''
        array = cls(parent_inventory = parent_inventory,
                    name = instance.name,
                    description = instance.description,
                    author_uri = instance.author_uri,
                    agency_uri = instance.agency_uri,
                    creation_time = instance.creation_time)

        for cur_station_tb in instance.stations:
            db_station = parent_inventory.get_station(name = cur_station_tb.name,
                                                      network = cur_station_tb.network,
                                                      location = cur_station_tb.location)[0]
            array.add_station(station = db_station,
                              start_time = cur_station_tb.start_time,
                              end_time = cur_station_tb.end_time)

        return array


    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        Array.__setattr__(self, attr, value)
        attr_map = {};
        attr_map['name'] = 'name'
        attr_map['description'] = 'description'
        attr_map['author_uri'] = 'author_uri'
        attr_map['agency_uri'] = 'agency_uri'
        attr_map['creation_time'] = 'creation_time'

        if attr in iter(attr_map.keys()):
            if 'orm' in self.__dict__:
                setattr(self.orm, attr_map[attr], value)


    def add_station(self, station, start_time, end_time, ignore_orm = False):
        ''' Add a station to the array.

        Parameters
        ----------
        station : :class:`DbStation`
            The station instance to add to the network.

        start_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time from which on the sensor has been operating at the station.

        end_time : :class:`obspy.core.utcdatetime.UTCDateTime`
            The time up to which the sensor has been operating at the station. "None" if the station is still running.

        ignore_orm : Boolean
            Control if the component assignment is added to the orm or not. This is usefull
            when creating an instance from a orm mapper using the from_sqlalchemy_orm
            class method.
        '''
        added_station = Array.add_station(self,
                                          station = station,
                                          start_time = start_time,
                                          end_time = end_time)

        if ignore_orm is False:
            if added_station is not None:
                if start_time is not None:
                    try:
                        start_time_timestamp = UTCDateTime(start_time).timestamp
                    except:
                        start_time_timestamp = None
                else:
                    start_time_timestamp = None

                if end_time is not None:
                    try:
                        end_time_timestamp = UTCDateTime(end_time).timestamp
                    except:
                        end_time_timestamp = None
                else:
                    end_time_timestamp = None

                orm_class = self.parent_inventory.project.db_tables['geom_stat_to_array']
                stat_to_array_orm = orm_class(self.name,
                                              added_station.id,
                                              start_time_timestamp,
                                              end_time_timestamp)
                stat_to_array_orm.station = added_station.orm
                self.orm.stations.append(stat_to_array_orm)

        return added_station


    def remove_station(self, start_time = None, end_time = None, **kwargs):
        ''' Remove a station from the array.
        '''
        removed_stations = Array.remove_station(self,
                                               start_time = start_time,
                                               end_time = end_time,
                                               **kwargs)
        for cur_stat in removed_stations:
            self.orm.stations.remove(cur_stat.orm)
            self.parent_inventory.db_session.expunge(cur_stat.orm)

        return removed_stations


class DbStation(Station):
    ''' The database station.

    Parameters
    ----------
    orm: :class:`mss_data_server.geometry.databaseFactory.GeomStation`
        The sqlalchemy table mapper class for the network.

    
    Keyword Arguments
    -----------------
    **kwargs
        The keyword arguments passed to the __init__ method of the parent class :class:`mss_dataserver.geoemetry.inventory.Station`.
    '''
    
    def __init__(self, orm = None, **kwargs):
        Station.__init__(self, **kwargs)

        if orm is None:
            # Create a new database station instance.
            orm_class = self.parent_inventory.project.db_tables['geom_station']
            self.orm = orm_class(name = self.name,
                                 location = self.location,
                                 x = self.x,
                                 y = self.y,
                                 z = self.z,
                                 coord_system = self.coord_system,
                                 description = self.description,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri,
                                 creation_time = self.creation_time)
        else:
            self.orm = orm

    @property
    def id(self):
        ''' int: The database id.
        '''
        if self.orm is not None:
            return self.orm.id
        else:
            return None


    @classmethod
    def from_sqlalchemy_orm(cls, parent_network, orm):
        ''' Create a :class:`DbStation` instance from a :class:`mss_data_server.geometry.databaseFactory.GeomStation` table mapper class.

        Parameters
        ----------
        parent_inventory: :class:`mss_data_server.geometry.inventory.Network`
            The parent network to which the instance is related.

        orm: :class:`mss_data_server.geometry.databaseFactory.GeomStation`
            The sqlalchemy table mapper class for the network.
        
        Returns
        -------
        :class:`DbStation`
            The created instance.
        '''
        station = cls(parent_network = parent_network,
                      name = orm.name,
                      location = orm.location,
                      x = orm.x,
                      y = orm.y,
                      z = orm.z,
                      coord_system = orm.coord_system,
                      description = orm.description,
                      author_uri = orm.author_uri,
                      agency_uri = orm.agency_uri,
                      creation_time = orm.creation_time,
                      orm = orm)

        for cur_channel in orm.channels:
            station.add_channel(DbChannel.from_sqlalchemy_orm(station, cur_channel))

        return station



    @classmethod
    def from_inventory_instance(cls, parent_network, instance):
        ''' Create a :class:`DbStation` instance from a :class:`mss_data_server.geometry.inventory.Station` instance.

        Parameters
        ----------
        parent_network: :class:`mss_data_server.geometry.inventory.Network`
            The parent network to which the DbStation is related.

        instance: :class:`mss_data_server.geometry.inventory.Station`
            The instance used to create the :class:`DbStation` instance.
        
        Returns
        -------
        :class:`DbSation`
            The created station.
        '''
        station =  cls(parent_network = parent_network,
                       name = instance.name,
                       location = instance.location,
                       x = instance.x,
                       y = instance.y,
                       z = instance.z,
                       coord_system = instance.coord_system,
                       description = instance.description,
                       author_uri = instance.author_uri,
                       agency_uri = instance.agency_uri,
                       creation_time = instance.creation_time)

        for cur_channel in instance.channels:
            station.add_channel(cur_channel)

        return station


    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        attr_map = {};
        attr_map['name'] = 'name'
        attr_map['location'] = 'location'
        attr_map['x'] = 'x'
        attr_map['y'] = 'y'
        attr_map['z'] = 'z'
        attr_map['coord_system'] = 'coord_system'
        attr_map['description'] = 'description'
        attr_map['author_uri'] = 'author_uri'
        attr_map['agency_uri'] = 'agency_uri'
        attr_map['creation_time'] = 'creation_time'

        self.__dict__[attr] = value

        if attr in iter(attr_map.keys()):
            if 'orm' in self.__dict__:
                setattr(self.orm, attr_map[attr], value)

        self.__dict__['has_changed'] = True


    def add_channel(self, cur_channel):
        ''' Add a channel to the station.

        Parameters
        ----------
        cur_channel: :class:`DbChannel`
            The channel to add to the station.

        Returns
        -------
        :class:`DbChannel`
            The added channel.
        '''
        if cur_channel.__class__ is Channel:
            cur_channel = DbChannel.from_inventory_instance(self, cur_channel)

        added_channel = Station.add_channel(self, cur_channel)
        if added_channel is not None:
            if cur_channel.orm not in self.orm.channels:
                self.orm.channels.append(cur_channel.orm)

        return added_channel




class DbRecorder(Recorder):
    ''' The database recorder.

    Parameters
    ----------
    orm: :class:`mss_data_server.geometry.databaseFactory.GeomRecorder`
        The sqlalchemy table mapper class.

    
    Keyword Arguments
    -----------------
    **kwargs
        The keyword arguments passed to the __init__ method of the parent class :class:`mss_dataserver.geoemetry.inventory.Recorder`.
    '''

    def __init__(self, orm = None, **kwargs):
        Recorder.__init__(self, **kwargs)

        if orm is None:
            # Create a new database recorder instance.
            orm_class = self.parent_inventory.project.db_tables['geom_recorder']
            self.orm = orm_class(serial = self.serial,
                                 model = self.model,
                                 producer = self.producer,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri,
                                 creation_time = self.creation_time)
        else:
            self.orm = orm

    @property
    def id(self):
        '''int: The database id.
        '''
        if self.orm is not None:
            return self.orm.id
        else:
            return None


    @classmethod
    def from_sqlalchemy_orm(cls, parent_inventory, orm):
        ''' Create a :class:`DbRecorder` instance from a :class:`mss_data_server.geometry.databaseFactory.GeomRecorder` table mapper class.

        Parameters
        ----------
        parent_inventory: :class:`mss_data_server.geometry.inventory.Inventory`
            The parent inventory to which the instance is related.

        orm: :class:`mss_data_server.geometry.databaseFactory.GeomStation`
            The sqlalchemy table mapper class for the network.
        
        Returns
        -------
        :class:`DbRecorder`
            The created instance.
        '''
        recorder = cls(parent_inventory = parent_inventory,
                       serial = orm.serial,
                       model = orm.model,
                       producer = orm.producer,
                       description = orm.description,
                       author_uri = orm.author_uri,
                       agency_uri = orm.agency_uri,
                       creation_time = orm.creation_time,
                       orm = orm)

        for cur_stream in orm.streams:
            db_stream = DbRecorderStream.from_sqlalchemy_orm(recorder, cur_stream)
            recorder.add_stream(db_stream)

        return recorder


    @classmethod
    def from_inventory_instance(cls, parent_inventory, instance):
        ''' Create a :class:`DbRecorder` instance from a :class:`mss_data_server.geometry.inventory.Recorder` instance.

        Parameters
        ----------
        parent_network: :class:`mss_data_server.geometry.inventory.Inventory`
            The parent inventory to which the DbStation is related.

        instance: :class:`mss_data_server.geometry.inventory.Recorder`
            The instance used to create the :class:`DbRecorder` instance.
        
        Returns
        -------
        :class:`DbRecorder`
            The created recorder.
        '''
        recorder = cls(parent_inventory = parent_inventory,
                       serial = instance.serial,
                       model = instance.model,
                       producer = instance.producer,
                       author_uri = instance.author_uri,
                       agency_uri = instance.agency_uri,
                       creation_time = instance.creation_time,
                       description = instance.description)

        for cur_stream in instance.streams:
            recorder.add_stream(cur_stream)

        return recorder


    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        attr_map = {};
        attr_map['serial'] = 'serial'
        attr_map['model'] = 'model'
        attr_map['producer'] = 'producer'
        attr_map['description'] = 'description'
        attr_map['author_uri'] = 'author_uri'
        attr_map['agency_uri'] = 'agency_uri'
        attr_map['creation_time'] = 'creation_time'

        if attr in iter(attr_map.keys()):
            self.__dict__[attr] = value
            if 'orm' in self.__dict__:
                setattr(self.orm, attr_map[attr], value)
        else:
            self.__dict__[attr] = value


    def add_stream(self, cur_stream):
        ''' Add a stream to the recorder.

        Parameters
        ----------
        cur_stream: :class:`DbRecorderStream`
            The stream to add to the recorder.

        Returns
        -------
        :class:`DbRecorderStream` or :class:`mss_data_server.geometry.inventory.RecorderStream`
            The added recorder stream.
        '''
        if cur_stream.__class__ is RecorderStream:
            cur_stream = DbRecorderStream.from_inventory_instance(self, cur_stream)

        added_stream = Recorder.add_stream(self, cur_stream)
        if added_stream is not None:
            if cur_stream.orm not in self.orm.streams:
                self.orm.streams.append(cur_stream.orm)

        return added_stream


class DbRecorderStream(RecorderStream):
    ''' The database recorder stream.

    Parameters
    ----------
    orm: :class:`mss_data_server.geometry.databaseFactory.GeomRecorderStream`
        The sqlalchemy table mapper class.

    
    Keyword Arguments
    -----------------
    **kwargs
        The keyword arguments passed to the __init__ method of the parent class :class:`mss_dataserver.geoemetry.inventory.RecorderStream`.

    '''

    def __init__(self, orm = None, **kwargs):
        RecorderStream.__init__(self, **kwargs)

        if orm is None:
            orm_class = self.parent_inventory.project.db_tables['geom_rec_stream']
            self.orm = orm_class(name = self.name,
                                 label = self.label,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri,
                                 creation_time = self.creation_time)
        else:
            self.orm = orm

    @property
    def id(self):
        ''' The database id.
        '''
        if self.orm is not None:
            return self.orm.id
        else:
            return None


    @classmethod
    def from_sqlalchemy_orm(cls, parent_recorder, orm):
        stream =  cls(parent_recorder = parent_recorder,
                      name = orm.name,
                      label = orm.label,
                      author_uri = orm.author_uri,
                      agency_uri = orm.agency_uri,
                      creation_time = orm.creation_time,
                      orm = orm)

        for cur_component_to_stream in orm.components:
            cur_component = cur_component_to_stream.component
            tmp = stream.add_component(serial = cur_component.parent.serial,
                                 model = cur_component.parent.model,
                                 producer = cur_component.parent.producer,
                                 name = cur_component.name,
                                 start_time = cur_component_to_stream.start_time,
                                 end_time = cur_component_to_stream.end_time,
                                 ignore_orm = True)

        for cur_parameter in orm.parameters:
            stream.add_parameter(DbRecorderStreamParameter.from_sqlalchemy_orm(stream, cur_parameter))

        return stream


    @classmethod
    def from_inventory_instance(cls, parent_recorder, instance):
        cur_stream =  cls(parent_recorder = parent_recorder,
                          name = instance.name,
                          label = instance.label,
                          author_uri = instance.author_uri,
                          agency_uri = instance.agency_uri,
                          creation_time = instance.creation_time)

        for cur_timebox in instance.components:
            cur_stream.add_component(serial = cur_timebox.item.serial,
                                     model = cur_timebox.item.model,
                                     producer = cur_timebox.item.producer,
                                     name = cur_timebox.item.name,
                                     start_time = cur_timebox.start_time,
                                     end_time = cur_timebox.end_time)

        for cur_parameter in instance.parameters:
            cur_stream.add_parameter(cur_parameter)

        return cur_stream


    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        attr_map = {};
        attr_map['name'] = 'name'
        attr_map['label'] = 'label'
        attr_map['gain'] = 'gain'
        attr_map['bitweight'] = 'bitweight'
        attr_map['bitweight_units'] = 'bitweight_units'
        attr_map['author_uri'] = 'author_uri'
        attr_map['agency_uri'] = 'agency_uri'
        attr_map['creation_time'] = 'creation_time'
        if attr in iter(attr_map.keys()):
            self.__dict__[attr] = value
            if 'orm' in self.__dict__:
                setattr(self.orm, attr_map[attr], value)
        else:
            self.__dict__[attr] = value


    def add_component(self, serial, model, producer, name, start_time, end_time, ignore_orm = False):
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

        ignore_orm : Boolean
            Control if the component assignment is added to the orm or not. This is usefull
            when creating an instance from a orm mapper using the from_sqlalchemy_orm
            class method.
        '''
        added_component = RecorderStream.add_component(self,
                                                       serial = serial,
                                                       model = model,
                                                       producer = producer,
                                                       name = name,
                                                       start_time = start_time,
                                                       end_time = end_time)
        if ignore_orm is False:
            if added_component is not None:
                # Add the sensor to the database orm.
                if start_time is not None:
                    start_time_timestamp = start_time.timestamp
                else:
                    start_time_timestamp = None

                if end_time is not None:
                    end_time_timestamp = end_time.timestamp
                else:
                    end_time_timestamp = None

                geom_comp_to_stream_orm = self.parent_inventory.project.db_tables['geom_component_to_stream']
                geom_comp_to_stream = geom_comp_to_stream_orm(self.id,
                                                                  added_component.id,
                                                                  start_time_timestamp,
                                                                  end_time_timestamp)
                geom_comp_to_stream.component = added_component.orm
                self.orm.components.append(geom_comp_to_stream)

        return added_component


    def add_parameter(self, cur_parameter):
        ''' Add a parameter to the recorder_stream.
        '''
        if cur_parameter.__class__ is RecorderStreamParameter:
            cur_parameter = DbRecorderStreamParameter.from_inventory_instance(self, cur_parameter)

        added_parameter = RecorderStream.add_parameter(self, cur_parameter)

        if added_parameter is not None:
            if cur_parameter.orm not in self.orm.parameters:
                self.orm.parameters.append(cur_parameter.orm)

        return added_parameter


    def remove_parameter_by_instance(self, parameter_to_remove):
        ''' Remove a parameter.
        '''
        RecorderStream.remove_parameter_by_instance(self, parameter_to_remove)
        self.logger.debug('Removing DB parameter %d.', parameter_to_remove.id)
        self.orm.parameters.remove(parameter_to_remove.orm)
        self.parent_inventory.db_session.delete(parameter_to_remove.orm)


    def remove_component_by_instance(self, component_to_remove):
        ''' Remove a component from the stream.
        '''
        RecorderStream.remove_component_by_instance(self, component_to_remove)
        try:
            start_time = component_to_remove.start_time.timestamp
        except:
            start_time = None

        try:
            end_time = component_to_remove.end_time.timestamp
        except:
            end_time = None

        orm_to_remove = [x for x in self.orm.components if x.component_id == component_to_remove.id \
                                                        and x.start_time == start_time \
                                                        and x.end_time == end_time]

        for cur_orm in orm_to_remove:
            self.parent_inventory.db_session.delete(cur_orm)





    def change_sensor_start_time_OLD(self, sensor, start_time, end_time, new_start_time):
        ''' Change the sensor deployment start time

        Parameters
        ----------
        sensor : :class:`Sensor`
            The sensor which should be changed.

        start_time : :class:`~obspy.core.utcdatetime.UTCDateTime or String
            A :class:`~obspy.core.utcdatetime.UTCDateTime` instance or a data-time string which can be used by :class:`~obspy.core.utcdatetime.UTCDateTime`.
        '''
        sensor_2_change = [(s, b, e, k) for k, (s, b, e) in enumerate(self.sensors) if s == sensor and b == start_time and e == end_time]

        if len(sensor_2_change) == 1:
            sensor_2_change = sensor_2_change[0]
            position = sensor_2_change[3]
        elif len(sensor_2_change) > 1:
            msg = 'More than one sensor found in the station.'
            return(start_time, msg)
        else:
            msg = 'The sensor can''t be found in the station.'
            return (start_time, msg)

        msg = ''    


        if not isinstance(new_start_time, UTCDateTime):
            try:
                new_start_time = UTCDateTime(new_start_time)
            except:
                new_start_time = sensor_2_change[2]
                msg = "The entered value is not a valid time."


        if not sensor_2_change[2] or (sensor_2_change[2] and new_start_time < sensor_2_change[2]):
            self.sensors[position] = (sensor_2_change[0], new_start_time, sensor_2_change[2])
            # Change the start-time in the ORM.
            cur_geom_sensor_time = [x for x in self.geom_station.sensors if x.child is sensor.geom_sensor and UTCDateTime(x.start_time) == start_time]
            if len(cur_geom_sensor_time) == 1:
                if new_start_time is not None:
                    cur_geom_sensor_time[0].start_time = new_start_time.timestamp
                else:
                    cur_geom_sensor_time[0].start_time = new_start_time
            elif len(cur_geom_sensor_time) > 1:
                self.logger.error('Found more than two sensor ORM children.')
        else:
            new_start_time = sensor_2_change[1]
            msg = "The end-time has to be larger than the begin time."

        return (new_start_time, msg)


    def change_sensor_end_time_OLD(self, sensor, end_time):
        ''' Change the sensor deployment end time

        Parameters
        ----------
        sensor : :class:`Sensor`
            The sensor which should be changed.

        end_time : String
            A data-time string which can be used by :class:`~obspy.core.utcdatetime.UTCDateTime`.
        '''
        sensor_2_change = [(s, b, e, k) for k, (s, b, e) in enumerate(self.sensors) if s == sensor]

        if sensor_2_change:
            sensor_2_change = sensor_2_change[0]
            position = sensor_2_change[3]
        else:
            msg = 'The sensor can''t be found in the station.'
            return (None, msg)

        msg = ''    

        if end_time == 'running':
            self.sensors[position] = (sensor_2_change[0], sensor_2_change[1], None)
            # Change the start-time in the ORM.
            cur_geom_sensor_time = [x for x in self.geom_station.sensors if x.child is sensor.geom_sensor]
            if len(cur_geom_sensor_time) == 1:
                    cur_geom_sensor_time[0].end_time = None
            elif len(cur_geom_sensor_time) > 1:
                self.logger.error('Found more than two sensor ORM children.')

        else:
            if not isinstance(end_time, UTCDateTime):
                try:
                    end_time = UTCDateTime(end_time)
                except:
                    end_time = sensor_2_change[2]
                    msg = "The entered value is not a valid time."


            if not sensor_2_change[1] or end_time > sensor_2_change[1]:
                self.sensors[position] = (sensor_2_change[0], sensor_2_change[1], end_time)
                # Change the start-time in the ORM.
                cur_geom_sensor_time = [x for x in self.geom_station.sensors if x.child is sensor.geom_sensor]
                if len(cur_geom_sensor_time) == 1:
                    if end_time is not None:
                        cur_geom_sensor_time[0].end_time = end_time.timestamp
                    else:
                        cur_geom_sensor_time[0].end_time = end_time
                elif len(cur_geom_sensor_time) > 1:
                    self.logger.error('Found more than two sensor ORM children.')
            else:
                end_time = sensor_2_change[2]
                msg = "The end-time has to be larger than the begin time."

        return (end_time, msg)



class DbRecorderStreamParameter(RecorderStreamParameter):
    def __init__(self, orm = None, **kwargs):
        RecorderStreamParameter.__init__(self, **kwargs)

        if orm is None:
            orm_class = self.parent_inventory.project.db_tables['geom_rec_stream_param']
            self.orm = orm_class(gain = self.gain,
                                 bitweight = self.bitweight,
                                 start_time = self.start_time.timestamp,
                                 end_time = None if self.end_time is None else self.end_time.timestamp,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri,
                                 creation_time = self.creation_time)
        else:
            self.orm = orm

    @property
    def id(self):
        ''' The database id.
        '''
        if self.orm is not None:
            return self.orm.id
        else:
            return None

    @classmethod
    def from_inventory_instance(cls, parent_recorder_stream, instance):
        return cls(parent_recorder_stream = parent_recorder_stream,
                   gain = instance.gain,
                   bitweight = instance.bitweight,
                   start_time = instance.start_time,
                   end_time = instance.end_time,
                   author_uri = instance.author_uri,
                   agency_uri = instance.agency_uri,
                   creation_time = instance.creation_time)

    @classmethod
    def from_sqlalchemy_orm(cls, parent_recorder_stream, orm):
        parameter = cls(parent_recorder_stream = parent_recorder_stream,
                        gain = orm.gain,
                        bitweight = orm.bitweight,
                        start_time = UTCDateTime(orm.start_time) if orm.start_time is not None else None,
                        end_time = UTCDateTime(orm.end_time) if orm.end_time is not None else None,
                        author_uri = orm.author_uri,
                        agency_uri = orm.agency_uri,
                        creation_time = orm.creation_time,
                        orm = orm)

        return parameter


    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        attr_map = {};
        attr_map['start_time'] = 'start_time'
        attr_map['end_time'] = 'end_time'
        attr_map['gain'] = 'gain'
        attr_map['bitweight'] = 'bitweight'
        attr_map['author_uri'] = 'author_uri'
        attr_map['agency_uri'] = 'agency_uri'
        attr_map['creation_time'] = 'creation_time'

        if attr in iter(attr_map.keys()):
            self.__dict__[attr] = value
            if 'orm' in self.__dict__:
                if (attr == 'start_time') or (attr == 'end_time'):
                    setattr(self.orm, attr_map[attr], value.timestamp)
                else:
                    setattr(self.orm, attr_map[attr], value)
        else:
            self.__dict__[attr] = value




class DbSensor(Sensor):
    def __init__(self, orm = None, **kwargs):
        Sensor.__init__(self, **kwargs)

        if orm is None:
            orm_class = self.parent_inventory.project.db_tables['geom_sensor']
            self.orm = orm_class(serial = self.serial,
                                 model = self.model,
                                 producer = self.producer,
                                 description = self.description,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri,
                                 creation_time = self.creation_time)
        else:
            self.orm = orm


    @property
    def id(self):
        ''' The database id.
        '''
        if self.orm is not None:
            return self.orm.id
        else:
            return None


    @classmethod
    def from_sqlalchemy_orm(cls, parent_inventory, orm):
        sensor =  cls(parent_inventory = parent_inventory,
                          serial = orm.serial,
                          model = orm.model,
                          producer = orm.producer,
                          description = orm.description,
                          author_uri = orm.author_uri,
                          agency_uri = orm.agency_uri,
                          creation_time = orm.creation_time,
                          orm = orm)

        for cur_component in orm.components:
            db_component = DbSensorComponent.from_sqlalchemy_orm(sensor, cur_component)
            sensor.add_component(db_component)

        return sensor



    @classmethod
    def from_inventory_instance(cls, parent_inventory, instance):
        sensor =  cls(parent_inventory = parent_inventory,
                      serial = instance.serial,
                      model = instance.model,
                      producer = instance.producer,
                      description = instance.description,
                      author_uri = instance.author_uri,
                      agency_uri = instance.agency_uri,
                      creation_time = instance.creation_time)

        for cur_component in instance.components:
            sensor.add_component(DbSensorComponent.from_inventory_instance(sensor, cur_component))
        return sensor


    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        attr_map = {};
        attr_map['serial'] = 'serial'
        attr_map['model'] = 'model'
        attr_map['producer'] = 'producer'
        attr_map['description'] = 'description'
        attr_map['author_uri'] = 'author_uri'
        attr_map['agency_uri'] = 'agency_uri'
        attr_map['creation_time'] = 'creation_time'

        if attr in iter(attr_map.keys()):
            self.__dict__[attr] = value
            if 'orm' in self.__dict__:
                setattr(self.orm, attr_map[attr], value)
        else:
            self.__dict__[attr] = value


    def add_component(self, cur_component):
        ''' Add a component to the sensor.

        Parameters
        ----------
        cur_component : :class:`DbSensorComponent`
            The stream to add to the recorder.
        '''
        if cur_component.__class__ is SensorComponent:
            cur_component = DbSensorComponent.from_inventory_instance(self, cur_component)

        added_component = Sensor.add_component(self, cur_component)
        if added_component is not None:
            if cur_component.orm not in self.orm.components:
                self.orm.components.append(cur_component.orm)

        return added_component


class DbSensorComponent(SensorComponent):

    def __init__(self, orm = None, **kwargs):
        SensorComponent.__init__(self, **kwargs)

        if orm is None:
            orm_class = self.parent_inventory.project.db_tables['geom_sensor_component']
            self.orm = orm_class(name = self.name,
                                 description = self.description,
                                 input_unit = self.input_unit,
                                 output_unit = self.output_unit,
                                 deliver_unit = self.deliver_unit,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri,
                                 creation_time = self.creation_time)
        else:
            self.orm = orm

    @property
    def id(self):
        ''' The database id.
        '''
        if self.orm is not None:
            return self.orm.id
        else:
            return None


    @classmethod
    def from_sqlalchemy_orm(cls, parent_sensor, orm):
        component =  cls(parent_sensor = parent_sensor,
                         name = orm.name,
                         description = orm.description,
                         input_unit = orm.input_unit,
                         output_unit = orm.output_unit,
                         deliver_unit = orm.deliver_unit,
                         author_uri = orm.author_uri,
                         agency_uri = orm.agency_uri,
                         creation_time = orm.creation_time,
                         orm = orm)

        for cur_param in orm.parameters:
            db_param = DbSensorComponentParameter.from_sqlalchemy_orm(component, cur_param)
            component.add_parameter(db_param)

        return component



    @classmethod
    def from_inventory_instance(cls, parent_sensor, instance):
        cur_sensor =  cls(parent_sensor = parent_sensor,
                          name = instance.name,
                          description = instance.description,
                          input_unit = instance.input_unit,
                          output_unit = instance.output_unit,
                          deliver_unit = instance.deliver_unit,
                          author_uri = instance.author_uri,
                          agency_uri = instance.agency_uri,
                          creation_time = instance.creation_time)

        for cur_parameter in instance.parameters:
            cur_sensor.add_parameter(DbSensorComponentParameter.from_inventory_instance(cur_sensor, cur_parameter))
        return cur_sensor




    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        attr_map = {};
        attr_map['name'] = 'name'
        attr_map['description'] = 'description'
        attr_map['input_unit'] = 'input_unit'
        attr_map['outpu_unit'] = 'output_unit'
        attr_map['deliver_unit'] = 'deliver_unit'
        attr_map['author_uri'] = 'author_uri'
        attr_map['agency_uri'] = 'agency_uri'
        attr_map['creation_time'] = 'creation_time'

        if attr in iter(attr_map.keys()):
            self.__dict__[attr] = value
            if 'orm' in self.__dict__:
                setattr(self.orm, attr_map[attr], value)
        else:
            self.__dict__[attr] = value


    def add_parameter(self, cur_parameter):
        ''' Add a parameter to the sensor

        Parameters
        ----------
        parameter : :class:`DbSensorComponentParameter`
            The parameter to add to the sensor.
        '''
        if cur_parameter.__class__ is SensorComponentParameter:
            cur_parameter = DbSensorComponentParameter.from_inventory_instance(self, cur_parameter)

        added_parameter = SensorComponent.add_parameter(self, cur_parameter)

        if added_parameter is not None:
            if cur_parameter.orm not in self.orm.parameters:
                self.orm.parameters.append(cur_parameter.orm)

                # Add the tf poles and zeros to the database orm.
                geom_tfpz_orm_class = self.parent_inventory.project.db_tables['geom_tf_pz']
                for cur_pole in cur_parameter.tf_poles:
                    cur_parameter.orm.tf_pz.append(geom_tfpz_orm_class(1, cur_pole.real, cur_pole.imag))
                for cur_zero in cur_parameter.tf_zeros:
                    cur_parameter.orm.tf_pz.append(geom_tfpz_orm_class(0, cur_zero.real, cur_zero.imag))

        return added_parameter


    def remove_parameter(self, parameter_to_remove):
        ''' Remove a parameter from the component.
        '''
        SensorComponent.remove_parameter(self, parameter_to_remove)
        self.orm.parameters.remove(parameter_to_remove.orm)
        self.parent_inventory.db_session.delete(parameter_to_remove.orm)






class DbSensorComponentParameter(SensorComponentParameter):

    def __init__(self, orm = None, **kwargs):

        SensorComponentParameter.__init__(self, **kwargs)

        if orm is None:
            orm_class = self.parent_inventory.project.db_tables['geom_component_param']
            self.orm = orm_class(component_id = self.parent_component.id,
                                 start_time = self.start_time.timestamp,
                                 end_time = None if self.end_time is None else self.end_time.timestamp,
                                 tf_normalization_factor = self.tf_normalization_factor,
                                 tf_normalization_frequency = self.tf_normalization_frequency,
                                 tf_type = self.tf_type,
                                 tf_units = self.tf_units,
                                 sensitivity = self.sensitivity,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri,
                                 creation_time = self.creation_time)
        else:
            self.orm = orm


    @property
    def id(self):
        ''' The database id.
        '''
        if self.orm is not None:
            return self.orm.id
        else:
            return None


    @classmethod
    def from_inventory_instance(cls, parent_component, instance):
        return cls(parent_component = parent_component,
                   start_time = instance.start_time,
                   end_time = instance.end_time,
                   tf_normalization_factor = instance.tf_normalization_factor,
                   tf_normalization_frequency = instance.tf_normalization_frequency,
                   tf_type = instance.tf_type,
                   tf_units = instance.tf_units,
                   tf_poles = instance.tf_poles,
                   tf_zeros = instance.tf_zeros,
                   sensitivity = instance.sensitivity,
                   author_uri = instance.author_uri,
                   agency_uri = instance.agency_uri,
                   creation_time = instance.creation_time)


    @classmethod
    def from_sqlalchemy_orm(cls, parent_component, orm):

        if orm.start_time is not None:
            start_time = UTCDateTime(orm.start_time)
        else:
            start_time = None

        if orm.end_time is not None:
            end_time = UTCDateTime(orm.end_time)
        else:
            end_time = None

        parameter = cls(parent_component = parent_component,
                        start_time = start_time,
                        end_time = end_time,
                        tf_normalization_factor = orm.tf_normalization_factor,
                        tf_normalization_frequency = orm.tf_normalization_frequency,
                        tf_type = orm.tf_type,
                        tf_units = orm.tf_units,
                        sensitivity = orm.sensitivity,
                        author_uri = orm.author_uri,
                        agency_uri = orm.agency_uri,
                        creation_time = orm.creation_time,
                        orm = orm)

        # Collect the poles and zeros of the transfer function.
        for cur_pz in orm.tf_pz:
            if cur_pz.type == 0:
                parameter.tf_zeros.append(complex(cur_pz.complex_real, cur_pz.complex_imag))
            elif cur_pz.type == 1:
                parameter.tf_poles.append(complex(cur_pz.complex_real, cur_pz.complex_imag))

        return parameter


    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        attr_map = {};
        attr_map['start_time'] = 'start_time'
        attr_map['end_time'] = 'end_time'
        attr_map['tf_normalization_factor'] = 'tf_normalization_factor'
        attr_map['tf_normalization_frequency'] = 'tf_normalization_frequency'
        attr_map['tf_type'] = 'tf_type'
        attr_map['tf_units'] = 'tf_units'
        attr_map['sensitivity'] = 'sensitivity'
        attr_map['author_uri'] = 'author_uri'
        attr_map['agency_uri'] = 'agency_uri'
        attr_map['creation_time'] = 'creation_time'

        if attr in iter(attr_map.keys()):
            self.__dict__[attr] = value
            if 'orm' in self.__dict__:
                if (attr == 'start_time') or (attr == 'end_time'):
                    setattr(self.orm, attr_map[attr], value.timestamp)
                else:
                    setattr(self.orm, attr_map[attr], value)
        else:
            self.__dict__[attr] = value




class DbChannel(Channel):

    def __init__(self, orm = None, **kwargs):
        Channel.__init__(self, **kwargs)

        if orm is None:
            orm_class = self.parent_inventory.project.db_tables['geom_channel']
            self.orm = orm_class(name = self.name,
                                 description = self.description,
                                 agency_uri = self.agency_uri,
                                 author_uri = self.author_uri,
                                 creation_time = self.creation_time)
        else:
            self.orm = orm


    @property
    def id(self):
        ''' The database id.
        '''
        if self.orm is not None:
            return self.orm.id
        else:
            return None


    @classmethod
    def from_sqlalchemy_orm(cls, parent_station, orm):
        channel =  cls(parent_station = parent_station,
                       name = orm.name,
                       description = orm.description,
                       author_uri = orm.author_uri,
                       agency_uri = orm.agency_uri,
                       creation_time = orm.creation_time,
                       orm = orm)

        for cur_stream_to_channel in orm.streams:
            cur_stream = cur_stream_to_channel.stream
            cur_start_time = UTCDateTime(cur_stream_to_channel.start_time)
            try:
                cur_end_time = UTCDateTime(cur_stream_to_channel.end_time)
            except:
                cur_end_time = None

            channel.add_stream(serial = cur_stream.parent.serial,
                               model = cur_stream.parent.model,
                               producer = cur_stream.parent.producer,
                               name = cur_stream.name,
                               start_time = cur_start_time,
                               end_time = cur_end_time,
                               ignore_orm = True)


        return channel



    @classmethod
    def from_inventory_instance(cls, parent_station, instance):
        channel =  cls(parent_station = parent_station,
                           name = instance.name,
                           description = instance.description,
                           author_uri = instance.author_uri,
                           agency_uri = instance.agency_uri,
                           creation_time = instance.creation_time)

        for cur_timebox in instance.streams:
            channel.add_stream(serial = cur_timebox.item.serial,
                               model = cur_timebox.item.model,
                               producer = cur_timebox.item.producer,
                               name = cur_timebox.item.name,
                               start_time = cur_timebox.start_time,
                               end_time = cur_timebox.end_time)

        return channel


    def __setattr__(self, attr, value):
        ''' Control the attribute assignements.
        '''
        attr_map = {};
        attr_map['name'] = 'name'
        attr_map['description'] = 'description'
        attr_map['author_uri'] = 'author_uri'
        attr_map['agency_uri'] = 'agency_uri'
        attr_map['creation_time'] = 'creation_time'

        if attr in iter(attr_map.keys()):
            self.__dict__[attr] = value
            if 'orm' in self.__dict__:
                setattr(self.orm, attr_map[attr], value)
        else:
            self.__dict__[attr] = value


    def add_stream(self, serial, model, producer, name, start_time, end_time, ignore_orm = False):
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

        ignore_orm : Boolean
            Control if the component assignment is added to the orm or not. This is usefull
            when creating an instance from a orm mapper using the from_sqlalchemy_orm
            class method.
        '''
        added_stream = Channel.add_stream(self,
                                          serial = serial,
                                          model = model,
                                          producer = producer,
                                          name = name,
                                          start_time = start_time,
                                          end_time = end_time)

        if ignore_orm is False:
            if added_stream is not None:
                # Add the streastream the the database orm.
                if start_time is not None:
                    start_time_timestamp = start_time.timestamp
                else:
                    start_time_timestamp = None

                if end_time is not None:
                    end_time_timestamp = end_time.timestamp
                else:
                    end_time_timestamp = None

                orm_class = self.parent_inventory.project.db_tables['geom_stream_to_channel']
                stream_to_channel_orm = orm_class(channel_id = self.id,
                                                  stream_id = added_stream.id,
                                                  start_time = start_time_timestamp,
                                                  end_time = end_time_timestamp)
                stream_to_channel_orm.stream = added_stream.orm
                self.orm.streams.append(stream_to_channel_orm)

        return added_stream


    def remove_stream_by_instance(self, stream_timebox):
        ''' Remove a stream timebox.
        '''
        Channel.remove_stream_by_instance(self, stream_timebox)
        try:
            start_time = stream_timebox.start_time.timestamp
        except:
            start_time = None

        try:
            end_time = stream_timebox.end_time.timestamp
        except:
            end_time = None

        orm_to_remove = [x for x in self.orm.streams if x.stream_id == stream_timebox.id \
                                                     and x.start_time == start_time \
                                                     and x.end_time == end_time]

        for cur_orm in orm_to_remove:
            self.parent_inventory.db_session.delete(cur_orm)


