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

''' Database change history
'''
import obspy

def databaseFactory(base):
    from sqlalchemy import Column, Integer, String, Float
    from sqlalchemy import ForeignKey, UniqueConstraint
    from sqlalchemy.orm import relationship

    tables = []

    # Create the geom_recorder table mapper class.
    class GeomRecorder(base):
        __tablename__ = 'geom_recorder'
        __table_args__ = (
                          UniqueConstraint('serial', 'model', 'producer'),
                          {'mysql_engine': 'InnoDB'}
                         )
        _version = '1.0.0'


        id = Column(Integer, primary_key=True, autoincrement=True)
        serial = Column(String(45), nullable=False)
        model = Column(String(100), nullable=False)
        producer = Column(String(100), nullable=False)
        description = Column(String(255), nullable=True)
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))

        streams = relationship('GeomRecorderStream',
                               cascade = 'all',
                               backref = 'parent')


        def __init__(self, serial, model, producer,
                agency_uri, author_uri, creation_time):
            self.serial = serial
            self.model = model
            self.producer = producer
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            if isinstance(creation_time, obspy.UTCDateTime):
                creation_time = creation_time.isoformat()
            self.creation_time = creation_time

        def __repr__(self):
            return "Recorder\nid: %d\nserial: %s\nmodel: %s\n" % (self.id, self.serial, self.model)

    tables.append(GeomRecorder)



    class GeomRecorderStream(base):
        __tablename__ = 'geom_rec_stream'
        __table_args__ = (
                          UniqueConstraint('recorder_id', 'name'),
                          {'mysql_engine': 'InnoDB'}
                         )
        _version = '1.0.0'

        id = Column(Integer, primary_key=True, autoincrement=True)
        recorder_id = Column(Integer,
                             ForeignKey('geom_recorder.id',
                                        onupdate='cascade',
                                        ondelete='set null'),
                             nullable=True)
        name = Column(String(20), nullable = False)
        label = Column(String(20), nullable = False)
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))

        # The channels to which the stream is associated.
        #channels = relationship('GeomStreamToChannel')

        components = relationship('GeomComponentToStream',
                                   backref = 'parent')
        parameters = relationship('GeomRecorderStreamParameter',
                                  backref = 'parent')

        def __init__(self, name, label,
                agency_uri, author_uri, creation_time):
            self.name = name
            self.label = label
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            if isinstance(creation_time, obspy.UTCDateTime):
                creation_time = creation_time.isoformat()
            self.creation_time = creation_time


        def __repr__(self):
            return "GeomRecorderStream\id: %d\nrecorder_id: %d\nname: %s\nlabel: %s\nagency_uri: %s\nauthor_uri: %s\ncreation_time: %s\n" % (self.id,
                        self.recorder_id, self.name, self.label, self.agency_uri, self.author_uri, self.creation_time)

    tables.append(GeomRecorderStream)


    class GeomRecorderStreamParameter(base):
        __tablename__ = 'geom_rec_stream_param'
        __table_args__ = (
                          UniqueConstraint('rec_stream_id', 'start_time', 'end_time'),
                          {'mysql_engine': 'InnoDB'}
                         )
        _version = '1.0.0'

        id = Column(Integer, primary_key=True, autoincrement=True)
        rec_stream_id = Column(Integer, ForeignKey('geom_rec_stream.id', onupdate='cascade'), nullable=True)
        start_time = Column(Float(53), nullable = False)
        end_time = Column(Float(53))
        gain = Column(Float)
        bitweight = Column(Float)
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))


        def __init__(self, start_time, end_time, gain, bitweight,
                     agency_uri, author_uri, creation_time):
            self.start_time = start_time
            self.end_time = end_time
            self.gain = gain
            self.bitweight = bitweight
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            if isinstance(creation_time, obspy.UTCDateTime):
                creation_time = creation_time.isoformat()
            self.creation_time = creation_time


    tables.append(GeomRecorderStreamParameter)



    class GeomSensor(base):
        __tablename__ = 'geom_sensor'
        __table_args__ = (
                          UniqueConstraint('serial', 'model', 'producer'),
                          {'mysql_engine': 'InnoDB'}
                         )
        _version = '1.0.0'

        id = Column(Integer, primary_key=True, autoincrement=True)
        serial = Column(String(45), nullable=False)
        model = Column(String(100), nullable=False)
        producer = Column(String(100), nullable = False)
        description = Column(String(255))
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))

        components = relationship('GeomSensorComponent',
                                  cascade = 'all',
                                  backref = 'parent')


        def __init__(self, serial, model, producer, description,
                agency_uri, author_uri, creation_time):
            self.serial = serial
            self.model = model
            self.producer = producer
            self.description = description
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            if isinstance(creation_time, obspy.UTCDateTime):
                creation_time = creation_time.isoformat()
            self.creation_time = creation_time

    tables.append(GeomSensor)


    # Create the geom_sensor_component table mapper class.
    class GeomSensorComponent(base):
        __tablename__ = 'geom_sensor_component'
        __table_args__ = (
                          UniqueConstraint('sensor_id', 'name'),
                          {'mysql_engine': 'InnoDB'}
                         )
        _version = '1.0.0'

        id = Column(Integer, primary_key=True, autoincrement=True)
        sensor_id = Column(Integer, ForeignKey('geom_sensor.id', onupdate='cascade'), nullable=True, default=-1)
        name = Column(String(45), nullable=False)
        description = Column(String(255))
        input_unit = Column(String(10))
        output_unit = Column(String(10))
        deliver_unit = Column(String(10))
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))

        parameters = relationship('GeomComponentParam',
                                  cascade = 'all',
                                  backref = 'parent')


        def __init__(self, name, description,
                input_unit, output_unit, deliver_unit,
                agency_uri, author_uri, creation_time):
            self.name = name
            self.description = description
            self.input_unit = input_unit
            self.output_unit = output_unit
            self.deliver_unit = deliver_unit
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            if isinstance(creation_time, obspy.UTCDateTime):
                creation_time = creation_time.isoformat()
            self.creation_time = creation_time

    tables.append(GeomSensorComponent)


    # Create the geom_sensor_param table mapper.
    class GeomComponentParam(base):
        __tablename__ = 'geom_component_param'
        __table_args__ = (
                          UniqueConstraint('component_id', 'start_time', 'end_time'),
                          {'mysql_engine': 'InnoDB'}
                         )
        _version = '1.0.0'

        id = Column(Integer, primary_key=True, autoincrement=True)
        component_id = Column(Integer, ForeignKey('geom_sensor_component.id', onupdate='cascade'), nullable=True, default=-1)
        start_time = Column(Float(53))
        end_time = Column(Float(53))
        tf_normalization_factor = Column(Float)
        tf_normalization_frequency = Column(Float)
        tf_type = Column(String(150))
        tf_units = Column(String(20))
        sensitivity = Column(Float(53))
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))

        tf_pz = relationship('GeomTfPz', cascade='all')


        def __init__(self, component_id, start_time, end_time, tf_normalization_factor,
                     tf_normalization_frequency, tf_type, tf_units, sensitivity,
                     agency_uri, author_uri, creation_time):
            self.component_id = component_id
            self.start_time = start_time
            self.end_time = end_time
            self.tf_normalization_factor = tf_normalization_factor
            self.tf_normalization_frequency = tf_normalization_frequency
            self.tf_type = tf_type
            self.tf_units = tf_units
            self.sensitivity = sensitivity
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            if isinstance(creation_time, obspy.UTCDateTime):
                creation_time = creation_time.isoformat()
            self.creation_time = creation_time


    tables.append(GeomComponentParam)


    # Create the geom_tf_pz table mapper.
    class GeomTfPz(base):
        __tablename__ = 'geom_tf_pz'
        __table_args__ = {'mysql_engine': 'InnoDB'}
        _version = '1.0.0'

        id = Column(Integer, primary_key=True, autoincrement=True)
        param_id = Column(Integer, ForeignKey('geom_component_param.id', onupdate='cascade'), nullable=False)
        type = Column(Integer, nullable=False, default=1)
        complex_real = Column(Float, nullable=False)
        complex_imag = Column(Float, nullable=False)


        def __init__(self, type, complex_real, complex_imag):
            self.type = type
            self.complex_real = complex_real
            self.complex_imag = complex_imag

    tables.append(GeomTfPz)



    class GeomComponentToStream(base):
        __tablename__ = 'geom_component_to_stream'
        __table_args__ = {'mysql_engine': 'InnoDB'}
        _version = '1.0.0'

        stream_id = Column(Integer, ForeignKey('geom_rec_stream.id', onupdate='cascade'), primary_key=True, nullable=False)
        component_id = Column(Integer, ForeignKey('geom_sensor_component.id', onupdate='cascade'), primary_key=True, nullable=False)
        start_time = Column(Float(53), primary_key=True, nullable=False)
        end_time = Column(Float(53))

        component = relationship('GeomSensorComponent')

        def __init__(self, stream_id, component_id, start_time, end_time):
            self.stream_id = stream_id
            self.component_id = component_id
            self.start_time = start_time
            self.end_time = end_time


    tables.append(GeomComponentToStream)


    # Create the geom_network table mapper.
    class GeomNetwork(base):
        __tablename__ = 'geom_network'
        __table_args__ = {'mysql_engine': 'InnoDB'}
        _version = '1.0.0'

        name = Column(String(10), primary_key=True, nullable=False)
        description = Column(String(255))
        type = Column(String(255))
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))

        stations = relationship('GeomStation', 
                                cascade = 'all', 
                                backref = 'parent')


        def __init__(self, name, description, type,
                agency_uri, author_uri, creation_time):
            self.name = name
            self.description = description
            self.type = type
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            if isinstance(creation_time, obspy.UTCDateTime):
                creation_time = creation_time.isoformat()
            self.creation_time = creation_time


    tables.append(GeomNetwork)


    # Create the geom_station table mapper class.
    class GeomStation(base):
        __tablename__ = 'geom_station'
        __table_args__ = (
                          UniqueConstraint('network', 'name', 'location'),
                          {'mysql_engine': 'InnoDB'}
                         )
        _version = '1.0.0'

        id = Column(Integer, primary_key=True, autoincrement=True)
        network = Column(String(10), ForeignKey('geom_network.name', onupdate='cascade'), nullable=True)
        name = Column(String(20), nullable=False)
        location = Column(String(3), nullable=False)
        x = Column(Float(53), nullable=False)
        y = Column(Float(53), nullable=False)
        z = Column(Float(53), nullable=False)
        coord_system = Column(String(50), nullable=False)
        description = Column(String(255))
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))

        channels = relationship('GeomChannel')
        arrays = relationship('GeomStatToArray',
                              viewonly = True)


        def __init__(self, name, location, x, y, z, coord_system,
                     description, agency_uri, author_uri, creation_time):
            self.name = name
            self.location = location
            self.x = x
            self.y = y
            self.z = z
            self.coord_system = coord_system
            self.description = description
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            if isinstance(creation_time, obspy.UTCDateTime):
                creation_time = creation_time.isoformat()
            self.creation_time = creation_time


        def __repr__(self):
            return "Station\nname: %s\nlocation: %s\nx: %f\ny: %f\nz: %f\ncoord_system: %s\ndescription: %s\n" % (self.name, self.location, self.x, self.y, self.z, self.coord_system, self.description)

    tables.append(GeomStation)


    class GeomChannel(base):
        __tablename__ = 'geom_channel'
        __table_args__ = (
                          UniqueConstraint('station_id', 'name'),
                          {'mysql_engine': 'InnoDB'}
                         )
        _version = '1.0.0'

        id = Column(Integer, primary_key=True, autoincrement=True)
        station_id = Column(Integer, ForeignKey('geom_station.id', onupdate='cascade'), primary_key=True, nullable=False)
        name = Column(String(20))
        description = Column(String(255))
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))

        streams = relationship('GeomStreamToChannel')


        def __init__(self, name, description, agency_uri,
                     author_uri, creation_time):
            self.name = name
            self.description = description
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            if isinstance(creation_time, obspy.UTCDateTime):
                creation_time = creation_time.isoformat()
            self.creation_time = creation_time

    tables.append(GeomChannel)




    class GeomStreamToChannel(base):
        __tablename__ = 'geom_stream_to_channel'
        __table_args__ = {'mysql_engine': 'InnoDB'}
        _version = '1.0.0'

        id = Column(Integer, primary_key=True, autoincrement=True)
        channel_id = Column(Integer, ForeignKey('geom_channel.id', onupdate='cascade'), primary_key=True, nullable=False)
        stream_id = Column(Integer, ForeignKey('geom_rec_stream.id', onupdate='cascade'), primary_key=True, nullable=False)
        start_time = Column(Float(53), nullable=False)
        end_time = Column(Float(53))

        stream = relationship('GeomRecorderStream')

        def __init__(self, channel_id, stream_id, start_time, end_time):
            self.channel_id = channel_id
            self.stream_id = stream_id
            self.start_time = start_time
            self.end_time = end_time


    tables.append(GeomStreamToChannel)


    class GeomArray(base):
        __tablename__ = 'geom_array'
        __table_args__ = {'mysql_engine': 'InnoDB'}
        _version = '1.0.0'

        name = Column(String(50), primary_key=True, nullable=False)
        description = Column(String(255))
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))

        stations = relationship('GeomStatToArray')

        def __init__(self, name, description,
                agency_uri, author_uri, creation_time):
            self.name = name
            self.description = description
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            if isinstance(creation_time, obspy.UTCDateTime):
                creation_time = creation_time.isoformat()
            self.creation_time = creation_time

    tables.append(GeomArray)


    class GeomStatToArray(base):
        __tablename__ = 'geom_stat_to_array'
        __table_args__ = {'mysql_engine': 'InnoDB'}
        _version = '1.0.0'

        array_name = Column(String(50),
                            ForeignKey('geom_array.name', onupdate = 'cascade'),
                            primary_key = True,
                            nullable = False)
        station_id = Column(Integer,
                            ForeignKey('geom_station.id', onupdate = 'cascade'),
                            primary_key = True,
                            nullable = False)
        start_time = Column(Float(53), nullable=False)
        end_time = Column(Float(53))

        station = relationship('GeomStation')

        def __init__(self, array_name, station_id, start_time, end_time):
            self.array_name = array_name
            self.station_id = station_id
            self.start_time = start_time
            self.end_time = end_time

    tables.append(GeomStatToArray)


    return tables
