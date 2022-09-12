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


def databaseFactory(base):
    from sqlalchemy import Column
    from sqlalchemy import Integer
    from sqlalchemy import String
    from sqlalchemy import Float
    from sqlalchemy import ForeignKey
    from sqlalchemy import UniqueConstraint
    from sqlalchemy.orm import relationship
    from sqlalchemy.orm import backref

    tables = []

    
    class Origin(base):
        __tablename__ = 'origin'
        __table_args__ = ({'mysql_engine': 'InnoDB'})
        _version = '1.0.0'

        id = Column(Integer,
                    primary_key = True,
                    autoincrement = True)
        ev_id = Column(Integer,
                       ForeignKey('event.id',
                                  onupdate = 'cascade',
                                  ondelete = 'set null'),
                       nullable = True)
        time = Column(Float(53),
                      nullable = True)
        x = Column(Float(53),
                   nullable = False)
        y = Column(Float(53),
                   nullable = False)
        z = Column(Float(53),
                   nullable = False)
        coord_system = Column(String(50),
                              nullable = False)
        method = Column(String(255),
                        nullable = False)
        comment = Column(String(255),
                         nullable = True)
        pref_magnitude_id = Column(Integer,
                                   nullable = True)
        agency_uri = Column(String(20),
                            nullable = True)
        author_uri = Column(String(20),
                            nullable = True)
        creation_time = Column(String(30),
                               nullable = True)

        event = relationship('EventDb',
                             back_populates = 'origins')
        magnitudes = relationship('Magnitude',
                                  back_populates = 'origin')

        def __init__(self, event_id, x, y, z, coord_system, method,
                     time = None, comment = None,
                     pref_magnitude_id = None,
                     agency_uri = None, author_uri = None,
                     creation_time = None):
            self.ev_id = event_id
            self.time = time
            self.x = x
            self.y = y
            self.z = z
            self.coord_system = coord_system
            self.method = method
            self.comment = comment
            self.pref_magnitude_id = pref_magnitude_id
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            self.creation_time = creation_time

    tables.append(Origin)


    class Magnitude(base):
        __tablename__ = 'magnitude'
        __table_args__ = ({'mysql_engine': 'InnoDB'})
        
        id = Column(Integer,
                    primary_key = True,
                    autoincrement = True)
        orig_id = Column(Integer,
                         ForeignKey('origin.id',
                                    onupdate = 'cascade',
                                    ondelete = 'set null'),
                         nullable = True)
        mag = Column(Float(53),
                     nullable = False)
        mag_type = Column(String(20),
                          nullable = False)
        comment = Column(String(255))
        agency_uri = Column(String(20))
        author_uri = Column(String(20))
        creation_time = Column(String(30))

        origin = relationship('Origin',
                              back_populates = 'magnitudes')

        def __init__(self, origin_id, mag, mag_type,
                     comment = None, agency_uri = None,
                     author_uri = None, creation_time = None):
            self.orig_id = origin_id
            self.mag = mag
            self.mag_type = mag_type
            self.comment = comment
            self.agency_uri = agency_uri
            self.author_uri = author_uri
            self.creation_time = creation_time

    tables.append(Magnitude)

    return tables
