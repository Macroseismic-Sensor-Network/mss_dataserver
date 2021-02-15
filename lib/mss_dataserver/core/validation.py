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
 # Copyright 2020 Stefan Mertl
##############################################################################

import enum

import pydantic
from pydantic import (
    confloat,
    constr,
    PositiveInt,
    PositiveFloat,
    validator
)

from typing import (
    List,
    Optional,
    Union,
)


class Event(pydantic.BaseModel):
    db_id: Union[None, PositiveInt]
    public_id: constr(regex=r'^\w+_\w+_\d{4}-\d{2}-\d{2}T\d{6,12}')
    start_time: constr(min_length=19, max_length=26)
    end_time: constr(min_length=19, max_length=26)
    length: confloat(ge=0)
    max_pgv: PositiveFloat
    num_detections: PositiveInt
    triggered_stations: List[constr(regex=r'^\w{1,10}:\w{1,10}:\w{1,4}')]
    description: Optional[constr(max_length=255)] = None
    comment: Optional[constr(max_length=255)] = None
    state: Optional[constr(max_length=20)] = None


class MsgClassEnum(str, enum.Enum):
    control = 'control'
    soh = 'soh'
    data = 'data'


class MsgControlIdEnum(str, enum.Enum):
    mode = 'mode'


class MsgRequestIdEnum(str, enum.Enum):
    event_supplement = 'event_supplement'


class MsgSohIdEnum(str, enum.Enum):
    connection = 'connection'
    server_state = 'server_state'


class MsgDataIdEnum(str, enum.Enum):
    current_pgv = 'current_pgv'
    pgv = 'pgv'
    pgv_archive = 'pgv_archive'
    detection_result = 'detection_result'
    event_data = 'event_data'
    event_archive = 'event_archive'
    event_warning = 'event_warning'
    key_data = 'key_data'
    supplement = 'supplement'


class WSMessageHeader(pydantic.BaseModel):
    msg_class: MsgClassEnum
    msg_id: Union[MsgControlIdEnum,
                  MsgDataIdEnum,
                  MsgSohIdEnum,
                  MsgRequestIdEnum]
    server_time: constr(regex=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?')

    @validator('msg_id')
    def check_msg_id(cls, v, values):
        ''' Check if the msg_id is in the allowed values for the related msg_class.
        '''
        msg_class = values['msg_class']
        class_enum = None

        if msg_class is MsgClassEnum.control:
            class_enum = MsgControlIdEnum
        elif msg_class is MsgClassEnum.soh:
            class_enum = MsgSohIdEnum
        elif msg_class is MsgClassEnum.data:
            class_enum = MsgDataIdEnum

        if v not in class_enum.__members__.values():
            raise ValueError('The msg_id "{msg_id}" is not allowed.'.format(msg_id = v))
        return v


class WSMessage(pydantic.BaseModel):
    header: WSMessageHeader
    payload: dict
