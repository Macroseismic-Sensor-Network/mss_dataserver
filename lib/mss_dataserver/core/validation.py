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

''' Pydantic data validation models.

The validation models are used to validate the data messages
sent and received using the websocket interface.
'''

import enum

import pydantic
from pydantic import (
    confloat,
    conlist,
    constr,
    PositiveInt,
    PositiveFloat,
    validator
)

from typing import (
    Dict,
    List,
    Optional,
    Union,
)


class Event(pydantic.BaseModel):
    ''' The Event validation model.
    '''
    db_id: Union[None, PositiveInt]
    public_id: constr(regex=r'^\w+_\w+_\d{4}-\d{2}-\d{2}T\d{6,12}')
    start_time: constr(min_length=19, max_length=26)
    end_time: constr(min_length=19, max_length=26)
    length: confloat(ge=0)
    max_pgv: PositiveFloat
    num_detections: PositiveInt
    triggered_stations: List[constr(regex=r'^\w{1,10}:\w{1,10}:\w{1,4}')]
    event_class: Optional[constr(max_length=20)] = None
    event_region: Optional[constr(max_length=30)] = None
    event_class_mode: Optional[constr(max_length=20)] = None
    hypo: Optional[conlist(float, min_items = 3, max_items = 3)] = None
    hypo_dist: Optional[Dict[str, float]] = None
    epi_dist: Optional[Dict[str, float]] = None
    magnitude: Optional[float] = None
    description: Optional[constr(max_length=255)] = None
    comment: Optional[constr(max_length=255)] = None
    state: Optional[constr(max_length=20)] = None
    pgv_3d: Optional[Dict[str, float]] = None
    f_dom: Optional[float] = None
    foreign_id: Optional[constr(max_length=10)] = None
    

class MsgClassEnum(str, enum.Enum):
    ''' The websocket message class enumeration.
    '''
    control = 'control'
    data = 'data'
    soh = 'soh'
    request = 'request'
    cancel = 'cancel'


class MsgControlIdEnum(str, enum.Enum):
    ''' The control message id enumeration.
    '''
    mode = 'mode'


class MsgRequestIdEnum(str, enum.Enum):
    ''' The request message id enumeration.
    '''
    event_supplement = 'event_supplement'
    pgv_timeseries = 'pgv_timeseries'


class MsgCancelIdEnum(str, enum.Enum):
    ''' The cancel message id enumeration.
    '''
    pgv_timeseries = 'pgv_timeseries'


class MsgSohIdEnum(str, enum.Enum):
    ''' The SOH message id enumeration.
    '''
    connection = 'connection'
    server_state = 'server_state'


class MsgDataIdEnum(str, enum.Enum):
    ''' The data message id enumeration.
    '''
    current_pgv = 'current_pgv'
    pgv_timeseries = 'pgv_timeseries'
    pgv_archive = 'pgv_timeseries_archive'
    detection_result = 'detection_result'
    event_data = 'event_data'
    recent_events = 'recent_events'
    event_archive = 'event_archive'
    event_warning = 'event_warning'
    keydata = 'keydata'
    event_supplement = 'event_supplement'
    station_metadata = 'station_metadata'


class WSMessageHeader(pydantic.BaseModel):
    ''' The websocket message header model.
    '''
    msg_class: MsgClassEnum
    msg_id: Union[MsgControlIdEnum,
                  MsgDataIdEnum,
                  MsgSohIdEnum,
                  MsgRequestIdEnum]
    server_time: Optional[constr(regex=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?')] = None

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
        elif msg_class is MsgClassEnum.request:
            class_enum = MsgRequestIdEnum
        elif msg_class is MsgClassEnum.cancel:
            class_enum = MsgCancelIdEnum

        if v not in class_enum.__members__.values():
            raise ValueError('The msg_id "{msg_id}" is not allowed.'.format(msg_id = v))
        return v


class WSMessage(pydantic.BaseModel):
    ''' The websocket message model.
    '''
    header: WSMessageHeader
    payload: dict


class MsgControlModeDataModeEnum(str, enum.Enum):
    ''' The control:datamode enumeration.
    '''
    pgv = 'pgv'
    keydata = 'keydata'


class MsgControlModePayload(pydantic.BaseModel):
    ''' The control:mode message model.
    '''
    data_mode: MsgControlModeDataModeEnum


class MsgRequestEventSupplementNameEnum(str, enum.Enum):
    ''' The request:event_supplement_name enumeration.
    '''
    pgvstation = 'pgvstation'
    pgvvoronoi = 'pgvvoronoi'
    simplices = 'simplices'
    isoseismalfilledcontours = 'isoseismalfilledcontours'


class MsgRequestEventSupplementCategoryEnum(str, enum.Enum):
    ''' The request:event_supplement_category enumeration.
    '''
    eventpgv = 'eventpgv'
    pgvsequence = 'pgvsequence'
    detectionsequence = 'detectionsequence'


class MsgRequestEventSupplementPayload(pydantic.BaseModel):
    ''' The request:event_supplement message payload model.
    '''
    public_id: constr(regex=r'^\w+_\w+_\d{4}-\d{2}-\d{2}T\d{6,12}')
    selection: List[Dict[str, str]]

    @validator('selection')
    def check_selection(cls, v, values):
        ''' Check if the dictionary contains valid category and name entries.
        '''
        for cur_supp_name in v:
            keys = list(cur_supp_name.keys())
            if sorted(keys) != ['category', 'name']:
                raise ValueError('Wrong dictionary keys for the supplement name.')

        return v


class MsgRequestPgvTimeseriesPayload(pydantic.BaseModel):
    ''' The request:pgv_timeseries message payload model.
    '''
    nsl_code: constr(regex=r'^\w{1,10}:\w{1,10}:\w{1,4}')


class MsgCancelPgvTimeseriesPayload(pydantic.BaseModel):
    ''' The cancel:pgv_timeseries message payload model.
    '''
    nsl_code: constr(regex=r'^\w{1,10}:\w{1,10}:\w{1,4}')
