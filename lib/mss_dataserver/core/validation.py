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

import pydantic
from pydantic import (
    PositiveInt,
    PositiveFloat,
    confloat,
    constr,
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
