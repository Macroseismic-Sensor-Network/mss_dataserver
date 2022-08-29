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
# Copyright 2021 Stefan Mertl
##############################################################################
''' The event classifyer.
'''

import logging


class EventClassifyer(object):
    ''' Classify an event.

    '''

    def __init__(self, public_id, meta, pgv_df):
        ''' Initialize the instance.
        '''
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # The public id of the event to classify.
        self.public_id = public_id

        # The metadata of the event to classify.
        self.meta = meta

        # The event instance to classify.
        self.event = None


    def classify(self):
        ''' Run the event classification.
        '''
        # Check if the event is a quarry blast.
        self.test_for_quarry_blast(self)


    def test_for_quarry_blast(self):
        ''' Test if the event is a quarry blast.
        '''
        # Test for quarry blasts of the quarry Dürnbach.
        pass
