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
@author: Stefan Mertl
'''
import unittest
import logging
import threading

import obspy.core
from obspy.core.utcdatetime import UTCDateTime

import mss_dataserver
import mss_dataserver.core.validation as validation


class ValidationTestCase(unittest.TestCase):
    """
    Test suite.
    """
    @classmethod
    def setUpClass(cls):
        # Configure the logger.
        cls.logger = logging.getLogger('mss_dataserver')
        cls.logger.addHandler(mss_dataserver.get_logger_handler(log_level = 'DEBUG'))


    @classmethod
    def tearDownClass(cls):
        pass


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_event_validation(self):
        ''' Test event validation.
        '''
        # Test a complete event.
        event = validation.Event(id = 1,
                                 start_time = UTCDateTime('2020-01-01T01:00:00').isoformat(),
                                 end_time = UTCDateTime('2020-01-01T01:01:00').isoformat(),
                                 description = 'Description',
                                 comment = 'Comment',
                                 max_pgv = 0.001,
                                 state = None)

        self.assertIsInstance(event.dict(), dict)

        event = validation.Event(id = None,
                                 start_time = UTCDateTime('2020-01-01T01:00:00').isoformat(),
                                 end_time = UTCDateTime('2020-01-01T01:01:00').isoformat(),
                                 description = 'Description',
                                 comment = 'Comment',
                                 max_pgv = 0.001,
                                 state = 'created')

        self.assertIsInstance(event.dict(), dict)

        event = validation.Event(id = None,
                                 start_time = UTCDateTime('2020-01-01T01:00:00').isoformat(),
                                 end_time = UTCDateTime('2020-01-01T01:01:00').isoformat(),
                                 description = None,
                                 comment = None,
                                 max_pgv = 0.001,
                                 state = None)

        self.assertIsInstance(event.dict(), dict)


    def test_ws_message_validation(self):
        ''' Test the validation of websocket messages.
        '''
        server_time = UTCDateTime('2020-01-01T01:00:00').isoformat()
        msg_header = validation.WSMessageHeader(msg_class = 'control',
                                                msg_id = 'mode',
                                                server_time = server_time)
        self.assertEqual(msg_header.msg_class, 'control')
        self.assertEqual(msg_header.msg_id, 'mode')
        self.assertIsInstance(msg_header.dict(), dict)

        msg_header = validation.WSMessageHeader(msg_class = 'soh',
                                                msg_id = 'connection',
                                                server_time = server_time)
        self.assertEqual(msg_header.msg_class, 'soh')
        self.assertEqual(msg_header.msg_id, 'connection')
        self.assertIsInstance(msg_header.dict(), dict)

        msg_header = validation.WSMessageHeader(msg_class = 'soh',
                                                msg_id = 'server_state',
                                                server_time = server_time)
        self.assertEqual(msg_header.msg_class, 'soh')
        self.assertEqual(msg_header.msg_id, 'server_state')
        self.assertIsInstance(msg_header.dict(), dict)

        msg_header = validation.WSMessageHeader(msg_class = 'data',
                                                msg_id = 'current_pgv',
                                                server_time = server_time)
        self.assertEqual(msg_header.msg_class, 'data')
        self.assertEqual(msg_header.msg_id, 'current_pgv')
        self.assertIsInstance(msg_header.dict(), dict)


        msg_header = validation.WSMessageHeader(msg_class = 'control',
                                                msg_id = 'mode',
                                                server_time = server_time)
        msg = validation. WSMessage(header = msg_header,
                                    payload = {})

def suite():
    return unittest.makeSuite(ValidationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
