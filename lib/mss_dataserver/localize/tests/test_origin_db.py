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

import unittest
import logging

import obspy

import mss_dataserver
import mss_dataserver.test.util as test_util


class OriginTestCase(unittest.TestCase):
    """
    Test suite for psysmon.packages.geometry.editGeometry.EditGeometryDlg
    """
    @classmethod
    def setUpClass(cls):
        # Configure the logger.
        cls.logger = logging.getLogger('mss_dataserver')
        handler = mss_dataserver.get_logger_handler(log_level = 'DEBUG')
        cls.logger.addHandler(handler)

        cls.project = test_util.create_db_test_project()
        test_util.clear_project_database_tables(cls.project)
        cls.project.load_inventory(update_from_xml = True)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_write_origin_to_database(self):
        ''' Test the adding of an origin to the database.
        '''
        print("Test-Test-Test")
        pass


def suite():
    return unittest.makeSuite(OriginTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

