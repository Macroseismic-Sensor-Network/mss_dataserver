#!/usr/bin/env python
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
Remove the database tables of a psysmon project.

:copyright:
    Stefan Mertl

:license:
    GNU General Public License, Version 3 
    (http://www.gnu.org/licenses/gpl-3.0.html)
'''
from __future__ import print_function

import sys
from mss_dataserver.core.test_util import drop_database_tables

def run():
    if len(sys.argv) <= 4:
        print("7 arguments required (db_dialect, db_user, db_host, db_name, db_pwd, db_driver).\n")
        sys.exit()

    db_dialect = sys.argv[1]
    db_user = sys.argv[2]
    db_host = sys.argv[3]
    db_name = sys.argv[4]

    if len(sys.argv) >= 6:
        db_pwd = sys.argv[5]
    else:
        db_pwd = ''

    if len(sys.argv) == 7:
        db_driver = sys.argv[6]
    else:
        db_driver = 'pymysql'

    drop_database_tables(db_dialect, db_driver, db_user, db_pwd, db_host, db_name) 

if __name__ == '__main__':
    run()




