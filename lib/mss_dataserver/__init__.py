# -*- coding: utf-8 -*-
##############################################################################
 # LICENSE
 #
 # This file is part of mss_dataserver.
 # 
 # If you use mss_vis in any program or publication, please inform and
 # acknowledge its authors.
 # 
 # mss_vis is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 # 
 # mss_vis is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 # 
 # You should have received a copy of the GNU General Public License
 # along with mss_vis. If not, see <http://www.gnu.org/licenses/>.
 #
 # Copyright 2019 Stefan Mertl
##############################################################################

__version__ = "0.0.1"
__author__ = "Stefan Mertl"
__authorEmail__ = "info@mertl-research.at"
__description__ = "MacroSeismicSensor websocket data server."
__longDescription__ = """ 
                      """
__website__ = "http://www.macroseismicsensor.at"
__downloadUrl__ = ""
__license__ = "GNU General Public Licence version 3"
__keywords__ = """raspberrypi seismic seismological data logger datalogger recorder macroseismic"""

import logging


def get_logger_handler(log_level = None):
    ch = logging.StreamHandler()
    if log_level is None:
        log_level = 'INFO'
    ch.setLevel(log_level)
    formatter = logging.Formatter("#LOG# - %(asctime)s - %(process)d - %(levelname)s - %(name)s: %(message)s")
    ch.setFormatter(formatter)

    return ch

