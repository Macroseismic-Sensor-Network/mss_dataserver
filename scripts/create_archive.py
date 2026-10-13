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

import gzip
import json
import logging
import os
import threading

import asyncio
import click
import obspy

import mss_dataserver.core.project
import mss_dataserver.core.util as util
import mss_dataserver.monitorclient.monitorclient as monitorclient


def create_monitor_client(project, config):
    # Get the logging instance.
    logger_name = 'mss_dataserver'
    logger = logging.getLogger(logger_name)

    logger.info('Creating the monitor client.')

    project.connect_to_db()
    project.load_inventory()

    if project.inventory is None:
        logger.error("No inventory found in the project. Quitting.")
        return

    # Create the monitor stream attributes
    # TODO: It seems that the monitor_stream and the stream_lock could be
    # defined in the MonitorClient class.
    monitor_stream = obspy.core.Stream()
    stream_lock = threading.Lock()
    host = config['seedlink']['host']
    port = config['seedlink']['port']
    stations = config['process']['stations']
    data_dir = config['output']['data_dir']
    event_dir = config['output']['event_dir']
    process_interval = config['process']['interval']
    pgv_sps = config['process']['pgv_sps']
    pgv_archive_time = config['process']['pgv_archive_time']
    trigger_thr = config['process']['trigger_threshold']
    warn_thr = config['process']['warn_threshold']
    valid_event_thr = config['process']['valid_event_threshold']
    event_archive_timespan = config['process']['event_archive_timespan']
    min_event_length = config['process']['min_event_length']
    min_event_detections = config['process']['min_event_detections']

    if len(stations) == 0:
        stations = None

    # Start the seedlink monitor client thread.
    logger.info('Creating the Seedlink client.')
    server_url = host + ':' + str(port)
    stop_event = threading.Event()
    loop = asyncio.get_event_loop()
    # TODO: Check the communication between the monitor client and the asyncio
    # loop using the event loop. Is this a good way?
    # TODO: Check the thread-safe access to the resources of the monitorclient
    # from the asyncio tasks.
    client = monitorclient.MonitorClient(project = project,
                                         asyncio_loop = loop,
                                         server_url = server_url,
                                         stations = stations,
                                         monitor_stream = monitor_stream,
                                         stream_lock = stream_lock,
                                         data_dir = data_dir,
                                         event_dir = event_dir,
                                         process_interval = process_interval,
                                         pgv_sps = pgv_sps,
                                         stop_event = stop_event,
                                         pgv_archive_time = pgv_archive_time,
                                         trigger_thr = trigger_thr,
                                         warn_thr = warn_thr,
                                         valid_event_thr = valid_event_thr,
                                         event_archive_timespan = event_archive_timespan,
                                         min_event_length = min_event_length,
                                         min_event_detections = min_event_detections)

    return client


@click.command()
@click.argument('config_file')
def cli(config_file):
    # Load the config file.
    if not os.path.exists(config_file):
        print('ERROR: Configuration file not found: %s.', config_file)
        raise click.Abort()
    config = util.load_configuration(config_file)

    # Create the logging instance.
    logger_name = 'mss_dataserver'
    logger = logging.getLogger(logger_name)
    logger.setLevel(config['log']['loglevel'])
    log_dir = config['log']['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filepath = os.path.join(log_dir, 'mss_dataserver.log')
    handler = util.get_logger_rotating_file_handler(filename = log_filepath,
                                                    log_level = config['log']['loglevel'],
                                                    max_bytes = config['log']['max_bytes'],
                                                    backup_count = config['log']['backup_count'])
    logger.addHandler(handler)

    #logging.basicConfig(level = config['log']['loglevel'],
    #                    format = "LOG - %(asctime)s - %(process)d - %(levelname)s - %(name)s: %(message)s")
    logger.info("Creating the project.")
    project = mss_dataserver.core.project.Project(**config)

    client = create_monitor_client(project = project,
                                   config = config)

    archive_dir = os.path.join(config['output']['data_dir'],
                               'archive')
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)

    supplements_dir = os.path.join(archive_dir,
                                   'supplements')
    if not os.path.exists(supplements_dir):
        os.makedirs(supplements_dir)
    
    # Get the station metadata.
    logger.info("Requesting the station_metadata.")
    station_metadata = client.get_station_metadata()
    out_file_name = os.path.join(archive_dir,
                                 'station_metadata.json.gz')
    logger.info("Saving the station metadata to %s.", out_file_name)
    with gzip.open(out_file_name, 'wt') as zip_file:
        json.dump(station_metadata, zip_file)

    # Get the recent events.
    logger.info("Requesting the recent events.")
    recent_events = client.get_recent_events()
    out_file_name = os.path.join(archive_dir,
                                 'recent_events.json.gz')
    logger.info("Saving the recent events to %s.", out_file_name)
    with gzip.open(out_file_name, 'wt') as zip_file:
        json.dump(recent_events, zip_file)


    # The supplement selection as used in mss_vis.
    supplement_selection = [{'category': 'eventpgv',
                             'name': 'pgvstation'}, 
                            {'category': 'eventpgv',
                             'name': 'pgvvoronoi'},
                            {'category': 'eventpgv',
                             'name': 'isoseismalfilledcontour'},
                            {'category': 'pgvsequence',
                             'name': 'pgvstation'},
                            {'category': 'pgvsequence',
                             'name': 'pgvvoronoi'},
                            {'category': 'localize',
                             'name': 'origins'}]

    logger.info("Archiving %d events.", len(recent_events))
    
    for cur_key, cur_event in recent_events.items():
        cur_year = cur_event['start_time'].split('-')[0]
        logger.info("Getting supplements for %s.", cur_key)
        cur_supplement = client.get_event_supplement(public_id = cur_event['public_id'],
                                                     selection = supplement_selection)

        cur_out_dir = os.path.join(supplements_dir,
                                   cur_year)
        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)
        cur_file_name = "{}_supplements.json.gz".format(cur_key)
        out_file_name = os.path.join(cur_out_dir,
                                     cur_file_name)
        logger.info("Saving the supplement data to %s.", out_file_name)
        with gzip.open(out_file_name, 'wt') as zip_file:
            json.dump(cur_supplement, zip_file)
    
if __name__ == '__main__':
    cli()




