#! /usr/bin/env python3

import datetime
import logging
import os
import threading
import time

import asyncio
import numpy as np
import obspy
import obspy.core.utcdatetime as utcdatetime
import obspy.clients.seedlink.easyseedlink as easyseedlink
from obspy.clients.seedlink.slpacket import SLPacket


class EasySeedLinkClientException(Exception):
    """
    A base exception for all errors triggered explicitly by EasySeedLinkClient.
    """
    # XXX Base on SeedLinkException?
    pass


class MonitorClient(easyseedlink.EasySeedLinkClient):
    """
    A custom SeedLink client
    """
    def __init__(self, server_url, stations, inventory, monitor_stream, stream_lock, data_dir, process_interval, stop_event, pgv_sps = 1, autoconnect = False):
        ''' Initialize the instance.
        '''
        easyseedlink.EasySeedLinkClient.__init__(self,
                                                 server_url = server_url,
                                                 autoconnect = autoconnect)
        self.logger = logging.getLogger('mss_data_server')

        # The stations to stream from the seedlink server.
        self.stations = stations

        # The recent, not yet processed data.
        self.monitor_stream = monitor_stream

        # The stream used for processing.
        self.process_stream = obspy.core.Stream()

        # The pgv stream holding the most recent data which has not been sent
        # over the websocket.
        self.pgv_stream = obspy.core.Stream()

        # The pgv archive stream holding the PGV data of the specified archive
        # time.
        self.pgv_archive_stream = obspy.core.Stream()

        # The length of the archive stream to keep [s].
        self.pgv_archive_time = 1800

        # The no-data value.
        self.nodata_value = -999999

        self.stream_lock = stream_lock

        self.data_dir = data_dir

        # The time interval [s] used to process the received data.
        self.process_interval = process_interval

        # The samples per second of the PGV data stream.
        self.pgv_sps = pgv_sps

        self.stop_event = stop_event

        # The PGV data state.
        self.pgv_data_available = asyncio.Event()

        # The psysmon geometry inventory.
        self.inventory = inventory

        self.recorder_map = self.get_recorder_mappings(station_nsl = self.stations)

        self.conn.timeout = 10

        self.connect()

        for cur_mss in self.recorder_map:
            self.select_stream(cur_mss[0],
                               cur_mss[1],
                               cur_mss[2] + cur_mss[3])

    def get_recorder_mappings(self, station_nsl = None):
        ''' Get the mappings of the seedlink SCNL to the MSS SCNL.
        '''
        recorder_map = {}
        if station_nsl is None:
            station_list = self.inventory.get_station()
        else:
            station_list = []
            for cur_nsl in station_nsl:
                cur_station = self.inventory.get_station(network = cur_nsl[0],
                                                         name = cur_nsl[1],
                                                         location = cur_nsl[2])
                if len(cur_station) > 1:
                    raise ValueError('There are more than one stations. This is not yet supported.')
                if len(cur_station) == 0:
                    raise ValueError('No station found for {0:s}. Check the input file.'.format(cur_nsl))
                cur_station = cur_station[0]
                station_list.append(cur_station)

        for station in station_list:

            for cur_channel in station.channels:
                stream_tb = cur_channel.get_stream(start_time = obspy.UTCDateTime())
                cur_loc = stream_tb[0].item.name.split(':')[0]
                cur_chan = stream_tb[0].item.name.split(':')[1]

                cur_key = ('XX',
                           stream_tb[0].item.serial,
                           cur_loc,
                           cur_chan)
                recorder_map[cur_key] = cur_channel.scnl

        self.logger.debug(recorder_map)
        return recorder_map

    def reconnect(self):
        ''' Reconnect to the server.'''
        self.logger.info('Reconnecting to the server.')
        self.close()

        self.connect()
        for cur_station in self.stations:
            self.select_stream(cur_station[0], cur_station[1], cur_station[2])

    def on_data(self, trace):
        """
        Override the on_data callback
        """
        #self.logger.debug('Received trace:')
        #self.logger.debug(str(trace))
        #self.logger.debug("on_data trace data: %s", trace.data)
        self.stream_lock.acquire()
        cur_scnl = self.recorder_map[tuple(trace.id.split('.'))]
        trace.stats.network = cur_scnl[2]
        trace.stats.station = cur_scnl[0]
        trace.stats.location = cur_scnl[3]
        trace.stats.channel = cur_scnl[1]
        self.monitor_stream.append(trace)
        #self.monitor_stream.merge(method = 1, fill_value = 'interpolate')
        #self.logger.debug(self.monitor_stream)
        self.stream_lock.release()
        #self.logger.debug('Leaving on_data.')

    def run(self):
        """
        Start streaming data from the SeedLink server.

        Streams need to be selected using
        :meth:`~.EasySeedLinkClient.select_stream` before this is called.

        This method enters an infinite loop, calling the client's callbacks
        when events occur.
        """
        # Note: This somewhat resembles the run() method in SLClient.

        # Check if any streams have been specified (otherwise this will result
        # in an infinite reconnect loop in the SeedLinkConnection)
        if not len(self.conn.streams):
            msg = 'No streams specified. Use select_stream() to select ' + \
                  'a stream.'
            raise EasySeedLinkClientException(msg)

        self.__streaming_started = True

        # Start the processing timer thread.
        #self.process_timer_thread = threading.Thread(name = 'process_timer',
        #                                             target = self.task_timer,
        #                                             args = (self.process_monitor_stream, ))

        #self.process_timer_thread.start()


        # Start the collection loop
        start = time.time()
        end = None
        while not self.stop_event.is_set():
            self.logger.debug('Waiting to collect data....')
            data = self.conn.collect()
            self.logger.debug('... received some data.')

            if data == SLPacket.SLTERMINATE:
                self.on_terminate()
                continue
            elif data == SLPacket.SLERROR:
                self.on_seedlink_error()
                continue

            # At this point the received data should be a SeedLink packet
            # XXX In SLClient there is a check for data == None, but I think
            #     there is no way that self.conn.collect() can ever return None
            assert(isinstance(data, SLPacket))

            packet_type = data.get_type()

            # Ignore in-stream INFO packets (not supported)
            if packet_type not in (SLPacket.TYPE_SLINF, SLPacket.TYPE_SLINFT):
                # The packet should be a data packet
                trace = data.get_trace()
                # Pass the trace to the on_data callback
                self.on_data(trace)
                end = time.time()


            # Start the data export thread if the write_interval has been
            # reached.
            #if end and (end - start) > self.write_interval:
                #process_thread = threading.Thread(target = process_monitor_stream, args = (self.monitor_stream, self.stream_lock, self.data_dir))
                #logging.info('Starting the processing thread.')
                #process_thread.start()
                #start = end
                #end = None

    def on_terminate(self):
        ''' Handle termination of the client.
        '''
        self.logger.warning('Terminating the client.')
        # Clean the state.
        self.stream_lock.acquire()
        self.monitor_stream.clear()
        self.stream_lock.release()

        # Reconnect to the server.
        try:
            self.reconnect()
        except Exception:
            self.logger.error("Can't reconnect to the server.")
            self.stop_event.set()

    def on_seedlink_error(self):
        ''' Handle client errors.
        '''
        self.logger.error('Got a seedlink error.')

    async def task_timer(self, callback):
        ''' A timer executing a task at regular intervals.
        '''
        self.logger.debug('Starting the timer.')
        interval = int(self.process_interval)
        now = obspy.UTCDateTime()
        delay_to_next_interval = interval - (now.timestamp % interval)
        self.logger.debug('Sleeping for %f seconds.', delay_to_next_interval)
        time.sleep(delay_to_next_interval)

        while not self.stop_event.is_set():
            try:
                self.logger.debug('Executing callback.')
                callback()
            except Exception as e:
                self.logger.exception(e)
                self.stop()

            now = obspy.UTCDateTime()
            delay_to_next_interval = interval - (now.timestamp % interval)
            self.logger.debug('Sleeping for %f seconds.', delay_to_next_interval)
            #time.sleep(delay_to_next_interval)
            await asyncio.sleep(delay_to_next_interval)

        self.logger.info("Leaving the task_timer method.")

    def process_monitor_stream(self):
        ''' Process the data in the monitor stream.

        '''
        logger = logging.getLogger('mss_data_server.write_monitor_stream')
        logger.info('Processing the monitor_stream.')
        self.stream_lock.acquire()
        monitor_stream_length = len(self.monitor_stream)
        self.monitor_stream.merge()
        self.stream_lock.release()
        if monitor_stream_length > 0:
            #now = obspy.UTCDateTime()
            #process_end_time = now - now.timestamp % self.process_interval
            #logger.debug('Trimming to end time: %s.', process_end_time)
            self.stream_lock.acquire()
            for cur_trace in self.monitor_stream:
                sec_remain = cur_trace.stats.endtime.timestamp % self.process_interval
                cur_end_time = obspy.UTCDateTime(round(cur_trace.stats.endtime.timestamp - sec_remain))
                cur_end_time = cur_end_time - cur_trace.stats.delta
                logger.debug('Trimming %s to end time: %f.', cur_trace.id, cur_end_time.timestamp)
                cur_slice_trace = cur_trace.slice(endtime = cur_end_time)
                if len(cur_slice_trace) > 0:
                    self.process_stream.append(cur_slice_trace)
                    cur_trace.trim(starttime = cur_end_time + cur_trace.stats.delta)
            self.stream_lock.release()

            logger.info('process_stream: %s', str(self.process_stream))

            with self.stream_lock:
                logger.info('monitor_stream: %s', str(self.monitor_stream))

            # Get a stream containing only equal length traces per station.
            el_stream = self.get_equal_length_traces()
            self.logger.debug('Got el_stream: %s.', el_stream)

            # Detrend to remove eventual offset.
            el_stream = el_stream.split()
            el_stream.detrend(type = 'constant')
            el_stream.merge()

            # Convert the amplitudes from counts to m/s.
            self.convert_to_physical_units(el_stream)

            # Compute the PGV values.
            self.compute_pgv(el_stream)

            # Trim the archive stream.
            self.trim_archive()

            # Signal available PGV data.
            self.pgv_data_available.set()
            return



            if len(set([len(x) for x in export_stream])) != 1:
                logger.warning('The length of the traces in the stream is not equal. No export.')
                logger.debug('Add the stream back to the monitor stream.')
                stream_lock.acquire()
                monitor_stream += export_stream
                monitor_stream.merge()
                logger.debug(monitor_stream)
                stream_lock.release()
            else:
                if len(export_stream) > 0:
                    logger.info('Exporting the stream.')
                    for cur_trace in export_stream:
                        logger.debug(cur_trace.get_id().replace('.', '_'))
                        logger.debug(cur_trace.stats.starttime.isoformat())
                        cur_filename = 'syscom_{0:s}_{1:s}.msd'.format(cur_trace.get_id().replace('.', '_'), cur_trace.stats.starttime.isoformat().replace('-', '').replace(':', '').replace('.', '_').replace('T', '_'))
                        cur_filename = os.path.join(data_dir, cur_filename)
                        export_stream.write(cur_filename, format = 'MSEED', reclen = 512, encoding = 11)
                    logger.info('Done.')
                else:
                    logger.info('No data in stream.')

            # Delete old files.
            now = utcdatetime.UTCDateTime()
            for cur_file in os.listdir(data_dir):
                if os.stat(os.path.join(data_dir, cur_file)).st_mtime < now - 60:
                    os.remove(os.path.join(data_dir, cur_file))

    def get_equal_length_traces(self):
        ''' Get a stream containing traces with equal length per station.
        '''
        unique_stations = [(x.stats.network, x.stats.station, x.stats.location) for x in self.process_stream]
        unique_stations = list(set(unique_stations))
        el_stream = obspy.core.Stream()
        for cur_station in unique_stations:
            self.logger.debug('Getting stream for %s.', cur_station)
            # Get all traces for the station.
            cur_stream = self.process_stream.select(network = cur_station[0],
                                                    station = cur_station[1],
                                                    location = cur_station[2])

            self.logger.debug('Selected stream: %s.', cur_stream)
            for cur_trace in cur_stream:
                self.process_stream.remove(cur_trace)
            cur_stream = cur_stream.merge()

            # Check if all required channels are present:
            missing_data = False
            required_scnl = [x for x in self.recorder_map.values()
                             if x[0] == cur_station[1]]
            available_scnl = [(x.stats.station,
                               x.stats.channel,
                               x.stats.network,
                               x.stats.location) for x in cur_stream]
            self.logger.debug('available_scnl: %s', available_scnl)
            self.logger.debug('required_scnl: %s', required_scnl)

            for cur_required_scnl in required_scnl:
                if cur_required_scnl not in available_scnl:
                    self.logger.debug('Data for channel %s is missing.',
                                      cur_required_scnl)
                    missing_data = True
                    break

            if missing_data:
                self.logger.debug('Skipping station because of missing channel data.')
                self.process_stream.extend(cur_stream)
                continue

            # Select a timespan with equal end time.
            min_end = np.min([x.stats.endtime for x in cur_stream])
            cur_el_stream = cur_stream.slice(endtime = min_end)
            cur_rem_stream = cur_stream.trim(starttime = min_end + cur_el_stream[0].stats.delta)
            self.process_stream.extend(cur_rem_stream)
            self.logger.debug('el_stream: %s', cur_el_stream)
            self.logger.debug('rem_stream: %s', cur_rem_stream)
            el_stream.extend(cur_el_stream)

        return el_stream

    def compute_pgv(self, stream):
        ''' Compute the PGV values of the stream.
        '''
        self.logger.debug('Computing the PGV.')
        unique_stations = [(x.stats.network, x.stats.station, x.stats.location) for x in stream]
        unique_stations = list(set(unique_stations))
        samp_interval = 1 / self.pgv_sps

        for cur_station in unique_stations:
            self.logger.debug('Getting stream for %s.', cur_station)
            # Get all traces for the station.
            cur_stream = stream.select(network = cur_station[0],
                                       station = cur_station[1],
                                       location = cur_station[2])

            self.logger.debug('Selected stream: %s.', cur_stream)

            min_start = np.min([x.stats.starttime for x in cur_stream])
            sec_remain = min_start.timestamp % samp_interval
            cur_offset = -sec_remain
            self.logger.debug('offset: %f.', cur_offset)
            cur_delta = cur_stream[0].stats.delta

            for win_st in cur_stream.slide(window_length = samp_interval - cur_delta,
                                           step = samp_interval,
                                           offset = cur_offset):
                self.logger.debug('Processing stream slice: %s.', win_st)
                x_st = win_st.select(channel = 'Hparallel')
                y_st = win_st.select(channel = 'Hnormal')

                cur_pgv = []
                for cur_x_trace, cur_y_trace in zip(x_st.traces, y_st.traces):
                    cur_x = cur_x_trace.data
                    cur_y = cur_y_trace.data

                    if len(cur_x) != len(cur_y):
                        self.logger.error("The x and y data lenght dont't match. Can't compute the res. PGV for this trace.")
                        continue

                    cur_sec_remain = cur_x_trace.stats.starttime.timestamp % samp_interval
                    cur_win_start = cur_x_trace.stats.starttime - cur_sec_remain
                    #cur_win_end = cur_win_start + self.process_interval - cur_x_trace.stats.delta
                    #cur_pgv_time = cur_x_trace.stats.starttime + \
                    #               (cur_x_trace.stats.endtime - cur_x_trace.stats.starttime) / 2
                    cur_pgv_time = cur_win_start + samp_interval / 2
                    cur_pgv_value = np.max(np.sqrt(cur_x**2 + cur_y**2))
                    cur_pgv.append([cur_pgv_time, cur_pgv_value])

                if cur_pgv:
                    cur_pgv = np.array(cur_pgv)
                    #self.logger.debug('cur_x: %s', cur_x)
                    #self.logger.debug('cur_y: %s', cur_y)
                    self.logger.debug('cur_pgv: %s', cur_pgv)
                    cur_stats = {'network': cur_x_trace.stats.network,
                                 'station': cur_x_trace.stats.station,
                                 'location': cur_x_trace.stats.location,
                                 'channel': 'pgv',
                                 'sampling_rate': self.pgv_sps,
                                 'starttime': cur_pgv[0, 0]}
                    self.logger.debug('data type: %s.', cur_pgv[:, 1].dtype)
                    cur_pgv_trace = obspy.core.Trace(data = cur_pgv[:, 1].astype(np.float32),
                                                     header = cur_stats)
                    # Write the data to the current pgv stream and the archive
                    # stream.
                    self.pgv_stream.append(cur_pgv_trace)
                    self.pgv_archive_stream.append(cur_pgv_trace)

        # Merge the current pgv stream.
        self.pgv_stream.merge()
        self.logger.debug('pgv_stream: %s.', self.pgv_stream)

        # Merge the archive stream.
        self.pgv_archive_stream.merge(fill_value = self.nodata_value)
        self.logger.info("pgv_archive_stream: %s", self.pgv_archive_stream)

        #self.pgv_archive_stream.write("/home/stefan/Schreibtisch/pgv_beben_neunkirchen.msd",
        #                             format = "MSEED",
        #                             reclen = 512)

    def convert_to_physical_units(self, stream):
        ''' Convert the counts to physical units.
        '''
        for tr in stream.traces:
            station = self.inventory.get_station(network = tr.stats.network,
                                                 name = tr.stats.station,
                                                 location = tr.stats.location)
            if len(station) > 1:
                raise ValueError('There are more than one stations. This is not yet supported.')
            station = station[0]

            channel = station.get_channel(name = tr.stats.channel)

            if len(channel) > 1:
                raise ValueError('There are more than one channels. This is not yet supported.')
            channel = channel[0]

            stream_tb = channel.get_stream(start_time = tr.stats.starttime,
                                           end_time = tr.stats.endtime)

            if len(stream_tb) > 1:
                raise ValueError('There are more than one recorder streams. This is not yet supported.')
            rec_stream = stream_tb[0].item

            rec_stream_param = rec_stream.get_parameter(start_time = tr.stats.starttime,
                                                        end_time = tr.stats.endtime)
            if len(rec_stream_param) > 1:
                raise ValueError('There are more than one recorder stream parameters. This is not yet supported.')
            rec_stream_param = rec_stream_param[0]


            components_tb = rec_stream.get_component(start_time = tr.stats.starttime,
                                                     end_time = tr.stats.endtime)

            if len(components_tb) > 1:
                raise ValueError('There are more than one components. This is not yet supported.')
            component = components_tb[0].item
            comp_param = component.get_parameter(start_time = tr.stats.starttime,
                                                 end_time = tr.stats.endtime)

            if len(comp_param) > 1:
                raise ValueError('There are more than one parameters for this component. This is not yet supported.')

            comp_param = comp_param[0]

            self.logger.debug("bitweight: %f", rec_stream_param.bitweight)
            self.logger.debug("gain: %f", rec_stream_param.gain)
            self.logger.debug("sensitivity: %f", comp_param.sensitivity)
            tr.data = tr.data * rec_stream_param.bitweight / (rec_stream_param.gain * comp_param.sensitivity)
            tr.stats.unit = component.output_unit.strip()

    def compute_pgv_res(st):
        ''' Compute the resultant of the peak-ground-velocity.
        '''
        x_st = st.select(channel = 'Hparallel').merge()
        y_st = st.select(channel = 'Hnormal').merge()

        res_st = obspy.core.Stream()
        for cur_x_trace, cur_y_trace in zip(x_st.traces, y_st.traces):
            cur_x = cur_x_trace.data
            cur_y = cur_y_trace.data

            if len(cur_x) != len(cur_y):
                logger.error("The x and y data lenght dont't match. Can't compute the res. PGV for this trace.")
                continue

            self.logger.debug("x: %s", cur_x)
            self.logger.debug("y: %s", cur_y)

            cur_res = np.sqrt(cur_x**2 + cur_y**2)

            cur_stats = {'network': cur_x_trace.stats['network'],
                         'station': cur_x_trace.stats['station'],
                         'location': cur_x_trace.stats['location'],
                         'channel': 'res',
                         'sampling_rate': cur_x_trace.stats['sampling_rate'],
                         'starttime': cur_x_trace.stats['starttime']}
            res_trace = obspy.core.Trace(data = cur_res, header = cur_stats)
            res_st.append(res_trace)

        res_st.split()

        return res_st

    def trim_archive(self):
        ''' Crop the archive to a specified length.
        '''
        max_end_time = np.max([x.stats.endtime for x in self.pgv_archive_stream])
        self.logger.info("max_end_time: %s.", max_end_time)
        crop_start_time = max_end_time - self.pgv_archive_time
        self.pgv_archive_stream.trim(starttime = crop_start_time)
        self.logger.info("Trimmed the archive stream to %s.", crop_start_time)

    def get_pgv_data(self):
        ''' Get the latest PGV data.
        '''
        pgv_data = {}
        for cur_trace in self.pgv_stream:
            cur_start_time = cur_trace.stats.starttime
            tmp = {}
            tmp['time'] = [x.isoformat() for (x, y) in zip(cur_trace.times(type = "utcdatetime"), cur_trace.data.tolist()) if y != self.nodata_value]
            tmp['data'] = [x for x in cur_trace.data.tolist() if x != self.nodata_value]
            pgv_data[cur_trace.get_id()] = tmp

        # Clear the pgv stream.
        self.pgv_stream.clear()
        return pgv_data

    def get_pgv_archive(self):
        ''' Get the archived PGV data.
        '''
        pgv_data = {}
        for cur_trace in self.pgv_archive_stream:
            try:
                tmp = {}
                tmp['time'] = [x.isoformat() for (x, y) in zip(cur_trace.times(type = "utcdatetime"), cur_trace.data.tolist()) if y != self.nodata_value]
                tmp['data'] = [x for x in cur_trace.data.tolist() if x != self.nodata_value]
                pgv_data[cur_trace.get_id()] = tmp
            except Exception as e:
                self.logger.exception("Error while preparing the pgv archive.")

        return pgv_data
