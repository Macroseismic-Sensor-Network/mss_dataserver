# LICENSE
#
# This file is part of pSysmon.
#
# If you use pSysmon in any program or publication, please inform and
# acknowledge its author Stefan Mertl (stefan@mertl-research.at).
#
# pSysmon is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
The inventory parser module.

:copyright:
    Stefan Mertl

:license:
    GNU General Public License, Version 3 
    http://www.gnu.org/licenses/gpl-3.0.html

This module contains parser classes to read inventory data from files.
'''

import logging
from lxml import etree

from geometry.inventory import Inventory
from geometry.inventory import Network
from geometry.inventory import Array
from geometry.inventory import Station
from geometry.inventory import Channel
from geometry.inventory import Recorder
from geometry.inventory import RecorderStream
from geometry.inventory import RecorderStreamParameter
from geometry.inventory import Sensor
from geometry.inventory import SensorComponent
from geometry.inventory import SensorComponentParameter
from obspy.core.utcdatetime import UTCDateTime

class InventoryXmlParser:
    '''
    Parse a pSysmon inventory XML file.
    '''
    def __init__(self):

        # the logger instance.
        logger_name = __name__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(logger_name)

        # The required attributes which have to be present in the tags.
        self.required_attributes = {}
        self.required_attributes['inventory'] = ('name', )
        self.required_attributes['sensor'] = ('serial', )
        self.required_attributes['component'] = ('name', )
        self.required_attributes['component_parameter'] = ()
        self.required_attributes['response_paz'] = ()
        self.required_attributes['recorder'] = ('serial', )
        self.required_attributes['stream'] = ('name', )
        self.required_attributes['stream_parameter'] = ()
        self.required_attributes['assigned_component'] = ()
        self.required_attributes['network'] = ('name', )
        self.required_attributes['station'] = ('name', )
        self.required_attributes['location'] = ('name', )
        self.required_attributes['channel'] = ('name', )
        self.required_attributes['assigned_stream'] = ()
        self.required_attributes['array'] = ('name', )
        self.required_attributes['array_station'] = ()

        # The required tags which have to be present in the inventory.
        self.required_tags = {}
        self.required_tags['sensor'] = ('model', 'producer')
        self.required_tags['component'] = ('description', 'input_unit', 'output_unit',
                                           'deliver_unit', 'component_parameter', )
        self.required_tags['component_parameter'] = ('start_time', 'end_time',
                                                     'sensitivity')
        self.required_tags['response_paz'] = ('type', 'units', 'A0_normalization_factor',
                                              'normalization_frequency')
        #self.required_tags['complex_zero'] = ('real_zero', 'imaginary_zero')
        #self.required_tags['complex_pole'] = ('real_pole', 'imaginary_pole')

        self.required_tags['recorder'] = ('model', 'producer', 'description')
        self.required_tags['stream'] = ('label', )
        self.required_tags['stream_parameter'] = ('start_time', 'end_time', 'gain',
                                        'bitweight')
        self.required_tags['assigned_component'] = ('sensor_serial', 'sensor_model', 'sensor_producer',
                                                    'component_name', 'start_time', 'end_time')

        self.required_tags['network'] = ('description', 'type')
        self.required_tags['station'] = ('location', )
        self.required_tags['location'] = ('x', 'y', 'z',
                                        'coord_system', 'description')
        self.required_tags['channel'] = ('description', )
        self.required_tags['assigned_stream'] = ('recorder_serial', 'recorder_model',
                                                 'recorder_producer','stream_name',
                                                 'start_time', 'end_time')
        self.required_tags['array'] = ()
        self.required_tags['array_station'] = ('network', 'name', 'location',
                                               'start_time', 'end_time')



    def parse(self, filename, inventory_name = 'new xml inventory'):
        import lxml.etree

        self.logger.debug("parsing file...\n")

        inventory = Inventory(inventory_name, type = 'xml')

        # Parse the xml file passed as argument.
        parser = lxml.etree.XMLParser(remove_comments = True)
        tree = lxml.etree.parse(filename, parser)
        inventory_root = tree.getroot()

        # Check if the root element is of type inventory.
        if inventory_root.tag != 'inventory':
            return
        else:
            self.logger.debug("found inventory root tag\n")

        # Set the name of the inventory.
        inventory.name = inventory_root.attrib['name']

        # Get the recorders and stations of the inventory.
        sensor_list = tree.findall('sensor_list')
        recorder_list = tree.findall('recorder_list')
        networks = tree.findall('network')
        arrays = tree.findall('array')

        # Process the sensors first.
        for cur_sensor_list in sensor_list:
            sensors = cur_sensor_list.findall('sensor')
            self.process_sensors(inventory, sensors)

        # Next process the recorders. These might depend on sensors.
        for cur_recorder_list in recorder_list:
            recorders = cur_recorder_list.findall('recorder')
            self.process_recorders(inventory, recorders)

        # Now process the networks which might depend on recorders.
        self.process_networks(inventory, networks)

        # Finally process the arrays which require all elements already added
        # to the inventory.
        self.process_arrays(inventory, arrays)

        self.logger.debug("Success reading the XML file.")

        return inventory


    def instance_to_xml(self, instance, root, name, attributes, tags, attr_map, converter, element_handler = {}):
        ''' Translate an inventory object into a xml element.
        '''
        attrib = {}
        for cur_key in attributes:
            attrib[cur_key] = getattr(instance, attr_map[cur_key])

        element = etree.SubElement(root, name, **attrib)

        for cur_key in tags:
            if cur_key in element_handler.keys():
                eh = element_handler[cur_key]
                eh(name = cur_key,
                   value = getattr(instance, attr_map[cur_key]),
                   root = element)
            else:
                tag = etree.SubElement(element, cur_key)
                if cur_key in converter.keys():
                    cur_text = converter[cur_key](getattr(instance, attr_map[cur_key]))
                else:
                    value = getattr(instance, attr_map[cur_key])
                    if value is not None:
                        cur_text = str(value)
                    else:
                        cur_text = ''

                tag.text = cur_text

        return element


    def clean_time_string(self, value):
        ''' Remove running and big bang string from time string.
        '''
        if value == 'big bang':
            value = ''
        elif value == 'running':
            value = ''
        return value


    def handle_element_pz(self, name, value, root):
        ''' Convert a list of complex values to a list of xml tree elements.
        '''
        for cur_pz in value:
            element = etree.SubElement(root, name)
            element.text = str(cur_pz).replace('(', '').replace(')','')



    def export_xml(self, inventory, filename):
        ''' Export an inventory to xml file.
        '''
        root = etree.Element('inventory', name = inventory.name)

        # sensor
        sensor_attributes = ['serial',]
        sensor_tags = ['model', 'producer', 'description']
        sensor_map = {'serial':'serial',
                      'model':'model',
                      'producer':'producer',
                      'description':'description'}
        sensor_converter = {}

        # sensor component
        component_attributes = ['name', ]
        component_tags = ['description', 'input_unit', 'output_unit',
                          'deliver_unit']
        component_map = {'name':'name',
                         'description':'description',
                         'input_unit':'input_unit',
                         'output_unit':'output_unit',
                         'deliver_unit':'deliver_unit'}
        component_converter = {}

        # sensor component parameter
        component_parameter_attributes = []
        component_parameter_tags = ['start_time', 'end_time', 'sensitivity']
        component_parameter_map = {'start_time':'start_time_string',
                                   'end_time':'end_time_string',
                                   'sensitivity':'sensitivity'}
        component_parameter_converter = {'start_time':self.clean_time_string,
                                         'end_time':self.clean_time_string}

        component_parameter_paz_attributes = []
        component_parameter_paz_tags = ['type', 'A0_normalization_factor', 'normalization_frequency',
                                        'complex_zero', 'complex_pole']
        component_parameter_paz_map = {'type':'tf_type',
                                       'A0_normalization_factor':'tf_normalization_factor',
                                       'normalization_frequency':'tf_normalization_frequency',
                                       'complex_zero':'tf_zeros',
                                       'complex_pole':'tf_poles'}
        component_parameter_paz_converter = {}
        component_parameter_paz_handler = {'complex_zero':self.handle_element_pz,
                                           'complex_pole':self.handle_element_pz}


        # recorder
        rec_attributes = ['serial',]
        rec_tags = ['model', 'producer', 'description']
        rec_map = {'serial':'serial',
                   'model':'model',
                   'producer':'producer',
                   'description':'description'}
        rec_converter = {}

        #stream
        stream_attributes = ['name',]
        stream_tags = ['label',]
        stream_map = {'name':'name',
                      'label':'label'}
        stream_converter = {}

        #stream_parameter
        stream_param_attributes = []
        stream_param_tags = ['start_time', 'end_time', 'gain', 'bitweight']
        stream_param_map = {'start_time':'start_time_string',
                            'end_time':'end_time_string',
                            'gain':'gain',
                            'bitweight':'bitweight'}
        stream_param_converter = {'start_time':self.clean_time_string,
                                  'end_time':self.clean_time_string}

        # stream assigned component
        stream_comp_attributes = []
        stream_comp_tags = ['sensor_serial', 'sensor_model', 'sensor_producer',
                            'component_name', 'start_time', 'end_time']
        stream_comp_map = {'sensor_serial':'serial',
                           'sensor_model':'model',
                           'sensor_producer':'producer',
                           'component_name':'name',
                           'start_time':'start_time_string',
                           'end_time':'end_time_string'}
        stream_comp_converter = {'start_time':self.clean_time_string,
                                 'end_time':self.clean_time_string}

        # network
        net_attributes = ['name',]
        net_tags = ['description', 'type']
        net_map = {'name':'name',
                   'description':'description',
                   'type':'type'}
        net_converter = {}

        # array
        array_attributes = ['name',]
        array_tags = ['description',]
        array_map = {'name': 'name',
                     'description': 'description'}
        array_converter = {}

        # array_station
        array_stat_attributes = []
        array_stat_tags = ['network', 'name', 'location',
                           'start_time', 'end_time']
        array_stat_map = {'network': 'network',
                          'name': 'name',
                          'location': 'location',
                          'start_time': 'start_time',
                          'end_time': 'end_time'}
        array_stat_converter = {}

        # station
        stat_attributes = ['name',]
        stat_tags = []
        stat_map = {}
        stat_converter = {}

        #location
        loc_attributes = ['name',]
        loc_tags = ['x', 'y', 'z', 'coord_system', 'description']
        loc_map = {'name':'location',
                    'x':'x',
                    'y':'y',
                    'z':'z',
                    'coord_system':'coord_system',
                    'description':'description'}
        loc_converter = {}

        # channel
        chan_attributes = ['name',]
        chan_tags = ['description',]
        chan_map = {'name':'name',
                    'description':'description'}
        chan_converter = {}

        # assigned stream
        chan_stream_attributes = []
        chan_stream_tags = ['recorder_serial', 'recorder_model', 'recorder_producer',
                            'stream_name', 'start_time', 'end_time']
        chan_stream_map = {'recorder_serial':'serial',
                           'recorder_model':'model',
                           'recorder_producer':'producer',
                           'stream_name':'name',
                           'start_time':'start_time_string',
                           'end_time':'end_time_string'}
        chan_stream_converter = {'start_time':self.clean_time_string,
                                 'end_time':self.clean_time_string}



        # Export the sensors.
        sensor_list = etree.SubElement(root, 'sensor_list')
        for cur_sensor in inventory.sensors:
            sensor_element = self.instance_to_xml(instance = cur_sensor,
                                                  root = sensor_list,
                                                  name = 'sensor',
                                                  attributes = sensor_attributes,
                                                  tags = sensor_tags,
                                                  attr_map = sensor_map,
                                                  converter = sensor_converter)
            for cur_component in cur_sensor.components:
                comp_element = self.instance_to_xml(instance = cur_component,
                                                    root = sensor_element,
                                                    name = 'component',
                                                    attributes = component_attributes,
                                                    tags = component_tags,
                                                    attr_map = component_map,
                                                    converter = component_converter)
                for cur_parameter in cur_component.parameters:
                    param_element = self.instance_to_xml(instance = cur_parameter,
                                                         root = comp_element,
                                                         name = 'component_parameter',
                                                         attributes = component_parameter_attributes,
                                                         tags = component_parameter_tags,
                                                         attr_map = component_parameter_map,
                                                         converter = component_parameter_converter)
                    paz_element = self.instance_to_xml(instance = cur_parameter,
                                                       root = param_element,
                                                       name = 'response_paz',
                                                       attributes = component_parameter_paz_attributes,
                                                       tags = component_parameter_paz_tags,
                                                       attr_map = component_parameter_paz_map,
                                                       converter = component_parameter_paz_converter,
                                                       element_handler = component_parameter_paz_handler)

        #Export the recorders.
        recorder_list = etree.SubElement(root, 'recorder_list')
        for cur_recorder in inventory.recorders:
            rec_element = self.instance_to_xml(instance = cur_recorder,
                                               root = recorder_list,
                                               name = 'recorder',
                                               attributes = rec_attributes,
                                               tags = rec_tags,
                                               attr_map = rec_map,
                                               converter = rec_converter)

            for cur_stream in cur_recorder.streams:
                stream_element = self.instance_to_xml(instance = cur_stream,
                                                      root = rec_element,
                                                      name = 'stream',
                                                      attributes = stream_attributes,
                                                      tags = stream_tags,
                                                      attr_map = stream_map,
                                                      converter = stream_converter)

                for cur_param in cur_stream.parameters:
                    param_element = self.instance_to_xml(instance = cur_param,
                                                         root = stream_element,
                                                         name = 'stream_parameter',
                                                         attributes = stream_param_attributes,
                                                         tags = stream_param_tags,
                                                         attr_map = stream_param_map,
                                                         converter = stream_param_converter)

                for cur_comp_tb in cur_stream.components:
                    comp_element = self.instance_to_xml(instance = cur_comp_tb,
                                                        root = stream_element,
                                                        name = 'assigned_component',
                                                        attributes = stream_comp_attributes,
                                                        tags = stream_comp_tags,
                                                        attr_map = stream_comp_map,
                                                        converter = stream_comp_converter)

        # Export the networks.
        for cur_network in inventory.networks:
            net_element = self.instance_to_xml(instance = cur_network,
                                               root = root,
                                               name = 'network',
                                               attributes = net_attributes,
                                               tags = net_tags,
                                               attr_map = net_map,
                                               converter = net_converter)

            # Get the unique station names.
            stat_names = [x.name for x in cur_network.stations]
            stat_names = list(set(stat_names))

            for cur_stat_name in stat_names:
                cur_station_list = cur_network.get_station(name = cur_stat_name)

                # Create the station element.
                stat_element = etree.SubElement(net_element, 'station', name = cur_stat_name)

                for cur_station in cur_station_list:
                    # Add the location elements.
                    loc_element = self.instance_to_xml(instance = cur_station,
                                                        root = stat_element,
                                                        name = 'location',
                                                        attributes = loc_attributes,
                                                        tags = loc_tags,
                                                        attr_map = loc_map,
                                                        converter = loc_converter)

                    for cur_channel in cur_station.channels:
                        chan_element = self.instance_to_xml(instance = cur_channel,
                                                            root = loc_element,
                                                            name = 'channel',
                                                            attributes = chan_attributes,
                                                            tags = chan_tags,
                                                            attr_map = chan_map,
                                                            converter = chan_converter)

                        for cur_stream_tb in cur_channel.streams:
                            self.instance_to_xml(instance = cur_stream_tb,
                                                 root = chan_element,
                                                 name = 'assigned_stream',
                                                 attributes = chan_stream_attributes,
                                                 tags = chan_stream_tags,
                                                 attr_map = chan_stream_map,
                                                 converter = chan_stream_converter)


        # Export the arrays
        for cur_array in inventory.arrays:
            array_element = self.instance_to_xml(instance = cur_array,
                                                 root = root,
                                                 name = 'array',
                                                 attributes = array_attributes,
                                                 tags = array_tags,
                                                 attr_map = array_map,
                                                 converter = array_converter)

            for cur_station_tb in cur_array.stations:
                # Create the station element.
                stat_element = self.instance_to_xml(instance = cur_station_tb,
                                                    root = array_element,
                                                    name = 'station',
                                                    attributes = array_stat_attributes,
                                                    tags = array_stat_tags,
                                                    attr_map = array_stat_map,
                                                    converter = array_stat_converter)


        # Write the xml string to a file.
        et = etree.ElementTree(root)
        et.write(filename, pretty_print = True, xml_declaration = True, encoding = 'UTF-8')
        #fid = open(filename, 'w')
        #fid.write(etree.tostring(root, pretty_print = True))
        #fid.close()



    def process_sensors(self, inventory, sensors):
        ''' Process the extracted sensor tags.

        Parameters
        ----------
        inventory : :class:`~psysmon.packages.geometry.inventory.Inventory`
            The inventory to which to add the parsed sensors.

        sensors : xml sensor nodes
            The xml sensor nodes parsed using the findall method.

        '''
        self.logger.debug("Processing the sensors.")
        for cur_sensor in sensors:
            sensor_content = self.parse_node(cur_sensor)

            if self.check_completeness(cur_sensor, sensor_content, 'sensor') is False:
                continue

            if 'component' in sensor_content.keys():
                sensor_content.pop('component')

            sensor_to_add = Sensor(serial = cur_sensor.attrib['serial'], **sensor_content)
            inventory.add_sensor(sensor_to_add)

            components = cur_sensor.findall('component')
            self.process_components(sensor_to_add, components)


    def process_components(self, sensor, components):
        ''' Process the component nodes of a sensor.

        Parameters
        ----------
        sensor : :class:`~psysmon.packages.geometry.inventory.Sensor`
            The sensor to which to add the components.

        components : xml component nodes
            The xml component nodes parsed using the findall method.
        '''
        for cur_component in components:
            component_content = self.parse_node(cur_component)

            if self.check_completeness(cur_component, component_content, 'component') is False:
                continue

            if 'component_parameter' in component_content:
                component_content.pop('component_parameter')
            component_to_add = SensorComponent(name = cur_component.attrib['name'],
                                               **component_content)
            sensor.add_component(component_to_add)

            parameters = cur_component.findall('component_parameter')
            self.process_component_parameters(component_to_add, parameters)



    def process_component_parameters(self, component, parameters):
        ''' Process the component_parameter nodes of a component.
        '''
        for cur_parameter in parameters:
            content = self.parse_node(cur_parameter)

            if self.check_completeness(cur_parameter, content, 'component_parameter') is False:
                continue

            if 'response_paz' in content:
                content.pop('response_paz')
            parameter_to_add = SensorComponentParameter(**content)
            component.add_parameter(parameter_to_add)

            response_paz = cur_parameter.findall('response_paz')
            self.process_response_paz(parameter_to_add, response_paz)



    def process_response_paz(self, parameter, response_paz):
        ''' Process the response_paz nodes of a component_paramter.

        '''
        for cur_paz in response_paz:
            content = self.parse_node(cur_paz)
            if self.check_completeness(cur_paz, content, 'response_paz') is False:
                continue

            self.logger.debug("Adding the tf to the parameter %s", parameter)
            parameter.set_transfer_function(tf_type = content['type'],
                                            tf_units = content['units'],
                                            tf_normalization_factor = float(content['A0_normalization_factor']),
                                            tf_normalization_frequency = float(content['normalization_frequency']))

            zeros = cur_paz.findall('complex_zero')
            self.process_complex_zero(parameter, zeros)
            poles = cur_paz.findall('complex_pole')
            self.process_complex_pole(parameter, poles)


    def process_complex_zero(self, parameter, zeros):
        ''' Process the complex_zero nodes in a response_paz.
        '''
        for cur_zero in zeros:
            self.logger.debug('Adding zero to the parameter %s', parameter)
            zero = cur_zero.text.replace(' ', '')
            parameter.tf_add_complex_zero(complex(zero))


    def process_complex_pole(self, parameter, poles):
        ''' Process the complex_poles nodes in a response_paz.
        '''
        for cur_pole in poles:
            pole = cur_pole.text.replace(' ', '')
            parameter.tf_add_complex_pole(complex(pole))



    def process_recorders(self, inventory, recorders):
        ''' Process the extracted recorder nodes.

        Parameters
        ----------
        inventory : :class:`~psysmon.packages.geometry.inventory.Inventory`
            The inventory to which to add the parsed sensors.

        recorders : xml recorder nodes
            The xml recorder nodes parsed using the findall method.

        '''
        self.logger.debug("Processing the recorders.")
        for cur_recorder in recorders:
            content = self.parse_node(cur_recorder)

            if self.check_completeness(cur_recorder, content, 'recorder') is False:
                continue

            if 'stream' in content.keys():
                content.pop('stream')

            # Create the Recorder instance.
            rec_to_add = Recorder(serial = cur_recorder.attrib['serial'], **content)
            inventory.add_recorder(rec_to_add)

            # Process the streams of the recorder.
            streams = cur_recorder.findall('stream')
            self.process_recorder_streams(rec_to_add, streams)


    def process_recorder_streams(self, recorder, streams):
        ''' Process the stream nodes of a recorder.

        Parameters
        ----------
        recorder : :class:`~psysmon.packages.geometry.inventory.Recorder`
            The recorder to which to add the streams.

        streams : xml stream nodes
            The xml stream nodes parsed using the findall method.

        '''
        for cur_stream in streams:
            content = self.parse_node(cur_stream)

            if self.check_completeness(cur_stream, content, 'stream') is False:
                continue

            if 'stream_parameter' in content.keys():
                content.pop('stream_parameter')

            if 'assigned_component' in content.keys():
                content.pop('assigned_component')

            # Create the stream instance.
            stream_to_add = RecorderStream(name = cur_stream.attrib['name'], **content)
            recorder.add_stream(stream_to_add)

            stream_parameters = cur_stream.findall('stream_parameter')
            self.process_stream_parameters(stream_to_add, stream_parameters)

            assigned_components = cur_stream.findall('assigned_component')
            self.process_assigned_components(stream_to_add, assigned_components)



    def process_stream_parameters(self, stream, parameters):
        ''' Process the stream_parameter nodes of a recorder stream.

        '''
        for cur_parameter in parameters:
            content = self.parse_node(cur_parameter)

            if self.check_completeness(cur_parameter, content, 'stream_parameter') is False:
                continue

            parameter_to_add = RecorderStreamParameter(**content)
            stream.add_parameter(parameter_to_add)


    def process_assigned_components(self, stream, components):
        ''' Process the components assigned to a recorder stream.

        '''
        for cur_component in components:
            content = self.parse_node(cur_component)

            if self.check_completeness(cur_component, content, 'assigned_component') is False:
                continue

            stream.add_component(serial = content['sensor_serial'],
                                 model = content['sensor_model'],
                                 producer = content['sensor_producer'],
                                 name = content['component_name'],
                                 start_time = content['start_time'],
                                 end_time = content['end_time'])


    def process_networks(self, inventory, networks):
        ''' Process the extracted network nodes.

        Parameters
        ----------
        inventory : :class:`~psysmon.packages.geometry.inventory.Inventory`
            The inventory to which to add the parsed sensors.

        networks : xml network nodes
            The xml network nodes parsed using the findall method.
        '''
        self.logger.debug("Processing the networks.")
        for cur_network in networks:
            content = self.parse_node(cur_network)

            if self.check_completeness(cur_network, content, 'network') is False:
                continue

            if 'station' in content.keys():
                content.pop('station')

            # Create the Recorder instance.
            net_to_add = Network(name=cur_network.attrib['name'], **content)

            # Add the network to the inventory.
            inventory.add_network(net_to_add)

            stations = cur_network.findall('station')
            self.process_stations(net_to_add, stations)


    def process_arrays(self, inventory, arrays):
        ''' Process the extracted array nodes.

        Parameters
        ----------
        inventory : :class:`~psysmon.packages.geometry.inventory.Inventory`
            The inventory to which to add the parsed sensors.

        arrays : xml array nodes
            The xml array nodes parsed using the findall method.
        '''
        self.logger.debug("Processing the networks.")

        for cur_array in arrays:
            content = self.parse_node(cur_array)
            self.check_completeness(cur_array, content, 'array')

            if 'station' in content.keys():
                content.pop('station')

            # Create the Array instance and add it to the inventory.
            array_to_add = Array(name = cur_array.attrib['name'], **content)
            inventory.add_array(array_to_add)

            # Process the stations to be added to the array.
            stations = cur_array.findall('station')
            for cur_station in stations:
                stat_content = self.parse_node(cur_station)
                self.check_completeness(cur_station, stat_content, 'array_station')
                station_to_add = inventory.get_station(network = stat_content['network'],
                                                       name = stat_content['name'],
                                                       location = stat_content['location'])
                if len(station_to_add) == 1:
                    station_to_add = station_to_add[0]
                    array_to_add.add_station(station_to_add,
                                             start_time = stat_content['start_time'],
                                             end_time = stat_content['end_time'])



    def process_stations(self, network, stations):
        ''' Process the station nodes of a network.

        Parameters
        ----------
        network : :class:`~psysmon.packages.geometry.inventory.Network`
            The network to which to add the stations.

        stations : xml station nodes
            The xml station nodes parsed using the findall method.
        '''
        for cur_station in stations:
            content = self.parse_node(cur_station)

            if self.check_completeness(cur_station, content, 'station') is False:
                continue

            locations = cur_station.findall('location')

            for cur_location in locations:
                loc_content = self.parse_node(cur_location)

                if 'channel' in loc_content.keys():
                    loc_content.pop('channel')

                station_to_add = Station(name = cur_station.attrib['name'],
                                         location = cur_location.attrib['name'],
                                         **loc_content)

                network.add_station(station_to_add)

                channels = cur_location.findall('channel')
                self.process_channels(station_to_add, channels)



    def process_channels(self, station, channels):
        ''' Process the channel nodes of a station.

        Parameters
        ----------
        station : :class:`~psysmon.packages.geometry.inventory.Station`
            The station to which to add the channels.

        channels : xml channel nodes
            The xml channel nodes parsed using the findall method.
        '''
        for cur_channel in channels:
            content = self.parse_node(cur_channel)

            if self.check_completeness(cur_channel, content, 'channel') is False:
                continue

            if 'assigned_stream' in content.keys():
                content.pop('assigned_stream')

            channel_to_add = Channel(name = cur_channel.attrib['name'], **content)

            station.add_channel(channel_to_add)

            assigned_streams = cur_channel.findall('assigned_stream')
            self.process_assigned_streams(channel_to_add, assigned_streams)



    def process_assigned_streams(self, channel, streams):
        ''' Process the assigned streams of a channel.

        Parameters
        ----------
        channel : :class:`~psysmon.packages.geometry.inventory.Channel`
            The channel to which to add the streams.

        streams : xml stream nodes
            The xml stream nodes parsed using the findall method.
        '''
        for cur_stream in streams:
            content = self.parse_node(cur_stream)

            if self.check_completeness(cur_stream, content, 'assigned_stream') is False:
                continue

            channel.add_stream(serial = content['recorder_serial'],
                               model = content['recorder_model'],
                               producer = content['recorder_producer'],
                               name = content['stream_name'],
                               start_time = content['start_time'],
                               end_time = content['end_time'])




    def get_node_text(self, xml_element, tag):
        node = xml_element.find(tag)
        if node is not None:
            return node.text
        else:
            return None

    def parse_node(self, xml_element):
        node_content = {}
        for cur_node in list(xml_element):
            if cur_node.text is not None:
                node_content[cur_node.tag] = cur_node.text.strip()
            else:
                node_content[cur_node.tag] = cur_node.text

        return node_content

    def keys_complete(self, node_content, required_keys):
        missing_keys = []
        for cur_key in required_keys:
            if cur_key in node_content:
                continue
            else:
                missing_keys.append(cur_key)

        return missing_keys


    def check_completeness(self, node, content, node_type):
            missing_attrib = self.keys_complete(node.attrib, self.required_attributes[node_type])
            missing_keys = self.keys_complete(content, self.required_tags[node_type]);
            if not missing_keys and not missing_attrib:
                self.logger.debug(node_type + " xml content:")
                self.logger.debug("%s", content)
                return True
            else:
                self.logger.error("Not all required fields present!\nMissing Keys:\n")
                self.logger.error("%s", missing_keys)
                self.logger.error("%s", missing_attrib)
                raise RuntimeError("Not all required fieds for node %s present." % node_type)
