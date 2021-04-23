mss_dataserver
==============

mss_dataserver collects the data from the Macroseismic-Sensors, computes the
PGV and runs the event detection. The event data is saved in a mariaDB database.
After the successful detection of an event, the program mssds_postprocess is called
to compute the event supplement data in geoJSON format.
The data PGV data and the event supplement data is served over websockets to
mss-vis clients.

mss_dataserver can also be used to perform some basic database management
tasks needed to initialize the mss_dataserver database.

The program is configured using a configuration file. A description
of the configuration file is given in the `configuration file example <https://github.com/Macroseismic-Sensor-Network/mss_dataserver/blob/main/example/mss_dataserver_config.ini>`_.

mss_dataserver is a command line program which accepts several commands
and command line arguments.

.. code-block:: bash
                
   mss_dataserver [OPTIONS] CONFIG_FILE COMMAND [ARGS]...

Options
-------

--help
   Show the program help.
   

Parameters
----------

CONFIG_FILE
   The filepath to the mss_dataserver configuration file.

COMMAND
   The command to execute.
   

Commands
--------

clear-db-tables
   Clear the project database tables.

create-db
   Create of update the project database.

load-geometry
   Load the geometry inventory file into the database.

start-server
   Start the data collection and websocket data server.


