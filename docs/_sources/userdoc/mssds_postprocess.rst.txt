mssds_postprocess
=================

mssds_postprocess computes the event supplement data in geoJSON format, creates map visualizations
of the supplement data and seismogram plots of the event data.

mssds_postprocess is a command line program which accepts several commands and command line arguments.

The configuration of the input and output settings is specified in the configuration file. A description
of the configuration file is given in the `configuration file example <https://github.com/Macroseismic-Sensor-Network/mss_dataserver/blob/main/example/mss_dataserver_config.ini>`_.

.. code-block:: bash
                
   mssds_postprocess [OPTIONS] CONFIG_FILE COMMAND [ARGS]...


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

create-maps
   Create map visualizations of the event metadata.

create-seismogram
   Create seismogram visualizations of the event data.

process-event
   Compute the event supplement data.



process-event
-------------------------------

Compute the event supplement data.

.. code-block:: bash
                
   mssds_postprocess process-event [OPTIONS]

   
Options
^^^^^^^

--public-id
   The public ID of the event to process.

--no-db
   Don't use the database to get event or geometry related data.
   All relevant information is loaded from the supplement data.

--no-meta
   Don't compute the metadata supplement.

--no-isoseismal
  Don't compute the PGV countours using kriging.

--no-pgv-sequence
  Don't compute the PGV sequence.

--no-detection-sequence
  Don't compute the detection sequence.

--nopgv-contour-sequence
  Don't compute the PGV contour sequence.

--help
  Show the program help.


create-maps
-------------------------------

Compute the map visualizations of the supplement data.

.. code-block:: bash
                
   mssds_postprocess create-maps [OPTIONS]

   
Options
^^^^^^^

--public-id
   The public ID of the event to process.

--no-db
   Don't use the database to get event or geometry related data.
   All relevant information is loaded from the supplement data.

--pgv-map / --no-pgv-map
  Enable/disable computation of the pgv map [enabled].

--detection-sequence / --no-detection-sequence
  Enable/disable the computation of the detection sequence images and movie [enabled].

--pgv-contour-sequence / --no-pgv-contour-sequence
  Enable/disable the computation of the PGV contour sequence images and movie [disabled].

--pgv-contour-map / --no-pgv-contour-map
  Enable/disable the computation of the PGV contour map.

--help
   Show the program help.

  
create-seismogram
-------------------------------

Compute the seismogram visualization of the event data.

.. code-block:: bash
                
   mssds_postprocess create-seismogram [OPTIONS]

   
Options
^^^^^^^

--public-id
   The public ID of the event to process.

--no-db
   Don't use the database to get event or geometry related data.
   All relevant information is loaded from the supplement data.

--hypocenter
   The hypocenter coordinates ("LAT [°],LON [°], DEPTH [m]"). e.g. "16.136,47.756,12000"
   If a hypocenter is given, the stations are sorted according to the hypo-distance.
   Otherwise they are sorted alphabetically.
   
--help
   Show the program help.
