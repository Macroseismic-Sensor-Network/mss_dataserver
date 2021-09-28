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
''' General utilities for the postprocessing.
'''

import datetime
import gzip
import json
import logging
import os

import geojson
import geopandas as gpd
import numpy as np
import obspy
import pykrige as pk
import pyproj
import shapely

import mss_dataserver.core.json_util as json_util


def event_dir_from_publicid(public_id):
    ''' Build the event directory from the public id.
    
    Parameters
    ----------
    public_id: str 
        The public id of the event.

    Returns
    -------
    str 
        The event directory.
    '''
    cur_parts = public_id.split('_')
    event_time = obspy.UTCDateTime(cur_parts[2][:17] + '.' + cur_parts[2][17:])
    year_dir = "{year:04d}".format(year = event_time.year)
    date_dir = os.path.join("{year:04d}_{month:02d}_{day:02d}".format(year = event_time.year,
                                                                      month = event_time.month,
                                                                      day = event_time.day))
    event_dir = os.path.join(year_dir,
                             date_dir,
                             public_id)

    return event_dir


def get_supplement_map():
    ''' Create the dictionary of the supplement data structure.

    Returns
    -------
    :obj:`dict`
        A dictionary containing the supplement data mappings.
    '''
    supplement_map = {}

    # Category detectiondata.
    tmp = {'detectiondata': {'name': 'detectiondata',
                             'format': 'json',
                             'subdir': 'detectiondata',
                             'encoder': json_util.SupplementDetectionDataDecoder},
           'geometryinventory': {'name': 'geometryinventory',
                                 'format': 'json',
                                 'subdir': 'detectiondata',
                                 'encoder': json_util.GeneralFileDecoder},
           'metadata': {'name': 'metadata',
                        'format': 'json',
                        'subdir': 'detectiondata',
                        'encoder': json_util.GeneralFileDecoder},
           'pgv': {'name': 'pgv',
                   'format': 'miniseed',
                   'subdir': 'detectiondata'},
           'velocity': {'name': 'velocity',
                        'format': 'miniseed',
                        'subdir': 'detectiondata'}}
    supplement_map['detectiondata'] = tmp

    # Category detectionsequence.
    tmp = {'simplices': {'name': 'simplices',
                         'format': 'geojson',
                         'subdir': 'detectionsequence'}}
    supplement_map['detectionsequence'] = tmp

    # Category eventpgv.
    tmp = {'pgvstation': {'name': 'pgvstation',
                          'format': 'geojson',
                          'subdir': 'eventpgv'},
           'pgvvoronoi': {'name': 'pgvvoronoi',
                          'format': 'geojson',
                          'subdir': 'eventpgv'},
           'isoseismalfilledcontour': {'name': 'isoseismalfilledcontour',
                                       'format': 'geojson',
                                       'subdir': 'eventpgv'},
           }
    supplement_map['eventpgv'] = tmp

    # Category pgvsequence.
    tmp = {'pgvstation': {'name': 'pgvstation',
                          'format': 'geojson',
                          'subdir': 'pgvsequence'},
           'pgvvoronoi': {'name': 'pgvvoronoi',
                          'format': 'geojson',
                          'subdir': 'pgvsequence'},
           'pgvcontour': {'name': 'pgvcontour',
                          'format': 'geojson',
                          'subdir': 'pgvsequence'}}
    supplement_map['pgvsequence'] = tmp

    return supplement_map


def get_supplement_data(public_id, category, name, directory):
    ''' Load the data from a supplement file.

    Parameters
    ----------
    public_id: str 
        The public id of the event.

    category: str 
        The supplement data category.
    
    name: str 
        The supplement data name.

    directory: str 
        The supplement data base directory.

    Returns
    -------
    :class:`geopandas.GeoDataFrame` or :class:`obspy.Stream` of :obj:`dict`
        The requested supplement data.
    '''
    supplement_map = get_supplement_map()
    event_dir = event_dir_from_publicid(public_id)
    supp_data = supplement_map[category][name]

    supplement_dir = os.path.join(directory,
                                  event_dir,
                                  supp_data['subdir'])

    cur_filename = public_id + '_' + category + '_' + supp_data['name']
    if supp_data['format'] == 'json':
        cur_filename += '.json.gz'
        cur_filepath = os.path.join(supplement_dir,
                                    cur_filename)

        with gzip.open(cur_filepath, 'rt', encoding = 'UTF-8') as json_file:
            cur_data = json.load(json_file,
                                 cls = supp_data['encoder'])
    elif supp_data['format'] == 'miniseed':
        cur_filename += '.msd.gz'
        cur_filepath = os.path.join(supplement_dir,
                                    cur_filename)
        # obspy handles handle the decompression of the files.
        cur_data = obspy.read(cur_filepath)
    elif supp_data['format'] == 'geojson':
        cur_filename += '.geojson.gz'
        cur_filepath = os.path.join(supplement_dir,
                                    cur_filename)
        cur_data = read_geojson(cur_filepath)
    else:
        cur_data = None

    return cur_data


def read_geojson(filepath):
    ''' Read data from a geojson file.
    
    Parameters
    ----------
    filepath: str 
        The filepath to the geojson file to load.

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        The data loaded from the geojson file.
    '''

    if not os.path.exists(filepath):
        return None

    # Read the dta using geojson to get the foreign members.
    # Geopandas neglects foreign members.
    with gzip.open(filepath, 'rt', encoding = 'UTF-8') as fid:
        full_data = geojson.load(fid)

    df = gpd.GeoDataFrame.from_features(full_data['features'],
                                        crs="EPSG:4326")

    # Add the foreign members to the dataframe attrs dictionary.
    df.attrs['name'] = full_data['name']
    df.attrs.update(full_data['properties'])

    return df


def write_geojson_file(geojson_instance, category, name, output_dir,
                       prefix = None, postfix = None):
    ''' Write a geojson data file.

    Parameters
    ----------
    geojson_instance:
        The geojson data.

    category: str 
        The supplement data category.
    
    name: str 
        The supplement data name.

    output_dir: str 
        The directory where to save the geojson file.

    prefix: str 
        The filename prefix.

    postfix: str 
        The filename postfix.

    Returns
    -------
    str
        The filepath of the saved file.

    '''
    # Write the feature collection to a geojson file.  
    if prefix is None:
        prefix = ''
    else:
        prefix = prefix + '_'

    if postfix is None:
        postfix = ''
    else:
        postfix = '_' + postfix

    filename = '{prefix}{category}_{name}{postfix}.geojson.gz'.format(prefix = prefix,
                                                                      category = category,
                                                                      name = name,
                                                                      postfix = postfix)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir,
                            filename)

    with gzip.open(filepath, 'wt', encoding = 'UTF-8') as json_file:
        geojson.dump(geojson_instance, json_file)

    return filepath


def save_supplement(public_id, df, output_dir,
                    category, name, props = None):
    ''' Save a geopandas dataframe to geojson file.

    Parameters
    ----------
    public_id: str 
        The public id of the event.

    df: :class:`geopandas.GeoDataFrame`
        The data to save.

    output_dir: str 
        The directory where to save the geojson file.

    category: str 
        The supplement data category.
    
    name: str 
        The supplement data name.

    props: :obj:`dict`
        Additional properties written to the feature collection properties.
    '''
    logger_name = __name__
    logger = logging.getLogger(logger_name)
    event_dir = event_dir_from_publicid(public_id)

    supp_map = get_supplement_map()
    if category not in supp_map.keys():
        logger.error('The category %s was not found in the available categories.', category)
        return

    supplement_sub_dir = supp_map[category][name]['subdir']
    supplement_name = supp_map[category][name]['name']
    output_dir = os.path.join(output_dir,
                              event_dir,
                              supplement_sub_dir)

    fc = geojson.loads(df.to_json())

    # Add foreign members to the feature collection.
    fc.name = supplement_name
    fc.properties = {
        'public_id': public_id,
        'computation_time': isoformat_tz(obspy.UTCDateTime())
    }

    if props is not None:
        fc.properties.update(props)

    filepath = write_geojson_file(fc,
                                  category = category,
                                  name = supplement_name,
                                  prefix = public_id,
                                  output_dir = output_dir)

    return filepath


def isoformat_tz(utcdatetime):
    ''' Convert a obspy UTCDateTime instance to a isoformat string
    including the UTC Timezone specifier +00:00.

    Parameters
    ----------
    utcdatetime: :class:`obspy.UTCDateTime`
        The UTCDateTime to convert.

    Returns
    -------
    str 
        The isoformat string with UTC timezone specifier.
    '''
    return utcdatetime.datetime.replace(tzinfo=datetime.timezone.utc).isoformat()


def contourset_to_shapely(cs):
    ''' Convert a matplotlib contourset to shapely data.

    '''
    contours = {}
    # Iterate over all collections. Each collection corresponds to a
    # contour level.
    for m, cur_col in enumerate(cs.collections):
        cur_level = cs.levels[m]
        poly_list = []
        # Iterate over the paths in the collection
        for cur_path in cur_col.get_paths():
            # Convert the path to polygons to handle holes in the polygon.
            for k, cur_poly_path in enumerate(cur_path.to_polygons()):
                cur_shape = shapely.geometry.Polygon(cur_poly_path)
                if k == 0:
                    cur_poly = cur_shape
                else:
                    # If a path contains more than one polygon, treat it as
                    # a hole and subtract it from the first polygon.
                    cur_poly = cur_poly.difference(cur_shape)
            poly_list.append(cur_poly)

        contours[cur_level] = poly_list

    return contours


def reproject_polygons(df, src_proj, dst_proj):
    ''' Reproject the coordinates of shapely polygons.

    Parameters
    ----------
    df: :class:`geopandas.GeoDataFrame`
        The dataframe containing the polygons.

    src_proj: :class:`pyproj.Proj`
        The source coordinate system projection.

    dst_proj: :class:`pyproj.Proj`
        The destination coordinate system projection.

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        The dataframe containing the polygons with the reprojected coordinates.
    '''
    for cur_id, cur_row in df.iterrows():
        cur_poly = cur_row.geometry
        # Convert the exterior.
        cur_xy = cur_poly.exterior.xy
        ext_lon, ext_lat = pyproj.transform(src_proj,
                                            dst_proj,
                                            cur_xy[0],
                                            cur_xy[1])

        # Convert the interiors.
        cur_int_list = []
        for cur_interior in cur_poly.interiors:
            cur_xy = cur_interior.xy
            cur_lon, cur_lat = pyproj.transform(src_proj,
                                                dst_proj,
                                                cur_xy[0],
                                                cur_xy[1])
            cur_ring = shapely.geometry.LinearRing(zip(cur_lon, cur_lat))
            cur_int_list.append(cur_ring)

        if len(cur_int_list) == 0:
            cur_int_list = None
        proj_poly = shapely.geometry.Polygon(zip(ext_lon, ext_lat),
                                             holes = cur_int_list)

        df.loc[cur_id, 'geometry'] = proj_poly

    # Set the CRS of the dataframe.
    df = df.set_crs(dst_proj.crs)

    return df


def intensity_to_pgv(intensity = None, relation = 'version_2'):
    ''' Compute the pgv and intensity values based on the MSS relationship.

    Parameters
    ----------
    intensity: :class:`numpy.ndarray`
        The intensity values for which to compute the PGV values.

    Returns
    -------
    :class:`numpy.ndarray`
        The intensity and the computed PGV values.
    '''
    if intensity is None:
        return

    intensity = np.array(intensity)
    intensity_I_pgv = 0.001e-3

    if relation == 'version_1':
        k_low = 0.5
        d_low = -5      # np.log10(0.00001)
        k_high = 1
        d_high = -7
        kink = 4

        low_ind = np.argwhere(intensity <= kink)
        intensity_low = intensity[low_ind]
        pgv_low = d_low + k_low * intensity_low

        high_ind = np.argwhere(intensity > kink)
        intensity_high = intensity[high_ind]
        pgv_high = d_high + k_high * intensity_high

        #intensity = np.hstack([intensity_low, intensity_high])
        #intensity_pgv = np.hstack([pgv_low, pgv_high])
        intensity_pgv = np.zeros(intensity.size)
        intensity_pgv[low_ind] = pgv_low
        intensity_pgv[high_ind] = pgv_high
        intensity_pgv = 10**intensity_pgv

        # Handle the intensity I values.
        # Set them to a fixed pgv value of 0.001 mm/s.
        intensity_pgv[intensity <= 1] = intensity_I_pgv

        int_pgv = np.hstack([intensity[:, np.newaxis],
                             intensity_pgv[:, np.newaxis]])
    else:
        pgv_mm = np.exp((intensity - 3.9) / 0.81)
        pgv = pgv_mm / 1000
        pgv[intensity <= 1] = intensity_I_pgv
        int_pgv = np.hstack([intensity[:, np.newaxis],
                             pgv[:, np.newaxis]])

    return int_pgv


def pgv_to_intensity(pgv = None, relation = 'version_2'):
    ''' Compute the pgv and intensity values based on the MSS relationship.

    Parameters
    ----------
    pgv: :class:`numpy.ndarray`
        The pgv values for which to compute the intensity data.

    Returns
    -------
    :class:`numpy.ndarray`
        The pgv data and the computed intensity values.
    '''
    if pgv is None:
        return

    pgv = np.array(pgv)

    if relation == 'version_1':
        pgv = np.log10(pgv)

        k_low = 2
        d_low = 10
        k_high = 1
        d_high = 7
        kink = np.log10(1e-3)

        low_ind = np.argwhere(pgv <= kink)
        pgv_low = pgv[low_ind]
        intensity_low = d_low + k_low * pgv_low

        high_ind = np.argwhere(pgv > kink)
        pgv_high = pgv[high_ind]
        intensity_high = d_high + k_high * pgv_high

        #pgv = np.hstack([pgv_low, pgv_high])
        #intensity = np.hstack([intensity_low, intensity_high])

        intensity = np.zeros(pgv.size)
        intensity[low_ind] = intensity_low
        intensity[high_ind] = intensity_high
        intensity[intensity < 1] = 1

        pgv_int = np.hstack([10**pgv[:, np.newaxis],
                             intensity[:, np.newaxis]])
    else:
        pgv_mm = pgv * 1000
        intensity = 0.81 * np.log(pgv_mm) + 3.9
        intensity[intensity < 1] = 1
        pgv_int = np.hstack([pgv[:, np.newaxis],
                             intensity[:, np.newaxis]])

    return pgv_int


def compute_pgv_krigging(x, y, z,
                         nlags = 6, weight = False,
                         verbose = False, enable_plotting = False):
    ''' Kriging of the pgv data.

    The kriging is done using :class:`pykrige.ok.OrdinaryKriging`.
        
    Parameters
    ----------
    x: :class:`numpy.ndarray`
        The x data.

    y: :class:`numpy.ndarray`
        The y data.

    z: :class:`numpy.ndarray`
        The z data.

    nlags: int
        The number of lags used for kriging.

    weight: bool
        If True, use weights for kriging.

    verbose: bool
        If True, set the kriging to verbose.

    enable_plotting: bool
        If True, enable the plotting of some kriging statistics.
    '''
    buffer = 10000
    x_lims = [548828.0, 655078.0]
    y_lims = [5257810.0, 5342810.0]
    x_lims[0] = x_lims[0] - buffer
    x_lims[1] = x_lims[1] + buffer
    y_lims[0] = y_lims[0] - buffer
    y_lims[1] = y_lims[1] + buffer
    grid_delta = 100.0
    grid_x = np.arange(x_lims[0], x_lims[1], grid_delta)
    grid_y = np.arange(y_lims[0], y_lims[1], grid_delta)
    
    krig_ok = pk.ok.OrdinaryKriging(x = x,
                                    y = y,
                                    z = z,
                                    variogram_model = 'linear',
                                    variogram_parameters = {'nugget': 0,
                                                            'slope': 1 / 750},
                                    nlags = nlags,
                                    weight = weight,
                                    verbose = verbose,
                                    enable_plotting = enable_plotting,
                                    exact_values = True,
                                    pseudo_inv = True)

    krig_z, krig_sigmasq = krig_ok.execute(style = 'grid',
                                           xpoints = grid_x,
                                           ypoints = grid_y)
    
    return krig_z, krig_sigmasq, grid_x, grid_y
