import datetime
import gzip
import json
import logging
import os

import geojson
import geopandas as gpd
import obspy

import mss_dataserver.core.json_util as json_util


def event_dir_from_publicid(public_id):
    ''' Build the event directory from the public id. '''
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
    ''' Create the dictionary of the supplement data structure.'''
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
           'vel': {'name': 'vel',
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
                          'subdir': 'eventpgv'}}
    supplement_map['eventpgv'] = tmp

    # Category pgvsequence.
    tmp = {'pgvstation': {'name': 'pgvstation',
                          'format': 'geojson',
                          'subdir': 'pgvsequence'},
           'pgvvoronoi': {'name': 'pgvvoronoi',
                          'format': 'geojson',
                          'subdir': 'pgvsequence'}}
    supplement_map['pgvsequence'] = tmp

    return supplement_map


def get_supplement_data(public_id, category, name, directory):
    ''' Load the data from a supplement file.
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
    ''' Read data from a geojson file.'''

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
    ''' Write a geojson data file. '''
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
    '''
    return utcdatetime.datetime.replace(tzinfo=datetime.timezone.utc).isoformat()
