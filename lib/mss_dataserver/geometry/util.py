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
# along with this program.  If not, see <http://www.gnu.org/licenses/>
'''
The geometry util module.

:copyright:
    Stefan Mertl

:license:
    GNU General Public License, Version 3 
    http://www.gnu.org/licenses/gpl-3.0.html

This module contains helper functions used in the geometry package.
'''
def lon2UtmZone(lon):
    '''
    Convert a longitude to the UTM zone.

    The formula is based on the wikipedia description:
    The UTM system divides the surface of Earth between 80S and 84N latitude 
    into 60 zones, each 6 of longitude in width. Zone 1 covers longitude 180 
    to 174 W; zone numbering increases eastward to zone 60 that covers 
    longitude 174 to 180 East.
    '''
    if lon < -180 or lon > 180:
        raise ValueError('The longitude must be between -180 and 180.')

    return (int((180 + lon) / 6.0) + 1) % 60


def zone2UtmCentralMeridian(zone):
    '''
    Compute the middle meridian of a given UTM zone.
    '''
    if zone < 1 or zone > 60:
        raise ValueError('The zone must be between 1 and 60.')

    return zone * 6 - 180 - 3

def epsg_from_srs(srs):
    ''' Extract the epsg code from a proj srs string.

    '''
    l = srs.split()
    for s in l:
        try:
            k,v = s.split('=')
        except:
            continue
        k = k.strip('+')
        if k == 'init':
            return v


def get_epsg_dict():
    ''' Create a dictionary for mapping proj projection arguments to epsg codes.

    This function is a modified version of the one included in mpl_toolkits.basemap.
    It reads the epsg file in the matplotlib data directory and creates a dictionary 
    with the epsg codes as the keys and the responding proj projection arguments as 
    the values.
    '''
    # create dictionary that maps epsg codes to Basemap kwargs.
    import os
    epsgf = open(os.path.join(os.path.dirname(__file__), 'epsg'))
    epsg_dict={}
    for line in epsgf:
        if line.startswith("#"):
            continue
        l = line.split()
        code = l[0].strip("<>")
        parms = ' '.join(l[1:-1])
        _kw_args={}
        for s in l[1:-1]:
            try:
                k,v = s.split('=')
            except:
                k = s.strip()
                v = None
            k = k.strip("+")
            if k=='proj':
                if v == 'longlat': v = 'cyl'
                k='projection'
            if k=='k':
                k='k_0'
            if k in ['projection','lat_1','lat_2','lon_0','lat_0',\
                     'a','b','k_0','lat_ts','ellps','datum', 'zone', 'units']:
                if k not in ['projection','ellps','datum', 'units']:
                    v = float(v)
                _kw_args[k]=v
            else:
                _kw_args[k] = True
        if 'projection' in _kw_args:
            if 'a' in _kw_args:
                if 'b' in _kw_args:
                    _kw_args['rsphere']=(_kw_args['a'],_kw_args['b'])
                    del _kw_args['b']
                else:
                    _kw_args['rsphere']=_kw_args['a']
                del _kw_args['a']
            if 'datum' in _kw_args:
                if _kw_args['datum'] == 'NAD83':
                    _kw_args['ellps'] = 'GRS80'
                elif _kw_args['datum'] == 'NAD27':
                    _kw_args['ellps'] = 'clrk66'
                elif _kw_args['datum'] == 'WGS84':
                    _kw_args['ellps'] = 'WGS84'
                del _kw_args['datum']
            # supported epsg projections.
            # omerc not supported yet, since we can't handle
            # alpha,gamma and lonc keywords.
            if _kw_args['projection'] != 'omerc':
                epsg_dict[code]=_kw_args
    epsgf.close()
    return epsg_dict


ellipsoids = {}
ellipsoids['wgs84'] = (6378137, 6356752.314245179)




