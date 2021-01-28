import numpy as np
import pyproj
import scipy
import scipy.spatial
import shapely


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def compute_wgs84_coordinates(coord):
    src_proj = pyproj.Proj(init = 'epsg:32633')
    dst_proj = pyproj.Proj(init = 'epsg:4326')

    lon, lat = pyproj.transform(src_proj,
                                dst_proj,
                                coord[:, 0],
                                coord[:, 1])

    coord_wgs84 = np.hstack((lon[:, np.newaxis],
                             lat[:, np.newaxis]))
    return coord_wgs84


def compute_voronoi_geometry(df, boundary = None):
    ''' Compute the Voronoi cells of the pgv data.
    '''
    has_data = ~np.isnan(df.pgv)
    coord_utm = df.loc[:, ['x_utm', 'y_utm']]
    coord = df.loc[:, ['x', 'y']]
    coord = coord[has_data]
    vor = scipy.spatial.Voronoi(coord_utm[has_data])
    regions, vertices = voronoi_finite_polygons_2d(vor, radius = 100000)
    vertices_wgs84 = compute_wgs84_coordinates(vertices)

    region_id = np.arange(len(regions))
    df['region_id'] = np.ones(len(has_data), dtype = np.int32) * np.nan
    df.loc[has_data, 'region_id'] = region_id

    # Compute the region polygons.
    for k, cur_region in enumerate(regions):
        cur_poly = shapely.geometry.Polygon(vertices_wgs84[cur_region])

        if boundary is not None:
            cur_poly = cur_poly.intersection(boundary)

        df.at[coord.iloc[k].name, 'geom_vor'] = cur_poly

    voronoi_dict = {
        'regions': regions,
        'vertices': vertices,
        'vertices_wgs84': vertices_wgs84
    }

    return voronoi_dict
