"""Methods for wrangling and pre-processing anpr data and related datasets."""
# ------------------------------------------------------------------------------

from   .utils               import log
from   .utils               import save_fig
from   .utils               import settings
from   .plot                import plot_G
from   .helpers             import get_quadrant
from   .helpers             import cut
from   .helpers             import common_words
from   .network             import add_edge_directions
from   .network             import edges_by_distance

import os
import re
import math
import time
import numpy                as np
import osmnx                as ox
import pandas               as pd
import pandas.api.types     as ptypes
import networkx             as nx
import geopandas            as gpd
import logging              as lg
import shapely              as shp

from   hashlib              import blake2b
from   functools            import reduce
from   itertools            import chain
from   functools            import reduce

# ------------------------------------------------------------------------------

g_direction_regex = (r'(East\/West|North\/South|Northbound|Eastbound|'
                     r'Southbound|Westbound|Northhbound|Southhbound)')
g_address_regex = (r'(East\/West|North\/South|Northbound \d|Eastbound \d|'
                   r'Southbound \d|Westbound \d|Northbound|Eastbound|'
                   r'Southbound|Westbound|Site \d|Camera \d|Camera|Site|L\d+|'
                   r'Lane \d|LT|RT|AH|&)')
g_road_ref_regex  = r'(A\d+|B\d+|C\d+)'
g_car_park_regex  = r'car park'
g_directions_separator = "/|&"
g_np_regex = (r'^[A-Za-z]{1,2}[ ]?[0-9]{1,4}$|'
              r'^[A-Za-z]{3}[ ]?[0-9]{1,3}$|'
              r'^[0-9]{1,3}[ ]?[A-Za-z]{3}$|'
              r'^[0-9]{1,4}[ ]?[A-Za-z]{1,2}$|'
              r'^[A-Za-z]{3}[ ]?[0-9]{1,3}[ ]?[A-Za-z]$|'
              r'^[A-Za-z][ ]?[0-9]{1,3}[ ]?[A-Za-z]{3}$|'
              r'^[A-Za-z]{2}[ ]?[0-9]{2}[ ]?[A-Za-z]{3}$|'
              r'^[A-Za-z]{3}[ ]?[0-9]{4}$')

NA_CAMERA = 999999

# ------------------------------------------------------------------------------

def infer_road_attr(
    descriptions,
    direction_regex      = g_direction_regex,
    address_regex        = g_address_regex,
    road_ref_regex       = g_road_ref_regex,
    car_park_regex       = g_car_park_regex,
    directions_separator = g_directions_separator,
):
    """
    Hello
    """
    directions = descriptions.str.extract(direction_regex,
                                          flags = re.IGNORECASE,
                                          expand = False)

    both_directions = directions.str.contains(directions_separator, na=False)

    directions_wrangled = directions \
        .dropna() \
        .str.split(directions_separator) \
        .apply(lambda x: x[0][0] if len(x) == 1 \
                         else "{}-{}".format(x[0][0], x[1][0]))

    directions = pd.concat([directions[directions.isnull()],
                            directions_wrangled])\
                    .sort_index()

    road_refs = descriptions.str.extract(road_ref_regex,
                                         flags = re.IGNORECASE,
                                         expand = False)

    addresses = descriptions.str.replace(address_regex, '',regex = True)\
                                .replace(' +', ' ', regex = True)\
                                .replace('^ +', '', regex = True)

    car_parks = descriptions.str.contains(car_park_regex,
                                          case = False, na = False)

    return pd.DataFrame({
        'direction'      : directions,
        'both_directions': both_directions,
        'ref'            : road_refs,
        'address'        : addresses,
        'is_carpark'     : car_parks
    })

def filter_by_attr_distance(
    row,
    df,
    filter_by_max_common     = True,
    filter_by_same_ref       = True,
    filter_by_same_direction = True,
    same_direction_filter    = "equals",
    distance_threshold       = 100.0,
    object_name              = ("Camera", "cameras")
):
    """
    Hello
    """
    valid_filters = {"equals", "isin"}

    sname = object_name[0]
    snames = object_name[1]

    cdf = df

    if 'address' in row:
        c_address = row['address']
        c_ref     = row['ref']

        cdf = cdf\
            .assign(common_address_words = cdf.address\
                .apply(lambda other: common_words(c_address, other, " ")))\
            .assign(same_ref = cdf.ref.apply(lambda other: c_ref == other))

        if filter_by_max_common:
            max_common = cdf['common_address_words'].max()
            cdf        = cdf[cdf.common_address_words == max_common]

        # beware if ref is na
        if filter_by_same_ref and not pd.isna(c_ref):
            cdf = cdf[cdf.same_ref == True]

        if len(cdf) == 0:
            log("{} {}: No other objects with the same address and ref."\
                    .format(sname, row['id']),
                level = lg.INFO)
            return cdf
        else:
            same_address_ids = list(cdf['id'])
            log(("{} {}: Found {} other {} ({}) with the same "
                 "address and ref.")\
                    .format(sname, row['id'], len(same_address_ids),
                            snames, same_address_ids),
                level = lg.INFO)

    # filter by direction
    if 'direction' in row:
        c_dir = row['direction']

        if same_direction_filter == "equals":
            cdf = cdf.assign(same_dir = cdf.direction\
                        .apply(lambda other: c_dir == other))

        elif same_direction_filter == "isin":
            cdf = cdf.assign(same_dir = cdf.direction\
                        .apply(lambda other: \
                            set(c_dir.split("-")).issubset(other.split("-"))))

        else:
            raise ValueError("Invalid same direction filter. Choose one of {}"\
                .format(valid_filters))

        if filter_by_same_direction:
            cdf = cdf[cdf.same_dir == True]

        if len(cdf) == 0:
            log(("{} {}: no other {} with the same direction")\
                    .format(sname, row['id'], snames),
                level = lg.INFO)
            return cdf
        else:
            shortlist_ids = list(cdf['id'])
            log(("{} {}: Shortlist of {} ({}) with the same "
                 "address, ref and direction.")\
                    .format(sname, row['id'], shortlist_ids, snames),
                level = lg.INFO)

    c_p = row['geometry']
    cdf = cdf.assign(dist = cdf.geometry\
                               .apply(lambda other: c_p.distance(other)))

    dists = [ "{:,.2f}".format(dist) for dist in cdf['dist'] ]
    cdf = cdf[cdf.dist <= distance_threshold]

    log_cols = ['id','address','ref','direction','lat','lon', 'dist']

    if len(cdf) == 0:
        log(("{} {}: no other {} within {} meters ({} meters).")\
                .format(sname, row['id'], snames, distance_threshold, dists),
            level = lg.INFO)
    else:
        row['dist'] = np.nan
        log(("{} {}: found {} matching {}\n{}")\
                .format(sname, row['id'], len(cdf), snames,
                        pd.concat([ pd.DataFrame(dict(row[log_cols]),
                                                index = [0]),
                                    cdf[log_cols] ],
                                  axis = 0, ignore_index=True)),
            level = lg.INFO)

    return cdf

def wrangle_objects(
    objects,
    is_test_col           = "name",
    is_commissioned_col   = "is_commissioned",
    road_attr_col         = "description",
    drop_car_park         = True,
    drop_na_direction     = True,
    direction_regex       = g_direction_regex,
    address_regex         = g_address_regex,
    road_ref_regex        = g_road_ref_regex,
    car_park_regex        = g_car_park_regex,
    directions_separator  = g_directions_separator,
    sort_by               = "id",
    utm_crs               = {'datum': 'WGS84',
                             'ellps': 'WGS84',
                             'proj' : 'utm',
                             'units': 'm'},
    object_name           = ("Camera", "cameras")
):
 """
 """
 sname = object_name[0]
 snames = object_name[1]

 nrows = len(objects)

 log("Wrangling {} dataset with {} rows and colnames: {}"\
         .format(snames, nrows, ",".join(objects.columns.values)),
     level = lg.INFO)

 mandatory_columns = {'id', 'lat', 'lon'}

 log("Checking if input dataframe contains mandatory columns {}."\
         .format(mandatory_columns),
     level = lg.INFO)

 cols = set(objects.columns.values)

 assert mandatory_columns.issubset(cols)

 other_cols = set(
     filter(lambda x: x is None,
            [is_test_col, is_commissioned_col, road_attr_col]))

 unrecognised_cols = cols - mandatory_columns - other_cols
 if len(unrecognised_cols) > 0:
     log("Skipping {} unrecognised columns: {}."\
             .format(len(unrecognised_cols), unrecognised_cols),
         level = lg.INFO)

 # Some asserts about the input data
 log("Checking if 'id' is unique."\
         .format(mandatory_columns),
     level = lg.INFO)
 assert len(objects['id']) == len(objects['id'].unique())

 # Find and drop cameras that are labelled as "test"
 if is_test_col:
     oldlen = len(objects)

     objects = objects.assign(
         is_test = objects[is_test_col].str.contains('test',
                                                         case = False))
     objects = objects[(objects.is_test == False)]
     objects = objects.drop(columns = 'is_test')

     log("Dropping {} rows with 'test' in name"\
             .format(oldlen - len(objects)),
         level = lg.INFO)

 # Find and drop cameras that not commissioned
 if is_commissioned_col:
     oldlen = len(objects)

     objects = objects[objects[is_commissioned_col] == True]
     objects = objects.drop(is_commissioned_col, axis = 1)

     log("Dropping {} rows with '{}' == False."\
             .format(oldlen - len(objects), is_commissioned_col),
         level = lg.INFO)

 # Infer road attributes
 if road_attr_col:
     log(("Inferring new cols 'direction', 'address', 'ref' and 'is_carpark'"
          " based on column '{}'.")\
             .format(road_attr_col),
         level = lg.INFO)

     road_attr = infer_road_attr(
        descriptions          = objects[road_attr_col],
        direction_regex       = direction_regex,
        address_regex         = address_regex,
        road_ref_regex        = road_ref_regex,
        car_park_regex        = car_park_regex,
        directions_separator  = directions_separator)

     objects   = pd.concat([objects, road_attr], axis = 1)
     # Find and drop cameras that are in car parks
     if drop_car_park:
         oldlen = len(objects)

         objects = objects[(objects.is_carpark == False)]
         objects = objects.drop(columns = 'is_carpark')

         log("Dropping {} rows with 'is_carpark' == True."\
                 .format(oldlen - len(objects)),
             level = lg.INFO)

     count_na_direction = len(objects[pd.isna(objects.direction)])
     log(("There are {} objects (ids = {}) with missing direction.")\
             .format(count_na_direction,
                     objects[pd.isna(objects.direction)]['id'].tolist()),
         level = lg.WARNING)

     if drop_na_direction and count_na_direction > 0:
         objects = objects[~pd.isna(objects.direction)]
         log(("Dropping {} {} with no direction.")\
                 .format(count_na_direction, snames),
             level = lg.WARNING)

 if len(objects) == 0:
     log("No more {} to process..".format(snames), level = lg.WARNING)
     return pd.DataFrame(columns = objects.columns.values)

 # Project coordinates
 log("Projecting {} to utm and adding geometry column.".format(snames),
     level = lg.INFO)

 objects.reset_index(drop=True, inplace = True)

 points = gpd.GeoSeries(
     [ shp.geometry.Point(x,y) for x, y in zip(
         objects['lon'],
         objects['lat'])
     ])

 geodf = gpd.GeoDataFrame(index    = objects.index,
                          geometry = points)
 geodf.crs = 'epsg:4326'

 avg_longitude = geodf['geometry'].unary_union.centroid.x
 utm_zone = int(math.floor((avg_longitude + 180) / 6.) + 1)
 utm_crs["zone"] = utm_zone

 proj_objects = geodf.to_crs(utm_crs)
 objects = proj_objects.join(objects, how = 'inner')

 return objects



def wrangle_cameras(
    cameras,
    is_test_col           = "name",
    is_commissioned_col   = "is_commissioned",
    road_attr_col         = "description",
    drop_car_park         = True,
    drop_na_direction     = True,
    direction_regex       = g_direction_regex,
    address_regex         = g_address_regex,
    road_ref_regex        = g_road_ref_regex,
    car_park_regex        = g_car_park_regex,
    directions_separator  = g_directions_separator,
    sort_by               = "id",
    utm_crs               = {'datum': 'WGS84',
                             'ellps': 'WGS84',
                             'proj' : 'utm',
                             'units': 'm'},
    distance_threshold    = 50.0,
    merge_cameras         = True
):
    """
    Wrangles a raw dataset of ANPR camera data.

    Parameters
    ----------
    cameras : pd.DataFrame
        dataframe

    Returns
    -------
    pd.DataFrame
        The same point given by latitude and longitude.

    """
    nrows = len(cameras)
    names = ("Camera", "cameras")
    start_time = time.time()

    cameras = wrangle_objects(
        cameras,
        is_test_col           = is_test_col,
        is_commissioned_col   = is_commissioned_col,
        road_attr_col         = road_attr_col,
        drop_car_park         = drop_car_park,
        drop_na_direction     = drop_na_direction,
        direction_regex       = direction_regex,
        address_regex         = address_regex,
        road_ref_regex        = road_ref_regex,
        car_park_regex        = car_park_regex,
        directions_separator  = directions_separator,
        sort_by               = sort_by,
        utm_crs               = utm_crs,
        object_name           = names
    )

    if merge_cameras:
        # Hard bit: merge cameras close by
        # ---------
        # First identify merges
        # ---------
        # Some roads have multiple cameras, side by side, one for each lane
        # When this happens, we should merge the nearby cameras, pointing in the
        # same direction, as a single camera. To do this, we compare the road attr
        # and compute the distance to every camera with the same road attrs

        to_merge = []
        for index, camera in cameras.iterrows():
            all_other_cameras = cameras.drop(index = index, axis = 0)

            within_distance = filter_by_attr_distance(
                camera,
                all_other_cameras,
                distance_threshold = distance_threshold,
                same_direction_filter = "equals",
                object_name = names)

            if len(within_distance) > 0:
                to_merge.append(
                    frozenset(within_distance.index.values.tolist() + [index]))

        # Remove any repeated tuples
        # list of indices
        to_merge = list(map(lambda x: tuple(x), set(to_merge)))

        # Remove combinations that are subsets of larger combinations
        duplicates = []

        # there's probably a more efficient way, but this is
        # good enough for now
        for i,tupl in enumerate(to_merge):
            for j,other in enumerate(to_merge):
                if i != j and set(tupl).issubset(set(other)):
                    duplicates.append(tupl)

        to_merge = list(set(to_merge) -  set(duplicates))

        # reduce list of tuples to a single list (without duplicate ) of indices
        all_indices = reduce(lambda x,y: x | y, map(lambda x: set(x), to_merge))

        # list of ids
        to_merge_ids = [ tuple(map(lambda x: cameras.loc[x, 'id'], tupl)) \
                         for tupl in to_merge ]

        log("Identified the following camera merges (ids): {}"\
                .format(to_merge_ids),
            level = lg.INFO)

        # ---------
        # Actual merge
        # ---------
        oldlen = len(cameras)

        # Use pandas concat to partition cameras and merge them back again
        unaffected_cameras = cameras.drop(index = all_indices, axis = 0)

        cameras_list = []
        for items in to_merge:
            ind1 = items[0]
            c1 = dict(cameras.loc[ind1])

            for i in range(1, len(items)):
                other = dict(cameras.loc[items[i]])

                c1['id'] = "{}-{}".format(c1['id'], other['id'])
                c1['name'] = "{}-{}".format(c1['name'], other['name'])
            # we use one of the geometries. Using the centroid of the 2 points might
            # not be a good idea as this might negatively impact the merge of
            # cameras onto the road network

            # inefficient but works
            newdf = pd.DataFrame(columns = c1.keys())
            newdf.loc[ind1] = list(c1.values())

            cameras_list.append(newdf)

        cameras_list.append(unaffected_cameras)
        cameras = pd.concat(cameras_list, axis = 0)

        log("Merged {} cameras that were in the same location as other cameras."\
                .format(oldlen - len(cameras)),
            level = lg.INFO)

    log("Sorting by {} and resetting index."\
            .format(sort_by),
        level = lg.INFO)

    # sort and reset_index
    cameras = cameras.sort_values(by=[sort_by])
    cameras = cameras.reset_index(drop=False)\
                     .rename(columns = {'id'    : 'old_id',
                                        'index' : 'id', })

    # convert bool cols to int type to avoid issues with read/write to/from disk
    cameras['both_directions'] = cameras['both_directions'].astype('int')

    log("Wrangled cameras in {:,.3f} seconds. Dropped {} rows, total is {}."\
            .format(time.time()-start_time, nrows - len(cameras), len(cameras)),
        level = lg.INFO)

    return gpd.GeoDataFrame(cameras)

# ------------------------------------------------------------------------------

def network_from_cameras(
    cameras,
    road_type = "all",
    # clean_intersections = False,
    tolerance = 30,
    min_bbox_length_km = 0.2,
    max_bbox_length_km = 50,
    bbox_margin = 0.10,
    retain_all = True,
    plot = False,
    **plot_kwargs
):
    """
    Get the road graph encompassing a set of ANPR cameras from OpenStreetMap.
    """
    log("Getting road network for cameras dataset of length {}"\
            .format(len(cameras)),
        level = lg.INFO)

    start_time = time.time()

    accepted_road_filters = ['all', 'primary', 'arterial']

    if road_type == "primary":
        osm_road_filter = \
            ('["area"!~"yes"]["highway"~"motorway|trunk|primary|'
             'secondary|tertiary"]["motor_vehicle"!~"no"]["motorcar"!~"no"]'
             '["access"!~"private"]')

    elif road_type == "arterial":
        osm_road_filter = \
            ('["area"!~"yes"]["highway"~"residential|living_street|'
             'unclassified|service"]["motor_vehicle"!~"no"]'
             '["motorcar"!~"no"]["access"!~"private"]')

    elif road_type == "all":
        osm_road_filter = None

    else:
        raise ValueError("Road filter has to be one of {}"\
                         .format(accepted_road_filters))

    xs = [p.x for p in cameras['geometry']]
    ys = [p.y for p in cameras['geometry']]

    center_x = min(xs) + (max(xs) - min(xs))/2
    center_y = min(ys) + (max(ys) - min(ys))/2

    dists_center = list(map(
        lambda p: math.sqrt((p.x - center_x) ** 2 + (p.y - center_y) ** 2),
        cameras['geometry']))

    length = max(dists_center)

    # 10% margin
    length = length + bbox_margin * length

    # if length is still very small
    if length < min_bbox_length_km * 1000:
        length = min_bbox_length_km * 1000

    elif length > max_bbox_length_km * 1000:
        raise ValueError("This exception prevents accidently querying large networks")

    # but now we need center point in lat,lon
    # points = [Point(lat,lng) for lat,lng in zip(cameras['lat'], cameras['lon'])]

    lat = cameras['lat']
    lon = cameras['lon']

    center_lat = min(lat) + (max(lat) - min(lat))/2
    center_lon = min(lon) + (max(lon) - min(lon))/2

    log("Center point = {}, distance = {}"\
            .format((center_lat, center_lon), length),
        level = lg.INFO)
    checkpoint = time.time()

    G = ox.graph_from_point(
        center_point = (center_lat, center_lon),
        distance = length,
        retain_all = retain_all,
        custom_filter = osm_road_filter
    )

    log("Returned road graph in {:,.3f} sec"\
            .format(time.time() - start_time),
        level = lg.INFO)
    checkpoint = time.time()

    # Make sure that every edge has a geometry attribute
    for u, v, data in G.edges(keys=False, data=True):
        if 'geometry' not in data:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            data['geometry'] = shp.geometry.LineString([(x1, y1), (x2, y2)])

    G = add_edge_directions(G)

    G = ox.project_graph(G)

    log("Added edge directions and projected graph in {:,.3f} sec"\
            .format(time.time() - checkpoint),
        level = lg.INFO)
    checkpoint = time.time()

    if plot:
        _, _, filename = plot_G(
            G,
            name = "road_graph",
            **plot_kwargs)

        log("Saved image of the road graph to disk {} in {:,.3f} sec"\
                .format(filename, time.time() - checkpoint),
            level = lg.INFO)
        checkpoint = time.time()

    # if clean_intersections:
    #     G = ox.clean_intersections(G, tolerance = tolerance)
    #
    #     log("Cleaned intersections (tol = {}) in {:,.3f} sec"\
    #             .format(tolerance, time.time() - checkpoint),
    #         level = lg.INFO)
    #     checkpoint = time.time()
    #
    #     if plot:
    #         _, _, filename = plot_G(
    #             G,
    #             name = "road_graph_cleaned",
    #             **plot_kwargs)
    #
    #         log("Saved image of cleaned road graph to disk {} in {:,.3f} sec"\
    #                 .format(filename, time.time() - checkpoint),
    #             level = lg.INFO)
    #         checkpoint = time.time()

    log("Retrieved road network from points in {:,.3f} sec"\
        .format(time.time() - start_time))

    if plot:
        if 'geometry' in cameras.columns:

            plot_kwargs['label'] = 'cameras'
            plot_kwargs['legend'] = True

            _, _, filename = plot_G(
                G,
                name = "road_graph_cameras",
                points = (cameras['geometry'].x, cameras['geometry'].y),
                **plot_kwargs)

            log("Saved image of cleaned road graph to disk {} in {:,.3f} sec"\
                    .format(filename, time.time() - checkpoint),
                level = lg.INFO)

            close_up_plots(G, cameras, **plot_kwargs)

        else:
            log(("Skipped making image of road graph with cameras because "
                "no geometry was available"),
                level = lg.WARNING)

    return G

def close_up_plots(
    G,
    cameras = None,
    bbox_distance = 400,
    **plot_kwargs
):
    """
    Close up plots of every camera.

    If cameras is provided, close-up plots of cameras are done using geometry
    coordinates for each row in dataframe (unmerged network).

    If cameras is not provided, then
    """
    filenames = []

    if cameras is None:
        merged = True

        if 'subdir' not in plot_kwargs:
            plot_kwargs['subdir'] = "cameras/merged"

        points = ([d['x'] for _, d in G.nodes(data = True) if d['is_camera']],
                  [d['y'] for _, d in G.nodes(data = True) if d['is_camera']])

        camera_nodes = [ data for node, data in G.nodes(data = True) \
                         if data['is_camera'] ]

        ids = [ data['id'] for node, data in G.nodes(data = True)\
                if data['is_camera'] ]

        node_ids = [ node for node, data in G.nodes(data = True)\
                     if data['is_camera'] ]

        cameras = pd.DataFrame(camera_nodes).assign(node = node_ids)
    else:
        merged = False

        if 'subdir' not in plot_kwargs:
            plot_kwargs['subdir'] = "cameras/unmerged"

        points = ([p.x for p in cameras['geometry']],
                  [p.y for p in cameras['geometry']])
        ids    = cameras['id'].tolist()

    for index, row in cameras.iterrows():

        if merged:
            x = row['x']
            y = row['y']
        else:
            x = row['geometry'].x
            y = row['geometry'].y

        bbox = (y + bbox_distance, y - bbox_distance,
                x + bbox_distance, x - bbox_distance)

        # filter points outside the bounding box
        poly = shp.geometry.box(x - bbox_distance, y - bbox_distance,
                            x + bbox_distance, y + bbox_distance)

        subpoints = [ (id,x,y) for id,x,y in zip(ids, points[0], points[1]) \
                      if shp.geometry.Point((x,y)).within(poly)]

        subids, tmp0, tmp1 = zip(*subpoints)
        subpoints = (tmp0, tmp1)

        checkpoint = time.time()

        _, _, filename = plot_G(
            G,
            name = row['id'],
            points = subpoints,
            bbox = bbox,
            labels = subids,
            **plot_kwargs
        )

        log("Saved image of close up camera {} to disk {} in {:,.3f} sec"\
                .format(row['id'], filename, time.time() - checkpoint),
            level = lg.INFO)

        filenames.append(filename)

    return filenames


def camera_candidate_edges(
    G,
    camera,
    camera_range = 45.0
):
    """
    Identify valid candidate edges
    """

    direction = camera['direction']
    address = camera['address']
    x = camera['geometry'].x
    y = camera['geometry'].y

    address_words = set(address.split(" "))

    # Get nearest edges to
    nedges = edges_by_distance(G, (y,x))

    # identify the edges that are in range
    distances = np.array(list(map(lambda x: x[1], nedges)))

    out_of_range_idx = np.argwhere(distances > camera_range)\
                         .reshape(-1)[0]
    in_range_slice = slice(0, (out_of_range_idx))

    candidate_edges = nedges[in_range_slice]

    if len(candidate_edges) == 0:
        log("No edges in range of camera {}. Closeste edge {} at {:,.2f} m"\
                .format(camera['id'], tuple(nedges[0][0][1:]), distances[0]),
            level = lg.WARNING)
        return candidate_edges

    # filter out candidates not pointing in same direction and re-arrange
    # by address
    geometries = list(map(lambda x: x[0][0], nedges[in_range_slice]))
    u_nodes = list(map(lambda x: x[0][1], candidate_edges))
    v_nodes = list(map(lambda x: x[0][2], candidate_edges))

    points_u = [ (G.nodes[u]['x'], (G.nodes[u]['y'])) for u in u_nodes]
    points_v = [ (G.nodes[v]['x'], (G.nodes[v]['y'])) for v in v_nodes]

    uv_vecs = [ (pv[0] - pu[0], pv[1] - pu[1]) \
                for pu,pv in zip(points_u, points_v)]

    uv_dirs = [ get_quadrant(np.rad2deg(
                    math.atan2(vec[1], vec[0]))) for vec in uv_vecs]

    same_dir = [ direction in uv_dir for uv_dir in uv_dirs ]

    # Names and refs might be None, str or list, must handle each case
    uv_refs = []
    uv_addresses = []
    for u,v in zip(u_nodes, v_nodes):
        attr = G.edges[u,v,0]
        if 'name' in attr:
            if isinstance(attr['name'], str):
                edge_address = attr['name'].split(" ")
            else:
                edge_address = " ".join(attr['name']).split(" ")
        else:
            edge_address = []
        uv_addresses.append(edge_address)

        if 'ref' in attr:
            if isinstance(attr['ref'], str):
                edge_ref = [attr['ref']]
            else:
                edge_ref = " ".join(attr['ref']).split(" ")
        else:
            edge_ref = []
        uv_refs.append(edge_ref)

    same_ref = [ len(set(uv_ref) & (address_words)) \
                 for uv_ref in uv_refs ]

    same_address = [len(set(uv_address) & (address_words)) \
                    for uv_address in uv_addresses ]

    candidates = \
        pd.DataFrame({
            'u' : u_nodes,
            'v' : v_nodes,
            'distance' : distances[in_range_slice],
            'point_u' : points_u,
            'point_v' : points_v,
            'geometry' : geometries,
            'dir_uv' : uv_dirs,
            'same_dir' : same_dir,
            'ref' : uv_refs,
            'same_ref' : same_ref,
            'address' : uv_addresses,
            'same_address' : same_address}
        )

    return candidates


def identify_cameras_merge(
    G,
    cameras,
    camera_range = 45.0
):
    """
    Identify camera merges
    """
    # Input validation
    required_cols = {'id', 'geometry', 'direction', 'address'}

    if not required_cols.issubset(set(cameras.columns.values)):
        raise ValueError("The following required columns are not available: {}"\
                         .format(required_cols))

    edges_to_remove = []
    edges_to_add = {}
    cameras_to_add = {}
    untreated = []
    untreatable = []

    # We assume that there is a road within 40 meters of each camera
    for index, row in cameras.iterrows():
        id = row['id']

        candidates = camera_candidate_edges(G, row, camera_range)

        if len(candidates) == 0:
            log(("({}) - Camera {} has no edge within {} meters. "
                 "Appending to untreatable list.")\
                    .format(index, id, camera_range),
                level = lg.WARNING)
            untreatable.append(id)
            continue

        # filter candidates not same dir and arrange by same address
        valid_candidates = candidates[candidates.same_dir == True]
        valid_candidates = valid_candidates.assign(
            same_ref_address = (valid_candidates.same_ref     > 0) &
                               (valid_candidates.same_address > 0))

        valid_candidates = valid_candidates.sort_values(
            by = ['same_ref_address', 'same_address', 'same_ref', 'distance'],
            ascending = [False, False, False, True])

        # If there was no suitable candidate
        if len(valid_candidates) == 0:
            log(("({}) - Camera {} has 0 valid candidate edges pointing in the "
                 "same direction {} as the camera. Flagging as untreatable.")\
                    .format(index, id, row['direction']),
                level = lg.ERROR)
            untreatable.append(id)
            continue

        log(("({}) - Camera {} has {}/{} edges pointing in the same direction "
             "{} and {} edges with the same reference and address, ({} same "
             "ref, {} same address). It's located on {}")\
                .format(index, id, len(valid_candidates), len(candidates),
                    row['direction'],
                    len(valid_candidates[valid_candidates.same_ref_address]),
                    len(valid_candidates[valid_candidates.same_ref     > 0]),
                    len(valid_candidates[valid_candidates.same_address > 0]),
                        row['address']),
            level = lg.INFO)

        chosen_edge = valid_candidates.iloc[0]

        line = chosen_edge['geometry']
        edge = (chosen_edge['u'], chosen_edge['v'])
        distance = chosen_edge['distance']
        ref = " ".join(chosen_edge['ref'])
        edge_address = " ".join(chosen_edge['address'])
        edge_dir = chosen_edge['dir_uv']

        log(("({}) - Camera {}: Picking top valid candidate edge {}. "
             "Distance: {:,.2f} meters, ref: {}, address: {}")\
                .format(index, id, edge, distance, ref, edge_address),
            level = lg.INFO)

        # Is this edge already assigned to a different camera?
        if edge in edges_to_remove:
            log(("({}) - Camera {}: another camera is already pointing at "
                "this edge. Appending to untreated list.")\
                    .format(index, id),
                level = lg.WARNING)

            untreated.append(index)
            continue

        # We get the attributes of G
        attr_uv = G.edges[edge[0], edge[1], 0]

        # Set the new node label
        camera_label = "c_{}".format(id)

        # It's this simple afterall
        camera_point = row['geometry']
        midpoint_dist = line.project(camera_point)
        sublines = cut(line, midpoint_dist)

        midpoint = line.interpolate(midpoint_dist).coords[0]

        # We have split the geometries and have the new point for the camera
        # belonging to both new geoms

        if len(sublines) == 1:
            # corner case:
            # camera overlaps with point in graph: cut again a few meters away
            # does it overlap with u or v?
            pu = shp.geometry.Point(chosen_edge['point_u'])
            pv = shp.geometry.Point(chosen_edge['point_v'])

            dists = (camera_point.distance(pu), camera_point.distance(pv))

            closest = np.argmin(dists)
            length = line.length

            if length > 6.000:
                cutoff = 5 if closest == 0 else line.length - 5
            else:
                cutoff = length/2

            sublines = cut(line, cutoff)

            midpoint = line.interpolate(cutoff).coords[0]

        # common case:
        # edge is split in two
        geom_u_camera = sublines[0]
        geom_camera_v = sublines[1]

        # Set the new edge attributes
        attr_u_camera = dict(attr_uv)
        attr_camera_v = dict(attr_uv)

        attr_u_camera['geometry'] = geom_u_camera
        attr_u_camera['length'] = attr_u_camera['geometry'].length

        attr_camera_v['geometry'] = geom_camera_v
        attr_camera_v['length'] = attr_camera_v['geometry'].length

        # I hate having to do this, but will do for now..
        new_row = dict(row)
        new_row['x'] = midpoint[0]
        new_row['y'] = midpoint[1]

        # Appending to output lists/dicts
        if camera_label in cameras_to_add.keys():
            # If this is a camera that sees in both directions, we still
            # just want to add one new node on the network but with edges
            # in both directions. We don't want duplicate camera entries
            # here, so if it already exists we update it's 'direction' key
            first_direction = cameras_to_add[camera_label]['direction']

            cameras_to_add[camera_label]['direction'] = \
                "{}-{}".format(first_direction, row['direction'])
        else:
            cameras_to_add[camera_label] = new_row

        edges_to_remove.append((edge[0], edge[1]))
        edges_to_add[(edge[0],camera_label)] = attr_u_camera
        edges_to_add[(camera_label,edge[1])] = attr_camera_v

        # Check if the resulting geom has the expected length
        if (geom_u_camera.length + geom_camera_v.length) - (line.length) > 1e-3:
            log(("({}) - Camera {}: There is a mismatch between the prior"
                 "and posterior geometries lengths:"
                 "{} -> {}, {} | {} != {} + {}")\
                    .format(index, id, edge, (edge[0],camera_label),
                            (camera_label,edge[1]), line.length,
                            geom_u_camera.length, geom_camera_v.length),
                level = lg.ERROR)

        log(("({}) - Camera {}: Scheduled the removal of "
             "edge {} and the addition of edges {}, {}.")\
                .format(index, id, edge,
                        (edge[0],camera_label), (camera_label,edge[1])),
            level = lg.INFO)


    return (cameras_to_add, edges_to_remove, edges_to_add,
            untreated, untreatable)

###
###

def merge_cameras_network(
    G,
    cameras,
    passes = 3,
    camera_range = 45.0,
    plot = True,
    **plot_kwargs
):
    """
    Merge
    """
    log("Merging {} cameras with road network with {} nodes and {} edges"\
            .format(len(cameras), len(G.nodes), len(G.edges)),
        level = lg.INFO)

    start_time = time.time()

    # Adding new attribute to nodes
    nx.set_node_attributes(G, False, 'is_camera')

    if 'both_directions' in cameras.columns.values:

        both_directions_mask = cameras.both_directions\
                                      .apply(lambda x: x == 1)

        tmp = cameras[both_directions_mask]

        tmp1 = tmp.assign(direction = tmp.direction.str.split("-").str[0])
        tmp2 = tmp.assign(direction = tmp.direction.str.split("-").str[1])

        to_merge = pd.concat([tmp1, tmp2, cameras[~both_directions_mask]])\
                     .reset_index(drop = True)

        log("Duplicated {} rows for cameras that see in both directions"\
                .format(len(cameras[both_directions_mask])),
            level = lg.INFO)

    else:
        to_merge = cameras

        log("No column 'both_directions' was found, dataframe is unchanged",
            level = lg.WARNING)

    all_untreatable = set()

    for i in range(passes):

        if len(to_merge) == 0:
            log("Pass {}/{}: Identifying edges to be added and removed."\
                    .format(i+1, passes),
                level = lg.INFO)
            break

        log("Pass {}/{}: Identifying edges to be added and removed."\
                .format(i+1, passes),
            level = lg.INFO)

        cameras_to_add, edges_to_remove, edges_to_add, untreated, untreatable =\
            identify_cameras_merge(G, to_merge, camera_range)

        all_untreatable.update(untreatable)

        log("Pass {}/{}: Adding {} cameras."\
                .format(i+1, passes, len(cameras_to_add)),
            level = lg.INFO)

        for label, row in cameras_to_add.items():
            d = dict(row)
            d['osmid'] = None
            d['is_camera'] = True

            G.add_node(label, **d)

        log("Pass {}/{}: Adding {} new edges."\
                .format(i+1, passes, len(edges_to_add)),
            level = lg.INFO)

        for edge, attr in edges_to_add.items():
            G.add_edge(edge[0], edge[1], **attr)

        log("Pass {}/{}: Removing {} stale edges."\
                .format(i+1, passes, len(edges_to_remove)),
            level = lg.INFO)

        for edge in edges_to_remove:
            G.remove_edge(*edge)

        log("Pass {}/{}: G has now {} nodes and {} edges."\
                .format(i+1, passes, len(G.nodes()), len(G.edges())),
            level = lg.INFO)

        to_merge = to_merge.loc[untreated]
        if len(to_merge) == 0:
            break
        else:
            log(("Pass {}/{}: {} cameras were not merged because their edge "
                 "overlapped with another camera.")\
                    .format(i+1, passes, len(untreated)),
                level = lg.INFO)


    checkpoint = time.time()
    log("Finished merging cameras with the road graph in {:,.2f} sec."\
            .format(checkpoint - start_time),
        level = lg.INFO)

    if len(all_untreatable) > 0:
        log(("{} cameras ({}) were flagged as 'untreatable' because there were "
             "no edges nearby that fit the distance and direction requirements."
             " Because of this they were not merged.")\
            .format(len(all_untreatable), all_untreatable),
        level = lg.WARNING)
    else:
        log(("No cameras were flagged as 'untreatable'."),
            level = lg.INFO)

    if len(to_merge) > 0:
        log("Cameras that could not be merged in {} passes: {}"\
                .format(passes, list(to_merge['id'])),
            level = lg.INFO)

    if plot:
        plot_kwargs['legend'] = True
        plot_kwargs['label'] = 'cameras'
        plot_kwargs['points_marker'] = '.'

        points = [ (data['x'],data['y']) for _, data in \
                    G.nodes(data = True) if data['is_camera'] ]

        _, _, filename = plot_G(
            G,
            name = "road_graph_merged",
            # key = "is_camera",
            points = ( list(map(lambda x: x[0], points)),
                       list(map(lambda x: x[1], points))),
            **plot_kwargs)

        log("Saved image of merged road graph to disk {} in {:,.2f} sec"\
                .format(filename, time.time() - checkpoint),
            level = lg.INFO)

        close_up_plots(G, **plot_kwargs)

    return G


def camera_pairs_from_graph(G):

    start_time = time.time()

    camera_nodes = [ data for node, data in G.nodes(data = True) \
                        if data['is_camera'] ]

    node_ids = [ node for node, data in G.nodes(data = True)\
                 if data['is_camera'] ]

    cameras = pd.DataFrame(camera_nodes)\
                .assign(node = node_ids)\
                .set_index('id')

    log(("Computing shortest paths and distances for {} + 1 cameras, total of "
         "{} pairs of cameras (including 1 dummy).")
            .format(len(cameras), (len(cameras) + 1) ** 2),
        level = lg.INFO)

    # Camera ids to serve as index
    cameras_id = cameras.index.tolist()
    # Add node to represent unknown origin/destination
    cameras_id.append(NA_CAMERA)

    # direction series
    direction = cameras['direction']

    # camera_pairs from cartesian product
    camera_pairs = pd.MultiIndex\
        .from_product([cameras_id, cameras_id],
                  names = ['origin', 'destination'])\
        .to_frame(index = False)\
        .merge(direction, how = 'left',
               left_on = 'origin', right_index = True)\
        .merge(direction, how = 'left',
               left_on = 'destination', right_index = True)\
        .rename(columns = {'direction_x' : 'direction_origin',
                           'direction_y' : 'direction_destination'})\
        .set_index(['origin', 'destination'], verify_integrity = True)\
        .sort_index()

    log(("Running shortest path algorithm. This may take a while."),
        level = lg.INFO)

    camera_pairs['valid'] = 1
    # invalid if opposite_directions
    camera_pairs.loc[\
        ((camera_pairs.direction_origin == 'N') & \
            (camera_pairs.direction_destination == 'S')) | \
        ((camera_pairs.direction_origin == 'S') & \
            (camera_pairs.direction_destination == 'N')) | \
        ((camera_pairs.direction_origin == 'E') & \
            (camera_pairs.direction_destination == 'W')) | \
        ((camera_pairs.direction_origin == 'W') & \
            (camera_pairs.direction_destination == 'E')), 'valid'] = 0

    # invalid if same origin and destination
    camera_pairs.loc[camera_pairs.index.get_level_values('origin') == \
                     camera_pairs.index.get_level_values('destination'),\
                    'valid'] = 0

    # invalid if unknown origin or destination
    camera_pairs.loc[
        (camera_pairs.index.get_level_values('origin')==NA_CAMERA) |\
        (camera_pairs.index.get_level_values('destination')==NA_CAMERA),\
        'valid'] = 0

    # Now we iterate through every row and run shortest path algorithm
    paths = []
    distances = []

    for origin, destination in camera_pairs.index:
        # corner cases
        if origin == destination or \
           origin == NA_CAMERA or \
           destination == NA_CAMERA:
            spath = np.nan
            distance = np.nan

        else:
            try:
                spath = nx.shortest_path(
                    G,
                    "c_{}".format(origin),
                    "c_{}".format(destination),
                    weight = 'length')

                edges = [(u,v) for u,v in zip(spath, spath[1:])]
                lengths = [ G.edges[u,v,0]['length'] for u,v in edges ]
                distance = reduce(lambda x,y: x+y, lengths)

                is_valid = camera_pairs.loc[(origin, destination), 'valid']

                if distance < 100.0 and is_valid == 1:
                    log(("Distance between cameras {} and {} is less than 100 "
                         "meters. Are these two cameras mergeable into one?")
                            .format(origin, destination),
                        level = lg.WARNING)

            except nx.NetworkXNoPath:
                spath = np.nan
                distance = np.inf
                camera_pairs.loc[(origin, destination), 'valid'] = 0
                log(("Could not find a path between {} and {}.")
                        .format(origin, destination),
                    level = lg.ERROR)

        paths.append(spath)
        distances.append(distance)

    camera_pairs = camera_pairs.assign(distance = distances, path = paths)
    expected_pairs = (len(cameras) + 1) ** 2

    # This should always be True unless I've coded something wrong
    assert len(camera_pairs) == len(paths) == len(distances)
    assert len(camera_pairs) == expected_pairs

    total_valid = len(camera_pairs[camera_pairs.valid == 1])

    log(("Out of {} possible camera pairs, {} were labelled as invalid, "
         "resulting in a total of {} valid camera pairs.")\
            .format(expected_pairs, expected_pairs - total_valid, total_valid),
        level = lg.INFO)

    log("Computed paths in {:,.1f} minutes"\
            .format((time.time() - start_time)/60.0),
        level = lg.INFO)

    log("Compute route geometries")

    camera_pairs['geometry'] = \
        camera_pairs.loc[~pd.isnull(camera_pairs.path), 'path']\
             .apply(lambda x: list(zip(iter(x), iter(x[1:]))))\
             .apply(lambda x: list(map(
                lambda uv: G.edges[uv[0],uv[1],0]['geometry'], x)))\
             .apply(lambda x: shp.ops.linemerge(shp.geometry.MultiLineString(x)))

    camera_pairs['is_contiguous'] = \
        camera_pairs['geometry']\
            .apply(lambda x: isinstance(x, shp.geometry.LineString))

    # turn paths into a string, so that we can write to file
    camera_pairs.loc[~pd.isnull(camera_pairs.path), 'path'] = \
        camera_pairs.loc[~pd.isnull(camera_pairs.path), 'path']\
        .apply(lambda x: ",".join(list(map(lambda y: str(y), x))))

    return gpd.GeoDataFrame(camera_pairs.reset_index())


def map_nodes_cameras(
    nodes,
    cameras,
    is_test_col           = "name",
    is_commissioned_col   = False,
    road_attr_col         = "description",
    drop_car_park         = True,
    drop_na_direction     = True,
    direction_regex       = g_direction_regex,
    address_regex         = g_address_regex,
    road_ref_regex        = g_road_ref_regex,
    car_park_regex        = g_car_park_regex,
    directions_separator  = g_directions_separator,
    sort_by               = "id",
    utm_crs               = {'datum': 'WGS84',
                             'ellps': 'WGS84',
                             'proj' : 'utm',
                             'units': 'm'},
    distance_threshold    = 150.0
):
    """
    Map 'nodes' to cameras by location, address and direction.
    """
    start_time = time.time()
    nrows = len(nodes)
    names = ("Node", "nodes")

    # Wrangle nodes: add new cols: address, direction, ref; filter cols:
    # is_carpark; project lat,lon
    nodes = wrangle_objects(
        nodes,
        is_test_col           = is_test_col,
        is_commissioned_col   = is_commissioned_col,
        road_attr_col         = road_attr_col,
        drop_car_park         = drop_car_park,
        drop_na_direction     = drop_na_direction,
        direction_regex       = direction_regex,
        address_regex         = address_regex,
        road_ref_regex        = road_ref_regex,
        car_park_regex        = car_park_regex,
        directions_separator  = directions_separator,
        sort_by               = sort_by,
        utm_crs               = utm_crs,
        object_name           = names
    )

    camera_map = pd.Series(index = nodes.index)

    for index, node in nodes.iterrows():
        id = node['id']

        within_distance = filter_by_attr_distance(
            node,
            cameras,
            same_direction_filter = "isin",
            distance_threshold = distance_threshold,
            object_name = ("Node", "cameras"))

        if len(within_distance) > 1:
            # Pick the top one?
            chosen_id = within_distance.iloc[0]['id']

            log(("Node {}: There are multiple cameras that can be mapped "
                 "(ids = {}) to this node. Picking the closest one: {}.")\
                    .format(id, within_distance['id'], chosen_id),
                level = lg.WARNING)

        elif len(within_distance) == 1:
            # We expect this to be the case more often than not
            chosen_id = within_distance.iloc[0]['id']

            log("Node {}: Mapping to camera {}."\
                    .format(id, chosen_id),
                level = lg.INFO)

        else:
            log("Node {}: Could not find a mapping to any camera."\
                    .format(id),
                level = lg.WARNING)

            chosen_id = NA_CAMERA

        camera_map.loc[index] = chosen_id

    log("Wrangled nodes in {:,.3f} seconds. Dropped {} rows, total is {}."\
            .format(time.time()-start_time, nrows - len(nodes), len(nodes)),
        level = lg.INFO)

    wnodes = nodes.assign(camera = camera_map)
    wnodes['camera'] = wnodes['camera'].astype('int')
    wnodes['both_directions'] = wnodes['both_directions'].astype('int')

    # Return wrangled nodes with 1:1 mapping to cameras
    return wnodes

def wrangle_raw_anpr(
    df,
    cameras = None,
    np_regex = g_np_regex,
    filter_low_confidence = True,
    confidence_threshold = 0.70,
    anonymise = True,
    digest_size = 10,
    digest_salt = os.urandom(10)
):
    """
    Wrangle raw anpr:

        a. Filter bad number plates
        b. Remove all sightings with confidence < 0.80
        c. Sort by Timestamp
        g. Anonymise
    """
    start_time = time.time()
    nrows = len(df)

    log("Wrangling raw ANPR dataset with {} rows and colnames: {}."\
            .format(nrows, ",".join(df.columns.values)),
        level = lg.INFO)

    # Assert dtypes
    str_cols = ['vehicle', 'camera']
    num_cols = ['confidence']
    dt_cols  = ['timestamp']

    log(("Checking if input dataframe contains mandatory columns {} "
         "and expected types.")\
            .format(str_cols + num_cols + dt_cols),
         level = lg.INFO)

    assert all(ptypes.is_string_dtype(df[col]) for col in str_cols)
    assert all(ptypes.is_numeric_dtype(df[col]) for col in num_cols)
    assert all(ptypes.is_datetime64_any_dtype(df[col]) for col in dt_cols)

    # Check if vehicle has any missing data (should not)
    na_vehicles = df['vehicle'].isna().sum()
    if na_vehicles > 0:
        log("Filtering {} na values in 'vehicle' column. Unique: {}."\
                .format(na_vehicles, df[df.vehicle.isna()]['vehicle'].unique()),
            level = lg.WARNING)
        df.dropna(axis = 0, subset = ['vehicle'], inplace = True)

    # Filter number plates that don't match regex
    df = df[df.vehicle.str.contains(np_regex)]

    frows = nrows - len(df)
    log(("Filtered {} rows ({:,.2f} %) containing a badly "
         "formatted plate number.")\
            .format(frows, frows/nrows*100),
        level = lg.INFO)

    cur_nrows = len(df)

    if cur_nrows == 0:
        return df

    # Filter low confidence observations
    if filter_low_confidence:
        df = df[df.confidence >= confidence_threshold]
        frows = cur_nrows - len(df)

        log("Filtered {} rows ({:,.2f} %) with low confidence."\
                .format(frows, frows/cur_nrows*100),
            level = lg.INFO)

    cur_nrows = len(df)

    if cur_nrows == 0:
        return df

    # Anonymise
    if anonymise:
        df['vehicle'] = df['vehicle']\
            .apply(lambda x: blake2b(x.encode(),
                                     key = digest_salt,
                                     digest_size = digest_size)\
                            .hexdigest())

        log("Anonymised plate numbers.", level = lg.INFO)

    # Sort by timestamp
    df = df.sort_values(by = ['timestamp'])

    # unique vehicles
    unique_vehicles = pd.Series(df['vehicle'].unique())
    veh2id = unique_vehicles.reset_index().set_index(0).squeeze()

    # Vehicle strings to integer (saves space!)
    df['vehicle'] = df['vehicle'].apply(lambda x: veh2id.loc[x])

    # Camera (old) id checks
    if cameras is not None:
        # Change camera column values to merged camera value
        merged_rows = cameras['old_id'].str.contains("-")
        merged_cameras = cameras[merged_rows]['old_id']

        unmerged_to_merged = {}
        for merged_camera in merged_cameras:
            original_cameras = merged_camera.split('-')
            for ocamera in original_cameras:
                unmerged_to_merged[ocamera] = merged_camera

        original_cameras = list(unmerged_to_merged.keys())

        is_merged_mask = (df['camera'].isin(original_cameras))
        is_merged_df = df[is_merged_mask]

        # Replacing unclustered with clustered old_id
        df.loc[is_merged_mask, 'camera'] = is_merged_df['camera']\
            .apply(lambda x: unmerged_to_merged[x])

        log(("Wrangled 'camera' column to new (merged) camera ids "
            "(affected {} rows).")\
                .format(len(is_merged_df)),
            level = lg.INFO)

        # Check if camera ids match in anpr and cameras dataframes
        unique_in_anpr = set(df['camera'].unique())
        unique_in_cameras = set(cameras['old_id'].unique())

        only_in_anpr = unique_in_anpr - unique_in_cameras

        if len(only_in_anpr) > 0:
            df = df[~df.camera.isin(only_in_anpr)]
            frows = cur_nrows - len(df)

            log(("Cameras {} show up in anpr data but not in cameras dataframe."
                 " Removing {} rows ({:,.2f} %) with these cameras.")\
                    .format(only_in_anpr, frows, frows/cur_nrows*100),
                level = lg.WARNING)

        only_in_cameras = unique_in_cameras - unique_in_anpr

        if len(only_in_cameras):
            log(("Cameras {} never show up in anpr data.")\
                    .format(only_in_cameras),
                level = lg.WARNING)

    frows = nrows - len(df)
    log(("Wrangled raw anpr dataset in {:,.3f} seconds. "
         "Dropped {} rows ({:,.2f} %), total is {}.")\
            .format(time.time()-start_time, frows, frows/len(df)*100, len(df)),
        level = lg.INFO)

    # map of old string ids to new integer ids
    old2new_id = cameras[['id', 'old_id']]\
                        .set_index('old_id')\
                        .squeeze() # make it a series

    # Replace camera old_id with id
    # (replacing string with integer - better for storage)
    df['camera'] = df['camera'].apply(lambda x: old2new_id.loc[x])

    df['confidence'] = df['confidence'].astype(int)

    return df.reset_index(drop = True)


def gdfs_from_network(G):
    """
    Convert a street network from a networkx MultiDiGraph object to a tuple
    of nodes and edges GeoDataFrames.
    """

    gdfs = ox.graph_to_gdfs(G)

    nodes_gdf = gdfs[0]

    if 'is_camera' in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.reset_index()\
                           .rename(columns = {'index' : 'node'})\
                           [['node', 'is_camera', 'geometry']]
    else:
        nodes_gdf = nodes_gdf.reset_index()\
                           .rename(columns = {'index' : 'node'})\
                           [['node', 'geometry']]


    edges_gdf = gdfs[1].drop(columns = ['osmid', 'service'])

    # Fix columns which have elements that are lists
    for col in edges_gdf.columns.tolist():
        # remove any instances of list: name
        col_has_list = edges_gdf[col].apply(lambda x: isinstance(x, list))

        # if there is at least one instance of type list:
        # make each element of list a string and join them
        if col_has_list.any():
            edges_gdf.loc[col_has_list, col] = \
                edges_gdf.loc[col_has_list][col].\
                    apply(lambda x: ",".join(list(map(lambda y: str(y), x))))

    # Transform direction tuples into objects
    edges_gdf = edges_gdf.assign(\
        direction = edges_gdf.direction.str[0] + '-' +\
                    edges_gdf.direction.str[1])

    # Need to transform to string otherwise there are mixed elements of type
    # int64 and string
    edges_gdf.u = edges_gdf.u.apply(lambda x: str(x))
    edges_gdf.v = edges_gdf.v.apply(lambda x: str(x))

    return (nodes_gdf, edges_gdf)
