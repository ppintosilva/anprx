"""Methods for wrangling and pre-processing anpr data and related datasets."""
# ------------------------------------------------------------------------------

from   .utils               import log

import re
import math
import time
import pandas               as pd
import geopandas            as gpd
import logging              as lg
from   shapely.geometry     import Point
from   collections          import OrderedDict

# ------------------------------------------------------------------------------

infer_direction_regex = \
('(East\/West|North\/South|Northbound|Eastbound|Southbound|Westbound)')

address_regex = \
(r'(East\/West|North\/South|Northbound|Eastbound|Southbound|Westbound|Site \d'
'|Camera \d|Camera|Site)')

road_category_regex = \
r'(A\d+|B\d+|C\d+)'

# ------------------------------------------------------------------------------
def wrangle_cameras(cameras,
                    infer_direction_col      = "description",
                    infer_direction_re       = infer_direction_regex,
                    drop_car_park            = "description",
                    drop_is_test             = "name",
                    drop_is_not_commissioned = True,
                    extract_address          = "description",
                    address_regex            = address_regex,
                    extract_road_category    = "description",
                    road_category_regex      = road_category_regex,
                    sort_by                  = "id",
                    project_coords           = True,
                    utm_crs                  = {'datum': 'WGS84',
                                               'ellps': 'WGS84',
                                               'proj' : 'utm',
                                               'units': 'm'}
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
    log("Wrangling cameras dataset with {} rows and colnames: {}"\
            .format(nrows, ",".join(cameras.columns.values)),
        level = lg.INFO)

    start_time = time.time()

    mandatory_columns = {'id', 'lat', 'lon'}
    optional_columns  = {'name', 'description', 'is_commissioned', 'type'}

    log("Checking if input dataframe contains mandatory columns {}."\
            .format(mandatory_columns),
        level = lg.INFO)

    cols = set(cameras.columns.values)

    assert mandatory_columns.issubset(cols)

    log("Detected {}/{} optional columns: {}"\
            .format(len(optional_columns & cols),
                    len(optional_columns),
                    optional_columns & cols),
        level = lg.INFO)

    log("Skipping {} unrecognised columns: {}."\
            .format(len(cols - optional_columns - mandatory_columns),
                    cols - optional_columns - mandatory_columns),
        level = lg.INFO)

    # Some asserts about the input data
    log("Checking if 'id' is unique."\
            .format(mandatory_columns),
        level = lg.INFO)
    assert len(cameras['id']) == len(cameras['id'].unique())

    # Find and drop cameras that are labelled as "test"
    if drop_is_test:
        oldlen = len(cameras)

        cameras = cameras.assign(
            is_test = cameras[drop_is_test].str.contains('test',
                                                         case = False))
        cameras = cameras[(cameras.is_test == False)]
        cameras = cameras.drop(columns = 'is_test')

        log("Dropping {} rows with 'test' in name"\
                .format(oldlen - len(cameras)),
            level = lg.INFO)


    # Find and drop cameras that not commissioned
    if drop_is_not_commissioned:
        oldlen = len(cameras)

        cameras = cameras[(cameras.is_commissioned == True)]
        cameras = cameras.drop('is_commissioned', axis = 1)

        log("Dropping {} rows with 'is_commissioned' == True."\
                .format(oldlen - len(cameras)),
            level = lg.INFO)

    # Find and drop cameras that are in car parks
    if drop_car_park:
        oldlen = len(cameras)

        cameras = cameras.assign(
            is_carpark = cameras[drop_car_park].str.contains('Car Park',
                                                             case = False))
        cameras = cameras[(cameras.is_carpark == False)]
        cameras = cameras.drop(columns = 'is_carpark')

        log("Dropping {} rows with 'is_carpark' == True."\
                .format(oldlen - len(cameras)),
            level = lg.INFO)

    if len(cameras) == 0:
        log("No more cameras to process..", level = lg.WARNING)
        return None

    # Get direction from other fields
    if infer_direction_col:
        log("Inferring direction based on column '{}'."\
                .format(infer_direction_col),
            level = lg.INFO)

        cameras = cameras.assign(direction = cameras[infer_direction_col]\
                    .str.extract(infer_direction_re, flags = re.IGNORECASE))

        # ugly code, but will have to do for now
        cameras.loc[~cameras['direction'].str.\
                contains("/", na=False), 'direction'] = \
            cameras.loc[~cameras['direction'].str.\
                contains("/", na=False)].direction.str[0] # get first char

        cameras.loc[cameras['direction'].str.\
                contains("/", na=False), 'direction'] = \
            cameras.loc[cameras['direction'].str.\
                contains("/", na=False)].direction.str.\
                split(pat = "/").apply(lambda x: (x[0][0], x[1][0]))
    else:
        log("Skipping inferring direction", level = lg.INFO)

    # Computing new column 'address'
    if extract_address:
        cameras = cameras.assign(
            address = cameras[extract_address]\
                        .str.replace(address_regex, '',regex = True))

        log("Extracting address from '{}'.".format(extract_address),
            level = lg.INFO)
    else:
        log("Skipping extracting address", level = lg.INFO)


    # Computing new column 'road_category'
    if extract_road_category:
        cameras = cameras.assign(
            road_category = cameras[extract_road_category]\
                                .str.extract(road_category_regex))
    else:
        log("Skipping extracting road category", level = lg.INFO)

    # Merge cameras:
    #   Combinations of lat/lon should be unique. If not, this may mean that
    #   we have multiple cameras in the same location. Furthermore, if these
    #   are pointing in the same direciton we should merge this into
    #   the same entity, otherwise it will cause problems later on
    oldlen = len(cameras)

    groups = cameras.groupby([cameras['lat'].apply(lambda x: round(x,8)),
                              cameras['lon'].apply(lambda x: round(x,8)),
                              cameras['direction']
                             ])
    ids = groups['id'].apply(lambda x: "-".join(x))\
                      .reset_index()\
                      .sort_values(by = ['lat', 'lon', 'direction'])['id']\
                      .reset_index(drop = True)

    names = groups['name'].apply(lambda x: "-".join(x))\
                          .reset_index()\
                          .sort_values(by = ['lat', 'lon', 'direction'])['name']\
                          .reset_index(drop = True)

    # Really janky way to do this, but I couldn't figure out the right way
    # to do this. I guess that should be done using the aggregate function.
    # But it's really unintuitive. Tidyverse is so much better.
    cameras  = cameras[groups['id'].cumcount() == 0]\
                    .sort_values(by = ['lat', 'lon', 'direction'])\
                    .reset_index(drop = True)

    cameras['id'] = ids
    cameras['name'] = names

    log("Merged {} cameras that were in the same location as other cameras."\
            .format(oldlen - len(cameras)),
        level = lg.INFO)


    log("Sorting by {} and resetting index."\
            .format(sort_by),
        level = lg.INFO)

    # sort and reset_index
    cameras = cameras.sort_values(by=[sort_by])
    cameras.reset_index(drop=True, inplace=True)

    if project_coords:
        log("Projecting cameras to utm and adding geometry column.",
            level = lg.INFO)

        camera_points = gpd.GeoSeries(
            [ Point(x,y) for x, y in zip(
                cameras['lon'],
                cameras['lat'])
            ])

        cameras_geodf = gpd.GeoDataFrame(index = cameras.index,
                                         geometry=camera_points)
        cameras_geodf.crs = {'init' :'epsg:4326'}

        avg_longitude = cameras_geodf['geometry'].unary_union.centroid.x
        utm_zone = int(math.floor((avg_longitude + 180) / 6.) + 1)
        utm_crs["zone"] = utm_zone,

        proj_cameras = cameras_geodf.to_crs(utm_crs)
        cameras = proj_cameras.join(cameras, how = 'inner')
    else:
        log("Skipping projecting coordinates to UTM", level = lg.INFO)

    log("Wrangled cameras in {:,.3f} seconds. Dropped {} rows, total is {}."\
            .format(time.time()-start_time, nrows - len(cameras), len(cameras)),
        level = lg.INFO)

    return cameras

# ------------------------------------------------------------------------------
