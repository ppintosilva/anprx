"""Methods for wrangling and pre-processing anpr data and related datasets."""
# ------------------------------------------------------------------------------

from   .utils               import log
from   .utils               import save_fig
from   .utils               import settings
from   .helpers             import add_edge_directions
from   .core                import Point
from   .core                import RelativeMargins
from   .core                import bbox_from_points

import re
import math
import time
import osmnx                as ox
import pandas               as pd
import geopandas            as gpd
import logging              as lg
import shapely.geometry     as geometry
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
def wrangle_cameras(
    cameras,
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

        cameras = cameras.assign(
            direction = cameras[infer_direction_col].str.extract(
                infer_direction_re,
                flags = re.IGNORECASE))

        cameras = cameras.assign(
            both_directions =
                (cameras.direction.str.contains("/|&", na=False))
        )

        # ugly code, but will have to do for now
        cameras.loc[~cameras.both_directions, 'direction'] = \
            cameras.loc[~cameras.both_directions].direction.str[0]

        cameras.loc[cameras.both_directions, 'direction'] = \
            cameras.loc[cameras.both_directions].direction.str\
                .split(pat = "/")\
                .apply(lambda x: "{}-{}".format(x[0][0], x[1][0]))
    else:
        log("Skipping inferring direction", level = lg.INFO)

    # Computing new column 'address'
    if extract_address:
        cameras = cameras.assign(
            address = cameras[extract_address]\
                        .str.replace(address_regex, '',regex = True)\
                            .replace(' +', ' ', regex = True))

        log("Extracting address from '{}'.".format(extract_address),
            level = lg.INFO)
    else:
        log("Skipping extracting address", level = lg.INFO)


    # Computing new column 'road category'
    if extract_road_category:
        cameras = cameras.assign(
            road_category = cameras[extract_road_category]\
                                .str.extract(road_category_regex))
        cameras = cameras.assign(road_category = cameras.road_category.str[0])
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
            [ geometry.Point(x,y) for x, y in zip(
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

def network_from_cameras(
    cameras,
    filter_residential = True,
    clean_intersections = False,
    tolerance = 30,
    make_plots = True,
    file_format = 'svg',
    margins = [0.1, 0.1, 0.1, 0.1],
    **plot_kwargs
):
    """
    Get the road graph encompassing a set of ANPR cameras from OpenStreetMap.
    """
    log("Getting road network for cameras dataset of length {}"\
            .format(len(cameras)),
        level = lg.INFO)

    start_time = time.time()

    if filter_residential:
        osm_road_filter = \
            ('["area"!~"yes"]["highway"~"motorway|trunk|primary|'
             'secondary|tertiary"]["motor_vehicle"!~"no"]["motorcar"!~"no"]'
             '["access"!~"private"]')
    else:
        osm_road_filter = None

    points = [Point(lat,lng) for lat,lng in zip(cameras['lat'], cameras['lon'])]

    log("{}".format(points))

    bbox = bbox_from_points(
        points = points,
        max_area = 1000000000,
        rel_margins = RelativeMargins(margins[0], margins[1],
                                      margins[2], margins[3])
    )

    log("Returning bbox from camera points: {}"\
            .format(bbox),
        level = lg.INFO)

    G = ox.graph_from_bbox(
            north = bbox.north,
            south = bbox.south,
            east = bbox.east,
            west = bbox.west,
            custom_filter = osm_road_filter
    )

    log("Returned road graph in {:,.3f} sec"\
            .format(time.time() - start_time),
        level = lg.INFO)
    checkpoint = time.time()

    G = add_edge_directions(G)

    G = ox.project_graph(G)

    log("Added edge directions and projected graph in {:,.3f} sec"\
            .format(checkpoint- start_time),
        level = lg.INFO)
    checkpoint = time.time()

    if make_plots:
        fig_height      = plot_kwargs.get('fig_height', 10)
        fig_width       = plot_kwargs.get('fig_width', 14)

        node_size       = plot_kwargs.get('node_size', 30)
        node_alpha      = plot_kwargs.get('node_alpha', 1)
        node_zorder     = plot_kwargs.get('node_zorder', 2)
        node_color      = plot_kwargs.get('node_color', '#66ccff')
        node_edgecolor  = plot_kwargs.get('node_edgecolor', 'k')

        edge_color      = plot_kwargs.get('edge_color', '#999999')
        edge_linewidth  = plot_kwargs.get('edge_linewidth', 1)
        edge_alpha      = plot_kwargs.get('edge_alpha', 1)


        fig, ax = ox.plot_graph(
            G, fig_height=fig_height, fig_width=fig_width,
            node_alpha=node_alpha, node_zorder=node_zorder,
            node_size = node_size, node_color=node_color,
            node_edgecolor=node_edgecolor, edge_color=edge_color,
            edge_linewidth = edge_linewidth, edge_alpha = edge_alpha,
            use_geom = True, annotate = False, save = False, show = False
        )

        image_name = "road_graph"
        save_fig(fig, ax, image_name, file_format = file_format, dpi = 320)

        filename = "{}/{}/{}.{}".format(
                        settings['app_folder'],
                        settings['images_folder_name'],
                        image_name, file_format)

        log("Saved image of the road graph to disk {} in {:,.3f} sec"\
                .format(filename, time.time() - checkpoint),
            level = lg.INFO)
        checkpoint = time.time()

    if clean_intersections:
        G = ox.clean_intersections(G, tolerance = tolerance)

        log("Cleaned intersections (tol = {}) in {:,.3f} sec"\
                .format(tolerance, time.time() - checkpoint),
            level = lg.INFO)
        checkpoint = time.time()

        if make_plots:
            image_name = "road_graph_cleaned"
            filename = "{}/{}/{}.{}".format(
                            settings['app_folder'],
                            settings['images_folder_name'],
                            image_name, file_format)

            fig, ax = ox.plot_graph(
                G, fig_height=fig_height, fig_width=fig_width,
                node_alpha=node_alpha, node_zorder=node_zorder,
                node_size = node_size, node_color=node_color,
                node_edgecolor=node_edgecolor, edge_color=edge_color,
                edge_linewidth = edge_linewidth, edge_alpha = edge_alpha,
                use_geom = True, annotate = False, save = False, show = False
            )

            save_fig(fig, ax, image_name, file_format = file_format, dpi = 320)

            log("Saved image of cleaned road graph to disk {} in {:,.3f} sec"\
                    .format(filename, time.time() - checkpoint),
                level = lg.INFO)
            checkpoint = time.time()

    if make_plots:
        if 'geometry' in cameras.columns:
            cameras_color  = plot_kwargs.get('cameras_color', '#D91A35')
            cameras_marker = plot_kwargs.get('cameras_marker', '*')
            cameras_size   = plot_kwargs.get('cameras_size', 100)
            cameras_zorder = plot_kwargs.get('cameras_zorder', 20)

            camera_points = ax.scatter(
                cameras['geometry'].x,
                cameras['geometry'].y,
                marker = cameras_marker,
                color = cameras_color,
                s = cameras_size,
                zorder = cameras_zorder,
                label = "cameras"
            )
            ax.legend()

            image_name = "road_graph_cleaned_cameras" if clean_intersections \
                         else "road_graph_cameras"
            filename = "{}/{}/{}.{}".format(
                            settings['app_folder'],
                            settings['images_folder_name'],
                            image_name, file_format)

            save_fig(fig, ax, image_name, file_format = file_format, dpi = 320)

            log("Saved image of cleaned road graph to disk {} in {:,.3f} sec"\
                    .format(filename, time.time() - checkpoint),
                level = lg.INFO)
        else:
            log(("Skipped making image of road graph with cameras because "
                "no geometry was available"),
                level = lg.WARNING)

    log("Retrieved road network from points in {:,.3f}"\
        .format(time.time() - start_time))
