################################################################################
# Module: core.py
# Description: Core functions
# License: MIT
# Web: https://github.com/pedroswits/anprx
################################################################################

import math
import numpy as np
import osmnx as ox
import networkx as nx
from statistics import mean
from collections import namedtuple
from typing import Dict, Tuple, List, NamedTuple

from . import constants as const

###
###

Point = namedtuple(
    'Point',
    [
        'lat',
        'lng'
    ])
"""
Represents a point on the surface of the Earth, given by latitude and longitude.
"""

BBox = namedtuple(
    'BBox',
    [
        'north',
        'south',
        'east',
        'west'
    ])
"""
Represents a bounding box, defined by 4 coordinate pairs. However, redundancy is avoid by providing instead 2 values of latitudes, north and south, and 2 values of longitude, east and west.
"""

RelativeMargins = namedtuple(
    'RelativeMargins',
    [
        'north',
        'south',
        'east',
        'west'
    ])
"""
Relative margins [0,1] for a given bounding box.
"""

class TooBigBBox(ValueError):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

###
###

def points_from_lists(latitudes, longitudes):
    """
    Merge two lists, containing latitudes and longitudes,
    to a list of Points

    Parameters
    ---------
    latitudes : List[float]
        list of latitudes

    longitudes : List[float]
        list of longitudes

    Returns
    -------
    List[Point]
    """
    return [ Point(lat, lng) for lat, lng in zip(latitudes, longitudes) ]

###
###

def points_from_tuples(points):
    """
    Transform a list of tuples to a list of points (named tuples).

    Parameters
    ---------
    points : List[Tuple]
        list of points

    Returns
    -------
    List[Point]
    """
    return [ Point(x[0], x[1]) for x in points ]

###
###

def latitudes_from_points(points):
    """
    Return the latitudes of a list of points.

    Parameters
    ---------
    points : List[Point]
        list of points

    Returns
    -------
    latitudes
        List[float]
    """
    return [ point.lat for point in points ]

###
###

def longitudes_from_points(points):
    """
    Return the longitudes of a list of points.

    Parameters
    ---------
    points : List[Point]
        list of points

    Returns
    -------
    longitudes
        List[float]
    """
    return [ point.lng for point in points ]

###
###

def get_bbox_area(bbox,
                  unit = const.SQUARED_KM,
                  method = const.METHOD_AREA_SIMPLE):
    """
    Calculate the area of a bounding boxself.
    Choose one of two possible methods:
        METHOD_AREA_SIMPLE :
            Calculate the area as a rectangle using length as latitude difference and width as longitude difference corrected by mean latitude point.
        METHOD_AREA_SINS :
            As explained in: http://mathfax.com/area-of-a-latitude-longitude-rectangle/

    Parameters
    ---------
    bbox : BBox
        bounding box (north, south, east, west)

    unit
        one of SQUARED_KM or SQUARED_M

    Returns
    -------
    float
        area in provided units
    """
    if unit not in {const.SQUARED_M, const.SQUARED_KM}:
        raise ValueError("unit must be one of: units.SQUARED_KM , units.SQUARED_M")

    if method == const.METHOD_AREA_SIMPLE:
        # Latitude difference in degrees
        deg_length = bbox.north - bbox.south

        # Intuitively, the width of the bounding box should be the same and be calculated using the longitude degree difference. However, distance per degree of longitude varies with latitude. Hence, using the north and south latitudes will yield different results. This effect is negligible for short distances and small bounding boxes (which is often the case when dealing with city-wide data). We therefore use the mid longitude (between north and south longitudes) to approximate width. However, a more accurate model might be needed for large bounding boxes.

        # Longitude width in degrees
        deg_width = math.cos(np.deg2rad(bbox.south + deg_length/2)) * (bbox.west - bbox.east) #

        # 1 degree = 111110 meters
        # 1 degree squared = 111119 meters * 111119 meters = 12347432161
        deg_to_distance_squared = const.DEG_TO_M_SQUARED if unit == const.SQUARED_M else const.DEG_TO_KM_SQUARED

        area = abs(deg_to_distance_squared * deg_length * deg_width)

    elif method == const.METHOD_AREA_SINS:
        earth_radius = const.EARTH_RADIUS_M if unit == const.SQUARED_M else const.EARTH_RADIUS_KM

        rrad = (math.pi/180) * earth_radius ** 2
        sin_lat_diff = math.sin(np.deg2rad(bbox.north)) - math.sin(np.deg2rad(bbox.south))
        lng_diff = bbox.west - bbox.east

        area = rrad * abs(sin_lat_diff) * abs(lng_diff)

    else:
        raise ValueError("No such method for calculating area of bounding box.")

    return area

###
###

def get_meanpoint(points):
    """
    Calculate the geometrical meanpoint from a list of points.

    Parameters
    ---------
    points : List[Point]
        list of points

    Returns
    -------
    Point
        The mid or 'mean' point of a set of points, geometrically speaking.
    """
    x = [ math.cos(np.deg2rad(point.lat)) * math.cos(np.deg2rad(point.lng)) for point in points ]
    y = [ math.cos(np.deg2rad(point.lat)) * math.sin(np.deg2rad(point.lng)) for point in points ]
    z = [ math.sin(np.deg2rad(point.lat)) for point in points ]

    mean_x = mean(x)
    mean_y = mean(y)
    mean_z = mean(z)

    return Point(lng = np.rad2deg(math.atan2(mean_y, mean_x)),
                 lat = np.rad2deg(math.atan2(mean_z, math.sqrt(mean_x * mean_x + mean_y * mean_y))))

###
###

def bbox_from_points(points,
                     rel_margins = RelativeMargins(0.025,0.025,0.025,0.025),
                     area_lower_threshold_km2 = 0.01, # 0.01 sq km
                     area_upper_threshold_km2 = 10.0): # 10 sq km
    """
    Get the bounding box that encompasses a set of points.

    Parameters
    ---------
    points : List[Point]
        list of points

    rel_margins : RelativeMargins
        margins as a proportion of latitude/longitude difference

    Returns
    -------
    longitudes
        List[float]
    """
    if len(points) == 0:
        raise ValueError("List of points is empty.")

    latitudes = latitudes_from_points(points)
    longitudes = longitudes_from_points(points)

    max_lat = max(latitudes)
    min_lat = min(latitudes)
    max_lng = max(longitudes)
    min_lng = min(longitudes)

    bbox = BBox(north = max_lat + (max_lat - min_lat) * rel_margins.north,
                south = min_lat - (max_lat - min_lat) * rel_margins.south,
                east = max_lng + (max_lng - min_lng) * rel_margins.east,
                west = min_lng - (max_lng - min_lng) * rel_margins.west)

    bbox_area = get_bbox_area(bbox, unit = const.SQUARED_KM)
    print(bbox_area)

    if bbox_area < area_lower_threshold_km2:
        midpoint = get_meanpoint(points)

        bbox_ = ox.core.bbox_from_point(
           point = (midpoint.lat, midpoint.lng),
           distance = math.sqrt(area_lower_threshold_km2 * 1e6))

        bbox = BBox(north = bbox_[0],
                    south = bbox_[1],
                    east = bbox_[2],
                    west = bbox_[3])

    elif bbox_area > area_upper_threshold_km2:
        # Too large network
        raise TooBigBBox("BBox is too big: area of bounding box exceeds the upper bound. This is a safety feature. You can surpass this by re-running with a larger upper bound.")

    return bbox

###
###

#
# def get_surrounding_network(points : List[Point],
#                             rel_margins = RelativeMargins(0.025,0.025,0.025,0.025),
#                             abs_margins = AbsoluteMargins(0,0,0,0),
#                             area_lower_threshold = 0.01, # 0.01 sq km (100m x 100m)
#                             area_upper_threshold = 10, # 10 sq km
#                             graph_name = None) -> nx.MultiDiGraph :
#     """
#     Get the drivable network that encompasses a set of cameras.
#
#     Parameters
#     ----------
#     points :
#         A list of named tuples containing the points coordinates (lng, lat)
#
#     margin:
#         Margin factor for calculating bounding box that encompasses the cameras
#
#     Returns
#     -------
#     street_network :
#         NetworkX MultiDiGraph
#     """
#
#
#
#     street_network = \
#         ox.graph_from_bbox(
#             north = bbox.north,
#             south = bbox.south,
#             east = bbox.east,
#             west = bbox.west,
#             network_type = "drive_service",
#             simplify = True,
#             retain_all = False,
#             truncate_by_edge = False,
#             name = graph_name,
#             timeout = 180,
#             memory = None,
#             clean_periphery = True,
#             infrastructure = 'way["highway"]',
#             custom_filter = None)
#
#     return street_network
#
#
# net = get_surrounding_network(points)
# ox.plot_graph(net)
#
#
# def plot_camera(camera_xy,
#                 edge = None):
#     """
#     Plot a camera on the road network and the edge it observes, if available.
#     """
#     pass
#
# def plot_cameras(cameras_xy,
#                  edges):
#     """
#     Plot cameras on the road network and the edge they observe, if available.
#     """
#     pass
