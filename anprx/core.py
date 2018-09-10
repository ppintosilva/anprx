################################################################################
# Module: core.py
# Description: Core functions
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import math
import numpy as np
import osmnx as ox
import networkx as nx
from statistics import mean
from collections import namedtuple

from .network import *
from .helpers import *
from .constants import *
from .nominatim import lookup_ways


###
###

class GiantBBox(ValueError):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

###
###

def get_bbox_area(bbox,
                  unit = Units.km,
                  method = BBoxAreaMethod.simple):
    """
    Calculate the area of a bounding boxself.
    Choose one of two possible methods:

    **anpx.constants.BBoxAreaMethod.simple**

    Calculate the area as a rectangle using length as latitude difference and width as longitude difference corrected by mean latitude point.

    **anpx.constants.BBoxAreaMethod.sins**

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

    if method == BBoxAreaMethod.simple:
        # Latitude difference in degrees
        deg_length = bbox.north - bbox.south

        # Intuitively, the width of the bounding box should be the same and be calculated using the longitude degree difference. However, distance per degree of longitude varies with latitude. Hence, using the north and south latitudes will yield different results. This effect is negligible for short distances and small bounding boxes (which is often the case when dealing with city-wide data). We therefore use the mid longitude (between north and south longitudes) to approximate width. However, a more accurate model might be needed for large bounding boxes.

        # Longitude width in degrees
        deg_width = math.cos(np.deg2rad(bbox.south + deg_length/2)) * (bbox.west - bbox.east) #

        # 1 degree = 111110 meters
        # 1 degree squared = 111119 meters * 111119 meters = 12347432161
        area = abs(deg2sq_distance(unit) * deg_length * deg_width)

    elif method == BBoxAreaMethod.sins:

        rrad = (math.pi/180) * earth_radius(unit) ** 2
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
                     unit = Units.km,
                     rel_margins = RelativeMargins(0.025,0.025,0.025,0.025),
                     min_area = 0.01, # 0.01 sq km
                     max_area = 10.0): # 10 sq km
    """
    Get the bounding box that encompasses a set of points.

    Parameters
    ---------
    points : List[Point]
        list of points

    unit : Units
        unit of distance (m, km)

    rel_margins : RelativeMargins
        margins as a proportion of latitude/longitude difference

    min_area : float
        minimum area of bounding box in squared km

    max_area : float
        maximum area of bounding box in squared km

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

    bbox_area = get_bbox_area(bbox, unit = unit)

    if bbox_area < min_area:
        midpoint = get_meanpoint(points)

        if unit == Units.m:
            length = math.sqrt(min_area)
        elif unit == Units.km:
            length = math.sqrt(min_area * 1e6)

        bbox_ = ox.core.bbox_from_point(
           point = (midpoint.lat, midpoint.lng),
           distance = length)

        bbox = BBox(north = bbox_[0],
                    south = bbox_[1],
                    east = bbox_[2],
                    west = bbox_[3])

    elif bbox_area > max_area:
        # Too large network
        raise GiantBBox("BBox is too big: area of bounding box exceeds the upper bound. This is a safety feature. You can surpass this by re-running with a larger upper bound.")

    return bbox

###
###

def get_surrounding_network(points,
                            rel_margins = RelativeMargins(0.025,0.025,0.025,0.025),
                            min_area = 0.01, # 0.01 sq km (100m x 100m)
                            max_area = 10, # 10 sq km
                            graph_name = None):
    """
    Get the drivable network that encompasses a set of points.

    Parameters
    ----------
    points : List[Point]
        list of points

    rel_margins : RelativeMargins
        margins as a proportion of latitude/longitude difference

    min_area : float
        minimum area of bounding box in squared km

    max_area : float
        maximum area of bounding box in squared km

    Returns
    -------
    street_network :
        NetworkX MultiDiGraph
    """

    bbox = bbox_from_points(
        points = points,
        rel_margins = rel_margins,
        min_area = min_area,
        max_area = max_area)


    street_network = \
        ox.graph_from_bbox(
            north = bbox.north,
            south = bbox.south,
            east = bbox.east,
            west = bbox.west,
            network_type = "drive_service",
            simplify = True,
            retain_all = False,
            truncate_by_edge = False,
            name = graph_name,
            timeout = 180,
            memory = None,
            clean_periphery = True,
            infrastructure = 'way["highway"]',
            custom_filter = None)

    return street_network

###
###

def edges_from_osmid(network, osmids):
    """
    Get the network edge that matches a given osmid.

    Parameters
    ---------
    network : nx.MultiDiGraph
        a street network

    osmids : list(int)
        osmids of network edge = OSM way = road segment

    Returns
    -------
    generator
        generator of edges (Edge)
    """
    properties = {"osmid" : set(osmids)}
    for u,v in list(edges_with_properties(network, properties)):
        yield Edge(from_ = u, to_ = v, osmid = network[u][v]["osmid"])

###
###

def distance_to_edge(network, edge, point):
    """
    Calculate the distance of a given point to an edge of the network (road segment)

    Parameters
    ---------
    network : nx.MultiDiGraph
        street network

    point : point
        point

    Returns
    -------
    distance to road segment
        float
    """
    pass

def orientation_by_address(network, camera):
    """
    Estimate the orientation of a camera given the address of the street that
    it observes.

    Parameters
    ---------
    network : nx.MultiDiGraph
        street network

    camera : Camera
        traffic camera

    Returns
    -------
    camera orientation
        Orientation
    """
    if not camera.has_address():
        raise ValueError("Given camera has no defined address.")

    osm_ids = lookup_ways(camera.address)

    if len(osm_ids) == 0:
        raise ValueError("No ways found for the given address. Is the address valid?")



    pass

def orientation_by_position(network, camera):
    """
    Estimate the orientation of a camera given the address of the street that
    it observes.

    Parameters
    ---------
    network : nx.MultiDiGraph
        street network

    camera : Camera
        traffic camera

    Returns
    -------
    camera orientation
        Orientation
    """
    pass

def plot_camera(network, camera):
    """
    Plot a camera on the road network and the edge it observes, if available.
    """
    pass
