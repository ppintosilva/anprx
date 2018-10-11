################################################################################
# Module: navigation.py
# Description: Computing nvectors, great circle distances, bearings and the like
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import math
import time
import numpy as np
import osmnx as ox
import pandas as pd
import logging as lg
import networkx as nx
from statistics import mean
from collections import namedtuple

from .helpers import *
from .constants import *
from .nominatim import lookup_ways
from .utils import settings, config, log

###
###

Point = namedtuple(
    'Point',
    [
        'lat',
        'lng'
    ])
"""
Represents a point on the surface of the Earth, given by latitude and longitude, in degrees.

Attributes
----------
lat : float
    latitude

lng : float
    longitude
"""

###
###

BBox = namedtuple(
    'BBox',
    [
        'north',
        'south',
        'east',
        'west'
    ])
"""
Represents a bounding box, defined by 4 coordinate pairs. Instead of providing 4 points as input, redundancy is avoided by providing 2 values of latitude, north (max) and south (min), and 2 values of longitude, east (max) and west (min).

Attributes
---------
north : float
    maximum latitude

south : float
    minimum latitude

east : float
    maximum longitude

west : float
    minimum longitude
"""

###
###

RelativeMargins = namedtuple(
    'RelativeMargins',
    [
        'north',
        'south',
        'east',
        'west'
    ])
"""
Relative margins [0,1] for a given bounding box. These are calculated as the proportion of the latitude/longitude interval and added in degrees to the respective side.

Attributes
---------
north : float
    relative margin for maximum latitude

south : float
    relative margin for minimum latitude

east : float
    relative margin for maximum longitude

west : float
    relative margin for minimum longitude
"""

###
###

Orientation = namedtuple(
    'Orientation',
    [
        'bearing',
        'osm_way'
    ])
"""
Orientation of a traffic camera.

Attributes
----------
bearing : float
    Mean bearing of camera

osm_way : int
    OpenStreetMap Id of the way (road segment) that is observed by the camera.
"""

###
###

Edge = namedtuple(
    'Edge',
    [
        'u',
        'v',
        'k'
    ])
"""
Directed edge of the street network which represents a OpenStreetMap way and a road segment.

Attributes
----------
u : Node or int
    from node

v : Node or int
    to node

k : int
    index in the list of edges between u and v.
"""

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

def to_point(nvector):
    """
    Converts a point represented by an n-vector to latitude and longitude.

    Parameters
    ----------
    nvector : np.ndarray (3,)
        the n-vector of a point given by latitude and longitude

    Returns
    -------
    Point
        The same point given by latitude and longitude.

    """
    return math.atan2(nvector[0], math.sqrt(nvector[1] ** 2 + nvector[2] ** 2))

def to_nvector(point):
    """
    Converts a point represented by latitude and longitude to an n-vector.

    Parameters
    ----------
    point : Point
        a point given by latitude and longitude

    Returns
    -------
    np.ndarray
        A numpy nd.array with shape (3,), representing the same point as an n-vector (vector in 3D space).

    """

    lat = np.deg2rad(point.lat)
    lng = np.deg2rad(point.lng)

    return np.array([
        math.sin(lat),
        math.sin(lng) * math.cos(lat),
        -math.cos(lng) * math.cos(lat)])

def great_circle_distance(origin, destination):
    """
    Computes the great_circle_distance between two points represented by nvectors.

    Parameters
    ----------
    origin : numpy nd.array with shape (3,)
        origin point nvector

    destination : numpy nd.array with shape (3,)
        destination point nvector

    Returns
    -------
    float64
        great circle distance ("as the crow flies")

    """

    inner_p = np.inner(origin, destination)
    outer_p = np.cross(origin, destination)

    atan = math.atan2(np.linalg.norm(outer_p, ord = 2),
                      inner_p)

    return earth_radius() * atan

def true_bearing(origin, destination):
    """
    Calculates the true bearing between two points represented as nvectors.

    Parameters
    ----------
    origin : numpy nd.array with shape (3,)
        origin point nvector

    destination : numpy nd.array with shape (3,)
        destination point nvector

    Returns
    -------
    float64
        bearing (angle) between the two points, in degrees

    """
    north = np.array([0,0,1])
    c1 = np.cross(origin, destination)
    c2 = np.cross(origin, north)
    c1c2 = np.cross(c1,c2)

    bearing_sine = np.linalg.norm(c1c2, ord=2) * np.sign(np.inner(c1c2,origin))
    bearing_cossine = np.inner(c1,c2)
    bearing = math.atan2(bearing_sine, bearing_cossine)
    return (np.rad2deg(bearing) + 360) % 360



def as_lvector(origin, point):
    """
    Represents a Point as a vector in a (local) cartesian coordinate system, where another Point is the origin.

    Consider the case of a traffic camera and nearby nodes in the road network (junctions). A nearby node can be represented as a Point with polar coordinates (r,phi), where the camera is the origin, r is the distance between the camera and the node, and phi is the bearing between the two (basis used in navigation: the 0 degrees axis is drawn vertically upwards and the angle increases for clockwise rotations). Edges (roads) in the network can be represented by vector addition. For that purpose, the point is converted and returned in cartesian coordinates, in the standard basis.

    Parameters
    ----------
    origin : Point
        nvector of the Point used as origin in the new coordinate system

    point : Point
        target Point to be represented in the new coordinate system

    Returns
    -------
    np.ndarray (2,)
        vector representing the target Point in the new (local) cartesian coordinate system

    """
    origin_nvec = to_nvector(origin)
    point_nvec = to_nvector(point)

    r = great_circle_distance(origin_nvec, point_nvec)
    phi = true_bearing(origin_nvec, point_nvec)

    x = r * math.cos(np.deg2rad(90 - phi))
    y = r * math.sin(np.deg2rad(90 - phi))

    return np.array([x,y])

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
    Get the network edge(s) that match a given osmid.

    Parameters
    ---------
    network : nx.MultiDiGraph
        a street network

    osmids : list(int)
        osmids of network edge = 1+ OSM ways = road segment

    Returns
    -------
    generator
        generator of Edge(u,v,k)
    """
    properties = {"osmid" : osmids}
    log("Looking for the edges with the osmids: {}".format(set(osmids)))

    for u,v,k in list(edges_with_properties(network, properties)):
        yield Edge(u, v, k)

###
###

def distance_to_edge(network,
                     edge,
                     point,
                     method = EdgeDistanceMethod.farthest_node):
    """
    Calculate the distance of a given point to an edge of the network (road segment)

    Parameters
    ---------
    network : nx.MultiDiGraph
        street network

    edge : Edge
        network edge

    point : point
        point

    method : anprx.constants.EdgeDistanceMethod
        metric used to compute distance to edge

    Returns
    -------
    distance to road segment
        float
    """
    distance_node_from = ox.great_circle_vec(
                                lat1 = point.lat,
                                lng1 = point.lng,
                                lat2 = network.node[edge.u]['y'],
                                lng2 = network.node[edge.u]['x'],
                                earth_radius = earth_radius(unit = Units.m))

    distance_node_to = ox.great_circle_vec(
                                lat1 = point.lat,
                                lng1 = point.lng,
                                lat2 = network.node[edge.v]['y'],
                                lng2 = network.node[edge.v]['x'],
                                earth_radius = earth_radius(unit = Units.m))

    distances = [ distance_node_to, distance_node_from ]

    if method == EdgeDistanceMethod.closest_node:
        return min(distances)

    elif method == EdgeDistanceMethod.farthest_node:
        return max(distances)

    elif method == EdgeDistanceMethod.sum_of_distances:
        return sum(distances)

    elif method == EdgeDistanceMethod.mean_of_distances:
        return mean(distances)

    else:
        raise ValueError("Invalid method for computing edge distance")


###
###

def get_balltree(network):
    """
    G

    Parameters
    ---------
    network : nx.MultiDiGraph
        a street network

    Returns
    -------
    tree
        instance of sklearn.neighbors.BallTree
    """
    nodes = pd.DataFrame({'x': nx.get_node_attributes(network, 'x'),
                          'y': nx.get_node_attributes(network, 'y')})

    nodes_rad = np.deg2rad(nodes[['y', 'x']].astype(np.float))

    tree = BallTree(nodes_rad, metric='haversine')

    return tree, nodes

###
###

def get_nodes_in_range(network,
                       points,
                       radius,
                       tree = None):
    """
    Get nodes whose distance is within radius meters of a point, for a bunch of points. Vectorised.

    Parameters
    ---------
    network : nx.MultiDiGraph
        street network

    points : array-like[Point]
        array of points

    radius : float
        maximum distance in meters

    tree : sklearn.neighbors.BallTree
        ball-tree for quick nearest-neighbor lookup using the haversine formula

    Returns
    -------
    nearest nodes and distances, sorted according to points
        (np.array[np.array[float]],
         np.array[np.array[float]])
    """
    start_time = time.time()

    log("{} input points: {}".format(len(points), points),
        level = lg.INFO)

    if tree is None:
        tree, nodes = get_balltree(network)
        log("BallTree instance is None.")

    points_rad = np.deg2rad(points)

    nn_node_idx, nn_node_distances = \
            tree.query_radius(points_rad,
                              r = radius/rad2distance(Units.m),
                              return_distance = True)

    if nodes is None:
        nodes = pd.DataFrame({'x': nx.get_node_attributes(network, 'x'),
                              'y': nx.get_node_attributes(network, 'y')})

    node_ids = np.array([ np.array(nodes.iloc[point_nn].index).astype(np.int64) for point_nn in nn_node_idx ])

    log("Found osmids: {}".format(node_ids))

    # nn = [ (ids, distances ) for ids, distances in zip(nn_ids, nn_node_distances) ]
    nn_node_distances = nn_node_distances * rad2distance(Units.m)

    log("Found nearest nodes to {} points in {:,.3f} seconds".format(len(points), time.time()-start_time))

    return node_ids, nn_node_distances

###
###

def get_edges_in_range(network,
                       nodes_in_range):
    """
    Get edges whose nodes' distance is within radius meters of a point, for a bunch of points. Vectorised.

    Parameters
    ---------
    network : nx.MultiDiGraph
        street network

    points_nodes_in_range : list[Points]
        list of points

    Returns
    -------
    nearest nodes and distances, sorted according to points
        (np.array[np.array[float]] , np.array[np.array[float]] ]
    """
    start_time = time.time()

    edges_in_range = list()

    for point in nodes_in_range:

        edges = set()
        for nn in point:
            nearest_edges = \
                list(network.in_edges(nn, keys = True)) + \
                list(network.out_edges(nn, keys = True,))

            for edge in nearest_edges:
                edges.add(Edge(edge[0], edge[1], edge[2]))

        edges_in_range.append(edges)

    log("Found edges in range in {:,.3f} seconds, for {} points."\
            .format(time.time()-start_time,
                    len(nodes_in_range)),
        level = lg.INFO)

    return edges_in_range

###
###

def filter_by_address(network,
                      edges,
                      address = []):
    """
    Filter edges by address.

    Parameters
    ---------
    network : nx.MultiDiGraph
        street network

    edges : np.array([(u,v,k)])
        array of edges

    address : str

    Returns
    -------
    edges
        np.array([(u,v,k)])
    """
    start_time = time.time()

    log("Filtering edges by address.",
        level = lg.INFO)

    osmway_ids = lookup_ways(address)
    address_edges = set(edges_from_osmid(
                            network = network,
                            osmids  = osmway_ids))

    candidate_edges = edges & address_edges

    log("Filtered {} out of {} edges in {:,.3f} seconds, based on address: {}."\
            .format(len(edges) - len(candidate_edges),
                    len(edges),
                    time.time()-start_time,
                    address),
        level = lg.INFO)

    return candidate_edges

###
###

def local_coordinate_system(network,
                            origin,
                            nodes,
                            edges):
    """
    Generate a local cartesian coordinate system from a set of nodes and edges.

    Parameters
    ---------
    network : nx.MultiDiGraph
        street network

    origin : Point
        point

    nodes : array-like
        ids of nodes

    edges : array-like
        edges (u,v,k) to represent in new cartesian coordinate system

    Returns
    -------
    nodes and edges represented in new coordinate system
        (np.array([np.array(float)]),
         np.array([np.array(float)]))
    """
    start_time = time.time()

    # Nodes as vectors
    nodes_lvectors = \
        {
            node_id :
                as_lvector(
                    origin = origin,
                    point = Point(
                            lat = network.node[node_id]['y'],
                            lng = network.node[node_id]['x']))

            for node_id in nodes
        }

    log("Nodes lvectors: {}"\
            .format(nodes_lvectors),
        level = lg.DEBUG)

    edges_lvectors = \
        {
            edge :
                nodes_lvectors[edge[0]] - nodes_lvectors[edge[1]]

            for edge in edges
        }

    log("Nodes lvectors: {}"\
            .format(edges_lvectors),
        level = lg.DEBUG)

    log("Obtained coordinate system with origin at point {} in {:,.3f} seconds"\
            .format(origin, time.time()-start_time),
        level = lg.INFO)

    return nodes_lvectors, edges_lvectors
