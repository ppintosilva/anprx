################################################################################
# Module: core.py
# Description: Core functions
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import math
import time
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
from statistics import mean
from collections import namedtuple

from .network import *
from .helpers import *
from .constants import *
from .nominatim import lookup_ways
from .utils import settings, config, log
from .navigation import as_lvector


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

def get_nodes_in_range(network,
                       points,
                       radius,
                       tree = None):
    """
    Get nodes whose distance is within radius meters of a point, for a bunch of points.

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

    log("Computing nodes in range for {} points with radius {}"\
        .format(len(points), radius))
    log("Input points: {}".format(points))

    if tree is None:
        tree, nodes = get_balltree(network)
        log("BallTree instance is None. Call get_balltree.")

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

def get_edges_in_range(network, points_nodes_in_range):
    """
    Get nodes whose distance is within radius meters of a point, for a bunch of points.

    Parameters
    ---------
    network : nx.MultiDiGraph
        street network

    points : list[Points]
        list of points

    radius : float
        maximum distance in meters

    tree : sklearn.neighbors.BallTree
        ball-tree for quick nearest-neighbor lookup using the haversine formula

    Returns
    -------
    nearest nodes and distances, sorted according to points
        (np.array[np.array[float]] , np.array[np.array[float]] ]
    """
    points_edges_in_range = list()

    for point_nodes_in_range in points_nodes_in_range:

        edges = set()
        for node in point_nodes_in_range:
            node_edges = \
                list(network.in_edges(node, keys = True)) + \
                list(network.out_edges(node, keys = True,))

            for edge in node_edges:
                edges.add(Edge(edge[0], edge[1], edge[2]))

        points_edges_in_range.append(edges)

    return points_edges_in_range

###
###

def estimate_orientation(network,
                         camera,
                         filter_by = Filter.address,
                         set_value = True):
    """
    Estimate the orientation of a camera.

    Parameters
    ---------
    network : nx.MultiDiGraph
        street network

    camera : Camera
        traffic camera

    filter_by : anprx.constants.Filter
         address - filter nearby roads using the address of the street that the camera observes.
         none - guess camera orientation based on camera location alone

    set_value : bool
        set camera.orientation to estimated/returned value

    Returns
    -------
    camera orientation
        Orientation
    """

    near_nodes, _ = \
        get_nodes_in_range(network = network,
                           points = np.array([camera.point]),
                           radius = camera.radius)

    near_edges = get_edges_in_range(network, near_nodes)[0]

    log("Found {} nodes and {} edges within {} meters \
        of camera {}.".format(
                             len(near_nodes),
                             len(near_edges),
                             camera.radius,
                             camera.id))

    if filter_by == Filter.address:

        if not camera.has_address():
            raise ValueError("The given camera has no defined address.")

        osmway_ids = lookup_ways(camera.address)
        address_edges = set(edges_from_osmid(
                                network = network,
                                ids = osmway_ids))

        if len(osm_ids) == 0:
            raise ValueError("No ways found for the given address. Is the address valid?")

        filtered_edges = near_edges & address_edges

        log("Filtered {} out of {} edges from camera {} based \
            on address: {}.".format(
                                len(near_edges) - len(filtered_edges), len(near_edges),
                                camera.id,
                                camera.address))

    else:
        filtered_edges = near_edges

    # Nodes as vectors
    # nodes_lvectors = \
    #     {
    #         node_id: as_lvector(
    #                     origin = camera.point,
    #                     point = Point(lat = network.node[node_id]['y'],
    #                                   lng = network.node[node_id]['x']))
    #         for node_id in near_nodes[0]
    #     }

    # edges_lvectors =

    # Determine edge that maximizes camera placement
    return filtered_edges

###
###

#
# def sample_orientation_vectors(camera,
#                                minimum_range = 10,
#                                maximum_range = 35,
#                                sample_rate = 1):
#     """
#     Get
#
#     Parameters
#     ---------
#     camera : Camera
#         traffic camera
#
#     Returns
#     -------
#     sample orientation vectors
#
#     """
#     vectors = [ as_vector(camera.point, ) for degree in range(0, 360-sample_rate, sample_rate) ]


###
###

def plot_edges(network, edges, fig = None, axis = None):
    pass

def plot_nodes(network, nodes, fig = None, axis = None):
    pass

def plot_camera(network, camera, fig = None, axis = None):
    """
    Plot a camera on the road network and the edge it observes, if available.
    """
    pass
