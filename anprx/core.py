################################################################################
# Module: core.py
# Description: The Core
# License: Apache v2.0
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import math
import time
import statistics
import adjustText
import collections
import numpy                as np
import osmnx                as ox
import pandas               as pd
import logging              as lg
import networkx             as nx
import matplotlib.colorbar  as colorbar
import matplotlib.pyplot    as plt
import matplotlib.colors    as colors

from sklearn.neighbors      import BallTree

from .constants             import Units
from .constants             import earth_radius
from .constants             import deg2distance
from .constants             import rad2distance
from .constants             import deg2sq_distance
from .constants             import EdgeDistanceMethod

from .helpers               import chunks
from .helpers               import as_undirected
from .helpers               import angle_between
from .helpers               import edges_with_properties

from .utils                 import log
from .utils                 import save_fig

import anprx.nominatim      as nominatim

###
###

Point = collections.namedtuple(
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

BBox = collections.namedtuple(
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

RelativeMargins = collections.namedtuple(
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

Edge = collections.namedtuple(
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

def to_point(nvector):
    """
    Converts a point represented by an n-vector to latitude and longitude.

    Parameters
    ----------
    nvector : np.ndarray
        the n-vector of a point given by latitude and longitude

    Returns
    -------
    Point
        The same point given by latitude and longitude.

    """
    return math.atan2(nvector[0], math.sqrt(nvector[1] ** 2 + nvector[2] ** 2))

###
###

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
        A numpy.ndarray with shape (3,), representing the same point as an n-vector (vector in 3D space).

    """

    lat = np.deg2rad(point.lat)
    lng = np.deg2rad(point.lng)

    return np.array([
        math.sin(lat),
        math.sin(lng) * math.cos(lat),
        -math.cos(lng) * math.cos(lat)])

###
###

def great_circle_distance(origin, destination):
    """
    Computes the great_circle_distance between two points represented by nvectors.

    Parameters
    ----------
    origin : np.ndarray
        origin point nvector

    destination : np.ndarray
        destination point nvector

    Returns
    -------
    float
        great circle distance ("as the crow flies")

    """

    inner_p = np.inner(origin, destination)
    outer_p = np.cross(origin, destination)

    atan = math.atan2(np.linalg.norm(outer_p, ord = 2),
                      inner_p)

    return earth_radius() * atan

###
###

def true_bearing(origin, destination):
    """
    Calculates the true bearing between two points represented as nvectors.

    Parameters
    ----------
    origin : np.ndarray
        origin point nvector

    destination : np.ndarray
        destination point nvector

    Returns
    -------
    float
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

###
###

def as_lvector(origin, point):
    """
    Represents a Point as a vector in a (local) cartesian coordinate system, where another Point is the origin.

    Consider the case of a traffic camera and nearby nodes in the road network (junctions). A nearby node can be represented as a Point with polar coordinates (r,phi), where the camera is the origin, r is the distance between the camera and the node, and phi is the bearing between the two (basis used in navigation: the 0 degrees axis is drawn vertically upwards and the angle increases for clockwise rotations). Edges (roads) in the network can be represented by vector addition. For that purpose, the point is converted and returned in cartesian coordinates, in the standard basis.

    Parameters
    ----------
    origin : Point
        point used as origin in the new coordinate system

    point : Point
        target Point to be represented in the new coordinate system

    Returns
    -------
    np.ndarray
        vector representing the target Point in the new (local) cartesian coordinate system

    """
    r = ox.great_circle_vec(lat1 = origin.lat,
                            lat2 = point.lat,
                            lng1 = origin.lng,
                            lng2 = point.lng,
                            earth_radius = earth_radius(Units.m))

    phi = ox.get_bearing(origin, point)

    x = r * math.cos(np.deg2rad(90 - phi))
    y = r * math.sin(np.deg2rad(90 - phi))

    return np.array([x,y])

###
###

def from_lvector(origin, lvector):
    """
    Reconstruct the lat-lng point from its lvector.

    Parameters
    ----------
    origin : Point
        point used as origin in the local cartesian coordinate system

    lvector : np.ndarray (2,)
        vector representing the target Point in the new local cartesian coordinate system

    Returns
    -------
    Point
    """
    x = lvector[0]
    y = lvector[1]

    r = math.sqrt(x ** 2 + y ** 2)
    phi = np.deg2rad(90) - math.atan2(lvector[1],lvector[0])

    d = r / earth_radius(unit = Units.m)

    lat = np.deg2rad(origin[0])
    lng = np.deg2rad(origin[1])

    p_lat = math.asin(math.sin(lat) * math.cos(d) + \
                      math.cos(lat) * math.sin(d) * math.cos(phi))
    p_lng = lng + \
            math.atan2(math.sin(phi) * math.sin(d) * math.cos(lat),
                       math.cos(d) - math.sin(lat) * math.sin(p_lat))

    return Point(np.rad2deg(p_lat), np.rad2deg(p_lng))

###
###

class GiantBBox(ValueError):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

###
###

def get_bbox_area(bbox,
                  unit = Units.km,
                  method = "simple"):
    """
    Calculate the area of a bounding box.

    Parameters
    ---------
    bbox : BBox
        bounding box (north, south, east, west)

    unit : Unit
        unit of distance

    method : string
        Method used to compute the area of the bounding box.

        'simple' - calculates the area as a rectangle using length as latitude difference and width as longitude difference corrected by mean latitude point.

        'sins' - calculates the area according to the method presented in http://mathfax.com/area-of-a-latitude-longitude-rectangle/ (the link is currently down).

    Returns
    -------
    float
        area in provided units
    """

    if method == 'simple':
        # Latitude difference in degrees
        deg_length = bbox.north - bbox.south

        # Intuitively, the width of the bounding box should be the same and be calculated using the longitude degree difference. However, distance per degree of longitude varies with latitude. Hence, using the north and south latitudes will yield different results. This effect is negligible for short distances and small bounding boxes (which is often the case when dealing with city-wide data). We therefore use the mid longitude (between north and south longitudes) to approximate width. However, a more accurate model might be needed for large bounding boxes.

        # Longitude width in degrees
        deg_width = math.cos(
            np.deg2rad(bbox.south + deg_length/2)) * (bbox.west - bbox.east)

        # 1 degree = 111110 meters
        # 1 degree squared = 111119 meters * 111119 meters = 12347432161
        area = abs(deg2sq_distance(unit) * deg_length * deg_width)

    elif method == 'sins':

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
    points : list of Point
        list of points

    Returns
    -------
    Point
        The mid or 'mean' point of a set of points, geometrically speaking.
    """
    x = [ math.cos(np.deg2rad(point.lat)) * math.cos(np.deg2rad(point.lng)) for point in points ]
    y = [ math.cos(np.deg2rad(point.lat)) * math.sin(np.deg2rad(point.lng)) for point in points ]
    z = [ math.sin(np.deg2rad(point.lat)) for point in points ]

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    mean_z = statistics.mean(z)

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
    points : list of Point
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
    list of float
        longitudes

    """
    if len(points) == 0:
        raise ValueError("List of points is empty.")

    latitudes = [ point.lat for point in points ]
    longitudes = [ point.lng for point in points ]

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
                            min_area = 0.03, # 0.01 sq km (173m x 173m)
                            max_area = 10, # 10 sq km
                            unit = Units.km,
                            graph_name = None):
    """
    Get the drivable network that encompasses a set of points. Uses osmnx for this purpose.

    Parameters
    ----------
    points : list of Point
        list of points

    rel_margins : RelativeMargins
        margins as a proportion of latitude/longitude difference

    min_area : float
        minimum area of bounding box in squared km

    max_area : float
        maximum area of bounding box in squared km

    Returns
    -------
    nx.MultiDiGraph
        a graph representing the street network
    """

    bbox = bbox_from_points(
        points = points,
        rel_margins = rel_margins,
        min_area = min_area,
        max_area = max_area,
        unit = unit)


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
    Get the network edges that match a given osmid, for several input osmids.

    Parameters
    ---------
    network : nx.MultiDiGraph
        a street network

    osmids : list of int
        osmids of network edge = 1+ OSM ways = road segment

    Returns
    -------
    generator of Edge
        edges that match a osmid, for each input osmid
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

    method : EdgeDistanceMethod
        metric used to compute distance to edge

    Returns
    -------
    float
        distance from point to edge according to distance metric
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
        return statistics.mean(distances)

    else:
        raise ValueError("Invalid method for computing edge distance")


###
###

def get_balltree(network):
    """
    Generate a BallTree for a network that allows for fast generalized N-point problems, namely nearest nodes/edges search.

    Parameters
    ---------
    network : nx.MultiDiGraph
        a street network

    Returns
    -------
    sklearn.neighbors.BallTree
        a spatial index for the network
    """
    start_time = time.time()

    nodes = pd.DataFrame({'x': nx.get_node_attributes(network, 'x'),
                          'y': nx.get_node_attributes(network, 'y')})

    nodes_rad = np.deg2rad(nodes[['y', 'x']].astype(np.float))

    tree = BallTree(nodes_rad, metric='haversine')

    log("Generated BallTree in {:,.3f} seconds"\
            .format(time.time()-start_time),
        level = lg.INFO)

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

    points : array-like of Point
        array of points

    radius : float
        maximum distance in meters

    tree : sklearn.neighbors.BallTree
        ball-tree for quick nearest-neighbor lookup using the haversine formula

    Returns
    -------
    np.ndarray, np.ndarray
        nearest nodes (ids) and distances, sorted according to points
    """
    start_time = time.time()

    log("Finding nodes in range for {} input points."\
            .format(len(points)),
        level = lg.INFO)

    log("Points: {}".format(points),
        level = lg.DEBUG)

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

    log("Found {} osmids, respectively."\
            .format([len(ids) for ids in node_ids]),
        level = lg.INFO)

    log("Osmids: {}".format(node_ids),
        level = lg.DEBUG)

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

    nodes_in_range : np.ndarray
        ids of nodes in range, for a bunch of points

    Returns
    -------
    list of list of Edge
        list of set of nearest edges, sorted according to input nodes
    """
    start_time = time.time()

    edges_in_range = list()

    for point in nodes_in_range:

        edges = list()
        for nn in point:
            nearest_edges = \
                list(network.in_edges(nn, keys = True)) + \
                list(network.out_edges(nn, keys = True,))

            for edge in nearest_edges:
                edges.append(Edge(edge[0], edge[1], edge[2]))

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

    edges : array-like of Edge
        array of edges

    address : str

    Returns
    -------
    list of Edge
        edges or a subset of edges
    """
    start_time = time.time()

    log("Filtering edges by address.",
        level = lg.INFO)

    osmway_ids = nominatim.search_address(address)
    address_edges = edges_from_osmid(
                        network = network,
                        osmids  = osmway_ids)

    candidate_edges = set(edges) & set(address_edges)

    log("Filtered {} out of {} edges in {:,.3f} seconds, based on address: {}."\
            .format(len(edges) - len(candidate_edges),
                    len(edges),
                    time.time()-start_time,
                    address),
        level = lg.INFO)

    return list(candidate_edges)

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
        edges (u,v) to represent in new cartesian coordinate system

    Returns
    -------
    np.ndarray, np.ndarray
        nodes and edges represented in the new cartesian coordinate system
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


###
###

def flow_of_closest_lane(u, v,
                         left_handed = True):
    """
    Calculates the direction of traffic flow of the nearest lane, on the road represented by the straight line that passes through points {u,v}. This depends on whether traffic is left or righ-handed.

    For instance, consider an observer watching traffic in 2 traffic lanes, running in opposite directions, between two points: point one at 45 degrees of bearing and point 2 at 135 degrees of bearing. Then, if the traffic is left-handed, the closest of the two lanes is the one for which traffic flows from point 2 to point 1. If traffic is right-handed, then traffic is flowing from point 1 to point 2 in the closest of the two lanes (in reference to the observer - the origin of the coordinate system).

    Parameters
    ---------
    u : np.ndarray (2,)
        cartesian coordinates of point 1 relative to observer

    v : np.ndarray (2,)
        cartesian coordinates of point 2 relative to observer

    left_handed : bool
        True if the traffic keeps to the left side of the road, false otherwise.

    Returns
    -------
    tuple
        (u,v) if the closest lane corresponds to traffic flowing from u to v, (v,u) otherwise.
    """
    start_time = time.time()

    log("Calculating how traffic flows between u = {} and v = {}, in a {} traffic system."\
            .format(u,v,
                    "left-handed" if left_handed else "right-handed"),
        level = lg.INFO)

    log("{:17} = {}".format("point u", u),
        level = lg.DEBUG)
    log("{:17} = {}".format("point v", v),
        level = lg.DEBUG)

    phi_u = np.rad2deg(math.atan2(u[1], u[0]))
    phi_v = np.rad2deg(math.atan2(v[1], v[0]))

    log("{:17} = {:.2f}".format("phi_u", phi_u),
        level = lg.DEBUG)
    log("{:17} = {:.2f}".format("phi_v", phi_v),
        level = lg.DEBUG)

    phi_diff = phi_v - phi_u
    log("{:17} = {:.2f}".format("phi_v - phi_u", phi_diff),
        level = lg.DEBUG)

    if abs(phi_diff) > 180:
        phi_diff = phi_diff - np.sign(phi_diff)*360

    log("{:17} = {:.2f}"\
            .format("new phi_v - phi_u", phi_diff),
        level = lg.DEBUG)

    if (phi_diff > 0 and left_handed) or \
       (phi_diff < 0 and not left_handed):
        direction = (u,v)
    else:
        direction = (v,u)

    log("Found that cars flow from {} to {} in {:,.3f} seconds"\
            .format("u" if tuple(direction[0]) == tuple(u) else "v",
                    "v" if tuple(direction[1]) == tuple(v) else "u",
                    time.time() - start_time),
        level = lg.INFO)

    return direction

###
###
###

def get_dead_end_nodes(network):
    """
    Get nodes representing dead ends in the street network.

    These are not necessarily nodes with degree of 1, in the undirected representation of the street network,

    Parameters
    ----------
    network : nx.MultiDiGraph
        A street network

    Returns
    -------
    network : nx.MultiDiGraph
        The same network, but without dead end nodes and edges
    """
    if not 'streets_per_node' in network.graph:
        network.graph['streets_per_node'] = ox.count_streets_per_node(network)

    streets_per_node = network.graph['streets_per_node']

    return [node for node, count in streets_per_node.items() if count <= 1]


def remove_dead_end_nodes(network):
    """
    Remove nodes, and corresponding edges, representing dead ends in the street network.

    Parameters
    ----------
    network : nx.MultiDiGraph
        A street network

    Returns
    -------
    network : nx.MultiDiGraph
        The same network, but without dead end nodes and edges
    """
    start_time_local = time.time()

    dead_end_nodes = get_dead_end_nodes(network)
    network.remove_nodes_from(dead_end_nodes)

    log("Removed {} dead ends (nodes) in {:,.3f} seconds"\
            .format(len(dead_end_nodes), time.time() - start_time_local),
        level = lg.INFO)

###
###
###

def add_address_details(network,
                        drop_keys = ['place_id', 'license', 'osm_type',
                                     'osm_id', ' lat', 'lon', 'display_name',
                                     'country', 'country_code', 'state',
                                     'state_district', 'county', 'city'],
                        email = None):
    """
    Lookup and the address details of every edge in the network and
    add them as attributes.

    Depending on the size of the network, this method may incur a large number of requests and time to run. If anprx.settings['cache'] is set to True, as is by default, responses will be cached and subsequent calls to this method, for the same or intersecting networks, should be faster.

    Parameters
    ----------
    network : nx.MultiDiGraph
        A street network

    drop_keys : list
        keys to ignore from the nominatim response containing address details

    email : string
        Valid email address in case you are making a large number of requests

    Returns
    -------
    network : nx.MultiDiGraph
        The same network, but with additional edge attributes
    """
    start_time_local = time.time()

    unetwork = network.to_undirected(reciprocal = False)

    # Generator of groups of 50 edges
    edge_groups = chunks(l = list(unetwork.edges(keys = True, data = "osmid")),
                         n = 50)

    for group in edge_groups:
        # For edges with multiple osmids, pick the first osmid
        # Is there a better approach or is this a good enough approximation?
        osmids = [ osmid[0] if isinstance(osmid, collections.Iterable)
                   else osmid for osmid in map(lambda x: x[3], group) ]

        address_details = nominatim.lookup_address(
                                osmids = osmids,
                                entity = 'W',
                                drop_keys = drop_keys,
                                email = email)

        for edge, details  in zip(group, address_details):
            edge_uv = edge[:3]
            if network.has_edge(*edge_uv):
                network[edge_uv[0]][edge_uv[1]][edge_uv[2]].update(details)

            edge_vu = (edge[1], edge[0], edge[2])
            if network.has_edge(*edge_vu):
                network[edge_vu[0]][edge_vu[1]][edge_vu[2]].update(details)


    log("Added address details to {} groups of 50 edges in {:,.3f} seconds"\
        .format(len(list(edge_groups)), time.time() - start_time_local),
    level = lg.INFO)

    return network

###
###
###

def enrich_network(network,
                   clean_dead_ends = True,
                   elevation_api_key = None,
                   drop_keys = ['place_id', 'license', 'osm_type',
                                'osmid', ' lat', 'lon', 'display_name',
                                'country', 'country_code', 'state',
                                'state_district', 'county', 'city'],
                   email = None,
                   postcode_delim = ' '):
    """
    Enrich a street network by adding further attributes to the edges in the network. These can then be used in clustering, compression, graph embeddings, shortest paths, etc.

    Depending on the size of the network, this method may incur a large number of requests and time to run. If anprx.settings['cache'] is set to True, as is by default, responses will be cached and subsequent calls to this method, for the same or intersecting networks, should be faster.

    Parameters
    ----------
    network : nx.MultiDiGraph
        a street network

    clean_dead_ends : bool
        true if dead end nodes should be removed from the graph

    elevation_api_key : string
        Google API key necessary to access the Elevation API. If None, elevation.

    drop_keys: list
        keys to ignore from the nominatim response containing address details

    email : string
        Valid email address in case you are making a large number of requests.

    postcode_delim : string
        postcode delimiter used to split the main postcode into two parts: outer and inner. Use None to skip postcode splitting.

    Returns
    -------
    network :  nx.MultiDiGraph
        The same network, but with additional edge attributes
    """
    start_time = time.time()

    log("Enriching network with {} nodes and {} edges. This may take a while.."\
            .format(len(network), network.number_of_edges()),
        level = lg.INFO)

    if clean_dead_ends:
        remove_dead_end_nodes(network)

    # Add bearings
    network = ox.add_edge_bearings(network)

    # Elevation
    if elevation_api_key:
        start_time_local = time.time()
        # add elevation to each of the  nodes,using the google elevation API
        network = ox.add_node_elevations(network, api_key = elevation_api_key)
        # then calculate edge grades
        network = ox.add_edge_grades(network)
        log("Added node elevations and edge grades in {:,.3f} seconds"\
                .format(time.time() - start_time_local),
            level = lg.INFO)

    # lookup addresses
    network = add_address_details(network, drop_keys, email)

    # Split post code into outward and inward
    # assume that there is a space that can be used for string split
    for (u,v,k,postcode) in network.edges(keys = True, data = 'postcode'):
        if postcode:
            postcode_l = postcode.split(postcode_delim)
            if len(postcode_l) != 2:
                log("Could not split postcode {}".format(postcode),
                    level = lg.WARNING)
            else:
                network[u][v][k]['out_postcode'] = postcode_l[0]
                network[u][v][k]['in_postcode'] = postcode_l[1]

    log("Enriched network in {:,.3f} seconds"\
        .format(time.time() - start_time),
    level = lg.INFO)

    return network

###
###
###

def gen_lsystem(network,
                origin,
                radius,
                address = None):
    """
    Generate a local cartesian coordinate system from a street network, centered around origin, whose nodes and edges within radius are represented as points and vectors in the new coordinate system.

    Parameters
    ----------
    network : nx.MultiDiGraph
        a street network

    origin : Point
        point representing the origin of the new coordinate system (e.g. a traffic camera).

    radius : float
        range of the local coordinate system, in meters.

    address: string
        only include 'candidate' edges that match the given address

    Returns
    -------
    lsystem : dict
        local coordinate system with the following key-values

        :nnodes: list of int
            nodes near the camera. These are composed of the nodes that are within the range the camera and nodes whose edges have a node that is within the range of the camera.

        :nedges: list of Edge
            edges near the camera. Edges which have at least 1 node within the range of the camera.

        :cedges: list of Edge
            edges considered as candidates for self.edge - the edge observed by the camera

        :lnodes: dict( int : np.ndarray )
            nnodes represented in a cartesian coordinate system, whose origin is the camera

        :ledges: dict( Edge : np.ndarray )
            cedges represented in a cartesian coordinate system, whose origin is the camera
    """
    start_time = time.time()

    near_nodes, _ = \
        get_nodes_in_range(network = network,
                           points = np.array([origin]),
                           radius = radius)

    log("Near nodes: {}"\
            .format(near_nodes),
        level = lg.DEBUG)

    near_edges = get_edges_in_range(network, near_nodes)[0]

    log("Near nodes: {}"\
            .format(near_edges),
        level = lg.DEBUG)

    log("Found {} nodes and {} edges within {} meters of camera {}."\
            .format(len(near_nodes),
                    len(near_edges),
                    radius,
                    origin),
        level = lg.INFO)

    # Add nodes that where not initially detected as neighbors, but that are included in near_edges
    all_nodes = { edge[0] for edge in near_edges } | \
                { edge[1] for edge in near_edges }

    log("All nodes: {}"\
            .format(all_nodes),
        level = lg.DEBUG)

    log("Added {} out of range nodes that are part of nearest edges. Total nodes: {}."\
            .format(len(all_nodes - set(near_nodes[0])),
                    len(all_nodes)),
        level = lg.INFO)

    if address:
        candidate_edges = \
            filter_by_address(network,
                              near_edges,
                              address)
    else:
        candidate_edges = near_edges

    log("Candidate edges: {}"\
            .format(candidate_edges),
        level = lg.DEBUG)

    nodes_lvectors, edges_lvectors = \
        local_coordinate_system(network,
                                origin = origin,
                                nodes = all_nodes,
                                edges = candidate_edges)

    log("Generated local coordinate system for camera in {:,.3f} seconds".format(time.time()-start_time),
        level = lg.INFO)

    lsystem = dict()
    lsystem['nnodes'] = list(all_nodes)
    lsystem['nedges'] = list(near_edges)
    lsystem['cedges'] = list(candidate_edges)
    lsystem['lnodes'] = nodes_lvectors
    lsystem['ledges'] = edges_lvectors

    return lsystem

###
###
###

def estimate_camera_edge(network,
                         lsystem,
                         nsamples = 100,
                         radius = 40,
                         max_angle = 40,
                         left_handed_traffic = True,
                         return_samples = False):
    """
    Estimate the edge of the road network that the camera is observing.

    Points are sampled from each candidate edge and filtered based on whether the distance and angle to the camera is below the allowed maximum or not. With this, we can calculate the proportion of sampled points that fit this criteria and pick the edge(s) that maximises this proportion.

    Parameters
    ----------
    network : nx.MultiDiGraph
        a street network

    lsystem : dict
        local coordinate system obtained using `gen_lsystem`

    nsamples : int
        number of road points to sample when estimating the camera's observed edge.

    radius : int
        range of the camera, in meters. Usually limited to 50 meters.

    max_angle : int
        max angle between the camera and the cars (plate number) travelling on the road, at which the ANPR camera can reliably operate.

    left_handed_traffic : bool
        True if traffic flows on the left-hand side of the road, False otherwise.

    return_samples : bool
        True if you want the sampled points to be returned together with the estimated edge and calculated proportions

    Returns
    -------
    camera_edge, p_cedges, samples : Edge, dict, dict
        the estimated camera edge, the calculated proportions for each of the candidate edges and, if return_samples, a dict with the sampled point for each candidate edge
    """
    start_time = time.time()
    p_cedges = dict()
    samples = dict()

    for candidate in lsystem['cedges']:
        start_point = lsystem['lnodes'][candidate.u]
        finish_point = lsystem['lnodes'][candidate.v]
        line = lsystem['ledges'][candidate]
        step = -line/nsamples

        points = np.array([
                    start_point + step*i
                    for i in range(0, nsamples + 1)
                ])

        distances = np.linalg.norm(points, ord = 2, axis = 1)

        line_rep = np.repeat(np.reshape(line, (1,2)), nsamples + 1, axis = 0)
        angles = angle_between(points, line_rep)

        filter_point = np.vectorize(
            lambda d, a: True if d < radius and a < max_angle else False)

        unfiltered_points = filter_point(distances, angles)

        p_cedge = sum(unfiltered_points)/len(unfiltered_points)
        p_cedges[candidate] = p_cedge

        if return_samples:
            samples[candidate] = (points, unfiltered_points)

        log("Proportion for candidate {} : {:,.4f}"\
                .format(candidate, p_cedge),
            level = lg.INFO)

        log("start = {} ".format(start_point) +
            "finish = {} ".format(finish_point) +
            "step = {}\n".format(step) +
            "points = {}\n".format(points) +
            "distances = {}\n".format(distances) +
            "angles = {}".format(angles),
            level = lg.DEBUG)

    edge_maxp = max(p_cedges.keys(),
                    key=(lambda key: p_cedges[key]))

    # Is the street one way or two ways?
    reverse_edge = Edge(edge_maxp.v, edge_maxp.u, edge_maxp.k)

    if network.has_edge(*reverse_edge):
        # Two way street - figure out which of the lanes is closer based on left/right-handed traffic system
        point_u = lsystem['lnodes'][edge_maxp.u]
        point_v = lsystem['lnodes'][edge_maxp.v]

        flow = flow_of_closest_lane(point_u, point_v,
                                    left_handed_traffic)
        flow_from = flow[0]

        if tuple(flow_from) == tuple(point_u):
            camera_edge = edge_maxp
        else:
            camera_edge = reverse_edge
    else:
        # One way street - single edge between nodes
        camera_edge = edge_maxp

    log("The best guess for the edge observed by the camera is: {}"\
            .format(camera_edge))

    log("Estimated the edge observed by camera, using {} nsamples for each candidate, in {:,.3f} seconds"\
            .format(nsamples, time.time()-start_time),
        level = lg.INFO)

    if return_samples:
        return camera_edge, p_cedges, samples
    else:
        return camera_edge, p_cedges

###
###
###

class Camera(object):
    """
    A traffic camera located on the side of a drivable street.

    Attributes
    ----------
    network : nx.MultiDiGraph
        a street network

    id : string
        a camera identifier

    point : Point
        location of the camera

    edge : Edge
        edge observed by the camera.

        The edge is estimated using the method `estimate_edge`, and corresponds to the candidate edge that maximises the proportion of sampled points that meet the criteria (< radius and < max_angle) - max of values in p_cedges. Evidently, this value is the same for directed edges (u,v) and (v,u). Therefore, the edge (u,v) or (v,u) is picked according to which of the two traffic lanes is closer to the camera, in a left or right handed traffic system. Relative to the traffic camera (imagine it as the origin of a cartesian graph), in a left-handed traffic system, the inner traffic lane moves anticlockwise, except for roundabouts. Cameras are likely to be osbserving the inner of the two lanes to get a cleaner shot of the plate numbers (without vehicles crossing the camera's frame - or line of sight - in the opposite direction).

    address : str
        address of the street observed by the camera as labelled by a human

    radius : float
        range of the camera, in meters. Usually limited to 50 meters

    max_angle : float
        max angle, in degrees, between the camera and the vehicle's plate number, at which the ANPR camera can operate reliably. Usually up to 40 degrees

    lsystem : dict

        :nnodes: list of int
            nodes near the camera. These are composed of the nodes that are within the range the camera and nodes whose edges have a node that is within the range of the camera.

        :nedges: list of Edge
            edges near the camera. Edges which have at least 1 node within the range of the camera.

        :cedges: list of Edge
            edges considered as candidates for self.edge - the edge observed by the camera

        :lnodes: dict( int : np.ndarray )
            nnodes represented in a cartesian coordinate system, whose origin is the camera

        :ledges: dict( Edge : np.ndarray )
            cedges represented in a cartesian coordinate system, whose origin is the camera

    p_cedges : dict(Edge : float)
        proportion of sampled points from each candidate edge that meet the criteria (< radius and < max_angle)
    """
    def __init__(self,
                 network,
                 id,
                 point,
                 address = None,
                 radius = 40,
                 max_angle = 40,
                 nsamples = 100,
                 left_handed_traffic = True):
        """

        Parameters
        ---------
        network : nx.MultiDiGraph
            a street network

        id : string
            a camera identifier

        point : Point
            location of the camera

        address : str
            address of the street observed by the camera as labelled by a human. Used to excludes candidate edges whose address is different than this.

        radius : int
            range of the camera, in meters. Usually limited to 50 meters.

        max_angle : int
            max angle between the camera and the cars (plate number) travelling on the road, at which the ANPR camera can reliably operate.

        nsamples : int
            number of road points to sample when estimating the camera's observed edge.

        left_handed_traffic : bool
            True if traffic flows on the left-hand side of the road, False otherwise.
        """
        self.network = network
        # @TODO - Check if the camera location is encompassed by the network's bounding box?

        self.id = id
        self.point = point
        self.address = address
        self.radius = radius
        self.max_angle = max_angle
        self.left_handed_traffic = left_handed_traffic

        lsystem = gen_lsystem(network, point, radius, address)
        edge, p_cedges = \
            estimate_camera_edge(network,
                                 lsystem,
                                 nsamples,
                                 radius,
                                 max_angle,
                                 left_handed_traffic)

        self.lsystem = lsystem
        self.edge = edge
        self.p_cedges = p_cedges

###
###

    def plot(self,
             bbox_side = 100,
             camera_color = "#FFFFFF",
             camera_marker = "*",
             camera_markersize = 10,
             annotate_camera = True,
             draw_radius = False,
             #
             fig_height = 6,
             fig_width = None,
             margin = 0.02,
             bgcolor='k',
             node_color='#999999',
             node_edgecolor='none',
             node_zorder=2,
             node_size=50,
             node_alpha = 1,
             edge_color='#555555',
             edge_linewidth=1.5,
             edge_alpha=1,
             #
             probability_cmap = plt.cm.Oranges,
             show_colorbar_label = True,
             draw_colorbar = True,
             #
             nn_color = '#66B3BA',
             nedge_color = '#D0CE7C',
             labels_color = "white",
             annotate_nn_id = False,
             annotate_nn_distance = True,
             adjust_text = True,
             #
             save = False,
             file_format = 'png',
             filename = None,
             dpi = 300
             ):
        """
        Plot the camera on a networkx spatial graph.

        Parameters
        ----------
        bbox_side : int
            half the length of one side of the bbox (a square) in which to plot the camera. This value should usually be kept within small scales  (hundreds of meters), otherwise near nodes and candidate edges become imperceptible.

        camera_color : string
            the color of the point representing the location of the camera

        camera_marker : string
            marker used to represent the camera

        camera_markersize: int
            the size of the marker representing the camera

        annotate_camera : True
            whether to annotate the camera or not using its id

        draw_radius : bool
            whether to draw (kind of) a circle representing the range of the camera

        bgcolor : string
            the background color of the figure and axis - passed to osmnx's plot_graph

        node_color : string
            the color of the nodes - passed to osmnx's plot_graph

        node_edgecolor : string
            the color of the node's marker's border - passed to osmnx's plot_graph

        node_zorder : int
            zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot nodes beneath them or 3 to plot nodes atop them - passed to osmnx's plot_graph

        node_size : int
            the size of the nodes - passed to osmnx's plot_graph

        node_alpha : float
            the opacity of the nodes - passed to osmnx's plot_graph

        edge_color : string
            the color of the edges' lines - passed to osmnx's plot_graph

        edge_linewidth : float
            the width of the edges' lines - passed to osmnx's plot_graph

        edge_alpha : float
            the opacity of the edges' lines - passed to osmnx's plot_graph

        probability_cmap : matplotlib colormap
            Colormap used to color candidate edges by probability of observation.

        show_colorbar_label : bool
            whether to set the label of the colorbar or not

        draw_colorbar : bool
            whether to plot a colorbar as a legend for probability_cmap

        nn_color : string
            the color of near nodes - these are not necessarily in range of the camera, but they are part of edges that do

        nedge_color : string
            the color of candidate edges - nearby edges filtered by address or other condition

        labels_color : string
            the color of labels used to annotate nearby nodes

        annotate_nn_id : bool
            whether the text annotating near nodes should include their id

        annotate_nn_distance : bool
            whether the text annotating near nodes should include their distance from the camera

        adjust_text : bool
            whether to optimise the location of the annotations, using adjustText.adjust_text, so that overlaps are avoided. Notice that this incurs considerable computational cost. Turning this feature off will result in much faster plotting.

        save : bool
            whether to save the figure in the app folder's images directory

        file_format : string
            format of the image

        filename : string
            filename of the figure to be saved. The default value is the camera's id.

        dpi : int
            resolution of the image

        Returns
        -------
        fig, ax : tuple
        """
        if filename is None:
            filename = self.id

        bbox = ox.bbox_from_point(point = self.point,
                                  distance = bbox_side)


        # Set color of near nodes by index
        nodes_colors = [node_color] * len(self.network.nodes())

        i = 0
        for node in self.network.nodes(data = False):
            if node in self.lsystem['nnodes']:
                nodes_colors[i] = nn_color
            i = i + 1

        # Color near edges
        edges_colors = [edge_color] * len(self.network.edges())

        norm = colors.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=probability_cmap)
        pcolor = { edge : cmap.to_rgba(p)
                   for edge, p in self.p_cedges.items() }

        j = 0
        for u,v,k in self.network.edges(keys = True, data = False):
            edge = Edge(u,v,k)
            if edge in self.lsystem['cedges']:
                edges_colors[j] = pcolor[edge]
            j = j + 1

        # Plot it
        fig, axis = \
            ox.plot_graph(
                self.network,
                bbox = bbox,
                margin = margin,
                bgcolor = bgcolor,
                node_color = nodes_colors,
                node_edgecolor = node_edgecolor,
                node_zorder = node_zorder,
                edge_color = edges_colors,
                node_alpha = node_alpha,
                edge_linewidth = edge_linewidth,
                edge_alpha = edge_alpha,
                node_size = node_size,
                save = False,
                show = False,
                close = False,
                axis_off = True,
                fig_height = fig_height,
                fig_width = fig_width)

        if draw_colorbar:
            axis2 = fig.add_axes([0.3, 0.15, 0.15, 0.02])

            cb = colorbar.ColorbarBase(
                    axis2,
                    cmap=probability_cmap,
                    norm=norm,
                    orientation='horizontal')
            cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            if show_colorbar_label:
                cb.set_label("Probability of edge", color = labels_color, size = 9)
            cb.ax.xaxis.set_tick_params(pad=0,
                                        color = labels_color,
                                        labelcolor = labels_color,
                                        labelsize = 8)

        # Plot Camera
        camera_point = axis.plot(
                self.point.lng,
                self.point.lat,
                marker = camera_marker,
                color = camera_color,
                markersize = camera_markersize)


        if draw_radius:
            radius_circle = \
                plt.Circle((self.point.lng, self.point.lat),
                           radius = self.radius/deg2distance(unit = Units.m),
                           color=camera_color,
                           fill=False)

            axis.add_artist(radius_circle)

        if annotate_camera:
            camera_text = axis.annotate(
                            str(self.id),
                            xy = (self.point.lng, self.point.lat),
                            color = labels_color)

        if annotate_nn_id or annotate_nn_distance:
            # Annotate nearest_neighbors
            texts = []
            for id in self.lsystem['nnodes']:
                distance_x = self.lsystem['lnodes'][id][0]
                distance_y = self.lsystem['lnodes'][id][1]
                distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

                if distance < bbox_side:
                    s1 = ""
                    s2 = ""
                    if annotate_nn_id:
                        s1 = "{}: ".format(id)
                    if annotate_nn_distance:
                        s2 = "{:,.1f}m".format(distance)

                    text = axis.text(self.network.node[id]['x'],
                                     self.network.node[id]['y'],
                                     s = s1 + s2,
                                     color = labels_color)
                    texts.append(text)

            if annotate_camera:
                texts.append(camera_text)

            if adjust_text:
                additional_obj = []

                if draw_radius:
                    additional_obj.append(radius_circle)
                if annotate_camera:
                    additional_obj.append(camera_text)

                adjustText.adjust_text(
                    texts,
                    x = [ self.network.node[id]['x'] for id in self.lsystem['nnodes'] ],
                    y = [ self.network.node[id]['y'] for id in self.lsystem['nnodes'] ],
                    ax = axis,
                    add_objects = camera_point + additional_obj,
                    force_points = (0.5, 0.6),
                    expand_text = (1.2, 1.4),
                    expand_points = (1.4, 1.4))

        if save:
            save_fig(fig, axis, filename, file_format, dpi)

        return fig, axis
