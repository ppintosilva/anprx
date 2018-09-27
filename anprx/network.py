################################################################################
# Module: network.py
# Description: Useful classes and entities
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

from collections import namedtuple

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

class Node(object):
    """
    Node of a street network which represents a OpenStreetMap node and a road junction.

    Attributes
    ----------
    id : int
        node id (osmid)

    point : Point
        point that represents node (road junction)
    """
    def __init__(self, network, id):
        self.id = id
        self.point = Point(lat = network.node[id]['y'],
                           lng = network.node[id]['x'])


class Edge(object):
    """
    Directed edge of the street network which represents a OpenStreetMap way and a road segment.

    Attributes
    ----------
    from_ : Node or int
        from node

    to_ : Node or int
        to node

    key : int
        index in the list of edges between from and to nodesself.

    osmids : list
        OpenStreetMap ids of ways that are represented by this network edge.

    Node: In nx.MultiDiGraph graphs, nodes can have multiple edges between them, and so the key attribute is used to differentiate these.
    """
    def __init__(self, network, from_, to_, key):
        """
        """
        if isinstance(from_, Node):
            self.from_ = from_
        elif isinstance(from_, int):
            self.from_ = Node(network = network, id = from_)
        else:
            raise ValueError("from_ must be a Node or an int")

        if isinstance(to_, Node):
            self.to_ = to_
        elif isinstance(to_, int):
            self.to_ = Node(network = network, id = to_)
        else:
            raise ValueError("to_ must be a Node or an int")

        self.key = key

        self.osmids = network[self.from_.id][self.to_.id][self.key]["osmid"]

###
###

class Camera(object):
    """
    Represents a traffic camera located on the side of the road, observing the street.
    This may represent any type of camera recording a road segment, at a given
    bearing/angle. The orientation of the camera may be estimated by providing
    the address of the street it observes for labelled data, or solely based
    on it's location, for unlabelled data.

    Attributes
    ----------
    point : Point
        location of the camera

    address : str
        address of the street observed by the camera as labelled by a human

    orientation : dict of str : Orientation
        camera orientation
    """
    def __init__(self, point, address = None):
        """
        Parameters
        ---------
        point : Point
            location of the camera

        address : str
            address of the street observed by the camera as labelled by a human
        """
        self.point = point
        self.address = address

    def has_address(self):
        return self.address is None

###
###

# class CameraNetwork(nx.MultiDiGraph):
#     """
#     Represents a street network containing a set of traffic cameras.
#
#     Attributes
#     ----------
#     cameras : list(Camera)
#         traffic cameras
#     """
#     def __init__(self, cameras):
#         """
#         Parameters
#         ---------
#         cameras : Camera
#             traffic cameras
#         """
#         super(nx.MultiDiGraph, self).__init__()
#         self.cameras = cameras
#         #
#         # Get network
#         #
