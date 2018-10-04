################################################################################
# Module: navigation.py
# Description: Computing nvectors, great circle distances, bearings and the like
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import math
import numpy as np
from collections import namedtuple
from . import constants as const

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

    atan = math.atan2(y = np.linalg.norm(outer_p, ord = 2),
                      x = inner_p)

    return const.earth_radius() * atan

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

    bearing_sine = np.linalg.norm(c1c2, ord=2) * np.sign(np.inner(c1c2,a))
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
