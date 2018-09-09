################################################################################
# Module: core.py
# Description: Core functions
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

from .network import *

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
