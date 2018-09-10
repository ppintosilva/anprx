################################################################################
# Module: core.py
# Description: Core functions
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import networkx as nx
from .network import *
from .constants import *

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

def edges_with_at_least_one_property(network, properties):
    """
    Find edges that match at least once in all property sets: (key, values)

    Parameters
    ---------
    network : nx.MultiDiGraph
        a street network

    properties : dict(str : set)
        properties of edges to filter by

    Returns
    -------
    generator
        generator of edges (u,v)
    """
    for u,v,d in network.edges(data = True):
        for key, values in properties.items():
            if d[key] in values:
                yield (u,v)

###
###

def edges_with_all_properties(network, properties):
    """
    Find edges that match always in all property sets: (key, values)

    Parameters
    ---------
    network : nx.MultiDiGraph
        a street network

    properties : dict(str : set)
        properties of edges to filter by

    Returns
    -------
    generator
        generator of edges (u,v)
    """
    for u,v,d in network.edges(data = True):
        nmatches = 0
        for key, values in properties.items():
            if d[key] in values:
                nmatches = nmatches + 1
            else:
                break
        if nmatches == len(properties.keys()):
            yield (u,v)

###
###

def edges_with_properties(network, properties, match_by = PropertiesFilter.all):
    """
    Get edges with given properties

    Parameters
    ---------
    network : nx.MultiDiGraph
        a street network

    properties : dict(str : set)
        properties of edges to filter by

    match_by : int
        . One of const.FILTER_PROPERTIES.

    Returns
    -------
    generator
        generator of edges (Edge)
    """
    if match_by == PropertiesFilter.at_least_one:
        return edges_with_at_least_one_property(network, properties)

    elif match_by == PropertiesFilter.all:
        return edges_with_all_properties(network, properties)

    else:
        raise ValueError("Invalid 'match_by' value. Pick one of PropertiesFilter.{{{}}}.".format(PropertiesFilter.__order__))
