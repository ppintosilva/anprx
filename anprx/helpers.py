################################################################################
# Module: core.py
# Description: Core functions
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import collections
import networkx as nx
from .utils import log
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

def flatten(list_):
    """
    Flatten a list of objects which may contain other lists as elements.

    Parameters
    ---------
    list_ : object
        data dictionary

    Returns
    -------
    generator
    """
    for el in list_:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

###
###

def is_in(value, values_set):
    """
    Computes whether an object is present in, or has at least one element that is present in, values_set. Calculates if two sets are not disjoint, but value does not have to be a set.

    Parameters
    ---------
    value : object
        data dictionary

    values_set : set
        set of values

    Returns
    -------
    bool
    """
    try:
        iter(value)
        is_iterable = True
    except TypeError:
        is_iterable = False

    if is_iterable and not isinstance(value, (str, bytes)):
        return not set(value).isdisjoint(values_set)
    else:
        return value in values_set
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
        generator of edges (u,v,key)
    """
    for u,v,k,d in network.edges(keys = True, data = True):
        for key, values in properties.items():
            if key in d.keys() and is_in(d[key], values):
                yield (u,v,k)

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
        generator of edges (u,v,key)
    """

    for u,v,k,d in network.edges(keys = True, data = True):
        nmatches = 0
        for key, values in properties.items():

            if key in d.keys() and is_in(d[key], values):
                nmatches = nmatches + 1
            else:
                break

        if nmatches == len(properties.keys()):
            yield (u,v,k)

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
