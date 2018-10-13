################################################################################
# Module: core.py
# Description: Core functions
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import collections
import numpy as np
import pandas as pd
import networkx as nx
from .utils import log
from .constants import *
from sklearn.neighbors import BallTree

###
###

def flatten(list_):
    """
    Flatten a list of objects which may contain other lists as elements.

    Parameters
    ---------
    list_ : object
        list

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
    Computes whether an object is present in, or has at least one element that is present in, values_set. This is equivalent to computing whether two sets intersect (not disjoint), but where value does not have to be a set.

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

def edges_with_at_least_one_property(G, properties):
    """
    Find edges that match at least once in all property sets: (key, values)

    Parameters
    ---------
    G : nx.MultiDiGraph
        a (multidi)graph

    properties : dict(str : set)
        properties of edges to filter by

    Returns
    -------
    generator
        generator of edges (u,v,key)
    """
    for u,v,k,d in G.edges(keys = True, data = True):
        for key, values in properties.items():
            if key in d.keys() and is_in(d[key], values):
                yield (u,v,k)

###
###

def edges_with_all_properties(G, properties):
    """
    Find edges that match always in all property sets: (key, values)

    Parameters
    ---------
    G : nx.MultiDiGraph
        a (multidi)graph

    properties : dict(str : set)
        properties of edges to filter by

    Returns
    -------
    generator
        generator of edges (u,v,key)
    """

    for u,v,k,d in G.edges(keys = True, data = True):
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

def edges_with_properties(G, properties, match_by = PropertiesFilter.all):
    """
    Get edges with given properties

    Parameters
    ---------
    G : nx.MultiDiGraph
        a (multidi)graph

    properties : dict(str : set)
        properties of edges to filter by

    match_by : int
        One of const.FILTER_PROPERTIES.

    Returns
    -------
    generator
        generator of edges (u,v,key)
    """
    if   match_by == PropertiesFilter.at_least_one:
        return edges_with_at_least_one_property(G, properties)

    elif match_by == PropertiesFilter.all:
        return edges_with_all_properties(G, properties)

    else:
        raise ValueError("Invalid 'match_by' value. Pick one of PropertiesFilter.{{{}}}.".format(PropertiesFilter.__order__))

def unit_vector(v):
    """
    Calculate the unit vector of an array or bunch of arrays.

    Parameters
    ---------
    v : np.ndarray
        vector(s)

    Returns
    -------
    v_u
        unit vector(s) of v
    """
    norm = np.linalg.norm(v, axis = 1)
    return v / np.reshape(norm, (len(v), 1))
