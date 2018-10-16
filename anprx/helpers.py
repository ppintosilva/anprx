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
    generator of (u,v,k)
        generator of edges
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
    generator of (u,v,k)
        generator of edges
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
    generator of (u,v,k)
        generator of edges
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
    np.ndarray
        unit vector(s) of v
    """
    norm = np.linalg.norm(v, axis = 1)
    return v / np.reshape(norm, (len(v), 1))

def dot2d(v1, v2, method = "einsum"):
    """
    Vectorised dot product for 2d vectors.

    Parameters
    ---------
    v1 : np.ndarray
        vectors on the left side of the dot product

    v2 : np.ndarray
        vectors on the right side of the dot product

    method: string
        method used to compute the dot product between each pair of members in v1,v2. One of {'einsum', 'loop'}

    Returns
    -------
    np.ndarray
        result of the dot products
    """
    if np.shape(v1) != np.shape(v2):
        raise ValueError("Input vectors don't have the same shape: {}, {}".format(np.shape(v1), np.shape(v2)))

    if method == "einsum":
        return np.einsum("ij, ij -> i", v1, v2)
    elif method == "loop":
        return np.array([i.dot(j)
                         for i,j in zip(v1,v2)])
    else:
        raise ValueError("No such method for computing the dot product.")

def angle_between(v1, v2):
    """
    Calculate the acute angle, in degrees, between two vectors. Vectorised for an array of vectors.

    Parameters
    ---------
    v1 : np.ndarray
        first vectors of each pair of vectors

    v2 : np.ndarray
        second vectors of each pair of vectors

    Returns
    -------
    np.ndarray
        acute angles between each pair of vectors
    """
    if np.shape(v1) != np.shape(v2):
        raise ValueError("Input vectors don't have the same shape: {}, {}".format(np.shape(v1), np.shape(v2)))

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    vdots = dot2d(v1_u, v2_u)
    clipped = np.clip(vdots, -1.0, 1.0)
    angles = np.rad2deg(np.arccos(clipped))

    reduce_angles = np.vectorize(
        lambda x: 180 - x if x > 90 else x)

    return reduce_angles(angles)
