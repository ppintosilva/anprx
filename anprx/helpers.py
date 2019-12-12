"""A series of helper methods that are useful in a variety of contexts."""

import  collections.abc
import  numpy                   as np
import  osmnx                   as ox
import  pandas                  as pd
import  networkx                as nx

import  shapely.geometry        as geometry

from    .utils                  import log

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
        if isinstance(el, collections.abc.Iterable) and\
           not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

###
###

def is_in(value, values_set):
    """
    Computes whether an object is present in, or has at least one element
    that is present in, values_set. This is equivalent to computing whether two
    sets intersect (not disjoint), but where value does not have to be a set.

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

###
###

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
        method used to compute the dot product between each pair of members in
        v1,v2. One of {'einsum', 'loop'}

    Returns
    -------
    np.ndarray
        result of the dot products
    """
    if np.shape(v1) != np.shape(v2):
        raise ValueError("Input vectors don't have the same shape: {}, {}"\
                          .format(np.shape(v1), np.shape(v2)))

    if method == "einsum":
        return np.einsum("ij, ij -> i", v1, v2)
    elif method == "loop":
        return np.array([i.dot(j)
                         for i,j in zip(v1,v2)])
    else:
        raise ValueError("No such method for computing the dot product.")

###
###

def angle_between(v1, v2):
    """
    Calculate the acute angle, in degrees, between two vectors.
    Vectorised for an array of vectors.

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
        raise ValueError("Input vectors don't have the same shape: {}, {}"\
                         .format(np.shape(v1), np.shape(v2)))

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    vdots = dot2d(v1_u, v2_u)
    clipped = np.clip(vdots, -1.0, 1.0)
    angles = np.rad2deg(np.arccos(clipped))

    reduce_angles = np.vectorize(
        lambda x: 180 - x if x > 90 else x)

    return reduce_angles(angles)


###
###

def flatten_dict(dict_, parent_key='', sep='_', inherit_parent_key = True):
    """
    Flatten a dict of objects which may contain other dicts as elements.

    Parameters
    ---------
    dict_ : object
        dict
        Borrowed from https://stackoverflow.com/a/6027615

    Returns
    -------
    generator
    """
    items = []
    for k, v in dict_.items():
        new_key = parent_key + sep + k if parent_key and inherit_parent_key\
                  else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v,
                                      parent_key = new_key,
                                      sep = sep,
                                      inherit_parent_key = inherit_parent_key)\
                         .items())
        else:
            items.append((new_key, v))
    return dict(items)

###
###

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.

    Parameters
    ----------

    l : list

    n : size of chunk

    Returns
    -------
    generator
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

###
###

def get_quadrant(phi):
    """
    Get the quadrant of an angle.

    E.g. phi = 40 -> ('E', 'N')

    Parameters
    ---------
    phi : angle

    Returns
    -------
    tuple
    """
    if phi > 360:
        phi = phi - 360

    if phi < 0:
        phi = phi + 360

    if   phi >= 0 and phi <= 45:
        return ('E', 'N')
    elif phi > 45 and phi <= 90:
        return ('N', 'E')
    elif phi > 90 and phi <= 135:
        return ('N', 'W')
    elif phi > 135 and phi <= 180:
        return ('W', 'N')
    elif phi > 180 and phi <= 225:
        return ('W', 'S')
    elif phi > 225 and phi <= 270:
        return ('S', 'W')
    elif phi > 270 and phi <= 315:
        return ('S', 'E')
    elif phi > 315 and phi <= 360:
        return ('E', 'S')
    else:
        raise ValueError("Input angle is not between [-360,720] degrees")


def cut(line, distance):
    """
    Cuts a LineString in two LineStrings at a distance from its starting point.

    Parameters
    ---------
    line : shapely.geometry.LineString

    distance : float
        distance from the starting point at which the line should be cut

    Returns
    -------
    list of length 2
        original line cut into two

    """
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [geometry.LineString(line)]

    coords = list(line.coords)

    for i, p in enumerate(coords):
        pd = line.project(geometry.Point(p))

        if pd == distance:
            return [
                geometry.LineString(coords[:i+1]),
                geometry.LineString(coords[i:])]

        if pd > distance:
            cp = line.interpolate(distance)
            return [
                geometry.LineString(coords[:i] + [(cp.x, cp.y)]),
                geometry.LineString([(cp.x, cp.y)] + coords[i:])]


def common_words(str1, str2, delim = " ", tolower = True):
    """
    Count how many words two strings have in common

    Parameters
    ---------
    str1 : string one
    str2 : string two
    delim : split each string using this delimiter
    tolower : whether to lower case strings before comparison

    Returns
    -------
    int
    """
    if tolower:
        str1 = str1.lower()
        str2 = str2.lower()

    words1 = set(str1.split(delim))
    words2 = set(str2.split(delim))

    return len(words1 & words2)
