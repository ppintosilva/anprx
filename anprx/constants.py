################################################################################
# Module: constants.py
# Description: Constants and enumerations
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

from enum import Enum

class Units(Enum):
    __order__ = 'm km'
    m = 1
    km = 2


def earth_radius(unit = Units.m):
    if unit == Units.m:
        return 6371009

    elif unit == Units.km:
        return 6371

    else:
        valid_units = {Units.m.name, Units.km.name}
        raise ValueError("Unit must be in one of {}".format(valid_units))

def deg2distance(unit = Units.km):
    if unit == Units.m:
        return 111119

    elif unit == Units.km:
        return 111.119

    else:
        valid_units = {Units.m.name, Units.km.name}
        raise ValueError("Unit must be in one of {}".format(valid_units))

def rad2distance(unit = Units.m):
    if unit == Units.m:
        return 6367000

    elif unit == Units.km:
        return 6367

    else:
        valid_units = {Units.m.name, Units.km.name}
        raise ValueError("Unit must be in one of {}".format(valid_units))


def deg2sq_distance(unit = Units.m):
    if unit == Units.m:
        return 12347432161

    elif unit == Units.km:
        return 12347.432161

    else:
        valid_units = {Units.m.name, Units.km.name}
        raise ValueError("Unit must be in one of {}".format(valid_units))

class BBoxAreaMethod(Enum):
    __order__ = 'simple sins'
    simple = 1
    sins = 2

class PropertiesFilter(Enum):
    __order__ = 'all at_least_one'
    all = 1
    at_least_one = 2

class EdgeDistanceMethod(Enum):
    __order__ = "sum_of_distances, mean_of_distances, closest_node, farthest_node"
    sum_of_distances = 1
    mean_of_distances = 2
    closest_node = 3
    farthest_node = 4

class Filter(Enum):
    __order__ = "none, address"
    none = 1
    address = 2
