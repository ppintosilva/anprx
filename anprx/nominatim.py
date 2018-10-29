################################################################################
# Module: nominatim.py
# Description: Calls to external APIs
# License: Apache v2.0
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import time
import numpy as np
import osmnx as ox
import logging as lg
from collections import OrderedDict

from .helpers import flatten_dict
from .utils import settings, config, log

def search_address(address,
                   email = None):
    """
    Find the OpenStreetMap ways that match a given address.

    Parameters
    ----------
    address : string
        Address to search for

    email : string
        Valid email address in case you are making a large number of requests.

    Returns
    -------
    list of int
        List of osm ids for ways that match the given address query
    """

    params = OrderedDict()
    params['format'] = "json"
    params['address_details'] = 0

    if email:
        params['email'] = email

    params['q'] = address

    response_json = ox.nominatim_request(params = params, type = 'search')

    ways = filter(lambda x: x['osm_type'] == "way", response_json)
    osmids = map(lambda x: int(x["osm_id"]), ways)

    return list(osmids)


###
###
###


def lookup_address(osmids,
                   entity,
                   drop_keys = ['place_id', 'license', 'osm_type',
                                'osm_id', ' lat', 'lon', 'display_name',
                                'country', 'country_code', 'state',
                                'state_district', 'county', 'city'],
                   email = None):
    """
    Lookup the address of multiple OSM ids that share the same entity type.

    Parameters
    ----------
    osmids : array-like
        OSMids for address lookup. Hard limit of 50 ids as indicated in
        wiki.openstreetmap.org/wiki/Nominatim

    drop_keys : list
        keys to ignore from the nominatim response containing address details

    entity : string
        OSM entity of osmids. Valid values are
        'N' for Node, 'W' for Way and 'R' for Relation.

    email : string
        Valid email address in case you are making a large number of requests.

    Returns
    -------
    details : list of dict
        Address details for each input osmid
    """

    if entity not in {'N', 'W', 'R'}:
        raise ValueError("Not valid OSM entity. Choose one of 'N', 'W' or 'R'")

    if len(osmids) > 50:
        raise ValueError("Nominatim supports a maximum of 50 osmids in a lookup request.")

    params = OrderedDict()
    params['format'] = "json"
    params['address_details'] = 1

    if email:
        params['email'] = email

    osmids = np.array(list(map(str, osmids)))
    entities = np.repeat(entity, len(osmids))
    params['osm_ids'] = ",".join(np.core.defchararray.add(entities, osmids))

    response_json = ox.nominatim_request(params = params, type = 'lookup')

    details = []
    for dict_ in response_json:
        flattened_dict = flatten_dict(dict_, inherit_parent_key = False)
        if drop_keys:
            for key in drop_keys:
                flattened_dict.pop(key, None)
        details.append(flattened_dict)

    return details
