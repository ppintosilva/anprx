################################################################################
# Module: nominatim.py
# Description: Core functions
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import osmnx as ox
from collections import OrderedDict

def lookup_ways(address):
    """
    Find the OpenStreetMap ways that match a given address.

    Parameters
    ----------
    address:
        Address for reverse geocoding

    Returns
    -------
    osm_ids :
        List of osm ids for ways that match the given address query
    """

    params = OrderedDict()
    params['format'] = "json"
    params['address_details'] = 0
    params['q'] = address

    response_json = ox.nominatim_request(params = params)

    return list(map(lambda x: int(x["osm_id"]), filter(lambda x: x['osm_type'] == "way", response_json)))
