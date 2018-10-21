################################################################################
# Module: nominatim.py
# Description: Core functions
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import osmnx as ox
from collections import OrderedDict

def search_address(address,
                   email = None):
    """
    Find the OpenStreetMap ways that match a given address.

    Parameters
    ----------
    address :
        Address to search for

    email :
        Valid email address in case you are making large number of

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

    response_json = ox.nominatim_request(params = params)

    ways = filter(lambda x: x['osm_type'] == "way", response_json)
    osmids = map(lambda x: int(x["osm_id"]), ways)

    return list(osmids)
