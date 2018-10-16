################################################################################
# Module: nominatim.py
# Description: Core functions
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import osmnx as ox
from collections import OrderedDict

from .constants import OsmEntity

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


# def lookup_osmids(osmids,
#                   types = OsmEntity.W,
#                   email = None):
#     """
#     Retrieve the address details of one or more OpenStreetMap entities, by id.
#
#     Parameters
#     ----------
#     osmids: list(int)
#         OSM ids of the entities. Maximum of 50.
#
#     types: list(OsmEntity) or OsmEntity
#         Class of OSM entity of each osmid.
#
#     Returns
#     -------
#     details :
#         dict
#     """
#
#     url = 'https://nominatim.openstreetmap.org/lookup'
#
#     if len(osmids) > 50:
#         raise ValueError("Nominatim supports up to 50 osm ids in a single request.")
#
#     params = OrderedDict()
#     params['format'] = "json"
#     params['address_details'] = 1
#     params['osmids'] = osmids
#
#     response_json = ox.nominatim_request(
#                         params = params,
#                         service = ox.NominatimService.LOOKUP)
#
#     return list(map(lambda x: int(x["osm_id"]), filter(lambda x: x['osm_type'] == "way", response_json)))
