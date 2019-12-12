################################################################################
# Module: nominatim.py
# Description: Calls to external APIs
# License: Apache v2.0
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import time
import shapely
import numpy                as np
import pandas               as pd
import osmnx                as ox
import logging              as lg
import geopandas            as gpd
from collections            import OrderedDict

from .utils                 import log
from .helpers               import flatten_dict

###
###

class ExternalRequestError(Exception):
    """
    Raised when a request to an external API fails.

    Attributes:
        request -- request to external API
        API -- name or base url of external API
        reason -- reason why the request failed
    """

    def __init__(self, request, API, reason):
        self.request = request
        self.API = API
        self.message = \
            "Request to {}: {} -- {}."\
                .format(API, reason, request)

class EmptyResponseError(ExternalRequestError):
    """
    Raised when a request to an external API returns an empty response.

    Attributes:
        request -- request to external API
        API -- name or base url of external API
    """

    def __init__(self, request, API):
        super().__init__(request, API, reason = "empty response")

class NoSuchEntitiesError(ExternalRequestError):
    """
    Raised when a request to OpenStreetMap's lookup API does not find the desired Entities.

    Attributes:
        request -- request to OSM
        desired_entity -- desired OSM entity
        found_entities -- found OSM entities
    """

    def __init__(self, request, desired_entity, found_entity):
        self.request = request
        self.API = "https://nominatim.openstreetmap.org/"
        super().__init__(request, self.API,
                         reason = "wanted entity {}, but found {} instead"\
                            .format(desired_entity, found_entity))

def search_address(address,
                   entity = 'way',
                   email = None):
    """
    Find the OpenStreetMap entities that match a given address.

    Parameters
    ----------
    address : string
        Address to search for

    entity : string
        OSM entity of osmids. Valid values are
        'node', 'way' and 'relation'

    email : string
        Valid email address in case you are making a large number of requests.

    Returns
    -------
    list of int
        List of osm ids for ways that match the given address query

    Raises
    ------
    ValueError
        If entity is not in {'node', 'way' and 'relation'}

    EmptyResponseError
        If the response is empty

    NoSuchEntitiesError
        If the desired entity type is not found among the returned entities
    """
    if entity not in {'node', 'way', 'relation'}:
        raise ValueError("Not valid OSM entity. Choose one of 'node', 'way' or 'relation'")

    log("Searching for OSM entities of type '{}' matching the address '{}'"\
            .format(entity, address),
        level = lg.INFO)

    params = OrderedDict()
    params['format'] = "json"
    params['address_details'] = 0

    if email:
        params['email'] = email

    params['q'] = address

    request = '&'.join(['{0}={1}'.format(k, v)
                        for k,v in params.items()])
    log("Request: {}"\
            .format(request),
        level = lg.DEBUG)

    response_json = ox.nominatim_request(
                        params = params,
                        type = 'search')

    if len(response_json) == 0:
        raise EmptyResponseError(
                request = request,
                API = "https://nominatim.openstreetmap.org/search")

    entities = list(filter(lambda x: x['osm_type'] == entity, response_json))

    log("Found entities {}"\
            .format(list(entities)),
        level = lg.DEBUG)

    if len(entities) == 0:

        raise NoSuchEntitiesError(
                request = request,
                desired_entity = entity,
                found_entities = set(list(map(lambda x: x["osm_type"], response_json))))

    osmids = list(map(lambda x: int(x["osm_id"]), entities))

    # What if osmids are invalid? - not likely - let's assume not

    log("Found {} osmids: {}"\
            .format(len(osmids), osmids),
        level = lg.INFO)

    return osmids


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


###
###

default_amenities_by_category = {
    "sustenance"    :
        [
            "bar", "cafe", "fast_food", "food_court", "pub", "restaurant"
        ],

    "education"     :
        [
            "childcare", "college", "driving_school", "kindergarten",
            "library", "language_school", "music_school", "school", "university"
        ],

    "transportation":
        [
            "parking", "taxi", "bus_station", "car_rental", "car_wash",
            "vehicle_inspection"
         ],

    "healthcare"    :
        [
            "clinic", "dentist", "doctors", "hospital", "nursing_home",
            "pharmacy", "social_facility", "veterinary"
        ],
    "entertainment" :
        [
            "arts_centre", "casino", "cinema", "community_centre", "gambling",
            "nightclub", "planetarium", "social_centre", "studio", "theatre"
        ],
    "institutions"  :
        [
            "marketplace", "bank", "bureau_de_change", "courthouse", "police",
            "fire_station", "prison", "townhouse", "post_office",
            "place_of_worship"
        ]
}


def get_amenities(polygon,
                  amenities_by_category = default_amenities_by_category):
    """
    Get all amenity POIs, classified by category, within a polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        spatial polygon to run the search in

    amenities_by_category : dict
        list of OpenStreetMap amenity values per user-defined category

    Returns
    -------
    amenities : GeoDataFrame
        A spatial dataframe of amenities
    """

    amenities = []
    # return pois from osmnx, subset columns and assign new category column
    for category, am_values in amenities_by_category.items():
        poi_gdf = ox.create_poi_gdf(polygon, am_values)
        if 'name' in poi_gdf.columns:
            poi_gdf = poi_gdf[['geometry', 'amenity', 'name']]
        else:
            poi_gdf = poi_gdf[['geometry', 'amenity']].assign(name = np.nan)

        poi_gdf = poi_gdf.assign(category = category)
        amenities.append(poi_gdf)

    amenities_gpd = pd.concat(amenities)

    return(amenities_gpd)


def get_tynewear_polygon():
    """
    Get the spatial polygon for the county of Tyne and Wear.

    Returns
    -------
    polygon : shapely.geometry.Polygon
    """
    data = ox.osm_polygon_download("Tyne and Wear county")

    keep_cols = ['place_id','osm_type','osm_id','class',
                 'display_name','type','importance']

    df = pd.DataFrame(data)[keep_cols]

    geometry = shapely.geometry.Polygon(data[0]['geojson']['coordinates'][0])

    gdf = gpd.GeoDataFrame(df.assign(geometry = geometry))

    return gdf
