"""Methods for street networks represented as graphs."""
# ------------------------------------------------------------------------------

import math
import time
import collections
import numpy                as np
import osmnx                as ox
import logging              as lg
import networkx             as nx
import shapely.geometry     as geometry

from .helpers               import chunks
from .helpers               import is_in
from .helpers               import get_quadrant
from .utils                 import log
import anprx.nominatim      as nominatim

# ------------------------------------------------------------------------------

EARTH_RADIUS_METERS = 6371009
DEG_TO_METERS = 111119
DEG_TO_METERS_SQ = 12347432161
RAD_TO_METERS = 6367000

# ------------------------------------------------------------------------------

def as_undirected(edges):
    """
    Get the undirected representation of a list of edges.

    Parameters
    ---------
    edges : array-like
        array of directed edges (u,v,k)

    Returns
    -------
    list of tuples
    """
    edge_set = frozenset(
                  [ frozenset((edge[0], edge[1]))
                    for edge in edges])

    return [ tuple(edge) for edge in edge_set ]


def add_edge_directions(G):
    """
    Add the direction of each edge as an attribute

    Parameters
    ---------
    G : nx.MultiDiGraph

    Returns
    -------
    G
    """
    G = ox.add_edge_bearings(G)
    for u,v,k,data in G.edges(keys = True, data = True):
        bearing = data['bearing']
        phi = (360 - bearing) + 90
        if phi and not np.isnan(phi):
            data['direction'] = get_quadrant(phi)
        else:
            data['direction'] = None
    return G

def edges_with_any_property(G, properties):
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

def edges_from_osmid(G, osmids):
    """
    Get the network edges that match a given osmid, for several input osmids.

    Parameters
    ---------
    G : nx.MultiDiGraph
        a street network

    osmids : list of int
        osmids of network edge = 1+ OSM ways = road segment

    Returns
    -------
    generator of Edge
        edges that match a osmid, for each input osmid
    """
    properties = {"osmid" : osmids}
    log("Looking for the edges with the osmids: {}".format(set(osmids)))

    for u,v,k in list(edges_with_all_properties(G, properties)):
        yield (u, v, k)

###
###

def edges_by_distance(G, point):
    """
    Calculate the distance of every edge to point and sort them by distance.

    Parameters
    ---------
    G : nx.MultiDiGraph

    point : tuple
        The (lat, lng) or (y, x) point for which we will
        find the nearest edge in the graph

    Returns
    -------
    list
        edges in the graph sorted by distance to point
    """
    # graph to GeoDataFrame
    gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
    graph_edges = gdf[["geometry", "u", "v"]].values.tolist()

    edges_with_distances = [
        (
            graph_edge,
            geometry.Point(tuple(reversed(point))).distance(graph_edge[0])
        )
        for graph_edge in graph_edges
    ]

    edges_with_distances = sorted(edges_with_distances, key=lambda x: x[1])
    return edges_with_distances

###
###

def get_dead_end_nodes(G):
    """
    Get nodes representing dead ends in the street network.

    These are not necessarily nodes with degree of 1, in the undirected representation of the street network,

    Parameters
    ----------
    G : nx.MultiDiGraph
        A street network

    Returns
    -------
    G : nx.MultiDiGraph
        The same network, but without dead end nodes and edges
    """
    if not 'streets_per_node' in G.graph:
        G.graph['streets_per_node'] = ox.count_streets_per_node(G)

    streets_per_node = G.graph['streets_per_node']

    return [node for node, count in streets_per_node.items() if count <= 1]


def remove_dead_end_nodes(G):
    """
    Remove nodes, and corresponding edges,
    representing dead ends in the street network.

    Parameters
    ----------
    G : nx.MultiDiGraph
        A street network (modified in place)

    """
    start_time_local = time.time()

    dead_end_nodes = get_dead_end_nodes(G)
    G.remove_nodes_from(dead_end_nodes)

    log("Removed {} dead ends (nodes) in {:,.3f} seconds"\
            .format(len(dead_end_nodes), time.time() - start_time_local),
        level = lg.INFO)

###
###
###

def add_address_details(G,
                        drop_keys = ['place_id', 'license', 'osm_type',
                                     'osm_id', ' lat', 'lon', 'display_name',
                                     'country', 'country_code', 'state',
                                     'state_district', 'county', 'city'],
                        email = None):
    """
    Lookup and the address details of every edge in the network and
    add them as attributes.

    Depending on the size of the network, this method may incur a large number of requests and time to run. If anprx.settings['cache'] is set to True, as is by default, responses will be cached and subsequent calls to this method, for the same or intersecting networks, should be faster.

    Parameters
    ----------
    G : nx.MultiDiGraph
        A street network

    drop_keys : list
        keys to ignore from the nominatim response containing address details

    email : string
        Valid email address in case you are making a large number of requests

    Returns
    -------
    G : nx.MultiDiGraph
        The same network, but with additional edge attributes
    """
    start_time_local = time.time()

    uG = G.to_undirected(reciprocal = False)

    # Generator of groups of 50 edges
    edge_groups = chunks(l = list(uG.edges(keys = True, data = "osmid")),
                         n = 50)

    for group in edge_groups:
        # For edges with multiple osmids, pick the first osmid
        # Is there a better approach or is this a good enough approximation?
        osmids = [ osmid[0] if isinstance(osmid, collections.abc.Iterable)
                   else osmid for osmid in map(lambda x: x[3], group) ]

        address_details = nominatim.lookup_address(
                                osmids = osmids,
                                entity = 'W',
                                drop_keys = drop_keys,
                                email = email)

        for edge, details  in zip(group, address_details):
            edge_uv = edge[:3]
            if G.has_edge(*edge_uv):
                G[edge_uv[0]][edge_uv[1]][edge_uv[2]].update(details)

            edge_vu = (edge[1], edge[0], edge[2])
            if G.has_edge(*edge_vu):
                G[edge_vu[0]][edge_vu[1]][edge_vu[2]].update(details)


    log("Added address details to {} groups of 50 edges in {:,.3f} seconds"\
        .format(len(list(edge_groups)), time.time() - start_time_local),
    level = lg.INFO)

    return G

###
###
###

def enrich_network(G,
                   clean_dead_ends = True,
                   elevation_api_key = None,
                   drop_keys = ['place_id', 'license', 'osm_type',
                                'osmid', ' lat', 'lon', 'display_name',
                                'country', 'country_code', 'state',
                                'state_district', 'county', 'city'],
                   email = None,
                   postcode_delim = ' '):
    """
    Enrich a street network by adding further attributes to the edges in the network. These can then be used in clustering, compression, graph embeddings, shortest paths, etc.

    Depending on the size of the network, this method may incur a large number of requests and time to run. If anprx.settings['cache'] is set to True, as is by default, responses will be cached and subsequent calls to this method, for the same or intersecting networks, should be faster.

    Parameters
    ----------
    G : nx.MultiDiGraph
        a street network

    clean_dead_ends : bool
        true if dead end nodes should be removed from the graph

    elevation_api_key : string
        Google API key necessary to access the Elevation API. If None, elevation.

    drop_keys: list
        keys to ignore from the nominatim response containing address details

    email : string
        Valid email address in case you are making a large number of requests.

    postcode_delim : string
        postcode delimiter used to split the main postcode into two parts: outer and inner. Use None to skip postcode splitting.

    Returns
    -------
    G :  nx.MultiDiGraph
        The same network, but with additional edge attributes
    """
    start_time = time.time()

    log("Enriching network with {} nodes and {} edges. This may take a while.."\
            .format(len(G), G.number_of_edges()),
        level = lg.INFO)

    if clean_dead_ends:
        remove_dead_end_nodes(G)

    # Add bearings
    G = ox.add_edge_bearings(G)

    # Elevation
    if elevation_api_key:
        start_time_local = time.time()
        # add elevation to each of the  nodes,using the google elevation API
        G = ox.add_node_elevations(G, api_key = elevation_api_key)
        # then calculate edge grades
        G = ox.add_edge_grades(G)
        log("Added node elevations and edge grades in {:,.3f} seconds"\
                .format(time.time() - start_time_local),
            level = lg.INFO)

    # lookup addresses
    G = add_address_details(G, drop_keys, email)

    # Split post code into outward and inward
    # assume that there is a space that can be used for string split
    for (u,v,k,postcode) in G.edges(keys = True, data = 'postcode'):
        if postcode:
            postcode_l = postcode.split(postcode_delim)
            if len(postcode_l) != 2:
                log("Could not split postcode {}".format(postcode),
                    level = lg.WARNING)
            else:
                G[u][v][k]['out_postcode'] = postcode_l[0]
                G[u][v][k]['in_postcode'] = postcode_l[1]

    log("Enriched network in {:,.3f} seconds"\
        .format(time.time() - start_time),
    level = lg.INFO)

    return G

###
###
###
