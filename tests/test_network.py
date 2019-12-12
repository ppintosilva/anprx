import os
import sys
import math
import pytest
import numpy as np
import osmnx as ox
import logging as lg
import networkx as nx

import anprx.network as network
import anprx.helpers as helpers


def get_G(distance = 1000, center = (54.97351, -1.62545)):

    G_pickle_filename = "tests/data/test_network_USB_{}.pkl"\
                              .format(distance)

    if os.path.exists(G_pickle_filename):
        G = nx.read_gpickle(path = G_pickle_filename)
    else:
        G = ox.graph_from_point(
            center_point = center,
            distance = distance, #meters
            distance_type='bbox',
            network_type="drive_service")
        nx.write_gpickle(G = G, path = G_pickle_filename)

    return G

def test_edges_from_osmid():
    expected_osmids = \
        [37899441,
         461119586,
         4725926,
         4692270,
         4655478,
         2544439,
         31992849]

    G = get_G(distance = 1000)

    all_osmids = list(helpers.flatten(G.edges(data = "osmid")))

    assert not set(expected_osmids).isdisjoint(set(all_osmids))

    edges = list(network.edges_from_osmid(G, expected_osmids))

    returned_osmids = set(helpers.flatten(map(lambda edge: \
        G[edge[0]][edge[1]][edge[2]]["osmid"], edges)))

    assert not set(returned_osmids).isdisjoint(set(expected_osmids))


def test_get_dead_end_nodes():
    G = get_G(distance = 1000)
    dead_end_nodes = network.get_dead_end_nodes(G)

    assert len(dead_end_nodes) > 0

    network.remove_dead_end_nodes(G)

    for node in dead_end_nodes:
        assert not G.has_node(node)


def test_add_address_details(monkeypatch):
    dummy_address_details = {
        'road' : 'Spring Street',
        'suburb' : 'Arthur\'s Hill',
        'place_rank' : 26,
        'class' : 'highway',
        'type' : 'residential',
        'importance' : '0.1',
        'postcode' : 'NE4 5TB'
    }

    G = get_G(distance = 1000)

    subG = G.subgraph([4519161284, 4519161278])

    monkeypatch.setattr('anprx.nominatim.lookup_address',
                        lambda osmids,entity,drop_keys,email:
                        [dummy_address_details] * len(osmids))

    subG = network.add_address_details(subG)

    for (u,v,k,d) in subG.edges(keys = True, data = True):
        assert all(item in d.items() for item in dummy_address_details.items())


def test_enrich_network(monkeypatch):
    def mock_osmnx_elevation(G, api_key, max_locations_per_batch=350,
    pause_duration=0.02):
        nx.set_node_attributes(G, 100, 'elevation')
        return G

    dummy_address_details = {
        'road' : 'Spring Street',
        'suburb' : 'Arthur\'s Hill',
        'place_rank' : 26,
        'class' : 'highway',
        'type' : 'residential',
        'importance' : '0.1',
        'postcode' : 'NE4 5TB'
    }

    G = get_G(distance = 1000)

    monkeypatch.setattr('anprx.nominatim.lookup_address',
                        lambda osmids,entity,drop_keys,email:
                        [dummy_address_details] * len(osmids))

    monkeypatch.setattr('osmnx.add_node_elevations',
                        mock_osmnx_elevation)


    new_G = network.enrich_network(G, elevation_api_key = "dummy")

    for (u,v,k,d) in G.edges(keys = True, data = True):
        # dummy_address_details are contained in edge data
        assert all(item in d.items() for item in dummy_address_details.items())
        # edge data now contains bearing
        assert 'bearing' in d.keys()
        # edge data now contains grade
        assert d['grade'] == 0
