import os
import sys
import math
import pytest
import numpy as np
import osmnx as ox
import logging as lg
import networkx as nx

import anprx.core as core
import anprx.helpers as helpers
import anprx.exceptions as exceptions
from anprx.constants import Units

###
###

def get_lat_lng():
    latitudes = [54.97092396,54.97080711]
    longitudes = [-1.622966153, -1.622935367]

    return latitudes, longitudes

def get_points():
    latitudes, longitudes = get_lat_lng()

    point1 = core.Point(lat = latitudes[0],
                         lng = longitudes[0])
    point2 = core.Point(lat = latitudes[1],
                         lng = longitudes[1])

    return (point1, point2)

def get_bbox(size):

    if size == "small":
        return core.BBox(54.97092396, 54.97080711,
                         -1.622966153, -1.622935367)

    elif size == "medium":
        return core.BBox(*ox.bbox_from_point(
                                point= (54.97351405, -1.62545930208892),
                                distance = 500))

    elif size == "uk":
        return core.BBox(59.478568831926395, 49.82380908513249,
                          -10.8544921875, 2.021484375)

    else:
        raise ValueError("No such bbox size")

def assert_bbox_almost_equal(bbox1, bbox2, decimal = 5):
    np.testing.assert_almost_equal(bbox1.north, bbox2.north, decimal = decimal)
    np.testing.assert_almost_equal(bbox1.south, bbox2.south, decimal = decimal)
    np.testing.assert_almost_equal(bbox1.west, bbox2.west, decimal = decimal)
    np.testing.assert_almost_equal(bbox1.east, bbox2.east, decimal = decimal)

def get_network(distance = 1000, center = (54.97351, -1.62545)):

    network_pickle_filename = "tests/data/test_network_USB_{}.pkl".format(distance)

    if os.path.exists(network_pickle_filename):
        network = nx.read_gpickle(path = network_pickle_filename)
    else:
        network = ox.graph_from_point(
            center_point = center,
            distance = distance, #meters
            distance_type='bbox',
            network_type="drive_service")
        nx.write_gpickle(G = network, path = network_pickle_filename)

    return network

def test_bbox_area_small():
    bbox = get_bbox(size = "small")

    expected_area_km2 = 2.55e-05
    observed_area_km2_simple = core.get_bbox_area(
                            bbox = bbox,
                            unit = Units.km,
                            method = "simple")
    observed_area_km2_sins = core.get_bbox_area(
                            bbox = bbox,
                            unit = Units.km,
                            method = "sins")

    expected_area_m2 = 2.55e-05 * 1e6
    observed_area_m2_simple = core.get_bbox_area(
                            bbox = bbox,
                            unit = Units.m,
                            method = "simple")
    observed_area_m2_sins = core.get_bbox_area(
                            bbox = bbox,
                            unit = Units.m,
                            method = "sins")

    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_simple, decimal = 6)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_simple, decimal = 1)
    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_sins, decimal = 6)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_sins, decimal = 1)


def test_bbox_area_large():
    bbox = get_bbox(size = "uk")

    expected_area_km2 = 888000
    observed_area_km2_simple = core.get_bbox_area(
                            bbox = bbox,
                            unit = Units.km,
                            method = "simple")
    observed_area_km2_sins = core.get_bbox_area(
                            bbox = bbox,
                            unit = Units.km,
                            method = "sins")

    expected_area_m2 = 888000 * 1e6
    observed_area_m2_simple = core.get_bbox_area(
                            bbox = bbox,
                            unit = Units.m,
                            method = "simple")
    observed_area_m2_sins = core.get_bbox_area(
                            bbox = bbox,
                            unit = Units.m,
                            method = "sins")

    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_simple, decimal = -5)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_simple, decimal = -10)
    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_sins, decimal = -5)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_sins, decimal = -10)

def test_meanpoint():
    point1, point2 = get_points()

    meanpoint = core.get_meanpoint([point1, point2])

    np.testing.assert_almost_equal(54.97086, meanpoint.lat, decimal=5)
    np.testing.assert_almost_equal(-1.622945, meanpoint.lng, decimal=5)

def test_empty_bbox_from_points():
    with pytest.raises(ValueError):
        core.bbox_from_points([])

def test_small_bbox_from_points():
    point1, point2 = get_points()
    bbox = get_bbox(size = "small")

    nw = core.Point(bbox.north, bbox.west)
    sw = core.Point(bbox.south, bbox.west)
    ne = core.Point(bbox.north, bbox.east)
    se = core.Point(bbox.south, bbox.east)

    points = [nw, sw, ne, se]

    bbox = core.bbox_from_points(points)
    expected_bbox = core.BBox(*ox.bbox_from_point(
                            point= core.get_meanpoint([point1, point2]),
                            distance = 100))

    assert_bbox_almost_equal(bbox, expected_bbox)


def test_large_bbox_from_points():
    bbox = get_bbox(size = "uk")

    nw = core.Point(bbox.north, bbox.west)
    sw = core.Point(bbox.south, bbox.west)
    ne = core.Point(bbox.north, bbox.east)
    se = core.Point(bbox.south, bbox.east)

    points = [nw, sw, ne, se]

    with pytest.raises(exceptions.BBoxAreaSafetyError):
        core.bbox_from_points(points)

def test_bbox_from_points_no_margins():
    bbox = get_bbox(size = "medium")

    nw = core.Point(bbox.north, bbox.west)
    sw = core.Point(bbox.south, bbox.west)
    ne = core.Point(bbox.north, bbox.east)
    se = core.Point(bbox.south, bbox.east)

    points = [nw, sw, ne, se]

    bbox = core.bbox_from_points(points, rel_margins = core.RelativeMargins(0,0,0,0))
    expected_bbox = core.BBox(*ox.bbox_from_point(
                            point= (54.97351405, -1.62545930208892),
                            distance = 500))

    assert_bbox_almost_equal(bbox, expected_bbox)

def test_bbox_from_points_with_margins():
    bbox = get_bbox(size = "medium")

    nw = core.Point(bbox.north, bbox.west)
    sw = core.Point(bbox.south, bbox.west)
    ne = core.Point(bbox.north, bbox.east)
    se = core.Point(bbox.south, bbox.east)

    points = [nw, sw, ne, se]

    bbox = core.bbox_from_points(points)
    expected_bbox = core.BBox(*ox.bbox_from_point(
                            point= (54.97351405, -1.62545930208892),
                            distance = 500))

    assert_bbox_almost_equal(bbox, expected_bbox, decimal = 3)


def test_edges_from_osmid():
    expected_osmids = \
        [37899441,
         461119586,
         4725926,
         4692270,
         4655478,
         2544439,
         31992849]

    network = get_network(distance = 1000)

    all_osmids = list(helpers.flatten(network.edges(data = "osmid")))

    assert not set(expected_osmids).isdisjoint(set(all_osmids))

    edges = list(core.edges_from_osmid(network, expected_osmids))

    returned_osmids = set(helpers.flatten(map(lambda edge: network[edge.u][edge.v][edge.k]["osmid"], edges)))

    assert not set(returned_osmids).isdisjoint(set(expected_osmids))


def test_distance_to_edge():
    point1, point2 = get_points()

    network = get_network(distance = 1000)

    edge = core.Edge(u = 826286632,
                      v = 29825878,
                      k = 0)

    assert \
        core.distance_to_edge(
                network = network,
                edge = edge,
                point = point1,
                method = core.EdgeDistanceMethod.farthest_node) \
        < 100

    assert \
        core.distance_to_edge(
                network = network,
                edge = edge,
                point = point2,
                method = core.EdgeDistanceMethod.mean_of_distances) \
        < 100


def test_lvector():
    origin, actual_point = get_points()

    lvector = core.as_lvector(origin, actual_point)

    desired_point = core.from_lvector(origin, lvector)

    np.testing.assert_almost_equal(
        actual_point,
        desired_point,
        decimal = 7)


def test_nodes_and_edges_in_range():
    point1, point2 = get_points()

    network = get_network(distance = 1000)

    nn_ids, nn_distances = core.get_nodes_in_range(network, [point1, point2], 100)

    assert len(nn_ids) == 2
    assert len(nn_distances) == 2

    assert len(nn_ids[0]) > 0
    assert len(nn_ids[1]) > 0
    assert len(nn_distances[0]) == len(nn_ids[0])
    assert len(nn_distances[1]) == len(nn_ids[1])


    edges = core.get_edges_in_range(network, nn_ids)

    assert len(edges) == 2
    assert len(edges[0]) >= len(nn_ids[0])
    assert len(edges[1]) >= len(nn_ids[1])


def test_filter_by_address_and_get_local_coordinate_system():
    network = get_network(distance = 1000)
    address = "Pitt Street, Newcastle Upon Tyne, UK"
    point = core.Point(lat = 54.974537, lng = -1.625644)

    nn_ids, nn_distances = core.get_nodes_in_range(network, [point], 100)
    nn_edges = core.get_edges_in_range(network, nn_ids)[0]

    all_nodes = { edge[0] for edge in nn_edges } | \
                { edge[1] for edge in nn_edges }

    assert len(all_nodes) > len(nn_ids[0])

    candidate_edges = core.filter_by_address(network, nn_edges, address)

    assert len(candidate_edges) < len(nn_edges)

    candidate_nodes = { edge[0] for edge in candidate_edges } | \
                      { edge[1] for edge in candidate_edges }

    nodes_lvectors, edges_lvectors = \
        core.local_coordinate_system(
            network = network,
            origin = point,
            nodes = candidate_nodes,
            edges = candidate_edges)

    assert len(nodes_lvectors) == len(candidate_nodes)
    assert len(edges_lvectors) == len(candidate_edges)

    for id in candidate_nodes:
        ox_distance = ox.great_circle_vec(
            lat1 = network.node[id]['y'],
            lng1 = network.node[id]['x'],
            lat2 = point.lat,
            lng2 = point.lng)

        lvector = nodes_lvectors[id]
        lvector_distance = math.sqrt(lvector[0] ** 2 + lvector[1] ** 2)

        np.testing.assert_almost_equal(
            ox_distance,
            lvector_distance,
            decimal = 6)

def test_gen_lsystem_recursive():
    network = get_network(distance = 1000)

    neighborless_point = core.Point(lat=54.959224, lng=-1.663313)

    with pytest.raises(exceptions.ZeroNeighborsError):
        lsystem = core.gen_lsystem(
                    network,
                    origin = neighborless_point,
                    radius = 40)


def test_estimate_camera_edge():
    network = get_network(distance = 1000)
    point = core.Point(lat = 54.974537, lng = -1.625644)

    lsystem = core.gen_lsystem(network, point, 40)

    assert 'nnodes' in lsystem
    assert 'nedges' in lsystem
    assert 'cedges' in lsystem
    assert 'lnodes' in lsystem
    assert 'ledges' in lsystem

    camera_edge, p_cedges, samples = \
        core.estimate_camera_edge(network,
                                  lsystem,
                                  nsamples = 100,
                                  return_samples = True)

    assert camera_edge is not None
    assert p_cedges is not None
    assert samples is not None

    assert set(p_cedges.keys()) == set(lsystem['cedges'])
    assert set(samples.keys()) == set(lsystem['cedges'])
    for element in samples.values():
        assert len(element) == 2
        assert len(element[0]) == 100 + 1
        assert len(element[1]) == 100 + 1


##
##
##

points_1q = np.array([(2,2), (9,1), (1,9), (0,1), (3,0)],
                     dtype = [('x', 'i8'), ('y', 'i8')])
points_2q = np.array([(-2,2), (-9,1), (-1,9)],
                     dtype = [('x', 'i8'), ('y', 'i8')])
points_3q = np.array([(-2,-2), (-9,-1), (-1,-9), (0,-1), (-3,0)],
                     dtype = [('x', 'i8'), ('y', 'i8')])
points_4q = np.array([(2,-2), (9,-1), (1,-9)],
                     dtype = [('x', 'i8'), ('y', 'i8')])

def test_direction_of_flow_q1_q2():
    q1_q2 = np.array(np.meshgrid(points_1q, points_2q, indexing = 'xy')).T.reshape(-1,2)

    for q1,q2 in q1_q2:
        assert core.flow_of_closest_lane(q1,q2,
                                    left_handed = True) == (q1,q2)
        assert core.flow_of_closest_lane(q2,q1,
                                    left_handed = True) == (q1,q2)
        assert core.flow_of_closest_lane(q1,q2,
                                    left_handed = False) == (q2,q1)
        assert core.flow_of_closest_lane(q2,q1,
                                    left_handed = False) == (q2,q1)

def test_direction_of_flow_q2_q3():
    q2_q3 = np.array(np.meshgrid(points_2q, points_3q, indexing = 'xy')).T.reshape(-1,2)

    for q2,q3 in q2_q3:
        assert core.flow_of_closest_lane(q2,q3,
                                    left_handed = True) == (q2,q3)
        assert core.flow_of_closest_lane(q3,q2,
                                    left_handed = True) == (q2,q3)
        assert core.flow_of_closest_lane(q2,q3,
                                    left_handed = False) == (q3,q2)
        assert core.flow_of_closest_lane(q3,q2,
                                    left_handed = False) == (q3,q2)

def test_direction_of_flow_q3_q4():
    q3_q4 = np.array(np.meshgrid(points_3q, points_4q, indexing = 'xy')).T.reshape(-1,2)

    for q3,q4 in q3_q4:
        assert core.flow_of_closest_lane(q3,q4,
                                    left_handed = True) == (q3,q4)
        assert core.flow_of_closest_lane(q4,q3,
                                    left_handed = True) == (q3,q4)
        assert core.flow_of_closest_lane(q3,q4,
                                    left_handed = False) == (q4,q3)
        assert core.flow_of_closest_lane(q4,q3,
                                    left_handed = False) == (q4,q3)

def test_direction_of_flow_q4_q1():
    q4_q1 = np.array(np.meshgrid(points_4q, points_1q, indexing = 'xy')).T.reshape(-1,2)

    for q4,q1 in q4_q1:
        assert core.flow_of_closest_lane(q4,q1,
                                    left_handed = True) == (q4,q1)
        assert core.flow_of_closest_lane(q1,q4,
                                    left_handed = True) == (q4,q1)
        assert core.flow_of_closest_lane(q4,q1,
                                    left_handed = False) == (q1,q4)
        assert core.flow_of_closest_lane(q1,q4,
                                    left_handed = False) == (q1,q4)

def test_get_dead_end_nodes():
    network = get_network(distance = 1000)
    dead_end_nodes = core.get_dead_end_nodes(network)

    assert len(dead_end_nodes) > 0

    core.remove_dead_end_nodes(network)

    for node in dead_end_nodes:
        assert not network.has_node(node)


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

    network = get_network(distance = 1000)

    subnetwork = network.subgraph([4519161284, 4519161278])

    monkeypatch.setattr('anprx.nominatim.lookup_address',
                        lambda osmids,entity,drop_keys,email:
                        [dummy_address_details] * len(osmids))

    subnetwork = core.add_address_details(subnetwork)

    for (u,v,k,d) in subnetwork.edges(keys = True, data = True):
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

    network = get_network(distance = 1000)

    monkeypatch.setattr('anprx.nominatim.lookup_address',
                        lambda osmids,entity,drop_keys,email:
                        [dummy_address_details] * len(osmids))

    monkeypatch.setattr('osmnx.add_node_elevations',
                        mock_osmnx_elevation)


    new_network = core.enrich_network(network, elevation_api_key = "dummy")

    for (u,v,k,d) in network.edges(keys = True, data = True):
        # dummy_address_details are contained in edge data
        assert all(item in d.items() for item in dummy_address_details.items())
        # edge data now contains bearing
        assert 'bearing' in d.keys()
        # edge data now contains grade
        assert d['grade'] == 0
