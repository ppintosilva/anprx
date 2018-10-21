import os
import sys
import math
import anprx
import pytest
import numpy as np
import osmnx as ox
import logging as lg
import networkx as nx

def get_lat_lng():
    latitudes = [54.97092396,54.97080711]
    longitudes = [-1.622966153, -1.622935367]

    return latitudes, longitudes

def get_points():
    latitudes, longitudes = get_lat_lng()

    point1 = anprx.Point(lat = latitudes[0],
                         lng = longitudes[0])
    point2 = anprx.Point(lat = latitudes[1],
                         lng = longitudes[1])

    return (point1, point2)

def get_bbox(size):

    if size == "small":
        return anprx.BBox(54.97092396, 54.97080711,
                         -1.622966153, -1.622935367)

    elif size == "medium":
        return anprx.BBox(*ox.bbox_from_point(
                                point= (54.97351405, -1.62545930208892),
                                distance = 500))

    elif size == "uk":
        return anprx.BBox(59.478568831926395, 49.82380908513249,
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
    observed_area_km2_simple = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.Units.km,
                            method = anprx.BBoxAreaMethod.simple)
    observed_area_km2_sins = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.Units.km,
                            method = anprx.BBoxAreaMethod.sins)

    expected_area_m2 = 2.55e-05 * 1e6
    observed_area_m2_simple = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.Units.m,
                            method = anprx.BBoxAreaMethod.simple)
    observed_area_m2_sins = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.Units.m,
                            method = anprx.BBoxAreaMethod.sins)

    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_simple, decimal = 6)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_simple, decimal = 1)
    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_sins, decimal = 6)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_sins, decimal = 1)


def test_bbox_area_large():
    bbox = get_bbox(size = "uk")

    expected_area_km2 = 888000
    observed_area_km2_simple = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.Units.km,
                            method = anprx.BBoxAreaMethod.simple)
    observed_area_km2_sins = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.Units.km,
                            method = anprx.BBoxAreaMethod.sins)

    expected_area_m2 = 888000 * 1e6
    observed_area_m2_simple = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.Units.m,
                            method = anprx.BBoxAreaMethod.simple)
    observed_area_m2_sins = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.Units.m,
                            method = anprx.BBoxAreaMethod.sins)

    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_simple, decimal = -5)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_simple, decimal = -10)
    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_sins, decimal = -5)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_sins, decimal = -10)

def test_meanpoint():
    point1, point2 = get_points()

    meanpoint = anprx.get_meanpoint([point1, point2])

    np.testing.assert_almost_equal(54.97086, meanpoint.lat, decimal=5)
    np.testing.assert_almost_equal(-1.622945, meanpoint.lng, decimal=5)

def test_empty_bbox_from_points():
    with pytest.raises(ValueError):
        anprx.bbox_from_points([])

def test_small_bbox_from_points():
    point1, point2 = get_points()
    bbox = get_bbox(size = "small")

    nw = anprx.Point(bbox.north, bbox.west)
    sw = anprx.Point(bbox.south, bbox.west)
    ne = anprx.Point(bbox.north, bbox.east)
    se = anprx.Point(bbox.south, bbox.east)

    points = [nw, sw, ne, se]

    bbox = anprx.bbox_from_points(points)
    expected_bbox = anprx.BBox(*ox.bbox_from_point(
                            point= anprx.get_meanpoint([point1, point2]),
                            distance = 100))

    assert_bbox_almost_equal(bbox, expected_bbox)


def test_large_bbox_from_points():
    bbox = get_bbox(size = "uk")

    nw = anprx.Point(bbox.north, bbox.west)
    sw = anprx.Point(bbox.south, bbox.west)
    ne = anprx.Point(bbox.north, bbox.east)
    se = anprx.Point(bbox.south, bbox.east)

    points = [nw, sw, ne, se]

    with pytest.raises(anprx.GiantBBox):
        anprx.bbox_from_points(points)

def test_bbox_from_points_no_margins():
    bbox = get_bbox(size = "medium")

    nw = anprx.Point(bbox.north, bbox.west)
    sw = anprx.Point(bbox.south, bbox.west)
    ne = anprx.Point(bbox.north, bbox.east)
    se = anprx.Point(bbox.south, bbox.east)

    points = [nw, sw, ne, se]

    bbox = anprx.bbox_from_points(points, rel_margins = anprx.RelativeMargins(0,0,0,0))
    expected_bbox = anprx.BBox(*ox.bbox_from_point(
                            point= (54.97351405, -1.62545930208892),
                            distance = 500))

    assert_bbox_almost_equal(bbox, expected_bbox)

def test_bbox_from_points_with_margins():
    bbox = get_bbox(size = "medium")

    nw = anprx.Point(bbox.north, bbox.west)
    sw = anprx.Point(bbox.south, bbox.west)
    ne = anprx.Point(bbox.north, bbox.east)
    se = anprx.Point(bbox.south, bbox.east)

    points = [nw, sw, ne, se]

    bbox = anprx.bbox_from_points(points)
    expected_bbox = anprx.BBox(*ox.bbox_from_point(
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

    all_osmids = list(anprx.flatten(network.edges(data = "osmid")))

    assert not set(expected_osmids).isdisjoint(set(all_osmids))

    edges = list(anprx.edges_from_osmid(network, expected_osmids))

    returned_osmids = set(anprx.flatten(map(lambda edge: network[edge.u][edge.v][edge.k]["osmid"], edges)))

    assert not set(returned_osmids).isdisjoint(set(expected_osmids))


def test_distance_to_edge():
    point1, point2 = get_points()

    network = get_network(distance = 1000)

    edge = anprx.Edge(u = 826286632,
                      v = 29825878,
                      k = 0)

    assert \
        anprx.distance_to_edge(
                network = network,
                edge = edge,
                point = point1,
                method = anprx.EdgeDistanceMethod.farthest_node) \
        < 100

    assert \
        anprx.distance_to_edge(
                network = network,
                edge = edge,
                point = point2,
                method = anprx.EdgeDistanceMethod.mean_of_distances) \
        < 100


def test_nodes_and_edges_in_range():
    point1, point2 = get_points()

    network = get_network(distance = 1000)

    nn_ids, nn_distances = anprx.get_nodes_in_range(network, [point1, point2], 100)

    assert len(nn_ids) == 2
    assert len(nn_distances) == 2

    assert len(nn_ids[0]) > 0
    assert len(nn_ids[1]) > 0
    assert len(nn_distances[0]) == len(nn_ids[0])
    assert len(nn_distances[1]) == len(nn_ids[1])


    edges = anprx.get_edges_in_range(network, nn_ids)

    assert len(edges) == 2
    assert len(edges[0]) >= len(nn_ids[0])
    assert len(edges[1]) >= len(nn_ids[1])


def test_filter_by_address_and_get_local_coordinate_system():
    network = get_network(distance = 1000)
    address = "Pitt Street, Newcastle Upon Tyne, UK"
    point = anprx.Point(lat = 54.974537, lng = -1.625644)

    nn_ids, nn_distances = anprx.get_nodes_in_range(network, [point], 100)
    nn_edges = anprx.get_edges_in_range(network, nn_ids)[0]

    all_nodes = { edge[0] for edge in nn_edges } | \
                { edge[1] for edge in nn_edges }

    assert len(all_nodes) > len(nn_ids[0])

    candidate_edges = anprx.filter_by_address(network, nn_edges, address)

    assert len(candidate_edges) < len(nn_edges)

    candidate_nodes = { edge[0] for edge in candidate_edges } | \
                      { edge[1] for edge in candidate_edges }

    nodes_lvectors, edges_lvectors = \
        anprx.local_coordinate_system(
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
        assert anprx.flow_of_closest_lane(q1,q2,
                                    left_handed = True) == (q1,q2)
        assert anprx.flow_of_closest_lane(q2,q1,
                                    left_handed = True) == (q1,q2)
        assert anprx.flow_of_closest_lane(q1,q2,
                                    left_handed = False) == (q2,q1)
        assert anprx.flow_of_closest_lane(q2,q1,
                                    left_handed = False) == (q2,q1)

def test_direction_of_flow_q2_q3():
    q2_q3 = np.array(np.meshgrid(points_2q, points_3q, indexing = 'xy')).T.reshape(-1,2)

    for q2,q3 in q2_q3:
        assert anprx.flow_of_closest_lane(q2,q3,
                                    left_handed = True) == (q2,q3)
        assert anprx.flow_of_closest_lane(q3,q2,
                                    left_handed = True) == (q2,q3)
        assert anprx.flow_of_closest_lane(q2,q3,
                                    left_handed = False) == (q3,q2)
        assert anprx.flow_of_closest_lane(q3,q2,
                                    left_handed = False) == (q3,q2)

def test_direction_of_flow_q3_q4():
    q3_q4 = np.array(np.meshgrid(points_3q, points_4q, indexing = 'xy')).T.reshape(-1,2)

    for q3,q4 in q3_q4:
        assert anprx.flow_of_closest_lane(q3,q4,
                                    left_handed = True) == (q3,q4)
        assert anprx.flow_of_closest_lane(q4,q3,
                                    left_handed = True) == (q3,q4)
        assert anprx.flow_of_closest_lane(q3,q4,
                                    left_handed = False) == (q4,q3)
        assert anprx.flow_of_closest_lane(q4,q3,
                                    left_handed = False) == (q4,q3)

def test_direction_of_flow_q4_q1():
    q4_q1 = np.array(np.meshgrid(points_4q, points_1q, indexing = 'xy')).T.reshape(-1,2)

    for q4,q1 in q4_q1:
        assert anprx.flow_of_closest_lane(q4,q1,
                                    left_handed = True) == (q4,q1)
        assert anprx.flow_of_closest_lane(q1,q4,
                                    left_handed = True) == (q4,q1)
        assert anprx.flow_of_closest_lane(q4,q1,
                                    left_handed = False) == (q1,q4)
        assert anprx.flow_of_closest_lane(q1,q4,
                                    left_handed = False) == (q1,q4)
