import anprx
import pytest
import numpy as np
import osmnx as ox
import networkx as nx

latitudes = [54.97092396,54.97080711]
longitudes = [-1.622966153, -1.622935367]

point1 = anprx.Point(lat = latitudes[0],
                     lng = longitudes[0])
point2 = anprx.Point(lat = latitudes[1],
                     lng = longitudes[1])

def test_points_from_lists():
    assert anprx.points_from_lists(latitudes, longitudes) == [ point1, point2 ]

def test_points_from_tuples():
    points = [(latitudes[0], longitudes[0]), (latitudes[1], longitudes[1])]
    assert anprx.points_from_tuples(points) == [ point1, point2 ]

def test_latitudes_from_points():
    assert anprx.latitudes_from_points([point1,point2]) == latitudes

def test_longitudes_from_points():
    assert anprx.longitudes_from_points([point1,point2]) == longitudes


G = nx.Graph()
G.add_node(1, label='one')
G.add_node(2, label='fish')
G.add_node(3, label='two')
G.add_node(4, label='fish')

G.add_edge(1,2,color='red', size = "big")
G.add_edge(2,3,color='blue', size = "small")
G.add_edge(1,4,color='blue', size = "big")


def test_edges_with_at_least_one_property():
    blue = set(anprx.edges_with_properties(
                 network = G,
                 properties = {"color" : {"blue"}},
                 match_by = anprx.PropertiesFilter.at_least_one))

    assert blue == set([(2,3), (1,4)])

    big_or_red = set(anprx.edges_with_properties(
                 network = G,
                 properties = {"color" : {"red"}, "size" : {"big"}},
                 match_by = anprx.PropertiesFilter.at_least_one))

    assert big_or_red == set([(1,2), (1,4)])

    small_or_big = set(anprx.edges_with_properties(
                 network = G,
                 properties = {"size" : {"small", "big"}},
                 match_by = anprx.PropertiesFilter.at_least_one))

    assert small_or_big == set([(1,2), (2,3), (1,4)])

    fish_type = set(anprx.edges_with_properties(
                 network = G,
                 properties = {"type" : {"shark", "whale"}},
                 match_by = anprx.PropertiesFilter.at_least_one))

    assert fish_type == set([])


def test_edges_with_all_properties():
    blue = set(anprx.edges_with_properties(
                 network = G,
                 properties = {"color" : {"blue"}},
                 match_by = anprx.PropertiesFilter.all))

    assert blue == set([(2,3), (1,4)])

    big_and_red = set(anprx.edges_with_properties(
                 network = G,
                 properties = {"color" : {"red"}, "size" : {"big"}},
                 match_by = anprx.PropertiesFilter.all))

    assert big_and_red == set([(1,2)])

    small_or_big = set(anprx.edges_with_properties(
                 network = G,
                 properties = {"size" : {"small", "big"}},
                 match_by = anprx.PropertiesFilter.all))

    assert small_or_big == set([(1,2), (2,3), (1,4)])

    big_and_purple = set(anprx.edges_with_properties(
                 network = G,
                 properties = {"color" : {"purple"}, "size" : {"big"}},
                 match_by = anprx.PropertiesFilter.all))

    assert big_and_purple == set([])

    fish_type = set(anprx.edges_with_properties(
                 network = G,
                 properties = {"type" : {"shark", "whale"}},
                 match_by = anprx.PropertiesFilter.all))

    assert fish_type == set([])
