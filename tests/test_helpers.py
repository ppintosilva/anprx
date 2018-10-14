import anprx
import pytest
import numpy as np
import networkx as nx

def test_is_in():
    test_value_1 = 1
    test_value_2 = 0
    test_value_3 = [0,1]
    test_value_4 = [1,2]
    test_value_5 = [-1,0]

    values_set = {1,2,3,4,5}

    assert anprx.is_in(test_value_1, values_set)
    assert not anprx.is_in(test_value_2, values_set)
    assert anprx.is_in(test_value_3, values_set)
    assert anprx.is_in(test_value_4, values_set)
    assert not anprx.is_in(test_value_5, values_set)



G = nx.MultiDiGraph()
G.add_node(1, label='one')
G.add_node(2, label='fish')
G.add_node(3, label='two')
G.add_node(4, label='fish')

G.add_edge(1,2,color='red', size = "big")
G.add_edge(2,3,color='blue', size = "small")
G.add_edge(1,4,color=['blue', 'dark_blue'], size = "big")


def test_edges_with_at_least_one_property():
    blue = set(anprx.edges_with_properties(
                 G = G,
                 properties = {"color" : {"blue"}},
                 match_by = anprx.PropertiesFilter.at_least_one))

    assert blue == set([(2,3,0), (1,4,0)])

    big_or_red = set(anprx.edges_with_properties(
                 G = G,
                 properties = {"color" : {"red"}, "size" : {"big"}},
                 match_by = anprx.PropertiesFilter.at_least_one))

    assert big_or_red == set([(1,2,0), (1,4,0)])

    small_or_big = set(anprx.edges_with_properties(
                 G = G,
                 properties = {"size" : {"small", "big"}},
                 match_by = anprx.PropertiesFilter.at_least_one))

    assert small_or_big == set([(1,2,0), (2,3,0), (1,4,0)])

    fish_type = set(anprx.edges_with_properties(
                 G = G,
                 properties = {"type" : {"shark", "whale"}},
                 match_by = anprx.PropertiesFilter.at_least_one))

    assert fish_type == set([])


def test_edges_with_all_properties():
    blue = set(anprx.edges_with_properties(
                 G = G,
                 properties = {"color" : {"blue"}},
                 match_by = anprx.PropertiesFilter.all))

    assert blue == set([(2,3,0), (1,4,0)])

    big_and_red = set(anprx.edges_with_properties(
                 G = G,
                 properties = {"color" : {"red"}, "size" : {"big"}},
                 match_by = anprx.PropertiesFilter.all))

    assert big_and_red == set([(1,2,0)])

    small_or_big = set(anprx.edges_with_properties(
                 G = G,
                 properties = {"size" : {"small", "big"}},
                 match_by = anprx.PropertiesFilter.all))

    assert small_or_big == set([(1,2,0), (2,3,0), (1,4,0)])

    big_and_purple = set(anprx.edges_with_properties(
                 G = G,
                 properties = {"color" : {"purple"}, "size" : {"big"}},
                 match_by = anprx.PropertiesFilter.all))

    assert big_and_purple == set([])

    fish_type = set(anprx.edges_with_properties(
                 G = G,
                 properties = {"type" : {"shark", "whale"}},
                 match_by = anprx.PropertiesFilter.all))

    assert fish_type == set([])

def test_flatten():
    l1 = list(range(0,6))
    l2 = list(range(6,10))
    l3 = [l1, l2]

    assert list(anprx.flatten(l1)) == l1
    assert list(anprx.flatten(l2)) == l2
    assert list(anprx.flatten(l1 + [l2])) == l1 + l2
    assert list(anprx.flatten(l3)) == l1 + l2

def test_unit_vector():
    v = np.array([1,2,1,1,0,1])
    v = np.reshape(v, (3,2))
    vunit = anprx.unit_vector(v)

    assert np.shape(vunit) == np.shape(v)
    for i in range(0, len(vunit)):
        np.testing.assert_almost_equal(sum(vunit[i] ** 2), 1.0)

def test_dot2d():
    v1 = np.reshape(np.array([1,2,1,1,0,1]), (3,2))
    v2 = np.reshape(np.array([2,1,3,2,1,0]), (3,2))

    np.testing.assert_array_equal(
        anprx.dot2d(v1,v2, method = "einsum"),
        anprx.dot2d(v1,v2, method = "loop")
    )

    with pytest.raises(ValueError):
        anprx.dot2d(v1,v2, method = "geronimo")


def test_angle_between():
    v1 = np.reshape(np.array([1,2,1,1,0,1,2,2,1,2]), (5,2))
    v2 = np.reshape(np.array([2,1,3,2,1,0,1,1,-2,-1]), (5,2))

    angles = anprx.angle_between(v1,v2)

    np.testing.assert_array_almost_equal(
        np.array([36.8698976,
                  11.3099325,
                  90,
                  0,
                  36.8698976]),
        angles)
