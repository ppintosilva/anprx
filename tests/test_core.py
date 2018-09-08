import anprx
import pytest
import numpy as np

def test_imports():
    import osmnx


latitudes = [54.97092396,54.97080711]
longitudes = [-1.622966153, -1.622935367]

point1 = anprx.Point(lat = 54.97092396,
                     lng = -1.622966153)
point2 = anprx.Point(lat = 54.97080711,
                     lng = -1.622935367)

bbox_small = anprx.BBox(latitudes[0], latitudes[1],
                        longitudes[0], longitudes[1])

bbox_uk = anprx.BBox(59.478568831926395, 49.82380908513249,
                     -10.8544921875, 2.021484375)


def test_points_from_lists():
    assert anprx.points_from_lists(latitudes, longitudes) == [ point1, point2 ]

def test_points_from_tuples():
    points = [(latitudes[0], longitudes[0]), (latitudes[1], longitudes[1])]
    assert anprx.points_from_tuples(points) == [ point1, point2 ]

def test_latitudes_from_points():
    assert anprx.latitudes_from_points([point1,point2]) == latitudes

def test_longitudes_from_points():
    assert anprx.longitudes_from_points([point1,point2]) == longitudes

def test_bbox_area_small():
    bbox = bbox_small

    expected_area_km2 = 2.55e-05
    observed_area_km2_simple = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.constants.SQUARED_KM,
                            method = anprx.constants.METHOD_AREA_SIMPLE)
    observed_area_km2_sins = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.constants.SQUARED_KM,
                            method = anprx.constants.METHOD_AREA_SINS)

    expected_area_m2 = 2.55e-05 * 1e6
    observed_area_m2_simple = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.constants.SQUARED_M,
                            method = anprx.constants.METHOD_AREA_SIMPLE)
    observed_area_m2_sins = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.constants.SQUARED_M,
                            method = anprx.constants.METHOD_AREA_SINS)

    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_simple, decimal = 6)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_simple, decimal = 1)
    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_sins, decimal = 6)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_sins, decimal = 1)


def test_bbox_area_large():
    bbox = bbox_uk

    expected_area_km2 = 888000
    observed_area_km2_simple = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.constants.SQUARED_KM,
                            method = anprx.constants.METHOD_AREA_SIMPLE)
    observed_area_km2_sins = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.constants.SQUARED_KM,
                            method = anprx.constants.METHOD_AREA_SINS)

    expected_area_m2 = 888000 * 1e6
    observed_area_m2_simple = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.constants.SQUARED_M,
                            method = anprx.constants.METHOD_AREA_SIMPLE)
    observed_area_m2_sins = anprx.get_bbox_area(
                            bbox = bbox,
                            unit = anprx.constants.SQUARED_M,
                            method = anprx.constants.METHOD_AREA_SINS)

    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_simple, decimal = -5)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_simple, decimal = -10)
    np.testing.assert_almost_equal(expected_area_km2, observed_area_km2_sins, decimal = -5)
    np.testing.assert_almost_equal(expected_area_m2, observed_area_m2_sins, decimal = -10)
