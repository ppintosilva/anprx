import anprx
import pytest
import numpy as np
import osmnx as ox


latitudes = [54.97092396,54.97080711]
longitudes = [-1.622966153, -1.622935367]

point1 = anprx.Point(lat = latitudes[0],
                     lng = longitudes[0])
point2 = anprx.Point(lat = latitudes[1],
                     lng = longitudes[1])

bbox_small = anprx.BBox(latitudes[0], latitudes[1],
                        longitudes[0], longitudes[1])

bbox_medium = anprx.BBox(*ox.bbox_from_point(
                        point= (54.97351405, -1.62545930208892),
                        distance = 500))

bbox_uk = anprx.BBox(59.478568831926395, 49.82380908513249,
                     -10.8544921875, 2.021484375)


def test_bbox_area_small():
    bbox = bbox_small

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
    bbox = bbox_uk

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
    meanpoint = anprx.get_meanpoint([point1, point2])

    np.testing.assert_almost_equal(54.97086, meanpoint.lat, decimal=5)
    np.testing.assert_almost_equal(-1.622945, meanpoint.lng, decimal=5)

def assert_bbox_almost_equal(bbox1, bbox2, decimal = 5):
    np.testing.assert_almost_equal(bbox1.north, bbox2.north, decimal = decimal)
    np.testing.assert_almost_equal(bbox1.south, bbox2.south, decimal = decimal)
    np.testing.assert_almost_equal(bbox1.west, bbox2.west, decimal = decimal)
    np.testing.assert_almost_equal(bbox1.east, bbox2.east, decimal = decimal)

def test_empty_bbox_from_points():
    with pytest.raises(ValueError):
        anprx.bbox_from_points([])

def test_small_bbox_from_points():
    bbox = bbox_small

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
    bbox = bbox_uk

    nw = anprx.Point(bbox.north, bbox.west)
    sw = anprx.Point(bbox.south, bbox.west)
    ne = anprx.Point(bbox.north, bbox.east)
    se = anprx.Point(bbox.south, bbox.east)

    points = [nw, sw, ne, se]

    with pytest.raises(anprx.GiantBBox):
        anprx.bbox_from_points(points)

def test_bbox_from_points_no_margins():
    bbox = bbox_medium

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
    bbox = bbox_medium

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
