"""Test module for data wranlging methods."""
# ------------------------------------------------------------------------------

from   anprx.cameras import wrangle_cameras, NA_CAMERA
from   anprx.cameras import network_from_cameras
from   anprx.cameras import merge_cameras_network
from   anprx.cameras import camera_pairs_from_graph
from   anprx.cameras import map_nodes_cameras
from   anprx.cameras import wrangle_raw_anpr
from   anprx.cameras import gdfs_from_network
from   anprx.trips   import trip_identification
from   anprx.flows   import discretise_time

import os
import numpy               as     np
import pandas              as     pd
import geopandas           as     gpd
import shapely             as     shp
import pandas.api.types    as ptypes

# ------------------------------------------------------------------------------

"""
Test set 1 - assert:
    - Cameras 1 and 10 are merged (same location, same direction)
    - Cameras 1 and 9 are not merged (same location, different direction)
    - Camera 6 is dropped (is_carpark = 1)
    - Camera 7 is dropped (is_commissioned = 0)
    - Camera 8 is dropped (is_test = 1)
    - Cameras 2 and 3 see in both directions
    - Direction is inferred correctly
    - Address is extracted correctly
    - Road Category is extracted correctly
    - Resulting df has 6 rows
"""
raw_cameras_testset_1 = pd.DataFrame({
    'id'   : ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
    'lat'  : [54.972017, 54.975509, 54.974499, 54.974612, 54.974181,
              54.90, 54.89, 54.88, 54.972017, 54.972017, 54.974612],
    'lon'  : [-1.631206, -1.628498, -1.627997, -1.637108, -1.659476,
              -1.60, -1.61, -1.67, -1.631206, -1.631206, -1.637108],
    'name' : ["CA" , "CB" , "CC", "CD", "CE", "CF", "CG",
              "Test", "CA2", "CA3", "No Direction"],
    'desc' : ["Westbound A186", "East/West Stanhope St A13",
              "North/South Diana St B1", "Beaconsfield St Southbound A27",
              "Northbound B1305 Condercum Rd", "Car park in",
              "Disabled", "Camera Test",
              "Eastbound A186", "Westbound A186", "Directionless"],
    'is_commissioned' : [1,1,1,1,1,1,0,1,1,1,1]
})

raw_nodes_testset = pd.DataFrame({
    'id'   : [1, 2, 3, 4, 5, 6],
    'lat'  : [54.971859, 54.975552, 54.974684, 54.974896, 54.970954, 54.973475],
    'lon'  : [-1.630304, -1.628980, -1.627947, -1.637061, -1.660613,-1.621355],
    'name' : ["NA" , "NB" , "NC", "ND", "NE", "NF"],
    'desc' : ["Westbound A186", "Eastbound Stanhope St A13",
              "Southbound Diana St B1", "Beaconsfield St Southbound A27",
              "Northbound B1305 Condercum Rd", "St James Av A98 Northbound"]
})

#   Test raw ANPR dataset
#
#   vehicle | camera | timestamp | confidence
#   ------------------------------------------
#   AA00AAA |   1    |     0     |    90        (Camera 1 shd change to 1-10)
#   AA00AAA |   1    |     5     |    92        (duplicate)
#   AA00AAA |   1    |     6     |    91        (duplicate x2)
#   AA00AAA |   2    |     90    |    84        (valid step)
#   AA00AAA |   1    |    1e6    |    83        (new trip)
#   AA00AAA |   2    |  1e6 +90  |    98        (valid step)
#   ------------------------------------------
#   AA11AAA |   3    |     0     |    82        (ok)
#   AA11AAA |   10   |     100   |    96        (valid step, camera 10 -> 1-10)
#   AA11AAA |   5    |     101   |    84        (too fast, should be filtered)
#   AA11AAA |   2    |     105   |    88        (too fast x2)
#   AA11AAA |   4    |     200   |    84        (valid step)
#   AA11AAA |   3    |     1e5   |    92        (new trip)
#   ------------------------------------------
#   AA22AAA |   4    |     0     |    90        (ok)
#   AA22AAA |   4    |    1500   |    92        (new trip, same camera)
#   AA22AAA |   3    |    1600   |    84        (valid step)
#   AA22AAA |   3    |    1601   |    83        (duplicate)
#   AA22AAA |   2    |    1700   |    35        (low confidence)
#   ------------------------------------------
#   np.nan  |   2    |     0     |    75        (nan license plate)
#   aaw91291249  |   2    |     0     |    75   (bad regex plate number)
#   aa      |   2    |     0     |    75   (bad regex plate number)

baseline_date = pd.to_datetime('21000101', format='%Y%m%d', errors='coerce')

raw_anpr_testset_v1 = pd.DataFrame({
    'vehicle'    : ['AA00AAA'] * 6,
    'camera'     : ['1', '1', '1', '2', '1', '2'],
    'timestamp'  : [0.0, 5.0, 6.0, 90.0, 1e5/2, 1e5/2 + 90],
    'confidence' : [90 , 92, 91, 84 , 83, 98]
})

raw_anpr_testset_v2 = pd.DataFrame({
    'vehicle'    : ['AA11AAA'] * 6,
    'camera'     : ['3', '10', '5', '2', '4', '3'],
    'timestamp'  : [0.0, 100.0, 101.0, 105.0, 200.0, 1e5],
    'confidence' : [82 , 96, 84, 84, 88, 92]
})

raw_anpr_testset_v3 = pd.DataFrame({
    'vehicle'    : ['AA22AAA'] * 5,
    'camera'     : ['4', '4', '3', '3', '2'],
    'timestamp'  : [0.0, 1500.0, 1600.0, 1601.0, 1700.0],
    'confidence' : [90 , 92, 84 , 83, 35]
})

raw_anpr_testset_v4 = pd.DataFrame({
    'vehicle'    : [np.nan, 'aaw91291249', 'aa'],
    'camera'     : ['2'] * 3,
    'timestamp'  : [0.0] * 3,
    'confidence' : [75.0] * 3
})

raw_anpr_testset = pd.concat([raw_anpr_testset_v1, raw_anpr_testset_v2,
                              raw_anpr_testset_v3, raw_anpr_testset_v4],
                             axis = 0)\
                     .reset_index(drop = True)
#
raw_anpr_testset['timestamp'] = raw_anpr_testset['timestamp']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))

#   Expected Trips - Vehicle 1: 'AA00AAA' (index = 0)
#
#   vehicle | ori | dst | to | td | tt | d_ori | d_dst | trip | step | trip_len
#   ----------------------------------------------------------------------------
#   AA00AAA | NA  | 1-10| NA | 0  | NA | NA    | W     |  1   |  1   |    3
#   AA00AAA | 1-10|  2  | 0  | 90 | 90 | W     | N-S   |  1   |  2   |    3
#   AA00AAA | 2   |  NA | 90 | NA | NA | N-S   | NA    |  1   |  3   |    3
#   ----------------------------------------------------------------------------
#   AA00AAA | NA  | 1-10| NA | 1e6| NA | NA    | W     |  2   |  1   |    3
#   AA00AAA | 1-10|  2  | 1e5/2|1e5/2+90|90| W     | N-S   |  2   |  2   |    3
#   AA00AAA | 2   |  NA |1e5/2+90|NA |NA | N-S   | NA    |  2   |  3   |    3
#

expected_trips_v1 = pd.DataFrame({
    'vehicle'               : [0] * 6,
    'origin'                : [np.nan, '1-10', '2'] * 2,
    'destination'           : ['1-10', '2', np.nan] * 2,
    't_origin'              : [pd.NaT, 0, 90, pd.NaT, 1e5/2, 1e5/2 + 90.0],
    't_destination'         : [0, 90.0, pd.NaT, 1e5/2, 1e5/2 + 90.0, pd.NaT],
    'travel_time'           : [pd.NaT, 90.0, pd.NaT] * 2,
    'trip'                  : np.array([1] * 3 + [2] * 3, dtype=int),
    'trip_step'             : np.array([1,2,3] * 2,       dtype=int),
    'trip_length'           : np.array([3] * 6,           dtype = int),
    'rest_time'             : [pd.NaT] * 3 + [1e5/2-90.0, pd.NaT, pd.NaT]
})

#   Expected Trips - Vehicle 2: 'AA11AAA' (index = 1)
#
#   vehicle | ori | dst | to | td | tt | d_ori | d_dst | trip | step | trip_len
#   ----------------------------------------------------------------------------
#   AA11AAA | NA  |  3 | NA | 0  | NA  | NA    | N-S   |  1   |  1   |    4
#   AA11AAA | 3   |1-10| 0  | 100| 100 | N-S   | W     |  1   |  2   |    4
#   AA11AAA | 1-10|  4 | 100| 200| 100 | W     | S     |  1   |  3   |    4
#   AA11AAA | 4   | NA | 200| NA | NA  | S     | NA    |  1   |  4   |    4
#   ----------------------------------------------------------------------------
#   AA00AAA | NA  | 3  | NA | 1e5| NA | NA    | N-S    |  2   |  1   |    2
#   AA00AAA | 3   | NA |1e5 | NA |NA | N-S   | NA    |  2   |  2   |    2
#

expected_trips_v2 = pd.DataFrame({
    'vehicle'               : [1] * 6,
    'origin'                : [np.nan, '3', '1-10', '4', np.nan, '3'],
    'destination'           : ['3', '1-10', '4', np.nan, '3', np.nan],
    't_origin'              : [pd.NaT, 0, 100.0, 200.0, pd.NaT, 1e5],
    't_destination'         : [0, 100.0, 200.0, pd.NaT, 1e5, pd.NaT],
    'travel_time'           : [pd.NaT, 100.0, 100.0, pd.NaT, pd.NaT, pd.NaT],
    'trip'                  : np.array([1] * 4 + [2] * 2, dtype=int),
    'trip_step'             : np.array([1,2,3,4] + [1,2], dtype=int),
    'trip_length'           : np.array([4] * 4 + [2] * 2, dtype=int),
    'rest_time'             : [pd.NaT] * 4 + [1e5-200.0, pd.NaT]
})

expected_trips = pd.concat([expected_trips_v1, expected_trips_v2], axis = 0)\
                   .reset_index(drop = True)

expected_cameras_new_id = np.arange(6)
expected_cameras_old_id = ['1-10','2','3','4','5','9']

old2new_ids = pd.Series(
    data  = expected_cameras_new_id,
    index = expected_cameras_old_id,
    dtype = int
)

old2new = lambda x: old2new_ids.loc[x] if not pd.isnull(x) else NA_CAMERA

# Correcting camera id
expected_trips['origin'] = \
    expected_trips['origin'].apply(old2new)

expected_trips['destination'] = \
    expected_trips['destination'].apply(old2new)

# Correcting datetime, timedelta dtypes
expected_trips['t_origin'] = expected_trips['t_origin']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))
expected_trips['t_destination'] = expected_trips['t_destination']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))
expected_trips['travel_time'] = expected_trips['travel_time']\
    .apply(lambda x: pd.to_timedelta(x, unit = 's'))
expected_trips['rest_time'] = expected_trips['rest_time']\
    .apply(lambda x: pd.to_timedelta(x, unit = 's'))

#   Expected Discretised Trips - Vehicle 1: 'AA00AAA', freq = 30S
#
#   vehicle | ori | dst | to   | td   | period
#   ----------------------------------------------
#   AA00AAA | NA  | 1-10| NA   | 0    | 0
#   AA00AAA | 1-10|  2  | 0    | 90   | 0
#   AA00AAA | 1-10|  2  | 0    | 90   | 30
#   AA00AAA | 1-10|  2  | 0    | 90   | 60
#   AA00AAA | 2   |  NA | 90   | NA   | 90
#   ----------------------------------------------------------------------------
#   AA00AAA | NA  | 1-10| NA   | 1e5/2  | 1e5/2-20
#   AA00AAA | 1-10|  2  | 1e5/2|1e5/2+90| 1e5/2-20
#   AA00AAA | 1-10|  2  | 1e5/2|1e5/2+90| 1e5/2+10
#   AA00AAA | 1-10|  2  | 1e5/2|1e5/2+90| 1e5/2+ 40
#   AA00AAA | 1-10|  2  | 1e5/2|1e5/2+90| 1e5/2+ 70
#   AA00AAA | 2   |  NA |1e5/2+90| NA   | 1e5/2+ 70
#

expected_dtrips_v1 = pd.DataFrame({
    'vehicle'               : [0] * 11,
    'origin'                : [np.nan, '1-10', '1-10', '1-10', '2',
                               np.nan, '1-10', '1-10', '1-10', '1-10', '2'],
    'destination'           : ['1-10', '2', '2', '2', np.nan,
                               '1-10', '2', '2', '2','2', np.nan],
    't_origin'              : [pd.NaT, 0, 0, 0, 90, pd.NaT, 1e5/2,
                               1e5/2, 1e5/2, 1e5/2, 1e5/2+90.0],
    't_destination'         : [0, 90.0, 90.0, 90.0, pd.NaT, 1e5/2, 1e5/2+90.0,
                               1e5/2+90.0, 1e5/2+90.0, 1e5/2+90.0, pd.NaT],
    'period'                : [0.0, 0.0, 30.0, 60.0, 90.0, 1e5/2-20, 1e5/2-20,
                               1e5/2+10, 1e5/2+40, 1e5/2+70, 1e5/2+70]
})


#   Expected Discretised Trips - Vehicle 2: 'AA11AAA', freq = 30S
#
#   vehicle | ori | dst | to     | td     | period
#   ----------------------------------------------
#   AA11AAA | NA  |  3 | NA      | 0      | 0
#   AA11AAA | 3   |1-10| 0       | 100    | 0
#   AA11AAA | 3   |1-10| 0       | 100    | 30
#   AA11AAA | 3   |1-10| 0       | 100    | 60
#   AA11AAA | 3   |1-10| 0       | 100    | 90
#   AA11AAA | 1-10|  4 | 100     | 200    | 90
#   AA11AAA | 1-10|  4 | 100     | 200    | 120
#   AA11AAA | 1-10|  4 | 100     | 200    | 150
#   AA11AAA | 1-10|  4 | 100     | 200    | 180
#   AA11AAA | 4   | NA | 200     | NA     | 180
#   ----------------------------------------------------------------------------
#   AA00AAA | NA  | 3  | NA      | 1e5    | 1e5-10
#   AA00AAA | 3   | NA |1e5      | NA     | 1e5-10
#

expected_dtrips_v2 = pd.DataFrame({
    'vehicle'               : [1] * 12,
    'origin'                : [np.nan, '3', '3', '3', '3', '1-10',
                               '1-10', '1-10', '1-10', '4', np.nan, '3'],
    'destination'           : ['3', '1-10', '1-10', '1-10', '1-10', '4',
                               '4', '4', '4', np.nan, '3', np.nan],
    't_origin'              : [pd.NaT, 0, 0, 0, 0, 100.0,
                               100.0, 100.0, 100.0, 200.0, pd.NaT, 1e5],
    't_destination'         : [0, 100.0, 100.0, 100.0, 100.0, 200.0,
                               200.0, 200.0, 200.0, pd.NaT, 1e5, pd.NaT],
    'period'                : [0,0,30,60,90,90,120,150,180,180,1e5-10,1e5-10],
})

dtrips_freq = '30S'

expected_dtrips = pd.concat(
    [expected_dtrips_v1, expected_dtrips_v2], axis = 0)\
    .reset_index(drop = True)

# Correcting camera id
expected_dtrips['origin'] = \
    expected_dtrips['origin'].apply(old2new)

expected_dtrips['destination'] = \
    expected_dtrips['destination'].apply(old2new)

# Correcting datetime, timedelta dtypes
expected_dtrips['t_origin'] = expected_dtrips['t_origin']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))
expected_dtrips['t_destination'] = expected_dtrips['t_destination']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))
expected_dtrips['period'] = expected_dtrips['period']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))

# Using global variables to avoid having to compute the same stuff twice

wrangled_cameras = None
raw_G            = None
merged_G         = None
camera_pairs     = None
wrangled_anpr    = None
trips            = None
dtrips           = None

### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------

def get_wrangled_cameras():
    global wrangled_cameras

    if wrangled_cameras is None:
        wrangled_cameras = wrangle_cameras(
            cameras             = raw_cameras_testset_1,
            is_test_col         = "name",
            is_commissioned_col = "is_commissioned",
            road_attr_col       = "desc",
            drop_car_park       = True,
            drop_na_direction   = True,
            distance_threshold  = 50.0,
            sort_by             = "id")

    return wrangled_cameras


def get_raw_network(plot = False):
    global raw_G

    if raw_G is None:
        raw_G = network_from_cameras(
            cameras = get_wrangled_cameras(),
            road_type = "all",
            clean_intersections = True,
            tolerance = 5,
            plot = plot,
            file_format = 'png',
            fig_height = 12,
            fig_width = 12
        )

    return raw_G

def get_merged_network(plot = False):
    global merged_G

    if merged_G is None:
        merged_G = merge_cameras_network(
            G = get_raw_network(plot),
            cameras = get_wrangled_cameras(),
            plot = plot,
            file_format = 'png',
            fig_height = 12,
            fig_width = 12
        )

    return merged_G

def get_camera_pairs():
    global camera_pairs

    if camera_pairs is None:
        G = get_merged_network()
        camera_pairs = camera_pairs_from_graph(G)

    return camera_pairs


def get_wrangled_anpr():
    global wrangled_anpr

    if wrangled_anpr is None:
        wrangled_anpr = wrangle_raw_anpr(
            raw_anpr_testset,
            cameras = get_wrangled_cameras(),
            filter_low_confidence = True,
            confidence_threshold = 70,
            anonymise = False,
            digest_size = 10,
            digest_salt = b"ABC"
        )

    return wrangled_anpr

def get_trips():
    global trips

    if trips is None:
        trips = trip_identification(
            anpr = get_wrangled_anpr(),
            camera_pairs = get_camera_pairs(),
            speed_threshold = 3.0, # km/h : 3 km/h = 1 km/20h
            duplicate_threshold = 60.0,
            maximum_av_speed = 140.0
        )

    return trips

def get_dtrips():
    global dtrips

    if dtrips is None:
        dtrips = discretise_time(
            trips = get_trips(),
            freq = dtrips_freq,
            apply_pthreshold = True,
            sort = True
        )

    return dtrips

### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------

def test_wrangle_cameras():
    observed_cameras = get_wrangled_cameras()

    assert isinstance(observed_cameras, gpd.GeoDataFrame)

    expected_cameras_direction  = ['W','E-W','N-S','S','N','E']
    expected_cameras_ref        = ['A186','A13','B1','A27','B1305','A186']
    expected_cameras_address    = ['A186','Stanhope St A13','Diana St B1',
                                   'Beaconsfield St A27','B1305 Condercum Rd',
                                   'A186']

    expected_cameras = pd.DataFrame({
        'id'        : expected_cameras_new_id,
        'old_id'    : expected_cameras_old_id,
        'direction' : expected_cameras_direction,
        'ref'       : expected_cameras_ref,
        'address'   : expected_cameras_address
    })

    pd.testing.assert_frame_equal(
        observed_cameras[['id','old_id','direction','ref','address']],
        expected_cameras
    )

    assert 'geometry' in observed_cameras.columns.values


def test_wrangle_nodes():
    cameras = get_wrangled_cameras()

    nodes = map_nodes_cameras(
        raw_nodes_testset,
        cameras,
        is_test_col           = "name",
        is_commissioned_col   = False,
        road_attr_col         = "desc",
        drop_car_park         = True,
        drop_na_direction     = True,
        distance_threshold    = 100
    )

    assert isinstance(nodes, gpd.GeoDataFrame)

    # Cameras 5 and 6 don't map
    expected_nodes_camera_map = pd.DataFrame({
        'id' : [1,2,3,4,5,6],
        'camera' : [0,1,2,3,np.nan,np.nan]
    })

    observed_nodes_camera_map = nodes[['id', 'camera']]

    pd.testing.assert_frame_equal(
        expected_nodes_camera_map,
        observed_nodes_camera_map
    )

def test_wrangle_network():
    G = get_raw_network()

    # test that every edge has the geometry attribute
    geoms = [ pd.notnull(geom) \
              for u, v, geom in G.edges(keys=False, data="geometry") ]

    # assert no nulls
    assert all(geoms)

def test_wrangle_network_pairs(plot):
    """Test default behavior."""

    cameras = get_wrangled_cameras()
    pairs = get_camera_pairs()

    assert isinstance(pairs, gpd.GeoDataFrame)

    assert len(pairs) == (len(cameras) + 1) ** 2

    pairs = pairs.set_index(['origin','destination'])

    # old_ids: '1-10', '2'
    assert pairs.loc[0,1]['valid'] == 1
    # old_ids: '1-10', '3'
    assert pairs.loc[0,2]['valid'] == 1
    # old_ids: '4', '5'
    assert pairs.loc[3,4]['valid'] == 0
    # old_ids: '5', '4'
    assert pairs.loc[4,3]['valid'] == 0
    # old_ids: '5', '5'
    assert pairs.loc[4,4]['valid'] == 0

    # assert geometries are of type 'LineString' or 'MultiLineString'
    valid_geometries = pairs.loc[pairs.valid == 1, 'geometry']
    valid_contiguous = pairs.loc[pairs.valid == 1, 'is_contiguous']

    for geom,is_contiguous in zip(valid_geometries, valid_contiguous):
        is_linestring = isinstance(geom, shp.geometry.LineString)
        is_multilinestring = isinstance(geom, shp.geometry.MultiLineString)

        assert is_linestring or is_multilinestring

        if is_linestring:
            assert is_contiguous
        else:
            assert not is_contiguous

    # test that we can write GeoDataFrame to geojson file
    pairs.to_file('tests/data/pairs.geojson', driver = 'GeoJSON')


def test_wrangle_raw_anpr():

    cameras = get_wrangled_cameras()

    wrangled_anpr = get_wrangled_anpr()

    assert len(wrangled_anpr) == 16 # including 3 duplicates, 2 fast obs
    assert ptypes.is_numeric_dtype(wrangled_anpr['camera'])
    assert ptypes.is_numeric_dtype(wrangled_anpr['vehicle'])

    assert np.isin(wrangled_anpr['vehicle'], np.array([0,1,2])).all()


def test_trips():
    cols = expected_trips_v1.columns.values

    trips = get_trips()

    pd.testing.assert_frame_equal(
        trips.loc[trips.vehicle == 0, cols],
        expected_trips.loc[expected_trips.vehicle == 0, cols],
        check_dtype = True)

    pd.testing.assert_frame_equal(
        trips.loc[trips.vehicle == 1, cols],
        expected_trips.loc[expected_trips.vehicle == 1, cols],
        check_dtype = True)

def test_discretise_time():

    dtrips = get_dtrips()

    dtrips = dtrips.loc[dtrips.vehicle.isin([0,1]),
    ['vehicle','origin','destination', 't_origin','t_destination','period']]

    pd.testing.assert_frame_equal(
        dtrips,
        expected_dtrips,
        check_dtype = True)


def test_gdfs_from_network():
    raw_G = get_raw_network()

    nodes_gdf, edges_gdf = gdfs_from_network(raw_G)

    nodes_gdf.to_file('tests/data/test_nodes.geojson', driver = 'GeoJSON')
    edges_gdf.to_file('tests/data/test_edges.geojson', driver = 'GeoJSON')

    merged_G = get_merged_network()

    nodes_gdf, edges_gdf = gdfs_from_network(merged_G)

    nodes_gdf.to_file('tests/data/test_nodes.geojson', driver = 'GeoJSON')
    edges_gdf.to_file('tests/data/test_edges.geojson', driver = 'GeoJSON')
