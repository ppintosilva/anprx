"""Test module for data wranlging methods."""

from   anprx.preprocessing import wrangle_cameras
from   anprx.preprocessing import network_from_cameras
from   anprx.preprocessing import merge_cameras_network
from   anprx.preprocessing import camera_pairs_from_graph
from   anprx.preprocessing import map_nodes_cameras

import os
import pandas              as     pd

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
    'id'   : ['1', '2', '3', '4', '5', '6'],
    'lat'  : [54.971859, 54.975552, 54.974684, 54.974896, 54.970954, 54.973475],
    'lon'  : [-1.630304, -1.628980, -1.627947, -1.637061, -1.660613,-1.621355],
    'name' : ["NA" , "NB" , "NC", "ND", "NE", "NF"],
    'desc' : ["Westbound A186", "Eastbound Stanhope St A13",
              "Southbound Diana St B1", "Beaconsfield St Southbound A27",
              "Northbound B1305 Condercum Rd", "St James Av A98 Northbound"]
})

def test_pipeline(plot):
    """Test default behavior."""

    cameras = wrangle_cameras(
        cameras             = raw_cameras_testset_1,
        is_test_col         = "name",
        is_commissioned_col = "is_commissioned",
        road_attr_col       = "desc",
        drop_car_park       = True,
        drop_na_direction   = True,
        distance_threshold  = 50.0,
        sort_by             = "id"
    )

    assert len(cameras) == 6

    assert {'1-10', '2', '3', '4', '5', '9'}.issubset(cameras['id'].unique())

    assert cameras.loc[cameras.id == '1-10'].iloc[0]['direction'] == "W"
    assert cameras.loc[cameras.id == '2'].iloc[0]['direction'] == "E-W"
    assert cameras.loc[cameras.id == '3'].iloc[0]['direction'] == "N-S"
    assert cameras.loc[cameras.id == '9'].iloc[0]['direction'] == "E"

    assert cameras.loc[cameras.id == '1-10'].iloc[0]['ref'] == "A186"
    assert cameras.loc[cameras.id == '9'].iloc[0]['ref'] == "A186"
    assert cameras.loc[cameras.id == '2'].iloc[0]['ref'] == "A13"
    assert cameras.loc[cameras.id == '3'].iloc[0]['ref'] == "B1"
    assert cameras.loc[cameras.id == '4'].iloc[0]['ref'] == "A27"
    assert cameras.loc[cameras.id == '5'].iloc[0]['ref'] == "B1305"

    assert "Condercum Rd" in cameras[cameras['id'] == '5']['address'].iloc[0]

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

    # Same address, direction and within distance
    assert nodes.loc[nodes.id == '1'].iloc[0]['camera'] == '1-10'
    assert nodes.loc[nodes.id == '2'].iloc[0]['camera'] == '2'
    assert nodes.loc[nodes.id == '3'].iloc[0]['camera'] == '3'
    assert nodes.loc[nodes.id == '4'].iloc[0]['camera'] == '4'
    # Camera with same address but over distance_threshold
    assert pd.isna(nodes.loc[nodes.id == '5'].iloc[0]['camera'])
    # No camera with the same address
    assert pd.isna(nodes.loc[nodes.id == '6'].iloc[0]['camera'])

    G = network_from_cameras(
        cameras,
        filter_residential = False,
        clean_intersections = True,
        tolerance = 5,
        plot = plot,
        file_format = 'png',
        fig_height = 12,
        fig_width = 12
    )

    G = merge_cameras_network(
        G,
        cameras,
        plot = plot,
        file_format = 'png',
        fig_height = 12,
        fig_width = 12
    )

    pairs = camera_pairs_from_graph(G)

    assert len(pairs) < len(cameras) ** 2
