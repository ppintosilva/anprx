"""Test module for data wranlging methods."""

from   anprx.preprocessing import wrangle_cameras
from   anprx.preprocessing import network_from_cameras

import os
import pandas              as     pd

"""
Test set 1 - assert:
    - Cameras 1 and 10 are merged (same location)
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
    'id'   : ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'lat'  : [54.95, 54.94, 54.93, 54.92, 54.91,
              54.90, 54.89, 54.88, 54.95, 54.95],
    'lon'  : [-1.70, -1.71, -1.72, -1.73, -1.74,
              -1.75, -1.76, -1.77, -1.70, -1.70],
    'name' : ["CA" , "CB" , "CC", "CD", "CE", "CF", "CG", "Test1", "CA2", "CA3"],
    'desc' : ["Westbound A13", "East/West A13",
              "North/South B1", "Southbound A27",
              "Northbound A3 Milton Street", "Car park in",
              "Disabled", "Camera Test",
              "Eastbound A13", "Westbound A13"],
    'is_commissioned' : [1,1,1,1,1,1,0,1,1,1]
})

def test_pipeline():
    """Test default behavior."""
    cameras = wrangle_cameras(
        cameras = raw_cameras_testset_1,
        infer_direction_col      = "desc",
        drop_car_park            = "desc",
        drop_is_test             = "name",
        drop_is_not_commissioned = True,
        extract_address          = "desc",
        extract_road_category    = "desc",
        project_coords           = True,
        sort_by                  = "id"
    )

    assert {'1-10', '2', '3', '4', '5', '9'}.issubset(cameras['id'].unique())

    assert cameras[cameras['id'] == '1-10']['direction'].iloc[0] == "W"
    assert cameras[cameras['id'] == '2']['direction'].iloc[0] == "E-W"
    assert cameras[cameras['id'] == '3']['direction'].iloc[0] == "N-S"
    assert cameras[cameras['id'] == '9']['direction'].iloc[0] == "E"

    assert cameras[cameras['id'] == '1-10']['road_category'].iloc[0] == "A"
    assert cameras[cameras['id'] == '3']['road_category'].iloc[0] == "B"
    assert cameras[cameras['id'] == '5']['road_category'].iloc[0] == "A"

    assert "Milton Street" in cameras[cameras['id'] == '5']['address'].iloc[0]

    assert len(cameras) == 6

    network = network_from_cameras(
        cameras,
        filter_residential = True,
        clean_intersections = False,
        tolerance = 30,
        make_plots = True,
        file_format = 'svg'
    )
