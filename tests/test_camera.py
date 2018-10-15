import os
import anprx
import numpy as np
import osmnx as ox
import logging as lg
import networkx as nx


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

test_camera = anprx.Camera(
    network = get_network(distance = 1000),
    id = "fake_camera",
    point = anprx.Point(lat = 54.974537, lng = -1.625644),
    address = "Pitt Street, Newcastle Upon Tyne, UK")

#-----------#
#-----------#
#-----------#

def test_camera_p_cedges():
    camera = test_camera

    p_cedges = camera.p_cedges
    p_cedges_values = np.array(list(camera.p_cedges.values()))

    assert (p_cedges_values >= 0  ).all() and \
           (p_cedges_values <= 1.0).all()

    assert len(p_cedges) == len(camera.cedges)


def test_plot():
    camera = test_camera

    camera.plot(annotate_camera = False,
                draw_radius = True,
                adjust_text = False)
    camera.plot(annotate_nn_id = False,
                annotate_nn_distance = True)
    camera.plot()
