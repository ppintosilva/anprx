import os
import anprx
import osmnx as ox
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

#-----------#
#-----------#
#-----------#

def test_camera_plot():
    camera = anprx.Camera(
        network = get_network(distance = 1000),
        id = "fake_camera",
        point = anprx.Point(lat = 54.974537, lng = -1.625644),
        address = "Pitt Street, Newcastle Upon Tyne, UK")

    camera.plot(annotate_camera = False,
                draw_radius = True,
                adjust_text = False)
    camera.plot(annotate_nn_id = False,
                annotate_nn_distance = True)
    camera.plot()
