################################################################################
# Module: plot.py
# Description: Plot functions
# License: Apache v2.0
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import math
import adjustText
import osmnx                as ox
import matplotlib.pyplot    as plt
import matplotlib.colors    as colors
import matplotlib.colorbar  as colorbar

from .utils                 import save_fig
from .core                  import Edge
from .constants             import Units
from .constants             import deg2distance

#
#
#

def plot_camera(
    camera,
    bbox_side = 100,
    camera_color = "#FFFFFF",
    camera_marker = "*",
    camera_markersize = 10,
    annotate_camera = True,
    draw_radius = False,
    #
    fig_height = 6,
    fig_width = None,
    margin = 0.02,
    bgcolor='k',
    node_color='#999999',
    node_edgecolor='none',
    node_zorder=2,
    node_size=50,
    node_alpha = 1,
    edge_color='#555555',
    edge_linewidth=1.5,
    edge_alpha=1,
    #
    probability_cmap = plt.cm.Oranges,
    show_colorbar_label = True,
    draw_colorbar = True,
    #
    nn_color = '#66B3BA',
    nedge_color = '#D0CE7C',
    labels_color = "white",
    annotate_nn_id = False,
    annotate_nn_distance = True,
    adjust_text = True,
    #
    save = False,
    file_format = 'png',
    filename = None,
    dpi = 300):

    """
    Plot the camera on a networkx spatial graph.

    Parameters
    ----------
    bbox_side : int
        half the length of one side of the bbox (a square) in which to plot the camera. This value should usually be kept within small scales  (hundreds of meters), otherwise near nodes and candidate edges become imperceptible.

    camera_color : string
        the color of the point representing the location of the camera

    camera_marker : string
        marker used to represent the camera

    camera_markersize: int
        the size of the marker representing the camera

    annotate_camera : True
        whether to annotate the camera or not using its id

    draw_radius : bool
        whether to draw (kind of) a circle representing the range of the camera

    bgcolor : string
        the background color of the figure and axis - passed to osmnx's plot_graph

    node_color : string
        the color of the nodes - passed to osmnx's plot_graph

    node_edgecolor : string
        the color of the node's marker's border - passed to osmnx's plot_graph

    node_zorder : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot nodes beneath them or 3 to plot nodes atop them - passed to osmnx's plot_graph

    node_size : int
        the size of the nodes - passed to osmnx's plot_graph

    node_alpha : float
        the opacity of the nodes - passed to osmnx's plot_graph

    edge_color : string
        the color of the edges' lines - passed to osmnx's plot_graph

    edge_linewidth : float
        the width of the edges' lines - passed to osmnx's plot_graph

    edge_alpha : float
        the opacity of the edges' lines - passed to osmnx's plot_graph

    probability_cmap : matplotlib colormap
        Colormap used to color candidate edges by probability of observation.

    show_colorbar_label : bool
        whether to set the label of the colorbar or not

    draw_colorbar : bool
        whether to plot a colorbar as a legend for probability_cmap

    nn_color : string
        the color of near nodes - these are not necessarily in range of the camera, but they are part of edges that do

    nedge_color : string
        the color of candidate edges - nearby edges filtered by address or other condition

    labels_color : string
        the color of labels used to annotate nearby nodes

    annotate_nn_id : bool
        whether the text annotating near nodes should include their id

    annotate_nn_distance : bool
        whether the text annotating near nodes should include their distance from the camera

    adjust_text : bool
        whether to optimise the location of the annotations, using adjustText.adjust_text, so that overlaps are avoided. Notice that this incurs considerable computational cost. Turning this feature off will result in much faster plotting.

    save : bool
        whether to save the figure in the app folder's images directory

    file_format : string
        format of the image

    filename : string
        filename of the figure to be saved. The default value is the camera's id.

    dpi : int
        resolution of the image

    Returns
    -------
    fig, ax : tuple
    """
    if filename is None:
        filename = camera.id

    bbox = ox.bbox_from_point(point = camera.point,
                              distance = bbox_side)


    # Set color of near nodes by index
    nodes_colors = [node_color] * len(camera.network.nodes())

    i = 0
    for node in camera.network.nodes(data = False):
        if node in camera.lsystem['nnodes']:
            nodes_colors[i] = nn_color
        i = i + 1

    # Color near edges
    edges_colors = [edge_color] * len(camera.network.edges())

    norm = colors.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=probability_cmap)
    pcolor = { edge : cmap.to_rgba(p)
               for edge, p in camera.p_cedges.items() }

    j = 0
    for u,v,k in camera.network.edges(keys = True, data = False):
        edge = Edge(u,v,k)
        if edge in camera.lsystem['cedges']:
            edges_colors[j] = pcolor[edge]
        j = j + 1

    # Plot it
    fig, axis = \
        ox.plot_graph(
            camera.network,
            bbox = bbox,
            margin = margin,
            bgcolor = bgcolor,
            node_color = nodes_colors,
            node_edgecolor = node_edgecolor,
            node_zorder = node_zorder,
            edge_color = edges_colors,
            node_alpha = node_alpha,
            edge_linewidth = edge_linewidth,
            edge_alpha = edge_alpha,
            node_size = node_size,
            save = False,
            show = False,
            close = False,
            axis_off = True,
            fig_height = fig_height,
            fig_width = fig_width)

    if draw_colorbar:
        axis2 = fig.add_axes([0.3, 0.15, 0.15, 0.02])

        cb = colorbar.ColorbarBase(
                axis2,
                cmap=probability_cmap,
                norm=norm,
                orientation='horizontal')
        cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if show_colorbar_label:
            cb.set_label("Probability of edge", color = labels_color, size = 9)
        cb.ax.xaxis.set_tick_params(pad=0,
                                    color = labels_color,
                                    labelcolor = labels_color,
                                    labelsize = 8)

    # Plot Camera
    camera_point = axis.plot(
            camera.point.lng,
            camera.point.lat,
            marker = camera_marker,
            color = camera_color,
            markersize = camera_markersize)


    if draw_radius:
        radius_circle = \
            plt.Circle((camera.point.lng, camera.point.lat),
                       radius = camera.radius/deg2distance(unit = Units.m),
                       color=camera_color,
                       fill=False)

        axis.add_artist(radius_circle)

    if annotate_camera:
        camera_text = axis.annotate(
                        str(camera.id),
                        xy = (camera.point.lng, camera.point.lat),
                        color = labels_color)

    if annotate_nn_id or annotate_nn_distance:
        # Annotate nearest_neighbors
        texts = []
        for id in camera.lsystem['nnodes']:
            distance_x = camera.lsystem['lnodes'][id][0]
            distance_y = camera.lsystem['lnodes'][id][1]
            distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

            if distance < bbox_side:
                s1 = ""
                s2 = ""
                if annotate_nn_id:
                    s1 = "{}: ".format(id)
                if annotate_nn_distance:
                    s2 = "{:,.1f}m".format(distance)

                text = axis.text(camera.network.node[id]['x'],
                                 camera.network.node[id]['y'],
                                 s = s1 + s2,
                                 color = labels_color)
                texts.append(text)

        if annotate_camera:
            texts.append(camera_text)

        if adjust_text:
            additional_obj = []

            if draw_radius:
                additional_obj.append(radius_circle)
            if annotate_camera:
                additional_obj.append(camera_text)

            adjustText.adjust_text(
                texts,
                x = [ camera.network.node[id]['x'] for id in camera.lsystem['nnodes'] ],
                y = [ camera.network.node[id]['y'] for id in camera.lsystem['nnodes'] ],
                ax = axis,
                add_objects = camera_point + additional_obj,
                force_points = (0.5, 0.6),
                expand_text = (1.2, 1.4),
                expand_points = (1.4, 1.4))

    if save:
        save_fig(fig, axis, filename, file_format, dpi)

    return fig, axis
