################################################################################
# Module: animate.py
# Description: Animation funcions
# License: Apache v2.0
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import os
import math
import textwrap
import adjustText
import numpy                as np
import osmnx                as ox
import matplotlib.pyplot    as plt
import matplotlib.colors    as colors
import matplotlib.colorbar  as colorbar
import matplotlib.animation as animation
from progress.bar           import Bar

from .helpers               import as_undirected
from .core                  import Edge
from .core                  import estimate_camera_edge
from .core                  import from_lvector
from .constants             import Units
from .constants             import deg2distance
from .utils                 import settings
from .utils                 import log


def animate_camera(
    camera,
    bbox_side = 100,
    camera_color = "#FFFFFF",
    camera_marker = "*",
    camera_markersize = 10,
    annotate_camera = True,
    draw_radius = False,
    #
    fig_height = 9,
    fig_width = 9,
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
    save_mp4 = True,
    filename = None,
    dpi = 320,
    #
    time_per_scene = 5000, # ms
    time_per_frame = 250, # ms
    progress = True,
    colorbar_rect = [0.125, 0.20, 0.20, 0.02],
    subtitle_placement = (0.00, 0.00),
    sample_point_size = 4,
    sample_valid_color = "green",
    sample_invalid_color = "red",
    subtitle_fontsize = 10):

    """
    Generate an animation explaining the edge estimation procedure for the camera on a networkx spatial graph.

    Total number of scenes, in the animation, is 6 + number of candidate edges.

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

    save_mp4 : bool
        whether to save the animation in mp4 format, in the app folder's images directory

    filename : string
        filename of the figure to be saved. The default value is the camera's id.

    dpi : int
        resolution of the image

    time_per_scene = 5000 : int
        time per scene in milliseconds

    time_per_frame = 250 : int
        time per frame in milliseconds. If time_per_scene = 250, then each scene has 20 frames. Most scenes just repeat the same frame, except the scenes for candidate edges - which plot a new sampled point per frame.

    progress : bool
        if True then a bar will show the current progress of generating the animation.

    colorbar_rect : list
        rectangle position of the colorbar as used by matplotlib.figure.add_axes

    subtitle_placement : tuple
        (x,y) coordinates, in transformed axis, of where to place the subtitle text used to explain what is going on.

    sample_point_size : int
        marker size of points sampled in candidate edges

    sample_point_valid_color : string
        color of sample points, in candidate edges, that fit the criteria: < camera.radius and < camera.max_angle

    sample_invalid_color : string
        color of sample points, in candidate edges, that don't fit the criteria: < camera.radius and < camera.max_angle

    subtitle_fontsize : int
        fontsize of subtitle text

    Returns
    -------
    anim : FuncAnimation
    """

    # ----------------------------
    # Generate base fig
    # ----------------------------

    bbox = ox.bbox_from_point(point = camera.point,
                              distance = bbox_side)

    fig, axis = ox.plot_graph(
            camera.network,
            bbox = bbox,
            margin = margin,
            bgcolor = bgcolor,
            node_color = node_color,
            node_edgecolor = node_edgecolor,
            node_zorder = node_zorder,
            edge_color = edge_color,
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

    # ----------------------------
    # Compute colors
    # ----------------------------

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

    # ----------------------------
    # Plot camera point
    # Compute texts but disable them
    # ----------------------------
    texts = []

    camera_point = axis.plot(
        camera.point.lng,
        camera.point.lat,
        marker = camera_marker,
        color = bgcolor,
        markersize = camera_markersize)

    camera_text = axis.annotate(
                        str(camera.id),
                        xy = (camera.point.lng, camera.point.lat),
                        color = labels_color)

    texts.append(camera_text)

    # Nearest Neighbors
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

    axis.texts = []

    n_cedges = len(as_undirected(camera.lsystem['cedges']))
    # ----------------------------
    # Frame and time calculations
    # ----------------------------
    frames_per_scene = int(time_per_scene/time_per_frame)

    scene_names = [ 'network_only',
                    'camera',
                    'near_nodes'] + \
                  [ 'near_edges,{}'.format(id) for id in range(n_cedges) ] + \
                  ['ne_pause',
                   'colorbar',
                   'chosen_edge']

    scenes = np.repeat(scene_names, frames_per_scene)

    scene_frame_index = np.stack(
                            np.unravel_index(
                                np.ravel_multi_index([list(range(len(scenes)))], scenes.shape),
                                (len(scene_names), frames_per_scene)),
                            axis = 1)

    # ----------------------------
    # Get sampled points per edge
    # ----------------------------
    _, _, samples = estimate_camera_edge(
            camera.network,
            camera.lsystem,
            nsamples = frames_per_scene,
            radius = camera.radius,
            max_angle = camera.max_angle,
            left_handed_traffic = camera.left_handed_traffic,
            return_samples = True)


    # ----------------------------
    # Get sampled points per edge
    # ----------------------------

    subtitle = \
        axis.text(subtitle_placement[0],
                  subtitle_placement[1],
                  '',
                  color = bgcolor,
                  transform = axis.transAxes,
                  fontsize = subtitle_fontsize,
                  verticalalignment = 'center',
                  bbox = dict(boxstyle = 'round',
                              facecolor = labels_color,
                              alpha=0.5),
                  wrap = False,
                  family = "serif")

    # ----------------------------
    # Animate function
    # ----------------------------
    if progress:
        bar = Bar('Animating', max = len(scenes))

    def update(frame):
        scene = scenes[frame]
        relative_frame = scene_frame_index[frame][1]

        if scene == 'network_only':
            if relative_frame == 0:
                txtstr = """
                         This is a graph modelling the road network centered at the camera,
                         using a bounding box, with length/2 = {} meters. Nodes and edges
                         represent junctions and roads, respectively."""\
                            .format(bbox_side)

        elif scene == 'camera':
            if relative_frame == 0:
                # Small Cheat, because of adjustText - is there a better solution?
                camera_point[0].set_color(camera_color)
                axis.texts += [texts[0]]
                txtstr = """
                         The camera, with id '{}', has a maximum range of {} meters, and a maximum
                         angle of {} degrees (on the horizontal plane to the plate number)"""\
                            .format(camera.id, camera.radius, camera.max_angle)

        elif scene == 'near_nodes':
            if relative_frame == 0:
                axis.collections[1].set_color(nodes_colors)
                axis.texts += texts[1:]
                txtstr = """
                         The goal is to determine which road (graph edge) the camera is
                         pointing at. We start by finding, the nodes which are in range
                         of the camera, and the neighbors (of degree 1) of these."""

        elif scene.startswith('near_edges'):
            edge_id = int(scene.split(',')[1])
            sample = samples[camera.lsystem['cedges'][edge_id]][0][relative_frame]
            isvalid = samples[camera.lsystem['cedges'][edge_id]][1][relative_frame]

            if isvalid:
                color = sample_valid_color
            else:
                color = sample_invalid_color

            point = from_lvector(origin = camera.point,
                                 lvector = sample)

            sample_point = axis.plot(
                point.lng,
                point.lat,
                marker = 'o',
                color = color,
                markersize = sample_point_size)

            txtstr = """
                     The edges between these pairs of nodes considered candidate edges. We
                     sample points from each candidate edge - {} - and calculate its distance
                     and angle to the camera. Points are coloured in {} or {}, if the
                     distance and angle to the camera are below the maximum value or not."""\
                        .format(edge_id, sample_valid_color, sample_invalid_color)

        elif scene == 'colorbar':
            if relative_frame == 0:
                # Remove sampled points
                axis.lines = [axis.lines[0]]

                axis.collections[0].set_color(edges_colors)
                axis2 = fig.add_axes(colorbar_rect)

                cb = colorbar.ColorbarBase(
                        axis2,
                        cmap=probability_cmap,
                        norm=norm,
                        orientation='horizontal')
                cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                cb.set_label("Proportion of valid points", color = labels_color, size = 8)
                cb.ax.xaxis.set_tick_params(pad=0,
                                            color = labels_color,
                                            labelcolor = labels_color,
                                            labelsize = 6)

                txtstr = """
                         We then calculate the proportion of points that fit these criteria,
                         and pick the undirected edge that maximises this value."""\
                            .format(camera.radius, camera.max_angle)

        elif scene == 'chosen_edge':
            if relative_frame == 0:
                axis.collections[0].set_color(edge_color)

                base_x = camera.network.nodes[camera.edge.u]['x']
                base_y = camera.network.nodes[camera.edge.u]['y']

                end_x = camera.network.nodes[camera.edge.v]['x']
                end_y = camera.network.nodes[camera.edge.v]['y']

                color = pcolor[camera.edge]

                axis.annotate('',
                              xytext = (base_x, base_y),
                              xy = (end_x, end_y),
                              arrowprops=dict(arrowstyle="->", color=color),
                              size = 15)

                txtstr = """
                         Finally, we pick the directed edge that represents traffic moving in the
                         nearest of the two lanes (otherwise the images would be more noisy -
                         crossing traffic). This depends on whether the traffic system is left
                         or right-handed. In this case it is: {}."""\
                            .format("left-handed" if camera.left_handed_traffic else "right_handed")

        else:
            return

        if relative_frame == 0:
            txt = textwrap.dedent(txtstr)[1:]
            subtitle.set_text(txt)

        if progress:
            bar.next()
            # print("{:.1f}%".format(frame/len(scenes) * 100))

    # ----------------------------
    # Animate it!
    # ----------------------------
    if progress:
        bar.finish()

    anim = animation.FuncAnimation(fig, update,
                                  blit = False,
                                  frames = len(scenes),
                                  interval = time_per_frame,
                                  repeat = False)

    if save_mp4:
        if filename is None:
            filename = camera.id
        filepath = os.path.join(settings['app_folder'],
                                settings['images_folder_name'],
                                "{}.mp4".format(filename))
        anim.save(filepath,
                  dpi = dpi,
                  savefig_kwargs = {'facecolor' : bgcolor})

    return anim
