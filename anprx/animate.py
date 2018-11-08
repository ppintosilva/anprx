################################################################################
# Module: animate.py
# Description: Animation funcions
# License: Apache v2.0
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import os
import math
import time
import textwrap
import adjustText
import numpy                    as np
import osmnx                    as ox
import logging                  as lg
import matplotlib.pyplot        as plt
import matplotlib.colors        as colors
import matplotlib.colorbar      as colorbar
import matplotlib.animation     as animation

from progress.bar               import Bar

from .helpers                   import as_undirected
from .core                      import Edge
from .core                      import estimate_camera_edge
from .core                      import from_lvector
from .constants                 import Units
from .constants                 import deg2distance
from .utils                     import settings
from .utils                     import log


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
    labels_fontsize = 8,
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
    save_as = 'mp4',
    filename = None,
    dpi = 320,
    #
    time_per_scene = 5000, # ms
    time_per_frame = 250, # ms
    progress = True,
    colorbar_rect = [0.125, 0.20, 0.20, 0.02],
    colorbar_ticks_fontsize = 6,
    show_subtitle = True,
    subtitle_placement = (0.00, 0.05),
    subtitle_fontsize = 12,
    sample_point_size = 4,
    sample_valid_color = "green",
    sample_invalid_color = "red"):

    """
    Generate an animation explaining the edge estimation procedure for the camera on a networkx spatial graph. The generated animation is not

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

    save_as : string
        format in which to save the animation in the app folder's images directory. Choose 'mp4' to save the animation in mp4 format, using ffmpeg, 'gif' to save the animation in gif format, using imagemagick, or any other value to skip saving the animation.

    filename : string
        filename of the figure to be saved. The default value is the camera's id.

    dpi : int
        resolution of the image, if saving the animation in 'mp4' format.

    time_per_scene : int
        time per scene in milliseconds

    time_per_frame : int
        time per frame in milliseconds. If time_per_scene = 250, then each scene has 20 frames. Most scenes just repeat the same frame, except the scenes for candidate edges - which plot a new sampled point per frame.

    progress : bool
        if True then a bar will show the current progress of generating the animation.

    colorbar_rect : list
        rectangle position of the colorbar as used by matplotlib.figure.add_axes

    labels_fontsize : int
        fontsize of generic text labels (nodes, camera, colorbar)

    colorbar_ticks_fontsize : int
        fontsize of colorbar ticks text

    show_subtitle : bool
        if True show a text box explaining each scene

    subtitle_placement : tuple
        (x,y) coordinates, in transformed axis, of where to place the subtitle text

    subtitle_fontsize : int
        fontsize of subtitle text

    sample_point_size : int
        marker size of points sampled in candidate edges

    sample_point_valid_color : string
        color of sample points, in candidate edges, that fit the criteria: < camera.radius and < camera.max_angle

    sample_invalid_color : string
        color of sample points, in candidate edges, that don't fit the criteria: < camera.radius and < camera.max_angle

    Returns
    -------
    anim : FuncAnimation
    """

    start_time = time.time()

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
                        color = labels_color,
                        fontsize = labels_fontsize)

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
                             color = labels_color,
                             fontsize = labels_fontsize)
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

    # Not elegant at all -> think of better one in the future
    uedges = [ frozenset((cedge.u, cedge.v)) for cedge in camera.lsystem['cedges'] ]
    unique_uedges = list(set(uedges))

    # ----------------------------
    # Frame and time calculations
    # ----------------------------
    frames_per_scene = int(time_per_scene/time_per_frame)

    scene_names = [ 'network_only',
                    'camera',
                    'near_nodes'] + \
                  [ 'near_edges,{}'.format(id) for id in range(len(unique_uedges)) ] + \
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
    if show_subtitle:
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
                                  alpha = 0.5),
                      wrap = False,
                      family = "serif")

    log("Attempting to animate camera '{}' with {} undirected candidate edges: {} scenes, {} frames"\
            .format(camera.id, len(unique_uedges),
                    len(scene_names), len(scenes)),
        level = lg.INFO)

    if progress:
        bar = Bar('Animating as {}'.format(save_as), max = len(scenes) + 1)

    # ----------------------------
    # Animate function
    # ----------------------------
    def update(frame):
        scene = scenes[frame]
        relative_frame = scene_frame_index[frame][1]

        if progress:
            bar.next()

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
                         The camera, with id '{}', has a maximum range of {} meters, and
                         a maximum angle of {} degrees (on the horizontal plane to the plate
                         number)"""\
                            .format(camera.id, camera.radius, camera.max_angle)

        elif scene == 'near_nodes':
            if relative_frame == 0:
                axis.collections[1].set_color(nodes_colors)
                axis.texts += texts[1:]
                txtstr = """
                         The goal is to determine which road (graph edge) the camera is
                         pointing at. We start by finding, the nodes which are in range of
                         the camera, and the neighbors (of degree 1) of these."""

        elif scene.startswith('near_edges'):
            # Not elegant at all -> think of better one in the future
            unique_uedge_id = int(scene.split(',')[1])
            unique_uedge = unique_uedges[unique_uedge_id]
            cedge_id = uedges.index(unique_uedge)

            sample = samples[camera.lsystem['cedges'][cedge_id]][0][relative_frame]
            isvalid = samples[camera.lsystem['cedges'][cedge_id]][1][relative_frame]

            if isvalid:
                color = sample_valid_color
            else:
                color = sample_invalid_color

            point = from_lvector(origin = camera.point,
                                 lvector = sample)

            log("Edge {}, id = {}. sample = {}, isvalid = {}, point = {}"\
                    .format(unique_uedge_id, relative_frame, sample, isvalid, point),
                level = lg.DEBUG)

            sample_point = axis.plot(
                point.lng,
                point.lat,
                marker = 'o',
                color = color,
                markersize = sample_point_size)

            txtstr = """
                     The edges between these pairs of nodes considered candidate edges.
                     We sample points from each candidate edge - {} - and calculate its
                     distance and angle to the camera. Points are coloured in {} or {}, if
                     the distance and angle to the camera are below the maximum value or not."""\
                        .format(unique_uedge_id,
                                sample_valid_color,
                                sample_invalid_color)

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
                cb.set_label("Proportion of valid points",
                             color = labels_color,
                             size = labels_fontsize)
                cb.ax.xaxis.set_tick_params(pad=0,
                                            color = labels_color,
                                            labelcolor = labels_color,
                                            labelsize = colorbar_ticks_fontsize)

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
                              arrowprops=dict(arrowstyle="->",
                                              color=color,
                                              linewidth = edge_linewidth),
                              size = 15)

                txtstr = """
                         Finally, we pick the directed edge that represents traffic moving
                         in the nearest of the two lanes (avoiding crossing traffic in the
                         camera's frame). This depends on whether the traffic system is left
                         or right-handed. In this case it is: {}."""\
                            .format("left-handed" if camera.left_handed_traffic else "right_handed")

        else:
            return

        if relative_frame == 0 and show_subtitle:
            txt = textwrap.dedent(txtstr)[1:]
            subtitle.set_text(txt)

    # ----------------------------
    # Animate it!
    # ----------------------------
    anim = animation.FuncAnimation(fig, update,
                                  blit = False,
                                  frames = len(scenes),
                                  interval = time_per_frame,
                                  repeat = False)

    if filename is None:
        filename = camera.id

    savefig_kwargs = {
        'facecolor' : bgcolor,
        'frameon' : False,
        'bbox_inches' : 'tight',
        'pad_inches' : 0 }

    if save_as == 'mp4':
        filepath = os.path.join(settings['app_folder'],
                                settings['images_folder_name'],
                                "{}.mp4".format(filename))
        anim.save(filepath,
                  dpi = dpi,
                  savefig_kwargs = savefig_kwargs)

        if progress:
            bar.finish()

    elif save_as == 'gif':
        filepath = os.path.join(settings['app_folder'],
                                settings['images_folder_name'],
                                "{}.gif".format(filename))
        anim.save(filepath,
                  writer = 'imagemagick',
                  fps = 1000/time_per_frame,
                  savefig_kwargs = savefig_kwargs)

        if progress:
            bar.finish()

    log("Animated camera in {:,.3f} seconds {}"\
            .format(time.time() - start_time,
                    "and saved it to file {}".format(filepath) if save_as in {'mp4', 'gif'} else "but did not save it to file"),
        level = lg.INFO)

    return anim
