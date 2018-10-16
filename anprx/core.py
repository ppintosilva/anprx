################################################################################
# Module: core.py
# Description: Core functions
# License: MIT
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/anprx
################################################################################

import math
import time
import adjustText
import numpy as np
import osmnx as ox
import logging as lg
import networkx as nx
import matplotlib.colorbar as colorbar
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from .constants import *
from .helpers import angle_between
from .utils import settings, config, log, save_fig
from .navigation import Point, Edge, get_nodes_in_range, get_edges_in_range, local_coordinate_system, filter_by_address

###
###

class Camera(object):
    """
    Represents a traffic camera located on the roadside, observing the street.
    This may represent any type of camera recording a road segment with a given
    orientation in respect to the true North (bearing). The orientation of the camera
    may be estimated by providing the address of the street it observes, in the case
    of labelled data, or solely based on it's location, in the case of unlabelled data.

    Attributes
    ----------
    point : Point
        location of the camera

    address : str
        address of the street observed by the camera as labelled by a human

    orientation : dict of str : Orientation
        camera orientation
    """
    def __init__(self,
                 network,
                 id,
                 point,
                 address = None,
                 radius = 40,
                 max_angle = 40,
                 nsamples = 100,
                 edges_filter = Filter.address):
        """
        Parameters
        ---------
        point : Point
            location of the camera

        address : str
            address of the street observed by the camera as labelled by a human
        """
        self.network = network
        # @TODO - Check if network contains the camera?

        self.id = id
        self.point = point
        self.address = address
        self.radius = radius
        self.max_angle = max_angle
        self.nsamples = nsamples
        self.edges_filter = edges_filter

        self.gen_local_coord_system()
        self.estimate_orientation()

    def gen_local_coord_system(self):
        start_time = time.time()

        near_nodes, _ = \
            get_nodes_in_range(network = self.network,
                               points = np.array([self.point]),
                               radius = self.radius)

        log("Near nodes: {}"\
                .format(near_nodes),
            level = lg.DEBUG)

        near_edges = get_edges_in_range(self.network, near_nodes)[0]

        log("Near nodes: {}"\
                .format(near_edges),
            level = lg.DEBUG)

        log("Found {} nodes and {} edges within {} meters of camera {}."\
                .format(len(near_nodes),
                        len(near_edges),
                        self.radius,
                        self.point),
            level = lg.INFO)

        # Add nodes that where not initially detected as neighbors, but that are included in near_edges
        all_nodes = { edge[0] for edge in near_edges } | \
                    { edge[1] for edge in near_edges }

        log("All nodes: {}"\
                .format(all_nodes),
            level = lg.DEBUG)

        log("Added {} out of range nodes that are part of nearest edges. Total nodes: {}."\
                .format(len(all_nodes - set(near_nodes[0])),
                        len(all_nodes)),
            level = lg.INFO)

        if self.edges_filter == Filter.address:
            if self.address is None:
                log("Camera {} has no address defined.".format(self.id))
                raise ValueError("Given camera has no address defined")

            candidate_edges = \
                filter_by_address(self.network,
                                  near_edges,
                                  self.address)
        else:
            candidate_edges = near_edges

        log("Candidate edges: {}"\
                .format(candidate_edges),
            level = lg.DEBUG)

        nodes_lvectors, edges_lvectors = \
            local_coordinate_system(self.network,
                                    origin = self.point,
                                    nodes = all_nodes,
                                    edges = candidate_edges)

        log("Generated local coordinate system for camera in {:,.3f} seconds".format(time.time()-start_time),
            level = lg.INFO)

        self.nnodes = list(all_nodes)
        self.nedges = list(near_edges)
        self.cedges = list(candidate_edges)
        self.lnodes = nodes_lvectors
        self.ledges = edges_lvectors


    def estimate_orientation(self):
        """
        Algorithm:

            1. Sample points from each candidate edge, representing the points in the road that are potentially being observed by the camera.

            2. Filter all sampled points:
                - Whose distance to the camera is:
                    - greater than radius
                    - lower than 10 meters
                - Whose angle with the edge vector is greater than 30
                - Are intercepted by a nearby edge representing traffic moving in a different direction

            3. The probability that an edge is the 'correct' edge is equal to the proportion of unfiltered sampled points over the total of sampled points.

            4. Pick the candidate edge with highest probability.
        """
        start_time = time.time()
        p_cedges = dict()

        for candidate in self.cedges:
            start_point = self.lnodes[candidate.u]
            finish_point = self.lnodes[candidate.v]
            line = self.ledges[candidate]
            step = -line/self.nsamples

            points = np.array([
                        start_point + step*i
                        for i in range(0, self.nsamples + 1)
                    ])

            distances = np.linalg.norm(points, ord = 2, axis = 1)

            line_rep = np.repeat(np.reshape(line, (1,2)), self.nsamples + 1, axis = 0)
            angles = angle_between(points, line_rep)

            filter_point = np.vectorize(
                lambda d, a: True if d < self.radius and a < self.max_angle else False)

            unfiltered_points = filter_point(distances, angles)

            p_cedge = sum(unfiltered_points)/len(unfiltered_points)
            p_cedges[candidate] = p_cedge

            log("Probability of candidate {} : {:,.4f}"\
                    .format(candidate, p_cedge),
                level = lg.INFO)

            log("start = {} ".format(start_point) +
                "finish = {} ".format(finish_point) +
                "step = {}\n".format(step) +
                "points = {}\n".format(points) +
                "distances = {}\n".format(distances) +
                "angles = {}".format(angles),
                level = lg.DEBUG)

        self.p_cedges = p_cedges
        self.edge = max(p_cedges.keys(),
                        key=(lambda key: p_cedges[key]))

        log("Estimated the edge observed by camera {}, using {} nsamples for each candidate, in {:,.3f} seconds".format(self.id, self.nsamples, time.time()-start_time),
            level = lg.INFO)


    def plot(self,
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
             filename = "camera",
             dpi = 300
             ):
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
            filename of the figure to be saved

        dpi : int
            resolution of the image

        Returns
        -------
        fig, ax : tuple
        """

        bbox = ox.bbox_from_point(point = self.point,
                                  distance = bbox_side)


        # Set color of near nodes by index
        nodes_colors = [node_color] * len(self.network.nodes())

        i = 0
        for node in self.network.nodes(data = False):
            if node in self.nnodes:
                nodes_colors[i] = nn_color
            i = i + 1

        # Color near edges
        edges_colors = [edge_color] * len(self.network.edges())

        norm = colors.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=probability_cmap)
        pcolor = { edge : cmap.to_rgba(p)
                   for edge, p in self.p_cedges.items() }

        j = 0
        for u,v,k in self.network.edges(keys = True, data = False):
            edge = Edge(u,v,k)
            if edge in self.cedges:
                edges_colors[j] = pcolor[edge]
            j = j + 1

        # Plot it
        fig, axis = \
            ox.plot_graph(
                self.network,
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
                cb.set_label("Probability of camera orientation", color = labels_color, size = 9)
            cb.ax.xaxis.set_tick_params(pad=0,
                                        color = labels_color,
                                        labelcolor = labels_color,
                                        labelsize = 8)

        # Plot Camera
        camera_point = axis.plot(
                self.point.lng,
                self.point.lat,
                marker = camera_marker,
                color = camera_color,
                markersize = camera_markersize)


        if draw_radius:
            radius_circle = \
                plt.Circle((self.point.lng, self.point.lat),
                           radius = self.radius/deg2distance(unit = Units.m),
                           color=camera_color,
                           fill=False)

            axis.add_artist(radius_circle)

        if annotate_camera:
            camera_text = axis.annotate(
                            str(self.id),
                            xy = (self.point.lng, self.point.lat),
                            color = labels_color)

        if annotate_nn_id or annotate_nn_distance:
            # Annotate nearest_neighbors
            texts = []
            for id in self.nnodes:
                distance_x = self.lnodes[id][0]
                distance_y = self.lnodes[id][1]
                distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

                if distance < bbox_side:
                    s1 = ""
                    s2 = ""
                    if annotate_nn_id:
                        s1 = "{}: ".format(id)
                    if annotate_nn_distance:
                        s2 = "{:,.1f}m".format(distance)

                    text = axis.text(self.network.node[id]['x'],
                                     self.network.node[id]['y'],
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
                    x = [ self.network.node[id]['x'] for id in self.nnodes ],
                    y = [ self.network.node[id]['y'] for id in self.nnodes ],
                    ax = axis,
                    add_objects = camera_point + additional_obj,
                    force_points = (0.5, 0.6),
                    expand_text = (1.2, 1.4),
                    expand_points = (1.4, 1.4))

        if save:
            save_fig(fig, axis, filename, file_format, dpi)

        return fig, axis
