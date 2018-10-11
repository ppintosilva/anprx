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
import matplotlib.pyplot as plt

from .constants import *
from .utils import settings, config, log
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
                 radius = 50,
                 filter_edges_by = Filter.address):
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

        self.gen_local_coord_system(filter_edges_by)

    def gen_local_coord_system(self,
                               filter_by = Filter.address):
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

        log("Added {} out of range nodes that are part of nearest edges." +
            "Total nodes: {}."\
                .format(len(set(near_nodes[0]) & all_nodes),
                        len(all_nodes)),
            level = lg.INFO)

        if filter_by == Filter.address:
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

        log("Generated local coordinate system for camera",
            level = lg.INFO)

        self.nnodes = list(all_nodes)
        self.nedges = list(near_edges)
        self.cedges = list(candidate_edges)
        self.lnodes = nodes_lvectors
        self.ledges = edges_lvectors


    def plot(self,
             bbox_side = 100,
             camera_color = "#EB8258",
             camera_markersize = 10,
             annotate_camera = True,
             draw_radius = False,
             #
             bgcolor='k',
             node_color='#999999',
             node_edgecolor='none',
             node_zorder=2,
             node_size=50,
             edge_color='#555555',
             edge_linewidth=1.5,
             edge_alpha=1,
             #
             nn_color = '#009DDC',
             nedge_color = '#D0CE7C',
             labels_color = "white",
             annotate_nn_id = True,
             annotate_nn_distance = True,
             nn_id_arrow_color = 'r',
             adjust_text = True,
             ):
        """
        Plot a networkx spatial graph.

        Parameters
        ----------
        bbox_side : int
            half the length of one side of the bbox (a square) in which to plot the camera. This value should usually be kept within small scales  (hundreds of meters), otherwise near nodes and candidate edges become imperceptible.

        camera_color : string
            the color of the point representing the location of the camera

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

        node_size : int
            the size of the nodes - passed to osmnx's plot_graph

        node_alpha : float
            the opacity of the nodes - passed to osmnx's plot_graph

        node_edgecolor : string
            the color of the node's marker's border - passed to osmnx's plot_graph

        node_zorder : int
            zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot nodes beneath them or 3 to plot nodes atop them - passed to osmnx's plot_graph

        edge_color : string
            the color of the edges' lines - passed to osmnx's plot_graph

        edge_linewidth : float
            the width of the edges' lines - passed to osmnx's plot_graph

        edge_alpha : float
            the opacity of the edges' lines - passed to osmnx's plot_graph

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

        nn_id_arrow_color : string
            the color of the arrow linking the point and annotation of each near node

        adjust_text : bool
            whether to optimise the location of the annotations, using adjustText.adjust_text, so that overlaps are avoided. Notice that this incurs considerable computational cost. Turning this feature off will result in much faster plotting.

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

        j = 0
        for u,v,k in self.network.edges(keys = True, data = False):
            edge = Edge(u,v,k)
            if edge in self.cedges:
                edges_colors[j] = nedge_color
            j = j + 1

        # Plot it
        fig, axis = \
            ox.plot_graph(
                self.network,
                bbox = bbox,
                margin = 0,
                bgcolor = bgcolor,
                node_color = nodes_colors,
                node_edgecolor = node_edgecolor,
                node_zorder = node_zorder,
                edge_color = edges_colors,
                edge_linewidth = edge_linewidth,
                edge_alpha = edge_alpha,
                node_size = node_size,
                show = False,
                close = False)

        # Plot Camera
        camera_point = axis.plot(
                self.point.lng,
                self.point.lat,
                marker = 'o',
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
                            color = camera_color)

        if annotate_nn_id or annotate_nn_distance:
            # Annotate nearest_neighbors
            texts = []
            for id in self.nnodes:
                distance_x = self.lnodes[id][0]
                distance_y = self.lnodes[id][1]
                distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

                s1 = ""
                s2 = ""
                if annotate_nn_id:
                    s1 = "{}: ".format(self.nnodes.index(id))
                if annotate_nn_distance and distance < bbox_side:
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
                    expand_points = (1.4, 1.4),
                    arrowprops=dict(arrowstyle="->", color=nn_id_arrow_color, lw=0.5))

        return fig, axis


    def lplot(self):
        pass
