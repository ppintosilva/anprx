"""Some useful plot methods"""

import osmnx                as ox
from .utils                 import save_fig
from .utils                 import settings

def plot_G(
    G,
    name = None,
    points = None,
    labels = None,
    **plot_kwargs
):
    """
    Wrapper for ox._plot_graph
    """
    file_format     = plot_kwargs.get('file_format', 'png')
    fig_height      = plot_kwargs.get('fig_height', 6)
    fig_width       = plot_kwargs.get('fig_width', None)

    node_size       = plot_kwargs.get('node_size', 15)
    node_alpha      = plot_kwargs.get('node_alpha', 1)
    node_zorder     = plot_kwargs.get('node_zorder', 2)
    node_color      = plot_kwargs.get('node_color', '#66ccff')
    node_edgecolor  = plot_kwargs.get('node_edgecolor', 'k')

    edge_color      = plot_kwargs.get('edge_color', '#999999')
    edge_linewidth  = plot_kwargs.get('edge_linewidth', 1)
    edge_alpha      = plot_kwargs.get('edge_alpha', 1)

    points_color    = plot_kwargs.get('points_color', '#D91A35')
    points_edgecolor= plot_kwargs.get('points_edgecolor', 'k')
    points_marker   = plot_kwargs.get('points_marker', '*')
    points_size     = plot_kwargs.get('points_size', 100)
    points_zorder   = plot_kwargs.get('points_zorder', 20)
    points_label    = plot_kwargs.get('label', 'points')
    labels_color    = plot_kwargs.get('labels_color', 'k')

    bbox            = plot_kwargs.get('bbox', None)
    subdir          = plot_kwargs.get('subdir', None)
    legend          = plot_kwargs.get('legend', False)
    dpi             = plot_kwargs.get('dpi', 320)

    fig, ax = ox.plot_graph(
        G, bbox = bbox, fig_height=fig_height, fig_width=fig_width,
        node_alpha=node_alpha, node_zorder=node_zorder,
        node_size = node_size, node_color=node_color,
        node_edgecolor=node_edgecolor, edge_color=edge_color,
        edge_linewidth = edge_linewidth, edge_alpha = edge_alpha,
        use_geom = True, annotate = False, save = False, show = False
    )

    if points:
        ax.scatter(
            points[0],
            points[1],
            marker = points_marker,
            color = points_color,
            s = points_size,
            zorder = points_zorder,
            label = points_label,
            edgecolors = points_edgecolor
        )
        if labels:
            for s,x,y in zip(labels, points[0], points[1]):
                ax.annotate(s, xy = (x,y),
                            xytext = (x, y + 5), color = labels_color)

    if legend:
        ax.legend()

    if name:
        save_fig(fig, ax, name,
                 subdir = subdir, file_format = file_format, dpi = dpi)

        filename = "{}/{}/{}/{}.{}".format(
                        settings['app_folder'],
                        settings['images_folder_name'],
                        subdir,
                        name, file_format)

    return fig, ax, filename
