#!/usr/bin/env python
import matplotlib.pyplot as plt 
from numpy import array, arange, linspace, empty, hstack, zeros, repeat, ones
import matplotlib.cm as cm
from biom import table, load_table

def filter_to_other(bt, threshold, method='any', lfm_name='LFM_other'):
    '''Remove features that have low abundance and add lost mass to new feature.

    This function removes features who do not have attain the threshold value
    as evaluated by the method. The following methods are available:
        any - the feature must attain the threshold value in at least one
              sample. 
        all - the feature must attain the threshold value in all samples.
    Any feature mass that is lost is grouped into a new feature named 
    'LFM_other_' and that feature is merged to the filtered table.

    Paramaters
    ----------
    bt : biom table object
        Biom table to filter.
    threshold : float
        Value that the feature must attain according to method to avoid being
        removed by the filtration.
    method : optional, str
        One of 'any' or 'all'. This controls the way by which features are
        filtered based on the threshold. See description for details.
    lfm_name : optional, str
        The name or id to give the feature that contains the lost feature mass
        due to threshold removal.

    Returns
    -------
    filtered_biom_table : biom table object

    Raises
    ------
    ValueError
        If 1 < threshold, as this would filter out all features.
    ValueError
        If method string is not 'any' or 'all'.
    ValueError
        If lfm_name is already contained in the biom table.
    '''
    # error checks
    if threshold > 1.0:
        raise ValueError('All features will be removed unless threshold < 1.0')
    if method not in ['any', 'all']:
        raise ValueError('Unknown method (`%s`), must be `any` or `all`.'
                         % method)
    if lfm_name in bt.ids(axis = 'observation'):
        raise ValueError('lfm_name (`%s`), is already in the biom table. You '\
                         % lfm_name + \
                         ('must set lfm_name to something that is not an ' + 
                          'existing feature id.'))

    if method is 'any':
        def _f(obs_vals, obs_id, obs_md):
            return (obs_vals >= threshold).any()
    elif method is 'all':
        def _f(obs_vals, obs_id, obs_md):
            return (obs_vals >= threshold).all()
    
    # remove features which don't meet criteria
    fbt = bt.filter(_f, axis = 'observation', inplace = False)
    # calculate the lost feature mass
    lost_feature_mass = 1. - fbt.sum(axis = 'sample')
    # create a new table with the lost feature mass as feature 'Other'
    t = table.Table(lost_feature_mass, [lfm_name], fbt.ids(axis = 'sample'))
    # merge tables and have lfm feature first to make plotting easier
    return t.merge(fbt, sample = 'intersection', observation = 'union')

def assemble_axes(font_size, lf, ls, num_features, num_samples, left_pad = .1,
                  right_pad = .5, bottom_pad = .5, top_pad = .5,
                  bar_width = .15, bar_height = 1.0, spines_alpha = 0.0,
                  dpi = 300):
    '''Create axes that won't cut off feature or sample names.

    This function creates axes spines that are the proper size to accomodate the
    number of samples and features that the plot must contain without cutting
    off sample or feature names.

    Paramaters
    ----------
    font_size : numeric
        The size of the font to plot sample and feature names.
    lf : int
        The length in characters of the longest feature name. 
    ls : int
        The length in characters of the longest sample name.
    num_features : int
        The number of features.
    num_samples : int
        The number of samples.
    left_pad : numeric
        Minimum distance (inches) between longest feature name and left side of
        plot. 
    right_pad : numeric
        Distance (inches) between right spine of plot and edge of plot.
    bottom_pad : numeric
        Distance (inches) between bottom spine of plot and edge of plot.
    top_pad : numeric
        Distance (inches) between longest sample name and the top edge of the
        plot.
    bar_width : numeric
        Width (inches) of each bar. 
    bar_height : numeric
        This parameter controls the maximum height for a bar for a given feature
        in inches.
    spines_alpha : numeric
        Controls how transparent spines are (0. = invisible, 1.0 = totally
        opaque).
    dpi : int
        Dots per inch of the resulting figure.

    Returns
    -------
    ax : matplotlib axes object
    fig : matplotlib figure object
    '''
    # calculate the width of the figure
    feature_name_width = lf * font_size / 72.
    total_width_inches = (left_pad + feature_name_width + 
                          bar_width * num_samples + right_pad)
    # set relative positions of left and right spine with relation to edges of 
    # the figure
    left_spine_position_relative = ((left_pad + feature_name_width) / 
                                    total_width_inches)
    left_spine_width_relative = (bar_width * num_samples) / total_width_inches

    # calculate the height of the figure. sample names will be rotated 90
    # degrees.
    sample_name_width = ls * font_size / 72.
    
    # bar height might not be constant between features because we want to 
    # minimize the total height of the plot. all we care about is the total 
    # height, so we just pass the mean.
    total_height_inches = (bottom_pad + bar_height * num_features + 
                           sample_name_width + top_pad)
    # set relative positions of bottom and top spines with relation to edges of
    # the figure
    bottom_spine_position_relative = bottom_pad / total_height_inches
    bottom_spine_height_relative = (num_features * bar_height / 
                                    total_height_inches)
    
    # generate the figure and put axes in requested locations
    fig = plt.figure(figsize = (total_width_inches, total_height_inches),
                     dpi = dpi)
    rect = (left_spine_position_relative, bottom_spine_position_relative, 
            left_spine_width_relative, bottom_spine_height_relative)
    ax = fig.add_axes(rect)
    
    # Set the spines to transparent if desired.
    ax.spines['bottom'].set_alpha(spines_alpha)
    ax.spines['top'].set_alpha(spines_alpha) 
    ax.spines['right'].set_alpha(spines_alpha)
    ax.spines['left'].set_alpha(spines_alpha)

    return ax, fig


def comparable_taxa_plot(fbt, font_size, cm_name = 'Set1', compress = True):
    '''Create a comparable taxa plot.

    This function creates a taxa summary plot with samples as columns and
    features (OTUs) as rows.

    Paramaters
    ----------
    fbt : biom table object
        A filtered biom table that contains only the features to be plotted.
    font_size : numeric
        The size of the font to plot sample and feature names.
    cm_name : optional, matplotlib colormap name
        The colormap to use for feature colors.
    compress : optional, boolean
        Controls whether the height of the graph will be compressed by capping
        actual height of features at 1.0 * max relative abundance of that
        feature. 

    Returns
    -------
    fig : matplotlib figure object
        The figure with requested data plotted.
    '''
    num_features, num_samples = fbt.shape
    longest_feature_name = max(map(len, fbt.ids(axis = 'observation')))
    longest_sample_name = max(map(len, fbt.ids(axis = 'sample')))
    cmap = cm.get_cmap(cm_name)
    
    # x coordinates of samples are known in advance
    x_lb = 0
    x_ub = num_samples
    x_coords = arange(num_samples)
    
    # y lower bound is set but the compress option might change y_ub
    y_lb = 0
    # we determine the height of the bars and the bottoms of the bars based on
    # the height compression strategy the user has selected
    if compress:
        # if we compress the graph, the maximum height of any given feature will
        # be 1.0 * max feature relative abundance. 
        heights = fbt.max(axis = 'observation')
    else:
        # without compression, maximum bar height is 1.0.
        heights = ones(num_features)

    # bottoms are the cumulative sum of the first n-1 entires. 
    bottoms = hstack(([0], heights)).cumsum()[:-1]

    # we now know everything we need to assemply the axes.
    ax, fig = assemble_axes(font_size, longest_feature_name, 
                            longest_sample_name, num_features, num_samples,
                            bar_height = heights.mean())

    # have to plot the ylabels - the feature names - based on information
    # revealed to us from the `bottoms` generator that gets passed. 
    ylabel_coords = []
    for i, f_i in enumerate(fbt.iter_data(axis = 'observation')):
        b = ones(num_samples) * bottoms[i]
        ylabel_coords.append(b[0])
        # remove 0 height entries, otherwise graphed as tiny lines
        inds = f_i.nonzero()[0]
        ax.bar(x_coords[inds], f_i[inds], width=1.0, bottom = b[inds], lw=0., 
               color = cmap(i / float(num_features)), alpha = 1.0)
        #ax.hlines(ylabel_coords[-1], x_lb, x_ub, linewidth=.25, alpha=1.0)

    # final height of feature values in inches. 
    y_ub = ylabel_coords[-1] + f_i.max()

    # turn left and right yticks off, label them with feature names, and set
    # font size. 
    ax.yaxis.set_tick_params(tick1On = False, tick2On = False)
    ax.set_yticks(ylabel_coords)
    ax.set_yticklabels(fbt.ids(axis = 'observation'), size = font_size)

    # turn the x ticks off, force xlables to appear on the top of the plot, 
    # label the middle of the sample, set font size, and rotate.
    ax.xaxis.set_tick_params(labeltop = 'on', labelbottom = 'off', 
                             tick1On = False, tick2On = False)
    ax.set_xticks(x_coords + .5)  # this puts labels in the middle
    ax.set_xticklabels(fbt.ids(axis = 'sample'), rotation = 90,
                       size = font_size)

    # set the limit of the figure to override matplotlibs scaling
    ax.set_xlim(x_lb, x_ub)
    ax.set_ylim(y_lb, y_ub)

    return fig

# def check_sample_names_overlap(font_size, bar_width):
#     '''Return True if font_size will cause samples to overlap visually.

#     Paramaters
#     ----------
#     font_size : numeric
#         Size in points of the font used for plotting sample names.
#     bar_width : numeric
#         Width in inches of each bar in the barplots.

#     Returns
#     -------
#     boolean
#         Whether or not the sample names will overlap visually.
#     '''
#     return bar_width <= font_size / 72.

# def check_feature_names_overlap(font_size, bar_heights):
#     '''Not implemented yet.'''