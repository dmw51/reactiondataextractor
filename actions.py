from collections import namedtuple, Counter
import copy
from itertools import product, chain

import logging
import numpy as np

from scipy.ndimage import label
from scipy.signal import find_peaks
from skimage.transform import probabilistic_hough_line
from skimage.morphology import skeletonize as skeletonize_skimage
from skimage.morphology import binary_dilation, disk
from skimage.measure import regionprops
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV

from config import get_area
from models.arrows import SolidArrow
from models.exceptions import NotAnArrowException
from models.reaction import ReactionStep,Conditions,Reactant,Product,Intermediate
from models.segments import Rect, Panel, Figure, TextLine
from models.utils import Point, Line
from utils.processing import approximate_line, create_megabox, merge_rect, pixel_ratio, binary_close, binary_floodfill, pad
from utils.processing import binary_tag, get_bounding_box, postprocessing_close_merge, erase_elements, crop, belongs_to_textline, is_boundary_cc
from utils.processing import is_small_textline_character, crop_rect, transform_panel_coordinates_to_expanded_rect, transform_panel_coordinates_to_shrunken_region
from utils.processing import label_and_get_ccs
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
file_handler = logging.FileHandler('actions.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

log.addHandler(file_handler)
log.addHandler(stream_handler)

def segment(bin_fig, arrows):
    """
    Segments the image to return arrows and all remaining connected components
    :param Figure bin_fig: analysed figure with the image in binary form #Arrows are usually hidden for improved closing
    :param iterable arrows: list of arrow objects found in the image
    :return: list of connected components
    """
    bin_fig = copy.deepcopy(bin_fig)
    bbox = bin_fig.get_bounding_box()
    skel_pixel_ratio = skeletonize_area_ratio(bin_fig, bbox)

    log.debug(" The skeletonized pixel ratio is %s" % skel_pixel_ratio)

    # Choose kernel size according to skeletonized pixel ratio
    if 0.03 < skel_pixel_ratio:
        kernel = 2
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    elif 0.025 < skel_pixel_ratio <= 0.03:
        kernel = 4
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    elif 0.02 < skel_pixel_ratio <= 0.025:
        kernel = 6
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    elif 0.015 < skel_pixel_ratio <= 0.02:
        kernel = 10
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    elif 0.01 < skel_pixel_ratio  <=0.015:
        kernel = 15
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    else:
        kernel = 20
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    # Using a binary floodfill to identify panel regions
    fill_img = binary_floodfill(bin_fig)
    tag_img = binary_tag(fill_img)
    panels = get_bounding_box(tag_img)

    # Removing relatively tiny pixel islands that are determined to be noise
    area_threshold = bin_fig.get_bounding_box().area / 200
    width_threshold = bin_fig.get_bounding_box().width / 150
    panels = [panel for panel in panels if panel.area > area_threshold or panel.width > width_threshold]
    return set(panels)

def skeletonize(fig):
    """
    A convenience function operating on Figure objects working similarly to skimage.morphology.skeletonize
    :param fig: analysed figure object
    :return: figure object with a skeletonised image
    """
    fig = copy.deepcopy(fig)
    fig.img = skeletonize_skimage(fig.img)

    return fig

def skeletonize_area_ratio(fig, panel):
    """ Calculates the ratio of skeletonized image pixels to total number of pixels
    :param fig: Input figure
    :param panel: Original panel object
    :return: Float : Ratio of skeletonized pixels to total area (see pixel_ratio)
    """

    skel_fig = skeletonize(fig)
    return pixel_ratio(skel_fig, panel)



def find_solid_arrows(fig, thresholds=None,min_arrow_lengths=None):
    if min_arrow_lengths is None:
        min_arrow_length = int((fig.img.shape[0]+fig.img.shape[1])*0.05)
        min_arrow_lengths = [min_arrow_length, int(min_arrow_length/1.5)]

    if thresholds is None:
        threshold = int((fig.img.shape[0]+fig.img.shape[1])*0.1)
        thresholds = [threshold, int(threshold/1.5)]
    print('thresh :', thresholds)
    print('min length :', min_arrow_lengths)

    # Find arrows in a two-step search
    arrows = find_solid_arrows_main_routine(fig,threshold=thresholds[0],min_arrow_length=min_arrow_lengths[0])

    if not arrows:
        log.info('No arrows have been found the image on the first attempt.')
        arrows = find_solid_arrows_main_routine(fig, threshold=thresholds[1], min_arrow_length=min_arrow_lengths[1])
        if not arrows:
            log.warning('No arrows have been found in the image')

    return arrows


def find_solid_arrows_main_routine(fig,threshold=None, min_arrow_length=None):
    """
    This function takes in a binary figure object and aims to find all solid arrows.
    :param Figure fig: input figure objext
    :param int min_arrow_length: parameter to adjust minimum arrow length
    :return: list of arrow objects
    """
    img = copy.deepcopy(fig.img)
    skeleton = skeletonize(fig).img
    lines = probabilistic_hough_line(skeleton, threshold=threshold, line_length=min_arrow_length)
    #print(lines)
    labelled_img, _ = label(img)
    arrows =[]
    for l in lines:
        points = [Point(row=y, col=x) for x, y in l]
        # Choose one of points to find the label and pixels in the image
        p1 = points[0]
        #print('checking p1:...')
        #print(p1.row, p1. col)
        #print('should be (96, 226)')
        arrow_label = labelled_img[p1.row, p1.col]
        arrow_pixels = np.nonzero(labelled_img == arrow_label)
        arrow_pixels = list(zip(*arrow_pixels))
        try:
            arrows.append(SolidArrow(arrow_pixels, line=approximate_line(*points)))
        except NotAnArrowException:
            log.info('An arrow candidate was discarded ')
    # Filter poor arrow assignments based on aspect ratio
    arrows = [arrow for arrow in arrows if arrow.aspect_ratio >5]
    return arrows




def scan_all_reaction_steps(fig, all_arrows, all_conditions, panels,global_skel_pixel_ratio, stepsize=30):
    """
    Main subroutine for reaction step scanning
    :param Figure fig: Figure object being processed
    :param iterable all_arrows: List of all found arrows
    :param iterable panels: List of all connected components
    :param float global_skel_pixel_ratio: a float describing density of on-pixels
    :param int stepsize: size of a step in the line scanning subroutine
    :return iterable: List of ReactionStep objects containing assigned connected components
    """
    steps =[]
    control_set = set()

    for idx,arrow in enumerate(all_arrows):
        ccs_reacts_prods = find_step_reactants_and_products(fig,all_conditions[idx],arrow,all_arrows,panels)

        panels_dict = assign_to_nearest(panels,ccs_reacts_prods['all_ccs_reacts'],ccs_reacts_prods['all_ccs_prods'])

        #control_set.update(*(value for value in panels_dict.values())) # The unpacking looks ugly

        conditions = Conditions(connected_components=all_conditions[idx])
        reacts = panels_dict['reactants']
        prods = panels_dict['products']
        print(f'panels_dict: {panels_dict}')
        # if global_skel_pixel_ratio > 0.02 : #Original kernel size < 6
        #     reacts = postprocessing_close_merge(fig, reacts)
        #     prods = postprocessing_close_merge(fig, prods)
        #     log.debug('Postprocessing closing and merging finished.')

        reacts=Reactant(connected_components=reacts)
        prods = Product(connected_components=prods)
        steps.append(ReactionStep(arrow,reacts,prods, conditions))
        #print('panels:', panels)
    # if control_set != panels:
    #     log.warning('Some connected components remain unassigned following scan_all_reaction_steps.')
    # else:
    #     log.info('All connected components have been assigned following scan_all_reaction steps.')
    return steps

def find_step_reactants_and_products(fig, step_conditions, step_arrow, all_arrows, panels, stepsize=30):
    """
    :param Figure fig: figure object being processed
    :param iterable conditions: a list of bounding boxes containing conditions
    :param Arrow step_arrow: Arrow object connecting the reactants and products
    :param iterable all_arrow: a list of all arrows found
    :return: a list of all conditions bounding boxes
    """
    log.info('Looking for reactants and products around arrow %s', step_arrow)
    megabox_ccs = copy.deepcopy(step_conditions)
    megabox_ccs.add(step_arrow)
    megabox = create_megabox(megabox_ccs)
    top_left = Point(row=megabox.top, col=megabox.left)
    bottom_left = Point(row=megabox.bottom, col=megabox.left)
    top_right = Point(row=megabox.top, col=megabox.right)
    bottom_right = Point(row=megabox.bottom, col=megabox.right)
    # TODO: Decide how to treat the special case of horizontal arrows
    # where we need to form lines differently
    # print('top left, bottom left:')
    # print(top_left.row, top_left.col)
    # print(bottom_left.row, bottom_left.col)
    left_edge = approximate_line(top_left, bottom_left)
    # print(left_edge.pixels)
    right_edge = approximate_line(top_right, bottom_right)
    arrow_react_midpoint = Point(*np.mean(step_arrow.react_side, axis=0))
    arrow_prod_midpoint = Point(*np.mean(step_arrow.prod_side, axis=0))

    dist_left_edge_react_midpoint = left_edge.distance_from_point(arrow_react_midpoint)
    dist_left_edge_prod_midpoint = left_edge.distance_from_point(arrow_prod_midpoint)
    react_scanning_line = left_edge if dist_left_edge_react_midpoint < dist_left_edge_prod_midpoint else right_edge
    prod_scanning_line = right_edge if react_scanning_line is left_edge else left_edge
    if react_scanning_line is left_edge:
        log.debug('Reactant scanning line is the left edge, product scanning line is the right edge')
    else:
        log.debug('Reactant scanning line is the right edge, product scanning line is the left edge')

    if react_scanning_line is left_edge:
        #Set these to assign direction of expansion for the lines
        react_increment = -1
        prod_increment = 1
    else:
        react_increment = 1
        prod_increment = -1

    raw_reacts = set()
    raw_prods = set()

    # Find reactants
    p1_react = react_scanning_line.pixels[0]
    p2_react = react_scanning_line.pixels[-1]
    p1_react_scancol = p1_react.col
    p2_react_scancol = p2_react.col
    i=0 # for debugging only
    for step in range(fig.img.shape[1]//stepsize):
        p1_react_scancol += stepsize*react_increment
        p2_react_scancol += stepsize*react_increment
        line = approximate_line(Point(row=p1_react.row, col=p1_react_scancol),
                                Point(row=p2_react.row, col=p2_react_scancol))
        if any(arrow.overlaps(line) for arrow in all_arrows):
            log.debug('While scanning reactants, an arrow was encountered - breaking from the loop')

            log.debug('breaking at iteration %s'% i)
            break
        raw_reacts.update([panel for panel in panels if panel.overlaps(line)])
    log.info('Found the following connected components of reactants: %s' % raw_reacts)

    # Find products
    raw_prods = set()
    p1_prod = prod_scanning_line.pixels[0]
    p2_prod = prod_scanning_line.pixels[-1]
    p1_prod_scancol = p1_prod.col
    p2_prod_scancol = p2_prod.col
    #Need to access figure dimensions here
    i=0 # for debugging only
    for step in range(fig.img.shape[1]//stepsize):
        p1_prod_scancol += stepsize*prod_increment
        p2_prod_scancol += stepsize*prod_increment
        line = approximate_line(Point(row=p1_prod.row, col=p1_prod_scancol),
                                Point(row=p2_prod.row, col=p2_prod_scancol))
        if any(arrow.overlaps(line) for arrow in all_arrows):
            log.debug('While scanning products, an arrow was encountered - breaking from the loop')
            log.debug('breaking prod at iteration  %s' % i)
            break
        raw_prods.update([panel for panel in panels if panel.overlaps(line)])
    log.info('Found the following connected components of prods: %s ' % raw_prods)
    return  {'all_ccs_reacts':raw_reacts, 'all_ccs_prods':raw_prods}

def assign_to_nearest(all_ccs, reactants, products, threshold=None):
    """
    This postrocessing function takes in all panels and classified panels to perform a set difference.
    It then assings the remaining panels to the appropriate group based on the closest neighbour, subject to a threshold.
    :param iterable all_ccs: list of all connected components excluding arrow
    :param int threshold: maximum distance from nearest neighbour # Not used at this stage. Is it necessary?
    # :param iterable conditions: List of conditions' panels of a reaction step # not used
    :param iterable reactants: List of reactants' panels of a reaction step
    :param iterable products: List of products' panels of a reaction step
    :return dictionary: The modified groups
    """
    return {'reactants':reactants, 'products':products}
    #
    log.debug('Assigning connected components based on distance')
    #print('assign: conditions set: ', conditions)
    classified_ccs = set((*reactants, *products))
    #print('diagonal lengths: ')
    #print([cc.diagonal_length for cc in classified_ccs])
    threshold = np.mean(([cc.diagonal_length for cc in classified_ccs]))
    unclassified =  all_ccs.difference(classified_ccs)
    for cc in unclassified:
        classified_ccs = sorted(classified_ccs, key=lambda elem: elem.separation(cc))
        nearest = classified_ccs[0]
        groups = [reactants, products]
        for group in groups:
            if nearest in group and nearest.separation(cc) < threshold:
                group.add(cc)
                log.info('assigning %s to group %s based on distance' % (cc, group))


    return {'reactants':reactants, 'products':products}


# def find_small_characters(cropped_figure, ccs,threshold_size=None):
#     # Currently not used
#     if threshold_size is None:
#         threshold_size = np.mean([cc.area for cc in ccs])/2
#
#     small_characters = [cc for cc in ccs if cc.area < threshold_size]
#
#     return set(small_characters)


# def attempt_fit_textline(cropped_figure, main_textlines):
#     img = cropped_figure.img
#
#     main_textlines_top = np.min([textline.top for textline in main_textlines])
#     main_textlines_bottom = np.max([textline.bottom for textline in main_textlines])
#     mean_main_textline_height = np.mean([textline.height for textline in main_textlines])
#     mean_main_textline_center_horizontal = np.mean([textline.center[0] for textline in main_textlines])
#
#     crop_top_region = Rect(left=0, right=img.shape[1],
#                            top=0, bottom=main_textlines_top)
#
#     crop_bottom_region = Rect(left=0, right=img.shape[1],
#                            top=main_textlines_bottom, bottom=img.shape[0])
#
#     top_text_buckets = fit_textline_locally(cropped_figure,crop_top_region)
#     bottom_text_buckets = fit_textline_locally(cropped_figure, crop_bottom_region)
#     if not (top_text_buckets or bottom_text_buckets):
#         return None
#     print(f'top buckets: {top_text_buckets}')
#     print(bool(top_text_buckets))
#     print(f'bottom buckets: {bottom_text_buckets}')
#     print(bool(bottom_text_buckets))
#     text_candidate_buckets = [*top_text_buckets, *bottom_text_buckets]
#     tolerance = 10 # pixels
#     additional_conditions_text_buckets =[]
#     for bucket in text_candidate_buckets:
#         mean_new_textline_center_horizontal = np.mean([elem.center[0] for elem in bucket])
#         mean_new_textline_height = np.mean([elem.height for elem in bucket])
#         cond1 = abs(mean_new_textline_center_horizontal - mean_main_textline_center_horizontal) <= tolerance
#         cond2 = abs(mean_new_textline_height - mean_main_textline_height) <= tolerance
#         if cond1 and cond2:
#             additional_conditions_text_buckets.append(bucket)
#
#     return additional_conditions_text_buckets
#
#
#     # Perform kmeans with the following features
#     #difference height - mean_textline height?
#     #bottom
#     #varying number of clusters between 1 and 4?
#     #restricting cluster centre to around middle of the line (img.shape[1]//2)
#     height_squared_residuals = np.array([(cc.height - mean_textline_height)**2 for cc in unclassified_ccs]).reshape(-1,1)
#     bottom_boundaries_squared = np.array([cc.bottom**2 for cc in unclassified_ccs]).reshape(-1,1)
#     print(f'bottom boundaries: {bottom_boundaries}')
#     data = np.hstack((height_squared_residuals,bottom_boundaries))
#     print(f'data: {data}')


# def fit_textline_locally(main_crop, subcrop_region):
#     cropped_region = crop_rect(main_crop.img, subcrop_region)
#     if cropped_region['rectangle'] != subcrop_region:
#         subcrop_region = cropped_region['rectangle']
#
#     cropped_img = cropped_region['img']
#     ccs = label_and_get_ccs(Figure(cropped_img))
#     plt.imshow(cropped_img)
#     plt.title('attempt_fit_textline')
#     plt.show()
#     print(f'ccs: {ccs}')
#     print(f'len ccs: {len(ccs)}')
#     if len(ccs) < 2:
#         return []
#
#     upper, lower = identify_textlines(ccs, cropped_img)
#
#     new_textlines = [TextLine(left=0, right=subcrop_region.right, top=top_line, bottom=bottom_line)
#                      for top_line, bottom_line in zip(upper, lower)]
#     text_candidate_buckets = assign_characters_to_textlines(
#         cropped_img, new_textlines, ccs, transform_from_crop=True,crop_rect=subcrop_region)
#
#     return text_candidate_buckets








