from collections import namedtuple
import copy
from itertools import product, chain

import logging
import numpy as np

from scipy.ndimage import label
from skimage.transform import probabilistic_hough_line
from skimage.morphology import skeletonize as skeletonize_skimage
from skimage.measure import regionprops

from config import get_area
from models.arrows import SolidArrow
from models.exceptions import NotAnArrowException
from models.reaction import ReactionStep,Conditions,Reactant,Product,Intermediate
from models.segments import Panel
from models.utils import Point, Line
from utils.processing import approximate_line, create_megabox, merge_rect, pixel_ratio, binary_close, binary_floodfill
from utils.processing import binary_tag, get_bounding_box

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

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
    fill_img = binary_floodfill(closed_fig)
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



def find_reaction_conditions(fig, arrow, panels, global_skel_pixel_ratio,stepsize=10):
    """
    Given a reaction arrow_bbox and all other bboxes, this function looks for reaction information around it
    If dense_graph is true, the initial arrow's bbox is scaled down to avoid false positives
    Currently, the function only works for horizontal arrows - extended by implementing Bresenham's line algorithm
    :param Figure fig: Analysed figure
    :param Arrow arrow: Arrow object
    :param iterable panels: List of Panel object
    :param float global_skel_pixel_ratio: value describing density of on-pixels in a graph
    :param int stepsize: integer value decribing size of step between two line scanning operations
    """

    p1, p2 = copy.deepcopy(arrow.line[0]), copy.deepcopy((arrow.line[-1]))
    if global_skel_pixel_ratio > 0.15:
        p1.col *= 1.2
        p2.col *= 0.8

    overlapped = set()

    for direction in range(2):
        i = 0
        increment = (-1) ** direction
        p1_scanrow, p2_scanrow = copy.copy(p1.row), copy.copy(p2.row)
        for offset in range(fig.img.shape[0]//stepsize):
            if i >= 20: #Take no more than 20 steps
                break
            #print('p1: ',p1_scanrow,p1.col)
            #print('p2: ', p2_scanrow,p2.col)
            p1_scanrow += increment*stepsize
            p2_scanrow += increment*stepsize
            line = approximate_line(Point(row=p1_scanrow, col=p1.col), Point(row=p2_scanrow, col=p2.col))
            overlapped.update([panel for panel in panels if panel.overlaps(line)])
            i += 1
    conditions = create_megabox(overlapped)
    return {conditions} #Return as a set to allow handling along with product and reactant sets


def scan_all_reaction_steps(fig, all_arrows, panels,global_skel_pixel_ratio, stepsize=30):
    """

    :param Figure fig: Figure object being processed
    :param iterable all_arrows: List of all found arrows
    :param iterable panels: List of all connected components
    :param float global_skel_pixel_ratio: a float describing density of on-pixels
    :param int stepsize: size of a step in the line scanning subroutine
    :return iterable: List of ReactionStep objects containing assigned connected components
    """
    steps =[]
    control_set = set()
    all_conditions = []
    for arrow in all_arrows:
        conditions = find_reaction_conditions(fig, arrow, panels, global_skel_pixel_ratio)
        all_conditions.append(conditions)
    for idx,arrow in enumerate(all_arrows):
        ccs_reacts_prods = find_step_reactants_and_products(fig,all_conditions[idx],arrow,all_arrows,panels)

        panels_dict = assign_to_nearest(panels,all_conditions,ccs_reacts_prods['all_ccs_reacts'],ccs_reacts_prods['all_ccs_prods'])

        control_set.update(*(value for value in panels_dict.values())) # The unpacking looks ugly

        conditions = Conditions(connected_components=all_conditions[idx])
        reacts=Reactant(connected_components=panels_dict['reactants'])
        prods = Product(connected_components=panels_dict['products'])
        steps.append(ReactionStep(arrow,reacts,prods, conditions))
        print('panels:', panels)
    if control_set != panels:
        log.warning('Some connected components remain unassigned following scan_all_reaction_steps.')
    else:
        log.info('All connected components have been assigned following scan_all_reaction steps.')
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
    print('megabox ccs:', megabox_ccs)
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

def assign_to_nearest(all_ccs,conditions, reactants, products, threshold=None):
    """
    This postrocessing function takes in all panels and classified panels to perform a set difference.
    It then assings the remaining panels to the appropriate group based on the closest neighbour, subject to a threshold.
    :param iterable all_ccs: list of all connected components excluding arrow
    :param int threshold: maximum distance from nearest neighbour # Not used at this stage. Is it necessary?
    :param iterable conditions: List of conditions' panels of a reaction step
    :param iterable reactants: List of reactants' panels of a reaction step
    :param iterable products: List of products' panels of a reaction step
    :return dictionary: The modified groups
    """
    log.debug('Assigning connected components based on distance')
    print('assign: conditions set: ', conditions)
    conditions_ccs = [cc for inner_set in conditions for cc in inner_set]
    classified_ccs = set((*conditions_ccs, *reactants, *products))
    print('diagonal lengths: ')
    print([cc.diagonal_length for cc in classified_ccs])
    threshold = max([cc.diagonal_length for cc in classified_ccs])
    unclassified =  all_ccs.difference(classified_ccs)
    for cc in unclassified:
        classified_ccs = sorted(classified_ccs, key=lambda elem: elem.separation(cc))
        nearest = classified_ccs[0]
        groups = [ reactants, products]
        for group in groups:
            if nearest in group and nearest.separation(cc) < threshold:
                group.add(cc)
                log.info('assigning %s to group %s based on distance' % (cc, group))


    return {'reactants':reactants, 'products':products}

