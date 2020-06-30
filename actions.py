from collections import namedtuple, Counter
from functools import partial
import copy
from itertools import product, chain

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.ndimage import label
from scipy.signal import find_peaks
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.morphology import skeletonize as skeletonize_skimage
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans

from config import get_area
from models.arrows import SolidArrow
from models.exceptions import NotAnArrowException
from models.reaction import ReactionStep,Conditions,Reactant,Product,Intermediate
from models.segments import Rect, Panel, Figure, TextLine
from models.utils import Point, Line
from utils.processing import approximate_line, create_megabox, merge_rect, pixel_ratio, binary_close, binary_floodfill, pad
from utils.processing import (binary_tag, get_bounding_box, postprocessing_close_merge, erase_elements, crop, \
                              is_slope_consistent, label_and_get_ccs, isolate_patches, standardize, get_line_parameters)
from ocr import read_character

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

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
        kernel = 4
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
        kernel = 25
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
        min_arrow_length = int(np.sqrt(fig.img.shape[0]**2+fig.img.shape[1]**2)*0.01)
        # min_arrow_lengths = [min_arrow_length, int(min_arrow_length/1.5)]

    if thresholds is None:
        threshold = int(np.sqrt(fig.img.shape[0]**2+fig.img.shape[1]**2)*0.025)
        # thresholds = [threshold, int(threshold/1.5)]
    # print('thresh :', thresholds)
    # print('min length :', min_arrow_lengths)

    # Find arrows in a two-step search
    arrows = find_solid_arrows_main_routine(fig,threshold=threshold,min_arrow_length=min_arrow_length)

    # if not arrows:
    #     log.info('No arrows have been found the image on the first attempt.')
    #     arrows = find_solid_arrows_main_routine(fig, threshold=thresholds[1], min_arrow_length=min_arrow_lengths[1])
    #     if not arrows:
    #         log.warning('No arrows have been found in the image')
    #
    return arrows


def find_solid_arrows_main_routine(fig, threshold=None, min_arrow_length=None):
    """
    This function takes in a binary figure object and aims to find all solid arrows.
    :param Figure fig: input figure objext
    :param int min_arrow_length: parameter to adjust minimum arrow length
    :return: list of arrow objects
    """
    img = copy.deepcopy(fig.img)
    skeleton = skeletonize(fig).img
    ccs = label_and_get_ccs(fig)
    lines = []
    for cc in ccs:
        isolated_fig = isolate_patches(fig, [cc])
        cc_lines = probabilistic_hough_line(isolated_fig.img, threshold=threshold, line_length=min_arrow_length)
        if len(cc_lines) > 1:
            if not is_slope_consistent(cc_lines):
                continue
            lines.append(cc_lines[0])
    # lines = probabilistic_hough_line(skeleton, threshold=threshold, line_length=min_arrow_length)
    #print(lines)
    labelled_img, _ = label(img)
    arrows =[]
    # plt.imshow(fig.img, cmap=plt.cm.binary)
    # line1 = list(zip(*lines[0]))
    # line2 = list(zip(*lines[1]))
    # plt.plot(line1[0], line1[1])
    # plt.plot(line2[0], line2[1])
    # plt.axis('off')
    # plt.show()
    # plt.imshow(fig.img, cmap=plt.cm.binary)
    # for line in lines:
    #     x, y = list(zip(*line))
    #     plt.plot(x,y)
    # plt.title('detected lines')
    # plt.show()

    for l in lines:
        points = [Point(row=y, col=x) for x, y in l]
        # Choose one of points to find the label and pixels in the image
        p1 = points[0]
        p2 = points[1]
        # p1_label = labelled_img[p1.row, p1.col]
        # p2_label = labelled_img[p2.row, p2.col]
        # if p1_label != p2_label: # Hough transform can find lines spanning several close ccs; these are discarded
        #     log.info('A false positive was found when detecting a line. Discarding...')
        #     continue
        #print('checking p1:...')
        #print(p1.row, p1. col)
        #print('should be (96, 226)')
        arrow_label = labelled_img[p1.row, p1.col]

        arrow_pixels = np.nonzero(labelled_img == arrow_label)
        arrow_pixels = list(zip(*arrow_pixels))
        try:
            new_arrow = SolidArrow(arrow_pixels, line=approximate_line(*points))
        except NotAnArrowException as e:
            log.info('An arrow candidate was discarded - ' + str(e))
        else:
            arrows.append(new_arrow)
    # Filter poor arrow assignments based on aspect ratio
    # arrows = [arrow for arrow in arrows if arrow.aspect_ratio >5]  ## This is not valid for tilted arrows
    return list(set(arrows))






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
    fig = copy.deepcopy(fig)

    closed_panels = segment(fig, all_arrows)

    structure_backbones = detect_structures(fig, label_and_get_ccs(fig))
    structures = [panel for panel in closed_panels if
                  any(panel.overlaps(backbone) and panel.area >= backbone.area for backbone in structure_backbones)]
    # f, ax = plt.subplots()
    # ax.imshow(fig.img)
    # for panel in structures:
    #     offset=0
    #     rect_bbox = Rectangle((panel.left+offset, panel.top+offset),
    #     panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='m')
    #     ax.add_patch(rect_bbox)
    # plt.show()


    # fig_structures = isolate_patches(fig, structures)
    # plt.imshow(fig_structures.img)
    # plt.title('structures isolated')
    # plt.show()
    # f, ax = plt.subplots()
    # ax.imshow(fig.img)
    # for panel in closed_panels:
    #     offset=0
    #     rect_bbox = Rectangle((panel.left+offset, panel.top+offset),
    #     panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='m')
    #     ax.add_patch(rect_bbox)
    # plt.show()
    for idx, arrow in enumerate(all_arrows):
        ccs_reacts_prods = find_step_reactants_and_products(fig, all_conditions[idx], arrow, all_arrows, structures)

        panels_dict = assign_to_nearest(structures, ccs_reacts_prods['reactants'], ccs_reacts_prods['products'])
        # panels_dict = ccs_reacts_prods
        first_step_flag = ccs_reacts_prods['first step']
        reacts = panels_dict['reactants']
        prods = panels_dict['products']
        # print(f'panels_dict: {panels_dict}')
        # if global_skel_pixel_ratio > 0.02 : #Original kernel size < 6
        #     reacts = postprocessing_close_merge(fig, reacts)
        #     prods = postprocessing_close_merge(fig, prods)
        #     log.debug('Postprocessing closing and merging finished.')

        reacts = [Reactant(connected_components=react) for react in reacts]
        prods = [Product(connected_components=prod) for prod in prods]
        steps.append(ReactionStep(arrow, reacts, prods, all_conditions[idx], first_step_flag))
        # print('panels:', panels)
    # if control_set != panels:
    #     log.warning('Some connected components remain unassigned following scan_all_reaction_steps.')
    # else:
    #     log.info('All connected components have been assigned following scan_all_reaction steps.')
    return steps


def find_step_reactants_and_products(fig, step_conditions, step_arrow, all_arrows, structures, stepsize=30):
    """
    :param Figure fig: figure object being processed
    :param Conditions step_conditions: an object containing `text_lines` representing conditions connected components
    :param Arrow step_arrow: Arrow object connecting the reactants and products
    :param iterable all_arrows: a list of all arrows found
    :param iterable structures: detected structures
    :return: a list of all conditions bounding boxes ##
    """
    log.info('Looking for reactants and products around arrow %s', step_arrow)
    first_step = False
    megabox_ccs = copy.deepcopy(step_conditions.text_lines) if isinstance(step_conditions, Conditions) else []
    megabox_ccs.append(step_arrow)
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
    i = 0 # for debugging only
    all_point_pairs_react = []
    for step in range(fig.img.shape[1]//stepsize):
        p1_react_scancol += stepsize*react_increment
        p2_react_scancol += stepsize*react_increment
        cond1 = p1_react_scancol < 0 or p2_react_scancol < 0
        cond2 = p1_react_scancol > fig.img.shape[1] or p2_react_scancol > fig.img.shape[1]
        if cond1 or cond2:
            break
        line = approximate_line(Point(row=p1_react.row, col=p1_react_scancol),
                                Point(row=p2_react.row, col=p2_react_scancol))
        case_study_plot_pairs = [(p1_react_scancol, p1_react.row), (p2_react_scancol, p2_react.row)]
        all_point_pairs_react.append(case_study_plot_pairs)
        if any(arrow.overlaps(line) for arrow in all_arrows):
            log.debug('While scanning reactants, an arrow was encountered - breaking from the loop')

            log.debug('breaking at iteration %s'% i)
            break
        raw_reacts.update([structure for structure in structures if structure
                          .overlaps(line)])
    else:
        first_step = True
    log.info('Found the following connected components of reactants: %s' % raw_reacts)


    # Find products
    raw_prods = set()
    p1_prod = prod_scanning_line.pixels[0]
    p2_prod = prod_scanning_line.pixels[-1]
    p1_prod_scancol = p1_prod.col
    p2_prod_scancol = p2_prod.col
    #Need to access figure dimensions here
    i=0 # for debugging only
    all_point_pairs_prod = []
    for step in range(fig.img.shape[1]//stepsize):
        p1_prod_scancol += stepsize*prod_increment
        p2_prod_scancol += stepsize*prod_increment
        cond1 = p1_prod_scancol < 0 or p2_prod_scancol < 0
        cond2 = p1_prod_scancol > fig.img.shape[1] or p2_prod_scancol > fig.img.shape[1]
        if cond1 or cond2:
            break
        line = approximate_line(Point(row=p1_prod.row, col=p1_prod_scancol),
                                Point(row=p2_prod.row, col=p2_prod_scancol))
        case_study_plot_pairs = [(p1_prod_scancol, p1_prod.row), (p2_prod_scancol, p2_prod.row)]
        all_point_pairs_prod.append(case_study_plot_pairs)

        if any(arrow.overlaps(line) for arrow in all_arrows):
            log.debug('While scanning products, an arrow was encountered - breaking from the loop')
            log.debug('breaking prod at iteration  %s' % i)
            break
        raw_prods.update([structure for structure in structures if structure.overlaps(line)])
    log.info('Found the following connected components of prods: %s ' % raw_prods)


    f = plt.figure(figsize=(20,20))
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.imshow(fig.img, cmap=plt.cm.binary)
    for line in all_point_pairs_react:
        x, y = list(zip(*line))
        ax.plot(x, y, 'r')

    for line in all_point_pairs_prod:
        x, y = list(zip(*line))
        ax.plot(x, y, 'b')

    plt.savefig('roles.tif')

    plt.show()

    return {'reactants':raw_reacts, 'products':raw_prods, 'first step': first_step}


# def find_step_reactants_and_products(fig, step_arrow, all_arrows, structures):
#     """
#     Finds reactants and products from ``structures`` of a single reaction step (around a single arrow) using
#     scanning in ``fig.img``. Scanning is terminated early if any of ``all_arrows`` is encountered
#     :param Figure fig: figure object being processed
#     :param Arrow step_arrow: Arrow object connecting the reactants and products
#     :param iterable all_arrows: a list of all arrows found
#     :param iterable structures: detected structures
#     :return: a dictionary with ``reactants``, ``products`` and ``first step`` flag
#     """
#     slope, intercept = get_line_parameters(step_arrow.line)



def assign_to_nearest(structures, reactants, products, threshold=None):
    """
    This postrocessing function takes in unassigned structures and classified panels to perform a set difference.
    It then assings the structures to the appropriate group based on the closest neighbour, subject to a threshold.
    :param iterable structures: list of all detected structured
    :param int threshold: maximum distance from nearest neighbour # Not used at this stage. Is it necessary?
    # :param iterable conditions: List of conditions' panels of a reaction step # not used
    :param iterable reactants: List of reactants' panels of a reaction step
    :param iterable products: List of products' panels of a reaction step
    :return dictionary: The modified groups
    """

    log.debug('Assigning connected components based on distance')
    #print('assign: conditions set: ', conditions)
    classified_structures = [*reactants, *products]
    #print('diagonal lengths: ')
    #print([cc.diagonal_length for cc in classified_ccs])
    threshold =  0.5 * np.mean(([cc.diagonal_length for cc in classified_structures]))
    unclassified =  [structure for structure in structures if structure not in classified_structures]
    for cc in unclassified:
        classified_structures.sort(key=lambda elem: elem.separation(cc))
        nearest = classified_structures[0]
        groups = [reactants, products]
        for group in groups:
            if nearest in group and nearest.separation(cc) < threshold:
                group.add(cc)
                log.info('assigning %s to group %s based on distance' % (cc, group))

    return {'reactants':reactants, 'products':products}


def remove_redundant_characters(fig, ccs, chars_to_remove=None):
    """
    Removes reduntant characters such as '+' and '[', ']' from an image to facilitate resolution of diagrams and labels.
    This function takes in `ccs` which are ccs to be considered. It then closes all connected components in `fig.img1
    and compares connected components in `ccs` and closed image. This way, text characters belonging to structures are
    not considered.
    :param iterable ccs: iterable of Panels containing species to check
    :param chars_to_remove: characters to be removed
    :return: list connected components containing redundant characters
    """
    # TODO: Store closed image globally and use when needed?
    if chars_to_remove is None:
        chars_to_remove = '+[]'

    closed = binary_close(fig, size=3)
    closed_ccs = label_and_get_ccs(closed)
    ccs_to_consider = set(ccs).intersection(set(closed_ccs))
    f, ax = plt.subplots()
    ax.imshow(closed.img)
    for panel in ccs_to_consider:
        rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',
                              edgecolor='r')
        ax.add_patch(rect_bbox)
    plt.show()

    diags_to_erase = []
    for cc in ccs_to_consider:
        text_word = read_character(fig, cc)

        if text_word:
            text = text_word.text
            print(f'recognised char: {text}')

            if any(redundant_char is text for redundant_char in chars_to_remove):
                diags_to_erase.append(cc)

    return erase_elements(fig, diags_to_erase)


def remove_redundant_square_brackets(fig, ccs):
    """
    Remove large, redundant square brackets, containing e.g. reaction conditions. These are not captured when parsing
    conditions' text (taller than a text line).
    :param Figure fig: Analysed figure
    :param [Panels] ccs: Iterable of Panels to consider for removal
    :return: Figure with square brackets removed
    """
    candidate_ccs = filter(lambda cc: cc.aspect_ratio > 5 or cc.aspect_ratio < 1 / 5, ccs)

    detected_lines = 0
    bracket_ccs = []

    # transform
    for cc in candidate_ccs:
        cc_fig = isolate_patches(fig,
                                 [cc])  # Isolate appropriate connected components in preparation for Hough
        # plt.imshow(cc_fig.img)
        # plt.show()
        line_length = (cc.width + cc.height) * 0.5  # since length >> width or vice versa, this is equal to ~0.8*length
        line = probabilistic_hough_line(cc_fig.img, line_length=int(line_length))
        if line:
            detected_lines += 1
            bracket_ccs.append(cc)

    print(bracket_ccs)
    if len(bracket_ccs) % 2 == 0:
        fig = erase_elements(fig, bracket_ccs)

    return fig


def match_function_and_smiles(reaction_step, smiles):
    """
    Matches the resolved smiles from chemschematicresolver to functions (reactant, product) found by the segmentation
    algorithm.

    :param ReactionStep reaction_step: object containing connected components classified as `reactants` or `products`
    :param [[smile], [ccs]l] smiles: list of lists containing structures converted into SMILES format and recognised
     labels, and connected components depicting the structures in an image
    :return: Mutated ReactionStep object - with optically recognised structures as SMILES and labels/auxiliary text
    """

    for reactant in reaction_step.reactants:
        matching_record = [recognised for recognised, diag in zip(*smiles) if diag == reactant.connected_component]
        # This __eq__ is based on a flaw in csr - cc is of type `Label`, but inherits from Rect
        if matching_record:
            matching_record = matching_record[0]
            reactant.label = matching_record[0]
            reactant.smiles = matching_record[1]
        else:
            log.warning('No SMILES match was found for a reactant structure')

    for product in reaction_step.products:

        matching_record = [recognised for recognised, diag in zip(*smiles) if diag == product.connected_component]
        # This __eq__ is based on flaw in csr - cc is of type `Diagram`, but inherits from Rect
        if matching_record:
            matching_record = matching_record[0]
            product.label = matching_record[0]
            product.smiles = matching_record[1]
        else:
            log.warning('No SMILES match was found for a product structure')


def detect_structures(fig, ccs):
    """
    Detects structures based on parameters such as size, aspect ratio and number of detected lines

    :param Figure fig: analysed figure
    :param [Panels[ ccs: list of all connected components
    :return [Panels]: list of connected components classified as structures
    """
    fig = copy.deepcopy(fig)
    ccs = list(ccs)
    # Get a rough bond length (line length) value from the two largest structures
    ccs = sorted(ccs, key=lambda cc: cc.area, reverse=True)
    estimation_fig = skeletonize(isolate_patches(fig, ccs[:2]))
    length_scan_start = 0.02 * max(fig.width, fig.height)
    min_line_lengths = np.linspace(length_scan_start, 3*length_scan_start, 20)
    # print(min_line_lengths)
    # min_line_lengths = list(range(20, 60, 2))
    num_lines = [(length, len(probabilistic_hough_line(estimation_fig.img, line_length=int(length), threshold=15))**2)
                    for length in min_line_lengths]
    # Choose the value where the number of lines starts to drop most rapidly and assign it as the boundary length
    (boundary_length,_), (_, _) = min(zip(num_lines, num_lines[1:]), key= lambda pair: pair[1][1] - pair[0][1])  # the key is
                                                                        # difference in number of detected lines
                                                                        # between adjacent pairs


    # Use the length to find number of lines in each cc - this will be one of the used features
    cc_lines = []
    # all_lines = [] #Case study only
    for cc in ccs:
        isolated_cc_fig = isolate_patches(fig, [cc])
        isolated_cc_fig = skeletonize(isolated_cc_fig)
        # lines = probabilistic_hough_line(isolated_cc_fig.img, line_length=boundary_length, threshold=15)  # Case study
        # all_lines.extend(lines)  # case study
        # hspace, angles, dists = hough_line(isolated_cc_fig.img)
        # hspace, angles, dists = hough_line_peaks(hspace, angles, dists, threshold=boundary_length)
        # num_lines = len(dists)
        # print(f'num lines normal: {len(dists)}')
        angles = np.linspace(-np.pi, np.pi, 360)
        num_lines = len(probabilistic_hough_line(isolated_cc_fig.img,
                                                 line_length=int(boundary_length), threshold=10, theta=angles))
        # print(f'num lines prob: {num_lines}')
        cc_lines.append(num_lines)

    ##Case study only
    # skeleton_fig = skeletonize(fig)
    # f = plt.figure(figsize=(20, 20))
    # ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.imshow(fig.img, cmap=plt.cm.binary)
    # for line in all_lines:
    #
    #     x, y = list(zip(*line))
    #     ax.plot(x,y, 'r')
    #
    # plt.savefig('lines_structures.tif')
    # plt.show()
    ##Case study end

    cc_lines = np.array(cc_lines).reshape(-1,1)
    area = np.array([cc.area for cc in ccs]).reshape(-1, 1)
    aspect_ratio = np.array([cc.aspect_ratio for cc in ccs]).reshape(-1, 1)
    mean_area = np.mean(area)

    print(f'boundary: {boundary_length}')
    print(f'mean sqrt area: {np.sqrt(mean_area)}')

    data = np.hstack((cc_lines, area, aspect_ratio))
    # print(f'data: \n {data}')
    # print(f'data: {data}')
    data = MinMaxScaler().fit_transform(data)
    distances = np.array([(x, y, z, np.sqrt(np.sqrt(x**2 + y**2)+z**2)) for x,y,z in data])
    # print(f'transformed: \n {distances}')
    # print(f'transformed: {data}')
    # print(f'distances: {distances}')
    # data = data.clip(min=0)
    # data = cc_lines
    # print(f'data: {data}')

    labels = DBSCAN(eps=0.08, min_samples=20).fit_predict(data)

    colors = ['b', 'm', 'g', 'r']
    paired = list(zip(ccs, labels))
    paired = [(cc, label) if cc.area > mean_area else (cc,0) for cc, label in paired]

    if False:
        f = plt.figure(figsize=(20, 20))
        ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.imshow(fig.img, cmap=plt.cm.binary)
        # ax.set_title('structure identification')
        for panel, label in paired:
            rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor=colors[label])
            ax.add_patch(rect_bbox)
        #plt.savefig('backbones.tif')
        plt.show()
    structures = [panel for panel, label in paired if label == -1]

    return structures





    # # There are only two possible classes here: structures and text - arrows are excluded (for now?)
    # size = np.asarray([cc.area**2 for cc in ccs], dtype=float)
    # aspect_ratio = [cc.aspect_ratio for cc in ccs]
    # aspect_ratio = np.asarray([ratio + 1 / ratio for ratio in aspect_ratio],
    #                           dtype=float)  # Transform to weigh wide
    # print('aspect ratio: \n', aspect_ratio)
    # plt.hist(aspect_ratio)
    # plt.show()
    #
    # # and tall structures equally (as opposed to ratios around 1)
    # pixel_ratios = np.asarray([pixel_ratio(fig, cc) for cc in ccs])
    # data = np.vstack((size, aspect_ratio, pixel_ratios))
    # data = np.transpose(data)
    # print(np.mean(data, axis=0))
    # print(np.std(data, axis=0))
    # data -= np.mean(data, axis=0)
    # data /= np.std(data, axis=0)
    # data[:,2] = np.power(data[:, 2], 3)
    # # data[:, 2] = np.power(data[:, 2], 3)
    # f, ax = plt.subplots(2,2)
    # print(sorted(data[:,0]))
    # ax[0,0].scatter(data[:,0],data[:,1])
    # ax[0,1].scatter(data[:,1], data[:,2])
    # ax[1,0].scatter(data[:,0], data[:,2])
    # plt.show()
    # # print('size: \n', size)
    #
    # # print('pixel ratio: \n', pixel_ratio)
    # #
    # print(f'data:')
    # # print(data)
    # # print(data.shape)
    # #labels = KMeans(n_clusters=2, n_init=20).fit_predict(data)
    # eps = np.sum(np.std(data, axis=0))
    #
    # neigh = NearestNeighbors(n_neighbors=2)
    # nbrs = neigh.fit(data)
    # distances, indices = nbrs.kneighbors(data)
    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # print('distances: \n', distances)
    # _, bins, _ = plt.hist(distances)
    # plt.show()
    # eps = bins[1]
    # labels = DBSCAN(min_samples=5,eps=eps).fit_predict(data)
    # #labels = OPTICS().fit_predict(data)
    # print(labels)
    # return ccs, labels





def extend_line(line, extension=None):
    """
    Extends line in both directions. Output is a pair of points, each of which is further from an arrow (closer to
    reactants or products in the context of reactions).
    :param Line line: original Line object
    :param int extension: value dictated how far the new line should extend in each direction
    :return: new Line object
    """
    slope, intercept = get_line_parameters(line)
    if slope is np.inf:  # vertical line
        line.pixels.sort(key=lambda point: point.row)


        first_line_pixel = line.pixels[0]
        last_line_pixel = line.pixels[-1]
        if extension is None:
            extension = int((last_line_pixel.separation(first_line_pixel)) * 0.4)

        left_extended_point = Point(row=first_line_pixel.row - extension, col=first_line_pixel.col)
        right_extended_point = Point(row=last_line_pixel.row + extension, col=last_line_pixel.col)

    else:
        line.pixels.sort(key=lambda point: point.col)

        first_line_pixel = line.pixels[0]
        last_line_pixel = line.pixels[-1]
        if extension is None:
            extension = int((last_line_pixel.separation(first_line_pixel)) * 0.4)

        left_extended_last_y = slope*(first_line_pixel.col-extension) + intercept
        right_extended_last_y = slope*(last_line_pixel.col+extension) + intercept

        left_extended_point = Point(row=left_extended_last_y, col=first_line_pixel.col-extension)
        right_extended_point = Point(row=right_extended_last_y, col=last_line_pixel.col+extension)

    # extended = approximate_line(first_line_pixel, left_extended_point) +\
    #            approximate_line(last_line_pixel, right_extended_point) + line
    #
    #
    # new_line = Line(extended)

    return (left_extended_point, right_extended_point)


def find_nearby_ccs(start, all_relevant_ccs, max_initial_distance):
    """
    Find all structures close to ``start`` position. All found structures are added to a queue and
    checked again to form a cluster of nearby structures.
    :param Point or (x,y) start: point where the search starts
    :param [Panel,...] all_relevant_ccs: list of all found structures
    :param float max_initial_distance: maximum allowed distance from the starting point
    :return: List of all nearby structures
    """
    frontier = []
    frontier.append(start)
    found_structures = []
    visited = set()
    while frontier:
        reference = frontier.pop()
        visited.add(reference)
        max_distance = max_initial_distance + 1.5 * np.sqrt(reference.area) if isinstance(reference, Panel) else max_initial_distance
        seps = [structure.separation(reference) for structure in all_relevant_ccs]
        successors = [structure for structure in all_relevant_ccs if structure.separation(reference) < max_distance
                             and structure not in visited]
        new_structures = [structure for structure in successors if structure not in found_structures]
        frontier.extend(successors)
        print(frontier)
        found_structures.extend(new_structures)

    return found_structures


