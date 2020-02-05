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



def find_reaction_conditions(fig, arrow, panels, global_skel_pixel_ratio,stepsize=10,steps=10):
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
    panels = [panel for panel in panels if panel.area < 55**2]
    arrow = copy.deepcopy(arrow)
    sorted_pts_col = sorted(arrow.line,key= lambda pt: pt.col)
    p1, p2 = sorted_pts_col[0], sorted_pts_col[-1]
    #'Squeeze the two points in the second dimension to avoid overlap with structures on either side

    overlapped = set()
    rows = []
    columns = []
    for direction in range(2):
        boredom_index = 0 #increments if no overlaps found. if nothing found in 5 steps,
        # the algorithm breaks from a loop
        increment = (-1) ** direction
        p1_scancol, p2_scancol = p1.col, p2.col
        p1_scanrow, p2_scanrow = p1.row, p2.row

        for step in range(steps):
            #Reduce the scanned area proportionally to distance from arrow
            p1_scancol = int(p1_scancol * 1.01) # These changes dont work as expected - need to check them
            p2_scancol = int(p2_scancol * .99)

            #print('p1: ',p1_scanrow,p1.col)
            #print('p2: ', p2_scanrow,p2.col)
            p1_scanrow += increment*stepsize
            p2_scanrow += increment*stepsize
            rows.extend((p1_scanrow, p2_scanrow))
            columns.extend((p1_scancol, p2_scancol))
            # print(f'approximating between points {p1_scanrow, p1_scancol} and {p2_scanrow, p2_scancol}')
            line = approximate_line(Point(row=p1_scanrow, col=p1_scancol), Point(row=p2_scanrow, col=p2_scancol))
            overlapping_panels = [panel for panel in panels if panel.overlaps(line)]

            if overlapping_panels:
                overlapped.update(overlapping_panels)
                boredom_index = 0
            else:
                boredom_index +=1

            if boredom_index >= 5: #Nothing found in the last five steps
                break
    # print(f'rows: {rows}')
    # print(f'cols: {columns}')
    # plt.imshow(fig.img, cmap=plt.cm.binary)
    # plt.scatter(columns, rows, c='y', s=1)
    # plt.savefig('destination_path.jpg', format='jpg', dpi=1000)
    # plt.show()

    conditions_text = scan_conditions_text2(fig,overlapped,arrow)

    return set(conditions_text) #Return as a set to allow handling along with product and reactant sets


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

def scan_conditions_text(fig, conditions,arrow):
    """
    Crops a larger area around raw conditions to look for additional text elements that have
    not been correctly recognised as conditions
    :param Figure fig: analysed figure with binarised image object
    :param iterable of Panels conditions: set or list of raw conditions (output of `find_reaction_conditions`)
    :return: Set of Panels containing all recognised conditions
    """

    fig = copy.deepcopy(fig)
    extended_boundary_vertical = 150
    extended_boundary_horizontal = 75
    conditions = create_megabox(conditions)

    fig = erase_elements(fig,[arrow])
    raw_conditions_region = crop(fig.img, conditions.left, conditions.right,
               conditions.top, conditions.bottom)['img']

    p = 5 #padding amount
    raw_conditions_region = pad(raw_conditions_region,[(p,p),(p,p)],'constant',constant_values=0)

    padded_conditions = Rect(conditions.left - p, conditions.right + p,
                             conditions.top - p, conditions.bottom + p)
    #raw_conditions_region = binary_close(Figure(raw_conditions_region),1).img
    labelled = binary_tag(Figure(raw_conditions_region))
    ccs = get_bounding_box(labelled)

    upper, lower = identify_textlines(ccs,raw_conditions_region)

    initial_ccs_transformed = []
    #upper = [bottom_textline-mean_height for bottom_textline in lower]


    print(f'lower before transformation: {lower}')
    search_region = Rect(conditions.left-extended_boundary_horizontal, conditions.right+extended_boundary_horizontal,
               conditions.top-extended_boundary_vertical, conditions.bottom+extended_boundary_vertical)
    print(f'search region: {search_region}')
    crop_dct = crop_rect(fig.img, search_region)
    roi = crop_dct['img']
    #roi = binary_close(Figure(roi),1)
    #roi = roi.img
    boundaries = crop_dct['rectangle'] #This is necessary in case a smaller area is cropped
    print(f'actual boundaries :{boundaries}')

    #Move lines to the new extended region
    vertical_shift = extended_boundary_vertical if search_region.top > 0 else conditions.top#if still within the image after expansion,
    # then choose the extended boundary, else the shift is the distance from top as crop cannot goe beyond 0
    # print(f'horizontal, vertical: {horizontal_shift, vertical_shift}')
    horizontal_shift = extended_boundary_horizontal if search_region.left > 0 else conditions.left
    lower = [row+vertical_shift-p for row in lower]
    upper = [row+vertical_shift-p for row in upper]


    initial_ccs_transformed = transform_panel_coordinates_to_expanded_rect(padded_conditions,boundaries,ccs,absolute=True)
    print(f'the new boundaries: {boundaries}')


    # f, ax = plt.subplots()
    # ax.imshow(roi)
    # for line in lower:
    #    ax.plot([i for i in range(roi.shape[1])],[line for i in range(roi.shape[1])],color='r')
    # for line in upper:
    #    ax.plot([i for i in range(roi.shape[1])],[line for i in range(roi.shape[1])],color='b')

    # for line in kde_lower:
    #     ax.plot([i for i in range(roi.shape[1])], [line for i in range(roi.shape[1])], color='m')
    # plt.show()

    upper.sort()
    lower.sort()
    textlines = [TextLine(left=0, right=roi.shape[1],top=top_line, bottom=bottom_line)
                 for top_line, bottom_line in zip(upper,lower)]
    #print('textlines:...')
    #print(textlines)


    print(f'all ccs: {initial_ccs_transformed}')
    roi = attempt_remove_structure_parts(roi, initial_ccs_transformed)
    # roi = roi.img
    print(type(roi))
    labelled = binary_tag(roi)


    ccs = set(get_bounding_box(labelled))
    # small_characters = find_small_characters(roi, ccs)
    # ccs = ccs.difference(small_characters)
    mean_character_area = np.average([cc.area for cc in ccs])
    text_candidate_buckets =[]
    # print(f'textlines: {textlines}')


    text_candidate_buckets = assign_characters_to_textlines(roi.img,textlines, ccs)



    text_candidates = [element for textline in text_candidate_buckets for element in textline]
    # print(f'text candidates: {text_candidates}' )
    text_elements = [cc for cc in text_candidates if not is_boundary_cc(roi.img,cc)]

    # Find unclassified characters
    remaining_elements = set(ccs).difference(text_elements)


    to_filter_out = set()
    for element in remaining_elements:

        assigned = attempt_assign_to_nearest_text_element(roi, element, mean_character_area)
        if assigned:
            print(f'assigned: {assigned}')
            small_cc, assigned_closest_cc = assigned

            for textline_elements in text_candidate_buckets :
                if assigned_closest_cc in textline_elements:
                    to_filter_out.add(small_cc)
                    textline_elements.append(small_cc)
                    print(f'after adding: {textline_elements}')
    remaining_elements = remaining_elements.difference(to_filter_out)

    additional_detected_textlines = attempt_fit_textline(roi, textlines)
    print(f'additional textlines: {additional_detected_textlines}')
    if additional_detected_textlines:
        text_candidate_buckets.extend(additional_detected_textlines)
    text_elements = [element for textline in text_candidate_buckets for element in textline]
    # f, ax = plt.subplots()
    # ax.imshow(roi.img)
    # ax.set_title('remaining elements')
    # for panel in remaining_elements:
    #     rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
    #                           facecolor='none', edgecolor='y')
    #     ax.add_patch(rect_bbox)
    # plt.show()

    # f, ax = plt.subplots()
    # ax.imshow(roi.img)
    # colors =['g','r','b','m','w','y','k','r','g','b','m']*2
    # c=-1
    # for textline in text_candidate_buckets:
    #     c+=1
    #     for panel in textline:
    #         rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
    #                               facecolor='none', edgecolor=colors[c])
    #         ax.add_patch(rect_bbox)
    # print(f'textline buckets: {text_candidate_buckets}')
    # for panel in text_elements:
    #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='b')
    #     ax.add_patch(rect_bbox)
    # plt.show()
    # TODO: Add another condition that entire original cc is in the crop
    text_elements = isolate_full_text_block(text_candidate_buckets)
    #Transform boxes back into the main image coordinate system
    print(f'text elements: {text_elements}')
    text_elements_transformed =[]
    for element in text_elements:
        height = element.bottom - element.top
        width = element.right - element.left
        p = Panel(left=boundaries.left+element.left, right=boundaries.left+element.left+width,
                           top=boundaries.top+element.top, bottom=boundaries.top+element.top+height)
        text_elements_transformed.append(p)


    return text_elements_transformed

def identify_textlines(ccs,img):

    bottom_boundaries = [cc.bottom for cc in ccs]
    bottom_boundaries.sort()

    bottom_count = Counter(bottom_boundaries)
    # bottom_count = Counter({value:count for value, count in bottom_count.items() if count>1})
    bottom_boundaries = np.array([item for item in bottom_count.elements()]).reshape(-1, 1)

    little_data = len(ccs) < 10
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.005, 2.0, 100)},
                        cv=(len(ccs) if little_data else 10))  # 10-fold cross-validation
    grid.fit(bottom_boundaries)
    best_bw = grid.best_params_['bandwidth']
    kde = KernelDensity(best_bw, kernel='exponential')
    kde.fit(bottom_boundaries)

    # print(f'params: {kde.get_params()}')
    rows= np.linspace(0, img.shape[0], img.shape[0] + 1)
    logp = kde.score_samples(rows.reshape(-1, 1))


    heights = [cc.bottom - cc.top for cc in ccs]
    mean_height = np.mean(heights, dtype=np.uint32)
    kde_lower, _ = find_peaks(logp, distance=mean_height*1.2)
    kde_lower.sort()
    plt.plot(rows, logp)
    plt.xlabel('Row')
    plt.ylabel('logP(textline)')
    plt.scatter(kde_lower, [0 for elem in kde_lower])
    plt.show()
    line_buckets = []
    for peak in kde_lower:
        bucket = [cc for cc in ccs if cc.bottom in range(peak - 3, peak + 3)]
        line_buckets.append(bucket)

    # print(f'list of buckets: {line_buckets}')
    # print(len(line_buckets))
    top_lines = []
    # print(kde_lower)
    for bucket, peak in zip(line_buckets, kde_lower):
        mean_height = np.max([elem.bottom-elem.top for elem in bucket])
        top_line = peak - mean_height
        top_lines.append(top_line)

    bottom_lines = kde_lower
    top_lines.sort()
    bottom_lines.sort()
    #textlines = [TextLine(None,top,bottom) for top, bottom in zip(top_lines,bottom_lines)]
    return (top_lines, bottom_lines)

def filter_distant_text_character(ccs, textline):
    if not ccs:
        return []

    data = np.array([cc.center for cc in ccs]).reshape(-1,2)
    char_size = np.max([cc.diagonal_length for cc in ccs]) # This is about 25-30 usually

    initial_cluster = np.array(textline.center).reshape(1,-1)
    db = DBSCAN(eps=char_size*2, min_samples=2).fit(data)
    db_labels = db.labels_

    # labels = db.labels_
    # n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise = list(labels).count(-1)
    # print(f'number of noise ccs detected: {n_noise}')
    # print(f'number of detected_clusters: {n_labels}')
    # print(f'number of original ccs: {len(ccs)}')

    # import matplotlib.pyplot as plt
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # # Black removed and is used for noise instead.
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = data[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], [0 for elem in xy[:,0]], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=14)
    #
    #     xy = data[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], [0 for elem in xy[:,0]], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=6)
    #
    # plt.show()
    # print(f'labels: {db_labels}')
    return [cc for cc, label in zip(ccs, db_labels) if label == 0]

def attempt_remove_structure_parts(cropped_img, text_ccs):
    crop_no_letters = erase_elements(Figure(cropped_img),text_ccs)
    # f, ax = plt.subplots()
    # ax.imshow(crop_no_letters.img)
    # for panel in text_ccs:
    #    rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='g',edgecolor='b',alpha=0.7)
    #    ax.add_patch(rect_bbox)
    # ax.set_title('characters removed')
    # plt.show()
    skel_pixel_ratio = skeletonize_area_ratio(Figure(cropped_img),Panel(0,cropped_img.shape[1], 0, cropped_img.shape[0]))
    print(f'the skel-pixel ratio is {skel_pixel_ratio}')
    closed = binary_dilation(crop_no_letters.img,selem=disk(4)) #Decide based on skel-pixel ratio
    labelled = binary_tag(Figure(closed))
    ccs = get_bounding_box(labelled)
    structure_parts = [cc for cc in ccs if is_boundary_cc(cropped_img,cc)]
    crop_no_structures = erase_elements(Figure(cropped_img),structure_parts)

    # f, ax = plt.subplots()
    # ax.imshow(crop_no_structures.img)
    # for panel in text_ccs:
    #    rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='b')
    #    ax.add_patch(rect_bbox)
    # ax.set_title('structures removed')
    # plt.show()

    return crop_no_structures


def assign_characters_to_textlines(img, textlines, ccs, transform_from_crop=False, crop_rect=None):
    text_candidate_buckets =[]
    for textline in textlines:
        textline_text_candidates = []

        for cc in ccs:
            if belongs_to_textline(img,cc,textline):
                textline_text_candidates.append(cc)

            # elif is_small_textline_character(roi,cc,mean_character_area,textline):
            #     if cc not in textline_text_candidates: #avoid doubling
            #         textline_text_candidates.append(cc)

        textline_text_candidates = filter_distant_text_character(textline_text_candidates,textline)
        print(f'textline text cands: {textline_text_candidates}')
        if transform_from_crop:
            textline_text_candidates = transform_panel_coordinates_to_expanded_rect(
                crop_rect, Rect(0,0,0,0), textline_text_candidates) #Relative only so a placeholder Rect is the input

        if textline_text_candidates: #If there are any candidates
            textline.connected_components = textline_text_candidates
            print(f'components: {textline.connected_components}')
            text_candidate_buckets.append(textline)

    return text_candidate_buckets
def find_small_characters(cropped_figure, ccs,threshold_size=None):
    # Currently not used
    if threshold_size is None:
        threshold_size = np.mean([cc.area for cc in ccs])/2

    small_characters = [cc for cc in ccs if cc.area < threshold_size]

    return set(small_characters)

def attempt_assign_to_nearest_text_element(fig, cc, mean_character_area, small_cc=True):
    """
    Crops `fig.img` and does a simple proximity search to determine the closest character.
    :param Figure fig: figure object containing image with the cc panel
    :param Panel cc: unassigned connected component
    :param float mean_character_area: average area of character in the main crop
    :return: tuple (cc, nearest_neighbour) if close enough, else return None (inconclusive search)
    """
    mean_character_diagonal = np.sqrt(2 * mean_character_area)
    expansion = int(2 * mean_character_diagonal)
    crop_region = Rect(cc.left-expansion, cc.right+expansion, cc.top-expansion, cc.bottom+expansion)
    cropped_img = crop_rect(fig.img, crop_region)
    if cropped_img['rectangle'] != crop_region:
        crop_region = cropped_img['rectangle']

    cropped_img = cropped_img['img']
    cc_in_shrunken_region = transform_panel_coordinates_to_shrunken_region(crop_region,cc)[0]
    ccs = label_and_get_ccs(Figure(cropped_img))
    print(ccs)
    small_cc = True if cc.area < mean_character_area else False

    if small_cc:
        print(cc.area)
        closest_ccs = sorted([(other_cc, other_cc.separation(cc_in_shrunken_region))
                             for other_cc in ccs if other_cc.area > 1.3 * cc.area], key=lambda elem : elem[1])
                        #Calculate separation, sort,
                        # then choose the smallest non-zero (index 1) separation
    else:
        closest_ccs = sorted([(other_cc, other_cc.separation(cc_in_shrunken_region))
                             for other_cc in ccs], key=lambda elem: elem[1])

    if len(closest_ccs) > 1:
        closest_cc = closest_ccs[1]
    else:
        return None


    if closest_cc[1] > 2 * mean_character_diagonal:
        return None # Too far away

    closest_cc_transformed = transform_panel_coordinates_to_expanded_rect(crop_region,fig.img,[closest_cc[0]])[0]
    return (cc, closest_cc_transformed)




def attempt_fit_textline(cropped_figure, main_textlines):
    img = cropped_figure.img

    main_textlines_top = np.min([textline.top for textline in main_textlines])
    main_textlines_bottom = np.max([textline.bottom for textline in main_textlines])
    mean_main_textline_height = np.mean([textline.height for textline in main_textlines])
    mean_main_textline_center_horizontal = np.mean([textline.center[0] for textline in main_textlines])

    crop_top_region = Rect(left=0, right=img.shape[1],
                           top=0, bottom=main_textlines_top)

    crop_bottom_region = Rect(left=0, right=img.shape[1],
                           top=main_textlines_bottom, bottom=img.shape[0])

    top_text_buckets = fit_textline_locally(cropped_figure,crop_top_region)
    bottom_text_buckets = fit_textline_locally(cropped_figure, crop_bottom_region)
    if not (top_text_buckets or bottom_text_buckets):
        return None
    print(f'top buckets: {top_text_buckets}')
    print(bool(top_text_buckets))
    print(f'bottom buckets: {bottom_text_buckets}')
    print(bool(bottom_text_buckets))
    text_candidate_buckets = [*top_text_buckets, *bottom_text_buckets]
    tolerance = 10 # pixels
    additional_conditions_text_buckets =[]
    for bucket in text_candidate_buckets:
        mean_new_textline_center_horizontal = np.mean([elem.center[0] for elem in bucket])
        mean_new_textline_height = np.mean([elem.height for elem in bucket])
        cond1 = abs(mean_new_textline_center_horizontal - mean_main_textline_center_horizontal) <= tolerance
        cond2 = abs(mean_new_textline_height - mean_main_textline_height) <= tolerance
        if cond1 and cond2:
            additional_conditions_text_buckets.append(bucket)

    return additional_conditions_text_buckets


    # Perform kmeans with the following features
    #difference height - mean_textline height?
    #bottom
    #varying number of clusters between 1 and 4?
    #restricting cluster centre to around middle of the line (img.shape[1]//2)
    height_squared_residuals = np.array([(cc.height - mean_textline_height)**2 for cc in unclassified_ccs]).reshape(-1,1)
    bottom_boundaries_squared = np.array([cc.bottom**2 for cc in unclassified_ccs]).reshape(-1,1)
    print(f'bottom boundaries: {bottom_boundaries}')
    data = np.hstack((height_squared_residuals,bottom_boundaries))
    print(f'data: {data}')


def fit_textline_locally(main_crop, subcrop_region):
    cropped_region = crop_rect(main_crop.img, subcrop_region)
    if cropped_region['rectangle'] != subcrop_region:
        subcrop_region = cropped_region['rectangle']

    cropped_img = cropped_region['img']
    ccs = label_and_get_ccs(Figure(cropped_img))
    plt.imshow(cropped_img)
    plt.title('attempt_fit_textline')
    plt.show()
    print(f'ccs: {ccs}')
    print(f'len ccs: {len(ccs)}')
    if len(ccs) < 2:
        return []

    upper, lower = identify_textlines(ccs, cropped_img)

    new_textlines = [TextLine(left=0, right=subcrop_region.right, top=top_line, bottom=bottom_line)
                     for top_line, bottom_line in zip(upper, lower)]
    text_candidate_buckets = assign_characters_to_textlines(
        cropped_img, new_textlines, ccs, transform_from_crop=True,crop_rect=subcrop_region)

    return text_candidate_buckets


# def filter_textlines_height_criterion(textlines):
#     mean_textline_height = np.mean([textline.height for textline in textlines])
#     tolerance = 10 # pixels
#     cond = [abs(mean_textline_height - textline.height) <= tolerance for textline in textlines]
#     return [textlines[idx] for idx in range(len(textlines)) if cond[idx]]
#
# def filter_textlines_center_criterion(textlines):
#     mean_textline_center_horizontal = np.mean([textline.center[1] for textline in textlines])
#     tolerance = 40 # pixels
#     cond = [abs(mean_textline_center_horizontal - textline.center[1]) <= tolerance for textline in textlines]
#     return [textlines[idx] for idx in range(len(textlines)) if cond[idx]]

def scan_conditions_text2(fig, conditions,arrow):
    """
    Crops a larger area around raw conditions to look for additional text elements that have
    not been correctly recognised as conditions
    :param Figure fig: analysed figure with binarised image object
    :param iterable of Panels conditions: set or list of raw conditions (output of `find_reaction_conditions`)
    :return: Set of Panels containing all recognised conditions
    """

    fig = copy.deepcopy(fig)
    fig = erase_elements(fig, [arrow]) # erase arrow at the very beginning
    extended_boundary_vertical = 150
    extended_boundary_horizontal = 75
    conditions_box = create_megabox(conditions)

    search_region = Rect(conditions_box.left-extended_boundary_horizontal, conditions_box.right+extended_boundary_horizontal,
               conditions_box.top-extended_boundary_vertical, conditions_box.bottom+extended_boundary_vertical)
    print(f'search region: {search_region}')
    crop_dct = crop_rect(fig.img, search_region)
    if crop_dct['rectangle'] != search_region:
        search_region = crop_dct['img']

    initial_ccs_transformed = transform_panel_coordinates_to_shrunken_region(crop_dct['rectangle'],conditions)
    search_region = attempt_remove_structure_parts(search_region, initial_ccs_transformed)
    ccs = label_and_get_ccs(search_region)
    top_boundaries, bottom_boundaries  = identify_textlines(ccs, search_region.img)

    textlines = [TextLine(0, search_region.img.shape[1], upper, lower)
                for upper, lower in zip(top_boundaries, bottom_boundaries)]

    f, ax = plt.subplots()
    ax.imshow(search_region.img)
    for line in top_boundaries:
       ax.plot([i for i in range(search_region.img.shape[1])],[line for i in range(search_region.img.shape[1])],color='r')
    for line in bottom_boundaries:
       ax.plot([i for i in range(search_region.img.shape[1])],[line for i in range(search_region.img.shape[1])],color='b')
    plt.show()


    print(f'textlines:{textlines}')
    text_candidate_buckets = assign_characters_to_textlines(search_region.img, textlines, ccs)
    print(f'example textline ccs: {text_candidate_buckets[0].connected_components}')
    mixed_text_candidates = [element for textline in text_candidate_buckets for element in textline]


    remaining_elements = set(ccs).difference(mixed_text_candidates)
    text_candidate_buckets = assign_characters_proximity_search(search_region,
                                                                remaining_elements, text_candidate_buckets)
# ## New function starts here
#     to_filter_out = set()
#     for element in remaining_elements:
#
#         assigned = attempt_assign_to_nearest_text_element(search_region, element, mean_character_area)
#         if assigned:
#             print(f'assigned: {assigned}')
#             small_cc, assigned_closest_cc = assigned
#
#             for textline in text_candidate_buckets :
#                 if assigned_closest_cc in textline:
#                     to_filter_out.add(small_cc)
#                     textline.append(small_cc)
                    # print(f'after adding: {textline_elements}')
    ## New function ends here?
    print(f'buckets: {text_candidate_buckets}')
    #text_candidate_buckets = filter_textlines_center_criterion(text_candidate_buckets)
    #text_candidate_buckets = filter_textlines_height_criterion(text_candidate_buckets)
    print(f'buckets after filtering: {text_candidate_buckets}')
    print(f'example textline ccs2: {text_candidate_buckets[0].connected_components}')
    if len(text_candidate_buckets) > 2:
        text_candidate_buckets = isolate_full_text_block(text_candidate_buckets, arrow)
    #all_mixed_text = [elem for bucket in text_candidate_buckets for elem in bucket]
    print(f'buckets: {text_candidate_buckets}')

    transformed_textlines= []
    for textline in text_candidate_buckets:
        textline.connected_components = transform_panel_coordinates_to_expanded_rect(crop_dct['rectangle'],
                                                                          Rect(0,0,0,0), textline.connected_components)
        transformed_textlines.append(textline)


    return transformed_textlines



def isolate_full_text_block(textlines, arrow):
    for textline in textlines:
        textline.adjust_left_right()
    # mixed_text_elements = [elem for textline in text_buckets for elem in textline]
    mean_textline_height = np.mean([textline.height for textline in textlines])
    # char_areas = [elem.area for elem in mixed_text_elements]
    # mean_char_area = np.mean(char_areas)
    # std_area = np.std(char_areas)
    # data = np.array([(*elem.center, elem.area) for elem in mixed_text_elements]).reshape(-1,3)
    data = [textline.center for textline in textlines]
    data.append(arrow.center)
    print(data)
    #center_area_ratio = np.mean(mean_char_area + mean_char_size)
    # max_r = np.sqrt((mean_char_area+3*std_area)**2 + mean_char_size**2)
    db = DBSCAN(eps=mean_textline_height*4, min_samples=2).fit(data)
    labels = db.labels_
    print(f'labels: {labels}')
    main_cluster = [textline for textline, label in zip(textlines, labels) if label == 0]
    print(f'found cluster: {main_cluster}')

    return main_cluster


def assign_characters_proximity_search(img, chars_to_assign, textlines):
    """
    Crops `img` around each of `chars_to_assign` and performs a short-range proximity search. Assigns it to the same
    group as its nearest neighbours
    :param img:
    :param chars_to_assign:
    :param textlines
    :return:
    """
    mixed_text_ccs = [char for textline in textlines for char in textline]
    mean_character_area = np.mean([char.area for char in mixed_text_ccs])
    for element in chars_to_assign:
        assigned = attempt_assign_to_nearest_text_element(img, element, mean_character_area)
        if assigned:
            print(f'assigned: {assigned}')
            small_cc, assigned_closest_cc = assigned

            for textline in textlines :
                if assigned_closest_cc in textline:
                    #to_filter_out.add(small_cc)
                    textline.append(small_cc)

    #sanity check:
    # after_assignment = set([elem for textline in textlines for elem in textline])
    # remaining_elements =  chars_to_assign.difference(after_assignment)
    # print(f'remaining unassigned elements: {remaining_elements}')

    return textlines