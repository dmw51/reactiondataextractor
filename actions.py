from collections import namedtuple, Counter
import copy
from itertools import product, chain

import logging
import numpy as np

from scipy.ndimage import label
from scipy.signal import find_peaks
from skimage.transform import probabilistic_hough_line
from skimage.morphology import skeletonize as skeletonize_skimage
from skimage.measure import regionprops
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV

from config import get_area
from models.arrows import SolidArrow
from models.exceptions import NotAnArrowException
from models.reaction import ReactionStep,Conditions,Reactant,Product,Intermediate
from models.segments import Rect, Panel, Figure
from models.utils import Point, Line
from utils.processing import approximate_line, create_megabox, merge_rect, pixel_ratio, binary_close, binary_floodfill, pad
from utils.processing import binary_tag, get_bounding_box, postprocessing_close_merge, erase_elements, crop, belongs_to_textline, is_boundary_cc, find_textline_threshold
from utils.processing import is_small_textline_character
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
            print(f'approximating between points {p1_scanrow, p1_scancol} and {p2_scanrow, p2_scancol}')
            line = approximate_line(Point(row=p1_scanrow, col=p1_scancol), Point(row=p2_scanrow, col=p2_scancol))
            overlapping_panels = [panel for panel in panels if panel.overlaps(line)]

            if overlapping_panels:
                overlapped.update(overlapping_panels)
                boredom_index = 0
            else:
                boredom_index +=1

            if boredom_index >= 5: #Nothing found in the last five steps
                break
    print(f'rows: {rows}')
    print(f'cols: {columns}')
    plt.imshow(fig.img, cmap=plt.cm.binary)
    plt.scatter(columns, rows, c='y', s=1)
    plt.savefig('destination_path.jpg', format='jpg', dpi=1000)
    plt.show()

    conditions_text = scan_conditions_text(fig,overlapped,arrow)

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

        panels_dict = assign_to_nearest(panels,all_conditions,ccs_reacts_prods['all_ccs_reacts'],ccs_reacts_prods['all_ccs_prods'])

        control_set.update(*(value for value in panels_dict.values())) # The unpacking looks ugly

        conditions = Conditions(connected_components=all_conditions[idx])
        reacts = panels_dict['reactants']
        prods = panels_dict['products']
        if global_skel_pixel_ratio > 0.02 : #Original kernel size < 6
            reacts = postprocessing_close_merge(fig, reacts)
            prods = postprocessing_close_merge(fig, prods)
            log.debug('Postprocessing closing and merging finished.')

        reacts=Reactant(connected_components=reacts)
        prods = Product(connected_components=prods)
        steps.append(ReactionStep(arrow,reacts,prods, conditions))
        #print('panels:', panels)
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
    #print('assign: conditions set: ', conditions)
    conditions_ccs = [cc for inner_set in conditions for cc in inner_set]
    classified_ccs = set((*conditions_ccs, *reactants, *products))
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
    extended_boundary_horizontal = 50
    conditions = create_megabox(conditions)
    fig = erase_elements(fig,[arrow])
    raw_conditions_region = crop(fig.img, conditions.left, conditions.right,
               conditions.top, conditions.bottom)['img']
    p = 5 #padding amount
    raw_conditions_region = pad(raw_conditions_region,[(p,p),(p,p)],'constant',constant_values=0)
    #raw_conditions_region = binary_close(Figure(raw_conditions_region),1).img
    labelled = binary_tag(Figure(raw_conditions_region))
    ccs = get_bounding_box(labelled)
    # bottoms = [cc.bottom for cc in ccs]
    # bottoms.sort()
    #
    # bottom_count = Counter(bottoms)
    # #bottom_count = Counter({value:count for value, count in bottom_count.items() if count>1})
    # bottoms = np.array([item for item in bottom_count.elements()]).reshape(-1,1)
    #
    # #counts_values = np.array([[bottom, count] for bottom, count in bottom_count.items()]).reshape(-1,1)
    # # logprobs=[]
    # # bandwiths = np.linspace(.1,5,500)
    # # for bandwith in bandwiths:
    # #     kde = KernelDensity(bandwith,kernel='tophat')
    # #     kde.fit(bottoms)
    # #     logprob = kde.score(bottoms)
    # #     logprobs.append(logprob)
    #
    # grid = GridSearchCV(KernelDensity(),
    #                     {'bandwidth': np.linspace(0.005, 2.0, 100)},
    #                     cv=5)  # 20-fold cross-validation
    # grid.fit(bottoms)
    # best_bw = grid.best_params_['bandwidth']
    # kde = KernelDensity(best_bw,kernel='exponential')
    # kde.fit(bottoms)
    #
    # print(f'params: {kde.get_params()}')
    # s = np.linspace(0, raw_conditions_region.shape[0],raw_conditions_region.shape[0]+1)
    # e = kde.score_samples(s.reshape(-1,1))
    #
    # plt.hist(bottoms,bins=20,range=[0,raw_conditions_region.shape[0]])
    # plt.show()
    # # plt.plot(s, e)
    # # plt.show()
    #
    # heights = [cc.bottom-cc.top for cc in ccs]
    # mean_height = np.mean(heights,dtype=np.uint32)
    # kde_trial_lower, _ = find_peaks(e,distance=mean_height)
    # np_peaks = np.diff(np.sign(np.diff(e)))
    # np_peaks = np.where(np_peaks < 0)[0] #need to extract the first array
    # print(f'np peaks: {np_peaks}')
    # kde_trial_lower.sort()
    # plt.plot(s, e)
    # plt.scatter(kde_trial_lower,[0 for elem in kde_trial_lower])
    # plt.scatter(np_peaks, [0 for elem in np_peaks])
    # plt.show()
    # line_buckets = []
    # bottoms = bottoms.reshape(-1)
    # print(f'bottoms: {bottoms}')
    # for peak in kde_trial_lower:
    #     bucket = [elem for elem in bottoms if elem in range(peak-3, peak+3)]
    #     line_buckets.append(bucket)
    #
    # print(f'list of buckets: {line_buckets}')


    #print(f'bottom count: {bottom_count}')
    # plt.imshow(raw_conditions_region)
    # plt.show()

    upper, lower = identify_textlines(ccs,raw_conditions_region)

    #upper = [bottom_textline-mean_height for bottom_textline in lower]
    # TODO: Make sure there is equal number of upper and lower lines and that they have approximately consistent height
    print(f'lower before transformation: {lower}')
    search_region = Rect(conditions.left-extended_boundary_horizontal, conditions.right+extended_boundary_horizontal,
               conditions.top-extended_boundary_vertical, conditions.bottom+extended_boundary_vertical)
    print(f'search region: {search_region}')
    crop_dct = crop(fig.img, search_region.left, search_region.right, search_region.top, search_region.bottom)
    roi = crop_dct['img']
    #roi = binary_close(Figure(roi),1)
    #roi = roi.img
    boundaries = crop_dct['rectangle'] #This is necessary in case a smaller area is cropped
    print(f'actual boundaries :{boundaries}')
    #Move lines to the new extended region
    vertical_shift = extended_boundary_vertical if search_region.top > 0 else conditions.top #if still within the image after expansion,
    # then choose the extended boundary, else the shift is the distance from top as crop cannot goe beyond 0
    # print(f'horizontal, vertical: {horizontal_shift, vertical_shift}')
    lower = [row+vertical_shift-p for row in lower]
#    kde_lower = [kde_trial_lower+vertical_shift-p for row in kde_trial_lower]
    print(f'lower in the search region: {lower}')
    upper = [row+vertical_shift-p for row in upper]
    f, ax = plt.subplots()
    ax.imshow(roi)
    for line in lower:
       ax.plot([i for i in range(roi.shape[1])],[line for i in range(roi.shape[1])],color='r')
    for line in upper:
       ax.plot([i for i in range(roi.shape[1])],[line for i in range(roi.shape[1])],color='b')

    # for line in kde_lower:
    #     ax.plot([i for i in range(roi.shape[1])], [line for i in range(roi.shape[1])], color='m')
    # plt.show()

    upper.sort()
    lower.sort()
    textlines = [Panel(left=0, right=roi.shape[1],top=top_line, bottom=bottom_line)
                 for top_line, bottom_line in zip(upper,lower)]
    #print('textlines:...')
    #print(textlines)

    f, ax = plt.subplots()

    for panel in textlines:
       rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='g',edgecolor='b',alpha=0.7)
       ax.add_patch(rect_bbox)

    labelled = binary_tag(Figure(roi))
    ax.imshow(labelled.img)
    ccs = get_bounding_box(labelled)
    mean_character_area = np.average([cc.area for cc in ccs])
    text_candidate_buckets =[]
    # print(f'textlines: {textlines}')
    # ax.imshow(roi)
    for textline in textlines:
        textline_text_candidates = []

        for cc in ccs:
            # if belongs_to_textline(roi,cc,textline):
            if textline.overlaps(cc):
                textline_text_candidates.append(cc)

            # elif is_small_textline_character(roi,cc,mean_character_area,textline):
            #     if cc not in textline_text_candidates: #avoid doubling
            #         textline_text_candidates.append(cc)

        textline_text_candidates = filter_distant_text_character(textline_text_candidates,textline)
        text_candidate_buckets.append(textline_text_candidates)

    colors =['g','r','b','m','w','y','k','r','g']
    c=-1
    for textline in text_candidate_buckets:
        c+=1
        for panel in textline:
            rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                  facecolor='none', edgecolor=colors[c])
            ax.add_patch(rect_bbox)
    plt.show()
    print(f'textline buckets: {text_candidate_buckets}')
    text_candidates = [element for textline in text_candidate_buckets for element in textline]
    print(f'text candidates: {text_candidates}' )
    #text_candidates = [cc for cc in text_candidates if not is_boundary_cc(roi,cc)]
    # for panel in text_candidates:
    #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='b')
    #     ax.add_patch(rect_bbox)
    # plt.show()
    # TODO: Add another condition that entire original cc is in the crop

    #Transform boxes back into the main image coordinate system
    text_elements =[]
    for element in text_candidates:
        height = element.bottom - element.top
        width = element.right - element.left
        p = Panel(left=boundaries.left+element.left, right=boundaries.left+element.left+width,
                           top=boundaries.top+element.top, bottom=boundaries.top+element.top+height)
        text_elements.append(p)
    return text_elements

def identify_textlines(ccs,raw_cond):
    bottom_boundaries = [cc.bottom for cc in ccs]
    bottom_boundaries.sort()

    bottom_count = Counter(bottom_boundaries)
    # bottom_count = Counter({value:count for value, count in bottom_count.items() if count>1})
    bottom_boundaries = np.array([item for item in bottom_count.elements()]).reshape(-1, 1)

    little_data = len(ccs) < 10
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.005, 2.0, 100)},
                        cv=(len(ccs) if little_data else 10))  # 20-fold cross-validation
    grid.fit(bottom_boundaries)
    best_bw = grid.best_params_['bandwidth']
    kde = KernelDensity(best_bw, kernel='exponential')
    kde.fit(bottom_boundaries)

    # print(f'params: {kde.get_params()}')
    rows= np.linspace(0, raw_cond.shape[0], raw_cond.shape[0] + 1)
    logp = kde.score_samples(rows.reshape(-1, 1))

    # plt.hist(bottom_boundaries, bins=20, range=[0, raw_cond.shape[0]])
    # plt.show()
    # plt.plot(s, e)
    # plt.show()

    heights = [cc.bottom - cc.top for cc in ccs]
    mean_height = np.mean(heights, dtype=np.uint32)
    kde_trial_lower, _ = find_peaks(logp, distance=mean_height)
    kde_trial_lower.sort()
    plt.plot(rows, logp)
    plt.xlabel('Row')
    plt.ylabel('logP(textline)')
    plt.scatter(kde_trial_lower, [0 for elem in kde_trial_lower])
    plt.show()
    line_buckets = []
    for peak in kde_trial_lower:
        bucket = [cc for cc in ccs if cc.bottom in range(peak - 3, peak + 3)]
        line_buckets.append(bucket)

    # print(f'list of buckets: {line_buckets}')
    # print(len(line_buckets))
    top_lines = []
    # print(kde_trial_lower)
    for bucket, peak in zip(line_buckets, kde_trial_lower):
        mean_height = np.max([elem.bottom-elem.top for elem in bucket])
        top_line = peak - mean_height
        top_lines.append(top_line)

    bottom_lines = kde_trial_lower
    top_lines.sort()
    bottom_lines.sort()

    return (top_lines, bottom_lines)

def filter_distant_text_character(ccs, textline):
    data = np.array([cc.center for cc in ccs]).reshape(-1,2)
    char_size = np.max([cc.diagonal_length for cc in ccs])

    db = DBSCAN(eps=2*char_size, min_samples=3,metric='minkowski', p=4).fit(data)
    labels = db.labels_
    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f'number of noise ccs detected: {n_noise}')
    print(f'number of detected_clusters: {n_labels}')
    print(f'number of original ccs: {len(ccs)}')

    import matplotlib.pyplot as plt
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
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

    return [cc for cc, label in zip(ccs,labels) if label !=-1]

