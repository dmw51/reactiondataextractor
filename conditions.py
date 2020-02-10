# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from collections import namedtuple, Counter
import copy
from itertools import chain
import logging
import numpy as np
import matplotlib.pyplot as plt
import re

from scipy.signal import find_peaks
from skimage.morphology import binary_dilation, disk
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from actions import skeletonize_area_ratio
from models.segments import Rect, Figure, TextLine
from models.utils import Point
from utils.processing import approximate_line, create_megabox
from utils.processing import (binary_tag, get_bounding_box, erase_elements, belongs_to_textline, is_boundary_cc,
crop_rect, transform_panel_coordinates_to_expanded_rect, transform_panel_coordinates_to_shrunken_region, label_and_get_ccs)
from chemschematicresolver.ocr import read_conditions
from chemdataextractor.doc import Paragraph, Span
from matplotlib.patches import Rectangle

log = logging.getLogger(__name__)

DEFAULT_VALUES_STRING = r'((?:\d\.)?\d{1,2})'  # Used for parsing recognized text


def parse_conditions(fig, arrow, panels, scan_stepsize=10, scan_steps=10):
    """

    :param fig:
    :param arrow:
    :param panels:
    :param scan_stepsize:
    :param scan_steps:
    :return:
    """
    textlines = find_reaction_conditions(fig, arrow, panels, scan_stepsize, scan_steps)
    recognised_text = []
    for textline in textlines:
        textline.adjust_left_right()
        textline.adjust_top_bottom()
        recognised = read_conditions(erase_elements(fig, [arrow]),
                                 textline)  # list of tuples (TextBlock, ocr_confidence)
        recognised_text.append(recognised)
    print(f'recognised: {recognised_text}')
    conf_lines = [text_block for text_block, confidence in recognised_text if confidence >= 0.65]

    for line in conf_lines:
        print(line.text)

    for line in conf_lines:
        cems = identify_chemicals(line.text[0])
        chem_dct = contextualize_cems(line.text[0], cems)
        print(chem_dct)
    # text = Paragraph(text)
    # cems = identify_chemicals(text)
    # cems_info = contextualize_cems(text, cems)
    # print(cems_info)


def find_reaction_conditions(fig, arrow, panels, stepsize=10, steps=10):
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
    sorted_pts_col = sorted(arrow.line, key= lambda pt: pt.col)
    p1, p2 = sorted_pts_col[0], sorted_pts_col[-1]
    # Squeeze the two points in the second dimension to avoid overlap with structures on either side

    overlapped = set()
    rows = []
    columns = []
    for direction in range(2):
        boredom_index = 0 # increments if no overlaps found. if nothing found in 5 steps,
        # the algorithm breaks from a loop
        increment = (-1) ** direction
        p1_scancol, p2_scancol = p1.col, p2.col
        p1_scanrow, p2_scanrow = p1.row, p2.row

        for step in range(steps):
            # Reduce the scanned area proportionally to distance from arrow
            p1_scancol = int(p1_scancol * 1.01) # These changes dont work as expected - need to check them
            p2_scancol = int(p2_scancol * .99)

            # print('p1: ',p1_scanrow,p1.col)
            # print('p2: ', p2_scanrow,p2.col)
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

            if boredom_index >= 5: # Nothing found in the last five steps
                break
    # print(f'rows: {rows}')
    # print(f'cols: {columns}')
    # plt.imshow(fig.img, cmap=plt.cm.binary)
    # plt.scatter(columns, rows, c='y', s=1)
    # plt.savefig('destination_path.jpg', format='jpg', dpi=1000)
    # plt.show()

    conditions_text = scan_conditions_text2(fig,overlapped,arrow)

    return set(conditions_text) # Return as a set to allow handling along with product and reactant sets


def scan_conditions_text2(fig, conditions, arrow):
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
    top_boundaries, bottom_boundaries = identify_textlines(ccs, search_region.img)

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

    print(f'buckets: {text_candidate_buckets}')
    print(f'buckets after filtering: {text_candidate_buckets}')
    print(f'example textline ccs2: {text_candidate_buckets[0].connected_components}')
    if len(text_candidate_buckets) > 2:
        text_candidate_buckets = isolate_full_text_block(text_candidate_buckets, arrow)
    print(f'buckets: {text_candidate_buckets}')

    transformed_textlines= []
    for textline in text_candidate_buckets:
        textline.connected_components = transform_panel_coordinates_to_expanded_rect(crop_dct['rectangle'],
                                                                          Rect(0,0,0,0), textline.connected_components)
        transformed_textlines.append(textline)

    return transformed_textlines


def attempt_remove_structure_parts(cropped_img, text_ccs):
    crop_no_letters = erase_elements(Figure(cropped_img),text_ccs)
    # f, ax = plt.subplots()
    # ax.imshow(crop_no_letters.img)
    # for panel in text_ccs:
    #    rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left,
    #    panel.bottom-panel.top, facecolor='g',edgecolor='b',alpha=0.7)
    #    ax.add_patch(rect_bbox)
    # ax.set_title('characters removed')
    # plt.show()
    skel_pixel_ratio = skeletonize_area_ratio(Figure(cropped_img),Rect(0,cropped_img.shape[1], 0, cropped_img.shape[0]))
    print(f'the skel-pixel ratio is {skel_pixel_ratio}')
    closed = binary_dilation(crop_no_letters.img,selem=disk(4)) #Decide based on skel-pixel ratio
    labelled = binary_tag(Figure(closed))
    ccs = get_bounding_box(labelled)
    structure_parts = [cc for cc in ccs if is_boundary_cc(cropped_img,cc)]
    crop_no_structures = erase_elements(Figure(cropped_img),structure_parts)

    # f, ax = plt.subplots()
    # ax.imshow(crop_no_structures.img)
    # for panel in text_ccs:
    #    rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left,
    #    panel.bottom-panel.top, facecolor='none',edgecolor='b')
    #    ax.add_patch(rect_bbox)
    # ax.set_title('structures removed')
    # plt.show()

    return crop_no_structures


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
    # plt.plot(rows, logp)
    # plt.xlabel('Row')
    # plt.ylabel('logP(textline)')
    # plt.scatter(kde_lower, [0 for elem in kde_lower])
    # plt.show()
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
    # textlines = [TextLine(None,top,bottom) for top, bottom in zip(top_lines,bottom_lines)]
    return (top_lines, bottom_lines)


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

        textline_text_candidates = filter_distant_text_character(textline_text_candidates)
        print(f'textline text cands: {textline_text_candidates}')
        if transform_from_crop:
            textline_text_candidates = transform_panel_coordinates_to_expanded_rect(
                crop_rect, Rect(0,0,0,0), textline_text_candidates) #Relative only so a placeholder Rect is the input

        if textline_text_candidates: #If there are any candidates
            textline.connected_components = textline_text_candidates
            print(f'components: {textline.connected_components}')
            text_candidate_buckets.append(textline)

    return text_candidate_buckets


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
    # center_area_ratio = np.mean(mean_char_area + mean_char_size)
    # max_r = np.sqrt((mean_char_area+3*std_area)**2 + mean_char_size**2)
    db = DBSCAN(eps=mean_textline_height*4, min_samples=2).fit(data)
    labels = db.labels_
    print(f'labels: {labels}')
    main_cluster = [textline for textline, label in zip(textlines, labels) if label == 0]
    print(f'found cluster: {main_cluster}')

    return main_cluster


def filter_distant_text_character(ccs):
    if not ccs:
        return []

    data = np.array([cc.center for cc in ccs]).reshape(-1, 2)
    char_size = np.max([cc.diagonal_length for cc in ccs])   # This is about 25-30 usually

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
    cc_in_shrunken_region = transform_panel_coordinates_to_shrunken_region(crop_region, cc)[0]
    ccs = label_and_get_ccs(Figure(cropped_img))
    print(ccs)
    small_cc = True if cc.area < mean_character_area else False

    if small_cc:
        print(cc.area)
        closest_ccs = sorted([(other_cc, other_cc.separation(cc_in_shrunken_region))
                             for other_cc in ccs if other_cc.area > 1.3 * cc.area], key=lambda elem: elem[1])
        # Calculate separation, sort,
        # then choose the smallest non-zero (index 1) separation
    else:
        closest_ccs = sorted([(other_cc, other_cc.separation(cc_in_shrunken_region))
                             for other_cc in ccs], key=lambda elem: elem[1])

    if len(closest_ccs) > 1:
        closest_cc = closest_ccs[1]
    else:
        return None

    if closest_cc[1] > 2 * mean_character_diagonal:
        return None   # Too far away

    closest_cc_transformed = transform_panel_coordinates_to_expanded_rect(crop_region, fig.img, [closest_cc[0]])[0]
    return cc, closest_cc_transformed

def identify_chemicals(sentence):
    cems = sentence.cems
    #cems = [cem.text for cem in cems]
    other_identifiers = r'(?<!\w)([A-Z]+)(?!\w)(?!\))' # Up to two capital letters? Just a single one?
    number_identifiers = r'(?:^| )(?<!\w)([1-9])(?!\w)(?!\))(?:$|[, ])(?![A-Za-z])'
    #number_identifiers matches the following:
    #1, 2, 3, three numbers as chemical identifiers
    # CH3OH, 5, 6 (5 equiv) 5 and 6 in the middle only
    # 5 5 equiv  first 5 only
    # A 5 equiv -no matches
    entity_mentions_letters = re.finditer(other_identifiers, sentence.text)
    entity_mentions_numbers = re.finditer(number_identifiers, sentence.text)
    numbers_letters_span = [Span(e.group(1), e.start(), e.end()) for e in chain(entity_mentions_numbers, entity_mentions_letters)]
    all_mentions = [mention for mention in chain (cems, numbers_letters_span)
                    if mention]
    return all_mentions


def contextualize_cems(sentence, all_cems):
    catalysis_info = parse_catalysis(sentence, all_cems)
    print(f'catalysts: {catalysis_info}')
    remaining_species = [species for species in all_cems if all(species != cat['Species'] for cat in catalysis_info)]
    print(remaining_species)
    auxilliary_chemicals = parse_aux_chemicals(sentence, remaining_species)
    print(f'auxilliary chemicals: {auxilliary_chemicals}')
    remaining_species = [species for species in remaining_species if
                         all(species != aux['Species'] for aux in auxilliary_chemicals)]
    print(f'remaining chemicals: {remaining_species}')
    remaining_species = {'Species': remaining_species}

    return {'catalysts': catalysis_info, 'aux_chemicals':auxilliary_chemicals, 'other': remaining_species}


def parse_aux_chemicals(sentence, cems):
    aux_units = r'(equiv(?:alents?)?\.?)'
    aux_values = DEFAULT_VALUES_STRING
    aux_str = re.compile(aux_values + r'\s?' + aux_units)

    auxiliaries = []

    for entity in re.finditer(aux_str, sentence.text):
        print(f'entity: {entity}')
        closest_cem = sorted([(cem, entity.start() - cem.start) for cem in cems if cem.start < entity.start()],
                             key=lambda x: x[1])[0] #start is a callabe in re
        closest_cem = closest_cem[0]
        auxiliaries.append({'Species': closest_cem, 'Value': entity.group(1), 'Units': entity.group(2)})
        global parsed_tokens
        parsed_tokens.extend(group for group in entity.groups() if group)
        parsed_tokens.append(closest_cem.text)

    return auxiliaries

def parse_catalysis(sentence, cems):
    cat_units1= r'(mol\s?%)'

    cat_values = DEFAULT_VALUES_STRING

    cat_str = re.compile(cat_values + r'\s?' + cat_units1)


    print(cems)
    catalysts = []
    for entity in re.finditer(cat_str, sentence.text):

        closest_cem = sorted([(cem,entity.start() - cem.start) for cem in cems if cem.start < entity.start()],
                             key=lambda x: x[1])[0]
        closest_cem = closest_cem[0]
        print(type(closest_cem))
        print(f'closest cem: {closest_cem}')
        catalysts.append({'Species': closest_cem, 'Value': entity.group(1), 'Units': entity.group(2)})
        global parsed_tokens
        parsed_tokens.extend(group for group in entity.groups() if group)
        parsed_tokens.append(closest_cem.text)
    return catalysts