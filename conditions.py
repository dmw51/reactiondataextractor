# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pprint import pprint

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

from actions import skeletonize_area_ratio, detect_structures
from correct import Correct
from models.reaction import Conditions
from models.segments import Rect, Figure, TextLine
from models.utils import Point
from utils.processing import approximate_line, create_megabox
from utils.processing import (binary_tag, get_bounding_box, erase_elements, belongs_to_textline, is_boundary_cc,
                              isolate_patches, crop_rect, transform_panel_coordinates_to_expanded_rect, transform_panel_coordinates_to_shrunken_region,
                              label_and_get_ccs)


from ocr import read_conditions, read_isolated_conditions
from chemdataextractor.doc import Paragraph, Span
from matplotlib.patches import Rectangle

log = logging.getLogger(__name__)


class ConditionParser:
    """
    This class is used to parse conditions text. It is composed of several methods to facilitate parsing recognised text
    using formal grammars.
    """

    """
    The following strings define formal grammars to detect catalysts (cat) and co-reactants (co) based on their units.
    Species which fulfill neither criterion can be parsed as `other_chemicals`. `default_values` is also defined to help 
    parse both integers and floating-point values.
    """
    default_values = r'((?:\d\.)?\d{1,2})'
    cat_units = r'(mol\s?%)'
    co_units = r'(equiv(?:alents?)?\.?)'

    def __init__(self, sentences):
        self.sentences = sentences  # sentences are CDE Sentence objects

    def parse_conditions(self):
        parse_fns = [ConditionParser._parse_coreactants, ConditionParser._parse_catalysis,
                     ConditionParser._parse_other_species, ConditionParser._parse_other_conditions]
        conditions_dct = {'catalysts': None, 'coreactants': None, 'other species': None, 'temperature':None,
                          'pressure': None, 'time': None, 'yield': None}
        coreactants_lst = []
        catalysis_lst = []
        other_species_lst = []
        for sentence in self.sentences:
            parsed = [parse(sentence) for parse in parse_fns]

            coreactants_lst.extend(parsed[0])
            catalysis_lst.extend(parsed[1])
            other_species_lst.extend(parsed[2])
            conditions_dct.update(parsed[3])

        conditions_dct['coreactants'] = coreactants_lst
        conditions_dct['catalysts'] = catalysis_lst
        conditions_dct['other species'] = other_species_lst
        pprint(conditions_dct)
        return conditions_dct

    @staticmethod
    def _identify_species(sentence):

        formulae_identifiers = r'(?<!°)(\(?\b(?:[A-Z]+[a-z]{0,2}[0-9]{0,2}\)?\d?)+\b\)?\d?)'  # A sequence of capital
        # letters between which some lowercase letters and digits are allows, optional brackets
        # cems = [cem.text for cem in cems]
        letter_base_identifiers = r'((?<!°)\b[A-Z]{1,4}\b)'  # Up to four capital letters? Just a single one?

        number_identifiers = r'(?:^| )(?<!\w)([1-9])(?!\w)(?!\))(?:$|[, ])(?![A-Za-z])'
        # number_identifiers matches the following:
        # 1, 2, 3, three numbers as chemical identifiers
        # CH3OH, 5, 6 (5 equiv) 5 and 6 in the middle only
        # 5 5 equiv  first 5 only
        # A 5 equiv -no matches
        entity_mentions_formulae = re.finditer(formulae_identifiers, sentence.text)
        entity_mentions_letters = re.finditer(letter_base_identifiers, sentence.text)

        entity_mentions_numbers = re.finditer(number_identifiers, sentence.text)

        numbers_letters_span = [Span(e.group(1), e.start(), e.end()) for e in
                                chain(entity_mentions_formulae, entity_mentions_numbers, entity_mentions_letters)]

        all_mentions = [mention for mention in chain(entity_mentions_formulae, numbers_letters_span)
                        if mention]

        return list(set(all_mentions))

    @staticmethod
    def _parse_coreactants(sentence):
        co_values = ConditionParser.default_values
        co_str = re.compile(co_values + r'\s?' + ConditionParser.co_units)

        return ConditionParser._find_closest_cem(sentence, co_str)


    @staticmethod
    def _parse_catalysis(sentence):
        cat_values = ConditionParser.default_values
        cat_str = re.compile(cat_values + r'\s?' + ConditionParser.cat_units)

        return ConditionParser._find_closest_cem(sentence, cat_str)

    @staticmethod
    def _parse_other_species(sentence):
        cems = ConditionParser._identify_species(sentence)
        other_species_if_end = r'(?:,|\.|$|\s)\s?(?!\d)'

        other_species = []
        for cem in cems:
            species_str = re.compile('(' + cem.text + ')' + other_species_if_end)
            species = re.search(species_str, sentence.text)
            if species and species.group(1) == cem.text:
                other_species.append(cem.text)

        return other_species

    @staticmethod
    def _parse_other_conditions(sentence):
        other_dct = {}
        parsed = [ConditionParser._parse_temperature(sentence), ConditionParser._parse_time(sentence),
                  ConditionParser._parse_pressure(sentence), ConditionParser._parse_yield(sentence)]
        if parsed[0]:
            other_dct['temperature'] = parsed[0]  # Create the key only if temperature was parsed

        if parsed[1]:
            other_dct['time'] = parsed[1]

        if parsed[2]:
            other_dct['pressure'] = parsed[2]

        if parsed[3]:
            other_dct['yield'] = parsed[3]

        return other_dct

    @staticmethod
    def _find_closest_cem(sentence, parse_str):
        matches = []

        for match in re.finditer(parse_str, sentence.text):
            match_start = match.group(1)
            match_start_idx = [idx for idx, token in enumerate(sentence.tokens) if token.text == match_start][0]

            length_condition = len(sentence.tokens[:match_start_idx]) >= 2
            comma_delimiter_condition = sentence.tokens[match_start_idx-2].text != ','

            if length_condition and comma_delimiter_condition:
                species = sentence.tokens[match_start_idx - 2:match_start_idx]
                species = ' '.join(token.text for token in species)
            else:
                species = sentence.tokens[match_start_idx - 1]

            matches.append({'Species': species, 'Value': float(match.group(1)), 'Units': match.group(2)})

        return matches


    @staticmethod
    def _parse_time(sentence):  # add conditions to add the parsed data
        t_values = ConditionParser.default_values
        t_units = r'(h(?:ours?)?|m(?:in)?|s(?:econds)?|days?)'
        time_str = re.compile(r'(?<!\w)' + t_values + r'\s?' + t_units + r'(?=$|\s?,)')
        time = re.search(time_str, sentence.text)
        if time:
            return {'Value': float(time.group(1)), 'Units': time.group(2)}
        else:
            log.info('Time was not found for...')

    @staticmethod
    def _parse_temperature(sentence):
        # The following formals grammars for temperature and pressure are quite complex, but allow to parse additional
        # generic descriptors like 'heat' or 'UHV' in `.group(1)'
        t_units = r'\s?(?:o|O|0|°)C|K'   # match 0C, oC and similar, as well as K

        t_value1 = r'\d{1,4}' + r'\s?(?=' + t_units + ')'  # capture numbers only if followed by units
        t_value2 = r'rt'
        t_value3 = r'heat'

        # Add greek delta?
        t_or = re.compile('(' + '|'.join((t_value1, t_value2, t_value3 ))+ ')' + '(' + t_units + ')' + '?', re.I)
        temperature = re.search(t_or, sentence.text)
        if temperature:
            units = temperature.group(2) if temperature.group(2) else 'N/A'
            try:
                return {'Value': float(temperature.group(1)), 'Units': units}
            except ValueError:
                return {'Value': temperature.group(1), 'Units': units}   # if value is rt or heat
        else:
            log.info('Temperature was not found for...')


    @staticmethod
    def _parse_pressure(sentence):
        p_units = r'(?:m|h|k|M)?Pa|m?bar|atm'   # match bar, mbar, mPa, hPa, MPa and atm

        p_values1 = r'\d{1,4}' + r'\s?(?=' + p_units + ')'  # match numbers only if followed by units
        p_values2 = r'(?:U?HV)|vacuum'


        p_or = re.compile('(' + '|'.join((p_values1, p_values2 ))+ ')' + '(' + p_units + ')' + '?')
        pressure = re.search(p_or, sentence.text)
        if pressure:
            units = pressure.group(2) if pressure.group(2) else 'N/A'
            return {'Value': float(pressure.group(1)), 'Units': units}
        else:
            log.info('Pressure was not found for...')

    @staticmethod
    def _parse_yield(sentence):

        y_units = r'%'   # match 0C, oC and similar, as well as K

        y_value1 = r'\d{1,2}' + r'\s?(?=' + y_units + ')'  # capture numbers only if followed by units
        y_value2 = r'gram scale'

        # Add greek delta?
        y_or = re.compile('(' + '|'.join((y_value1, y_value2)) + ')' + '(' + y_units + ')' + '?')
        y = re.search(y_or, sentence.text)
        if y:
            units = y.group(2) if y.group(2) else 'N/A'
            try:
                return {'Value': float(y.group(1)), 'Units': units}
            except ValueError:
                return {'Value': y.group(1), 'Units': units}   # if value is 'gram scale'
        else:
            log.info('Yield was not found for...')


def get_conditions(fig, arrow, panels, scan_stepsize=10, scan_steps=10):
    """

    :param fig:
    :param arrow:
    :param panels:
    :param scan_stepsize:
    :param scan_steps:
    :return:
    """
    textlines = find_reaction_conditions(fig, arrow, panels, scan_stepsize, scan_steps)
    recognised = [read_conditions(fig, line, conf_threshold=0.6) for line in textlines]
    print(f'recognised text: {recognised}')
    spell_checked = [Correct(line).correct_text() for line in recognised if line]
    parser = ConditionParser(spell_checked)
    conditions_dct = parser.parse_conditions()
    return Conditions(textlines, conditions_dct)


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
    # panels = [panel for panel in panels if panel.area < 55**2]  # Is this necessary?

    arrow = copy.deepcopy(arrow)
    sorted_pts_col = sorted(arrow.line, key= lambda pt: pt.col)
    p1, p2 = sorted_pts_col[0], sorted_pts_col[-1]

    arrow_length = p1.dist(p2)
    shrink_constant = int(arrow_length/(2*steps))

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
            p1_scancol += shrink_constant # These changes dont work as expected - need to check them
            p2_scancol -= shrink_constant  #  int(p2_scancol * .99)

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
                boredom_index += 1

            if boredom_index >= 5:  # Nothing found in the last five steps
                break
    print(f'rows: {rows}')
    print(f'cols: {columns}')
    # f = plt.figure(figsize=(20,20))
    # ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.imshow(fig.img, cmap=plt.cm.binary)
    # pairs = list(zip(rows, columns))
    # for i in range(0,len(rows)-1,2):
    #     p1 = pairs[i]
    #     p2 = pairs[i+1]
    #     y, x = list(zip(*[p1, p2]))
    #
    #     ax.plot(x, y, 'r')
    # ax.axis('off')
    # plt.savefig('diamond2.tif')


    # plt.imshow(fig.img, cmap=plt.cm.binary)
    # plt.scatter(columns, rows, c='y', s=1)
    # plt.savefig('destination_path.jpg', format='jpg', dpi=1000)
    # plt.show()

    conditions_text = scan_conditions_text(fig, overlapped, arrow)

    return conditions_text # Return as a set to allow handling along with product and reactant sets


def scan_conditions_text(fig, conditions, arrow, debug=False):
    """
    Crops a larger area around raw conditions to look for additional text elements that have
    not been correctly recognised as conditions
    :param Figure fig: analysed figure with binarised image object
    :param iterable of Panels conditions: set or list of raw conditions (output of `find_reaction_conditions`)
    :padam bool debug: debugging mode on/off - enables additional plotting
    :return: Set of Panels containing all recognised conditions
    """

    fig = copy.deepcopy(fig)
    fig = erase_elements(fig, [arrow])  # erase arrow at the very beginning

    conditions_box = create_megabox(conditions)
    # print(f'height: {conditions_box.height}, width: {conditions_box.width}')
    extended_boundary_vertical = conditions_box.height
    extended_boundary_horizontal = 75

    search_region = Rect(conditions_box.left-extended_boundary_horizontal, conditions_box.right+extended_boundary_horizontal,
               conditions_box.top-extended_boundary_vertical, conditions_box.bottom+extended_boundary_vertical)
    # print(f'search region: {search_region}')
    crop_dct = crop_rect(fig.img, search_region)
    search_region = crop_dct['img']  # keep the rectangle boundaries in the other key



    # print('running scan text!')
    # plt.imshow(search_region)
    plt.show()

    initial_ccs_transformed = transform_panel_coordinates_to_shrunken_region(crop_dct['rectangle'],conditions)
    search_region = attempt_remove_structure_parts(search_region, initial_ccs_transformed)

    ccs = label_and_get_ccs(search_region)
    top_boundaries, bottom_boundaries = identify_textlines(ccs, search_region.img)

    textlines = [TextLine(0, search_region.img.shape[1], upper, lower)
                 for upper, lower in zip(top_boundaries, bottom_boundaries)]

    # f = plt.figure(figsize=(20, 20))
    # ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.imshow(search_region.img, cmap=plt.cm.binary)
    # for line in top_boundaries:
    #    ax.plot([i for i in range(search_region.img.shape[1])],[line for i in range(search_region.img.shape[1])],color='b')
    # for line in bottom_boundaries:
    #    ax.plot([i for i in range(search_region.img.shape[1])],[line for i in range(search_region.img.shape[1])],color='r')
    #
    #
    # # print(f'textlines:{textlines}')
    text_candidate_buckets = assign_characters_to_textlines(search_region.img, textlines, ccs)
    # # print(f'example textline ccs: {text_candidate_buckets[0].connected_components}')
    mixed_text_candidates = [element for textline in text_candidate_buckets for element in textline]
    # for panel in mixed_text_candidates:
    #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='r',edgecolor='b', alpha=0.35)
    #     ax.add_patch(rect_bbox)
    #
    # plt.savefig('cond_chars_initial.tif')
    #plt.show()



    remaining_elements = set(ccs).difference(mixed_text_candidates)
    text_candidate_buckets = assign_characters_proximity_search(search_region,
                                                                remaining_elements, text_candidate_buckets)

    # print(f'buckets: {text_candidate_buckets}')
    # print(f'buckets after filtering: {text_candidate_buckets}')
    # print(f'example textline ccs2: {text_candidate_buckets[0].connected_components}')
    if len(text_candidate_buckets) > 2:
        text_candidate_buckets = isolate_full_text_block(text_candidate_buckets, arrow)
    # print(f'buckets: {text_candidate_buckets}')

    transformed_textlines= []
    for textline in text_candidate_buckets:
        textline.connected_components = transform_panel_coordinates_to_expanded_rect(crop_dct['rectangle'],
                                                                          Rect(0,0,0,0), textline.connected_components)
        transformed_textlines.append(textline)

    return transformed_textlines


def attempt_remove_structure_parts(cropped_img, text_ccs):
    """
    Attempt to remove parts of structures from a cropped region containing conditions text.
    :param np.ndarray cropped_img: array representing the cropped region
    :param [Panels] text_ccs: text connected components detected during the raw line scan stage
    :return np.ndarray: crop without the structure parts
    """

    crop_no_letters = erase_elements(Figure(cropped_img),text_ccs)
    # f, ax = plt.subplots()
    # ax.imshow(crop_no_letters.img)
    # for panel in text_ccs:
    #    rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left,
    #    panel.bottom-panel.top, facecolor='g',edgecolor='b',alpha=0.7)
    #    ax.add_patch(rect_bbox)
    # ax.set_title('characters removed')
    # plt.show()
    # skel_pixel_ratio = skeletonize_area_ratio(Figure(cropped_img),Rect(0,cropped_img.shape[1], 0, cropped_img.shape[0]))
    # print(f'the skel-pixel ratio is {skel_pixel_ratio}')
    ccs = label_and_get_ccs(crop_no_letters)
    # print(ccs)
    # print(f'type of ccs: {type(ccs)}')
    # detect_structures(Figure(cropped_img), ccs)
    skel_pixel_ratio = skeletonize_area_ratio(crop_no_letters, crop_no_letters.get_bounding_box()) # currently not used

    # print(f'skel_pixel ratio: {skel_pixel_ratio}')
    closed = binary_dilation(crop_no_letters.img,selem=disk(3)) #Decide based on skel-pixel ratio
    # plt.imshow(closed)
    # plt.show()
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
    # kde_lower.sort()
    # plt.plot(rows, logp, 'r')
    # plt.xlabel('Row')
    # plt.ylabel('logP(textline)')
    # plt.scatter(kde_lower, [0 for elem in kde_lower], c='r', marker='+')
    # plt.savefig('kde_2.tif')
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
        # print(f'textline text cands: {textline_text_candidates}')
        if transform_from_crop:
            textline_text_candidates = transform_panel_coordinates_to_expanded_rect(
                crop_rect, Rect(0,0,0,0), textline_text_candidates) #Relative only so a placeholder Rect is the input

        if textline_text_candidates: #If there are any candidates
            textline.connected_components = textline_text_candidates
            # print(f'components: {textline.connected_components}')
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
        assigned = attempt_assign_small_to_nearest_text_element(img, element, mean_character_area)
        if assigned:
            small_cc, assigned_closest_cc = assigned
            for textline in textlines :
                if assigned_closest_cc in textline:
                    #to_filter_out.add(small_cc)
                    textline.append(small_cc)

    return textlines


def isolate_full_text_block(textlines, arrow):
    # mixed_text_elements = [elem for textline in text_buckets for elem in textline]
    mean_textline_height = np.mean([textline.height for textline in textlines])
    # char_areas = [elem.area for elem in mixed_text_elements]
    # mean_char_area = np.mean(char_areas)
    # std_area = np.std(char_areas)
    # std_area = np.std(char_areas)
    # data = np.array([(*elem.center, elem.area) for elem in mixed_text_elements]).reshape(-1,3)
    data = [textline.center for textline in textlines]
    data.append(arrow.center)
    # print(data)
    # center_area_ratio = np.mean(mean_char_area + mean_char_size)
    # max_r = np.sqrt((mean_char_area+3*std_area)**2 + mean_char_size**2)
    db = DBSCAN(eps=mean_textline_height*4, min_samples=2).fit(data)
    labels = db.labels_
    # print(f'labels: {labels}')
    main_cluster = [textline for textline, label in zip(textlines, labels) if label == 0]
    # print(f'found cluster: {main_cluster}')

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


def attempt_assign_small_to_nearest_text_element(fig, cc, mean_character_area, small_cc=True):
    """
    Crops `fig.img` and does a simple proximity search to determine the closest character.
    :param Figure fig: figure object containing image with the cc panel
    :param Panel cc: unassigned connected component
    :param float mean_character_area: average area of character in the main crop
    :return: tuple (cc, nearest_neighbour) if close enough, else return None (inconclusive search)
    """
    mean_character_diagonal = np.sqrt(2 * mean_character_area)
    expansion = int(3 * mean_character_diagonal)
    crop_region = Rect(cc.left-expansion, cc.right+expansion, cc.top-expansion, cc.bottom+expansion)
    cropped_img = crop_rect(fig.img, crop_region)
    if cropped_img['rectangle'] != crop_region:
        crop_region = cropped_img['rectangle']

    cropped_img = cropped_img['img']
    cc_in_shrunken_region = transform_panel_coordinates_to_shrunken_region(crop_region, cc)[0]
    ccs = label_and_get_ccs(Figure(cropped_img))
    # print(ccs)
    small_cc = True if cc.area < 1.2 * mean_character_area else False
    if small_cc:
        # Calculate the separation between the small cc and the boundary of nearest cc which is approximated by
        # separation minus half of diagonal length
        close_ccs= sorted([(other_cc, other_cc.separation(cc_in_shrunken_region))
                           for other_cc in ccs if other_cc.area > cc.area], key=lambda elem: elem[1])

        # Calculate separation, sort,
        # then choose the smallest non-zero (index 1) separation
        # Check whether they share a textline?
        if len(close_ccs) > 1:
            vertical_overlap = [other_cc[0].overlaps_vertically(cc_in_shrunken_region) for other_cc in close_ccs]
            filtered_close_ccs =[cc for cc, overlap in zip(close_ccs, vertical_overlap) if overlap]
            if not filtered_close_ccs:
                return None
            closest_cc = filtered_close_ccs[0]
        else:
            log.info('No suitable neighbours were found to assign a small character cc')
            return None

        if closest_cc[1] > 1.5 * mean_character_diagonal:
            return None  # Too far away
        closest_cc_transformed = transform_panel_coordinates_to_expanded_rect(crop_region, fig.img, [closest_cc[0]])[0]
        return cc, closest_cc_transformed

    else:
        return None
