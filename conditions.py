# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pprint import pprint

from collections import  Counter
import copy
from itertools import chain
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from scipy.signal import find_peaks, argrelmin
from skimage.morphology import binary_dilation, disk
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from actions import skeletonize_area_ratio, detect_structures, find_nearby_ccs, extend_line
from correct import Correct
from models.reaction import Conditions
from models.arrows import SolidArrow
from models.segments import Rect, Figure, TextLine, Crop, FigureRoleEnum, ReactionRoleEnum
from models.utils import Point, Line, DisabledNegativeIndices
from utils.processing import approximate_line, find_minima_between_peaks
from utils.rectangles import create_megabox
from utils.processing import (binary_tag, get_bounding_box, erase_elements, is_boundary_cc,
                              isolate_patches, crop_rect, transform_panel_coordinates_to_expanded_rect, transform_panel_coordinates_to_shrunken_region,
                              label_and_get_ccs)

import settings
from chemschematicresolver.actions import read_diagram_pyosra


from ocr import read_conditions, read_isolated_conditions
from chemdataextractor.doc import Paragraph, Span, Sentence
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
from chemschematicresolver.parse import ChemSchematicResolverTokeniser
from matplotlib.patches import Rectangle

log = logging.getLogger('extract.conditions')

SPECIES_FILE = os.path.join(os.getcwd(),'dict', 'species.txt')

class ConditionParser:
    """
    This class is used to parse conditions text. It is composed of several methods to facilitate parsing recognised text
    using formal grammars.
    """

    """
    The following strings define formal grammars to detect catalysts (cat) and coreactants (co) based on their units.
    Species which fulfill neither criterion can be parsed as `other_chemicals`. `default_values` is also defined to help 
    parse both integers and floating-point values.
    """
    default_values = r'((?:\d\.)?\d{1,3})'
    cat_units = r'(mol\s?%|M|wt\s?%)'
    co_units = r'(eq\.?(?:uiv(?:alents?)?\.?)?|m?L)'

    def __init__(self, sentences, ):

        self.sentences = sentences  # sentences are CDE Sentence objects

    def parse_conditions(self):
        parse_fns = [ConditionParser._parse_coreactants, ConditionParser._parse_catalysis,
                     ConditionParser._parse_other_species, ConditionParser._parse_other_conditions]
        conditions_dct = {'catalysts': None, 'coreactants': None, 'other species': None, 'temperature': None,
                          'pressure': None, 'time': None, 'yield': None}

        coreactants_lst = []
        catalysis_lst = []
        other_species_lst = []
        for sentence in self.sentences:
            parsed = [parse(sentence) for parse in parse_fns]

            coreactants_lst.extend(parsed[0])
            catalysis_lst.extend(parsed[1])
            other_species_lst.extend(ConditionParser._filter_species(parsed))
            conditions_dct.update(parsed[3])

        conditions_dct['coreactants'] = coreactants_lst
        conditions_dct['catalysts'] = catalysis_lst
        conditions_dct['other species'] = other_species_lst
        # pprint(conditions_dct)
        return conditions_dct

    @staticmethod
    def _identify_species(sentence):

        with open(SPECIES_FILE, 'r') as file:
            species_list = file.read().split('\n')

        #formulae_identifiers = r'(\(?\b(?:[A-Z]+[a-z]{0,1}[0-9]{0,2}\)?\d?)+\b\)?\d?)'  # A sequence of capital
        # letters between which some lowercase letters and digits are allowed, optional brackets
        # cems = [cem.text for cem in cems]
        formulae_brackets = r'((?:[A-Z]*\d?[a-z]\d?)\((?:[A-Z]*\d?[a-z]?\d?)*\)?\d?[A-Z]*[a-z]*\d?)*'
        formulae_bracketless = r'(?<!°)\b(?<!\)|\()((?:[A-Z]+\d?[a-z]?\d?)+)(?!\(|\))\b'
        letter_upper_identifiers = r'((?<!°)\b[A-Z]{1,4}\b)(?!\)|\.)'  # Up to four capital letters? Just a single one?
        letter_lower_identifiers = r'(\b[a-z]\b)(?!\)|\.)'  # Accept single lowercase letter subject to restrictions

        number_identifiers = r'(?:^| )(?<!\w)([1-9])(?!\w)(?!\))(?:$|[, ])(?![A-Za-z])'
        # number_identifiers matches the following:
        # 1, 2, 3, three numbers as chemical identifiers
        # CH3OH, 5, 6 (5 equiv) 5 and 6 in the middle only
        # 5 5 equiv  first 5 only
        # A 5 equiv -no matches
        entity_mentions_brackets = re.finditer(formulae_brackets, sentence.text)
        entity_mentions_bracketless = re.finditer(formulae_bracketless, sentence.text)
        entity_mentions_letters_upper = re.finditer(letter_upper_identifiers, sentence.text)
        entity_mentions_letters_lower = re.finditer(letter_lower_identifiers, sentence.text)

        entity_mentions_numbers = re.finditer(number_identifiers, sentence.text)

        spans = [Span(e.group(1), e.start(), e.end()) for e in
                 chain(entity_mentions_brackets, entity_mentions_bracketless,
                       entity_mentions_numbers, entity_mentions_letters_upper,
                       entity_mentions_letters_lower) if e.group(1)]
        slashed_names = []
        for token in sentence.tokens:
            if '/' in token.text:
                slashed_names.append(token)

        all_mentions = ConditionParser._resolve_spans(spans+slashed_names)
        # Add species from the list, treat them as seeds - allow more complex names
        # eg. based on 'pentanol' on the list, allow '1-pentanol'
        species_from_list = [token for token in sentence.tokens if any(species in token.text.lower() for species in species_list
                                                                       if species)]  # except ''
        all_mentions += species_from_list
        return list(set(all_mentions))

    @staticmethod
    def _parse_coreactants(sentence):
        co_values = ConditionParser.default_values
        co_str = co_values + r'\s?' + ConditionParser.co_units

        return ConditionParser._find_closest_cem(sentence, co_str)


    @staticmethod
    def _parse_catalysis(sentence):
        cat_values = ConditionParser.default_values
        cat_str = cat_values + r'\s?' + ConditionParser.cat_units

        return ConditionParser._find_closest_cem(sentence, cat_str)

    @staticmethod
    def _parse_other_species(sentence):
        cems = ConditionParser._identify_species(sentence)
        return [cem.text for cem in cems]
        # other_species_if_end = r'(?:,|\.|$|\s)\s?(?!\d)'
        #
        # other_species = []
        # for cem in cems:
        #     cem = cem.text
        #     species_str = re.compile('(' + re.escape(cem) + ')' + other_species_if_end)
        #     species = re.search(species_str, sentence.text)
        #     if species and species.group(1) == cem:
        #         other_species.append(cem)

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
        phrase = sentence.text
        matches = []
        cwt = ChemWordTokenizer()
        bracketed_units_pat = re.compile(r'\(\s*'+parse_str+r'\s*\)')
        bracketed_units = re.findall(bracketed_units_pat, sentence.text)
        if bracketed_units:   #  remove brackets
            phrase = re.sub(bracketed_units_pat, ' '.join(bracketed_units[0]), phrase)
        # else:
        #     parse_str = re.compile(parse_str)
        for match in re.finditer(parse_str, phrase):
            match_tokens = cwt.tokenize(match.group(0))
            phrase_tokens = cwt.tokenize(phrase)
            match_start_idx = [idx for idx, token in enumerate(phrase_tokens) if match_tokens[0] in token][0]
            match_end_idx = [idx for idx, token in enumerate(phrase_tokens) if match_tokens[-1] in token][0]
            # To simplify syntax above, introduce a new tokeniser that splits full stops more consistently
            #Accept two tokens, strip commas and full stops, especially if one of the tokens
            species = DisabledNegativeIndices(phrase_tokens)[match_start_idx-2:match_start_idx]
            species = ' '.join(token for token in species).strip('()., ')
            if not species:
                try:
                    species = DisabledNegativeIndices(phrase_tokens)[match_end_idx+1:match_start_idx+4]
                    # filter special signs and digits
                    species = map(lambda s: s.strip('., '), species)
                    species = filter(lambda token: token.isalpha(), species)
                    species = ' '.join(token for token in species)
                except IndexError:
                    log.debug('Closest CEM not found for a catalyst/coreactant key phrase')
                    species = ''

            if species:
                    matches.append({'Species': species, 'Value': float(match.group(1)), 'Units': match.group(2)})

    # length_condition = len(sentence.tokens[:match_start_idx]) >= 2
    # comma_delimiter_condition = sentence.tokens[match_start_idx-2].text != ','  # Error-prone; -ve indexing
    #
    # if length_condition and comma_delimiter_condition:  # Accept 2-token species if match preceded
    #     # by at least two tokens and first of them is not a comma
    #     species = sentence.tokens[match_start_idx - 2:match_start_idx]
    #     species = ' '.join(token.text for token in species)
    # else:
    #     species = sentence.tokens[match_start_idx - 1].text
        return matches

    @staticmethod
    def _filter_species(parsed):
        """ If a chemical species has been assigned as both catalyst or coreactant, and `other species`, remove if from
        the latter. Also remove special cases"""
        coreactants, catalysts, other_species, _ = parsed
        combined = [d['Species'] for d in coreactants] + [d['Species'] for d in catalysts]
        # if not coreactants or catalysts found, return unchanged
        if not combined:
            return other_species

        else:
            unaccounted = []
            combined = ' '.join(combined)
            for species in other_species:
                found = re.search(re.escape(species), combined)  # include individual tokens for multi-token names
                if not found and species != 'M':
                    unaccounted.append(species)
            return list(set(unaccounted))

    @staticmethod
    def _resolve_spans(spans):
        span_copy = spans.copy()
        # spans is ~10-15 elements long at most
        for span1 in spans:
            for span2 in spans:
                if span1.text != span2.text:
                    if span1.text in span2.text:
                        try:
                            span_copy.remove(span1)
                        except ValueError:
                            pass
                    elif span2.text in span1.text:
                        try:
                            span_copy.remove(span2)
                        except ValueError:
                            pass

        return span_copy


    @staticmethod
    def _parse_time(sentence):  # add conditions to add the parsed data
        t_values = ConditionParser.default_values
        t_units = r'(h(?:ours?)?|m(?:in)?|s(?:econds)?|days?)'
        time_str = re.compile(r'(?<!\w)' + t_values + r'\s?' + t_units + r'(?=$|\s?,)')
        time = re.search(time_str, sentence.text)
        if time:
            return {'Value': float(time.group(1)), 'Units': time.group(2)}


    @staticmethod
    def _parse_temperature(sentence):

        # The following formals grammars for temperature and pressure are quite complex, but allow to parse additional
        # generic descriptors like 'heat' or 'UHV' in `.group(1)'
        t_units = r'\s?(?:o|O|0|°)C|K'   # match 0C, oC and similar, as well as K

        t_value1 = r'-?\d{1,4}' + r'\s?(?=' + t_units + ')'  # capture numbers only if followed by units
        t_value2 = r'r\.?\s?t\.?'
        t_value3 = r'heat|reflux|room\s?temp'

        # Add greek delta?
        t_or = re.compile('(' + '|'.join((t_value1, t_value2, t_value3 ))+ ')' + '(' + t_units + ')' + '?', re.I)
        temperature = re.search(t_or, sentence.text)
        if temperature:
            units = temperature.group(2) if temperature.group(2) else 'N/A'
            try:
                return {'Value': float(temperature.group(1)), 'Units': units}
            except ValueError:
                return {'Value': temperature.group(1), 'Units': units}   # if value is rt or heat


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


def get_conditions(fig, arrow):
    """

    :param fig:
    :param arrow:
    :param panels:
    :param scan_stepsize:
    :param scan_steps:
    :return:
    """
    textlines, condition_structures = find_step_conditions(fig, arrow)
    [setattr(panel, 'role', ReactionRoleEnum.CONDITIONS) for panel in condition_structures]

    if textlines:
        recognised = [read_conditions(fig, line, conf_threshold=40) for line in textlines]
        print(recognised)
        # print(f'recognised text: {recognised}')
        # spell_checked = [Correct(line).correct_text() for line in recognised if line]
        recognised = [sentence for sentence in recognised if sentence]
        parser = ConditionParser(recognised)
        conditions_dct = parser.parse_conditions()
    else:
        conditions_dct = {}
    return Conditions(textlines, conditions_dct, arrow, condition_structures), condition_structures




def find_step_conditions(fig: Figure, arrow: SolidArrow):
    """
    Finds conditions of a step. Selects a region around an arrow. If the region contains text, scans the text.
    Otherwise it returns None (no conditions found).
    :param Figure fig: Analysed figure
    :param Arrow arrow: Arrow around which the conditions are to be looked for
    :param [Panel,...] panels: Collection of all connected components in ``fig.img``
    :return: Collection [Textline,...] containing characters grouped together as text lines
    """

    structure_panels = [cc.parent_panel for cc in fig.connected_components if cc.role == FigureRoleEnum.STRUCTUREBACKBONE
                        and cc.parent_panel]
    conditions_panels = [panel for panel in structure_panels if belongs_to_conditions(panel, arrow)]


    text_lines = mark_text_lines(fig, arrow, conditions_panels)
    # if not text_lines:
    #     return []

    for text_line in text_lines:
        collect_characters(fig, text_line)
    text_lines = [text_line for text_line in text_lines if text_line.connected_components]


    ##
    ##  temp
    # f = plt.figure()
    # ax = f.add_axes([0.2, 0.2, 0.9, 0.9])
    # ax.imshow(fig.img)
    # c = iter(['y', 'b', 'r', 'g', 'y', 'b', 'r', 'g'])
    # for t in text_lines:
    #     print(f'len cc: {len(t.connected_components)}')
    #     color = next(c)
    #     for _panel in t.connected_components:
    #         rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right - _panel.left, _panel.bottom - _panel.top,
    #                               facecolor='none', edgecolor=color)
    #         ax.add_patch(rect_bbox)
    # plt.show()
    ### end temp

    return text_lines, conditions_panels

##
##  temp
# f = plt.figure()
# ax = f.add_axes([0.2, 0.2, 0.9, 0.9])
# ax.imshow(fig.img)
# c = iter(['y', 'b', 'r', 'g'])
# for t in text_lines:
#     color = next(c)
#     for _panel in t.connected_components:
#         rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right - _panel.left, _panel.bottom - _panel.top,
#                               facecolor='none', edgecolor=color)
#         ax.add_patch(rect_bbox)
# plt.show()


def belongs_to_conditions(structure_panel, arrow):
    """
    Checks if a structure is part of the conditions

    Looks if the ``structure_panel`` center lies close to a line parallel to arrow. Two points equidistant to the arrow
    are chosen and the distance from these is compared to two extreme points of an arrow. If the centre is closer to
    either of the two points (subject to a maximum threshold distance) than to either of the extremes,
    the structure is deemed to be part of the conditions region

    :param Panel structure_panel: Panel object marking a structure (superatoms included)
    :param Arrow arrow: Arrow defining the conditions region
    :return: bool True if within the conditions region else close
    """

    pixels = arrow.pixels
    react_endpoint = pixels[0]
    prod_endpoint = pixels[-1]
    midpoint = pixels[len(pixels)//2]
    parallel_line_dummy = Line([midpoint])

    slope = arrow.line.slope
    parallel_line_dummy.slope = -1/slope if abs(slope) > 0.05 else np.inf
    parallel_1, parallel_2 = extend_line(parallel_line_dummy, extension=react_endpoint.separation(prod_endpoint) // 2)

    closest = min([parallel_1, parallel_2, react_endpoint, prod_endpoint],
                  key=lambda point: structure_panel.separation(point))

    if closest in [parallel_1, parallel_2] and structure_panel.separation(arrow.panel) < 1.0 * np.sqrt(structure_panel.area):
        return True
    else:
        return False


def mark_text_lines(fig, arrow, conditions_panels):
    """
    Isolates conditions around around ``arrow`` in ``fig``.

    Marks text lines first by finding obvious conditions' text characters around an arrow. This scan is also performed
    around `conditions_panels` if any. Using the found ccs, text lines are fitted with kernel density estimates.
    :param Figure fig: analysed figure
    :param SolidArrow arrow: arrow around which the region of interest is centered
    :return: Crop: Figure-like object containing the relevant crop with the arrow removed
    """

    average_height = np.median([cc.height for cc in fig.connected_components])

    areas = [cc.area for cc in fig.connected_components]
    areas.sort()
    condition1 = lambda cc: cc.role != FigureRoleEnum.STRUCTUREAUXILIARY
    if arrow.is_vertical:
        condition2 = lambda cc: cc.top > arrow.top and cc.bottom < arrow.bottom
    else:
        condition2 = lambda cc: cc.left > arrow.left and cc.right < arrow.right

    condition = condition1 and condition2
    middle_pixel = arrow.center_px
    distance_fn = lambda cc: 2.2 * cc.height
    core_ccs = find_nearby_ccs(middle_pixel, fig.connected_components, (3*average_height, distance_fn),
                               condition=condition)
    if not core_ccs:
        for pixel in arrow.pixels[::10]:
            core_ccs = find_nearby_ccs(pixel, fig.connected_components, (2*average_height, distance_fn),
                                       condition=condition)
            if len(core_ccs) > 1:
                break
        else:
            log.warning('No conditions were found in the initial scan. Aborting conditions search...')
            return []

    if conditions_panels:
        for panel in conditions_panels:
            core_ccs += find_nearby_ccs(panel, fig.connected_components, (3 * average_height, distance_fn),
                            condition=condition)
    # f = plt.figure()
    # ax = f.add_axes([0.1,0.1,0.8,0.8])
    # ax.imshow(fig.img)
    # for _panel in core_ccs:
    #     rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right - _panel.left, _panel.bottom - _panel.top,
    #                           facecolor='none', edgecolor='r')
    #     ax.add_patch(rect_bbox)
    # plt.show()
    conditions_region = create_megabox(core_ccs)

    # if arrow.is_vertical:
    #     extension_width = 3 * average_width
    #     extension_height = 3 * average_height
    # else:
    #     slope = arrow.line.slope
    #     extension_height = 3 * abs(np.cos(slope)) * average_height  # "number of text lines"
    #     extension_width = 3 * abs(np.sin(slope)) * average_width   # "number of text chars which go beyond the arrow on either side"
    #
    # extension_height = int(extension_height)
    # extension_width = int(extension_width)



    cropped_region = Crop(erase_elements(fig, conditions_panels), conditions_region) # Do not look at structures

    text_lines = [TextLine(None, None, top, bottom, crop=cropped_region, anchor=anchor) for (top, bottom, anchor) in
                  identify_text_lines(cropped_region)]

    text_lines = [text_line.in_main_figure for text_line in text_lines]

    # for text_line in text_lines:
    #     text_line.find_anchor(core_ccs)

    return text_lines


def collect_characters(fig, text_line):
    """
    Accurately assigns relevant characters in ``fig`` to ``text_line``
    :param Figure fig: analysed figure
    :param TextLine text_line: found text line object
    :return:
    """
    if text_line.crop:
        raise ValueError('Character collection can only be performed in the main figure')


    relevant_ccs = [cc for cc in fig.connected_components if cc.role != FigureRoleEnum.ARROW]
    initial_distance = np.sqrt(np.mean([cc.area for cc in relevant_ccs]))
    distance_fn = settings.DISTANCE_FN_CHARS

    proximity_coeff = lambda cc: .75 if cc.area < np.percentile([cc.area for cc in relevant_ccs], 65) else .4
    condition1 = lambda cc: (abs(text_line.panel.center[1] - cc.center[1]) < proximity_coeff(cc) * text_line.panel.height)
    condition2 = lambda cc: (cc.height < text_line.panel.height * 1.7)
    condition3 = lambda cc: abs(text_line.panel.bottom - cc.bottom) < 0.65 * text_line.panel.height
    condition = lambda cc: condition1(cc) and condition2(cc) and condition3(cc)
    # first condition is cc.center in y direction close to center of text_line. Second is that height is comparable to text_line
    found_ccs = find_nearby_ccs(text_line.anchor, relevant_ccs,(initial_distance, distance_fn),
                                FigureRoleEnum.CONDITIONSCHAR, condition)
    if found_ccs:
        text_line.connected_components = found_ccs
    # def scan_conditions_text2(conditions_crop, arrow):
#     """
#     Given the ``conditions region`` Crop containing step conditions, filters out all irrelevant elements.
#     Uses ``arrow`` in a clustering process
#     :param conditions_region:
#     :param arrow:
#     :return:
#     """
#     # fig = copy.deepcopy(fig)
#     # fig = erase_elements(fig, [arrow])  # erase arrow at the very beginning
#
#     # conditions_box = create_megabox(conditions)
#     # # print(f'height: {conditions_box.height}, width: {conditions_box.width}')
#     # extended_boundary_vertical = conditions_box.height
#     # extended_boundary_horizontal = 75
#     #
#     # search_region = Rect(conditions_box.left-extended_boundary_horizontal, conditions_box.right+extended_boundary_horizontal,
#     #            conditions_box.top-extended_boundary_vertical, conditions_box.bottom+extended_boundary_vertical)
#     # # print(f'search region: {search_region}')
#     # crop_dct = crop_rect(fig.img, search_region)
#     # search_region = crop_dct['img']  # keep the rectangle boundaries in the other key
#
#
#
#     # print('running scan text!')
#     # plt.imshow(search_region)
#     plt.show()
#
#     initial_ccs_transformed =conditions_region.connected_components
#         # transform_panel_coordinates_to_shrunken_region(conditions_region.cropped_rect,
#         #                                                                      conditions_region.connected_components)
#     search_region = attempt_remove_structure_parts(conditions_region.cropped_img, initial_ccs_transformed)
#
#     ccs = label_and_get_ccs(search_region)
#     top_boundaries, bottom_boundaries = identify_text_lines(ccs, search_region.img)
#
#     textlines = [TextLine(0, search_region.img.shape[1], upper, lower)
#                  for upper, lower in zip(top_boundaries, bottom_boundaries)]
#
#     # f = plt.figure(figsize=(20, 20))
#     # ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
#     # ax.imshow(search_region.img, cmap=plt.cm.binary)
#     # for line in top_boundaries:
#     #    ax.plot([i for i in range(search_region.img.shape[1])],[line for i in range(search_region.img.shape[1])],color='b')
#     # for line in bottom_boundaries:
#     #    ax.plot([i for i in range(search_region.img.shape[1])],[line for i in range(search_region.img.shape[1])],color='r')
#     #
#     #
#     # # print(f'textlines:{textlines}')
#     text_candidate_buckets = assign_characters_to_textlines(search_region.img, textlines, ccs)
#     # # print(f'example text_line ccs: {text_candidate_buckets[0].connected_components}')
#     mixed_text_candidates = [element for textline in text_candidate_buckets for element in textline]
#     # for _panel in mixed_text_candidates:
#     #     rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right-_panel.left, _panel.bottom-_panel.top, facecolor='r',edgecolor='b', alpha=0.35)
#     #     ax.add_patch(rect_bbox)
#     #
#     # plt.savefig('cond_chars_initial.tif')
#     #plt.show()
#
#
#
#     remaining_elements = set(ccs).difference(mixed_text_candidates)
#     text_candidate_buckets = assign_characters_proximity_search(search_region,
#                                                                 remaining_elements, text_candidate_buckets)
#
#     # print(f'buckets: {text_candidate_buckets}')
#     # print(f'buckets after filtering: {text_candidate_buckets}')
#     # print(f'example text_line ccs2: {text_candidate_buckets[0].connected_components}')
#     if len(text_candidate_buckets) > 2:
#         text_candidate_buckets = isolate_full_text_block(text_candidate_buckets, arrow)
#     # print(f'buckets: {text_candidate_buckets}')
#
#     transformed_textlines= []
#     for textline in text_candidate_buckets:
#         textline.connected_components = transform_panel_coordinates_to_expanded_rect(crop_dct['rectangle'],
#                                                                           Rect(0,0,0,0), textline.connected_components)
#         transformed_textlines.append(textline)
#
#     return transformed_textlines

# def scan_conditions_text(fig, conditions, arrow, debug=False):
#     """
#     Crops a larger area around raw conditions to look for additional text elements that have
#     not been correctly recognised as conditions
#     :param Figure fig: analysed figure with binarised image object
#     :param iterable of Panels conditions: set or list of raw conditions (output of `find_reaction_conditions`)
#     :padam bool debug: debugging mode on/off - enables additional plotting
#     :return: Set of Panels containing all recognised conditions
#     """
#
#     fig = copy.deepcopy(fig)
#     fig = erase_elements(fig, [arrow])  # erase arrow at the very beginning
#
#     conditions_box = create_megabox(conditions)
#     # print(f'height: {conditions_box.height}, width: {conditions_box.width}')
#     extended_boundary_vertical = conditions_box.height
#     extended_boundary_horizontal = 75
#
#     search_region = Rect(conditions_box.left-extended_boundary_horizontal, conditions_box.right+extended_boundary_horizontal,
#                conditions_box.top-extended_boundary_vertical, conditions_box.bottom+extended_boundary_vertical)
#     # print(f'search region: {search_region}')
#     crop_dct = crop_rect(fig.img, search_region)
#     search_region = crop_dct['img']  # keep the rectangle boundaries in the other key
#
#
#
#     # print('running scan text!')
#     # plt.imshow(search_region)
#     plt.show()
#
#     initial_ccs_transformed = transform_panel_coordinates_to_shrunken_region(crop_dct['rectangle'],conditions)
#     search_region = attempt_remove_structure_parts(search_region, initial_ccs_transformed)
#
#     ccs = label_and_get_ccs(search_region)
#     top_boundaries, bottom_boundaries = identify_text_lines(ccs, search_region.img)
#
#     textlines = [TextLine(0, search_region.img.shape[1], upper, lower)
#                  for upper, lower in zip(top_boundaries, bottom_boundaries)]
#
#     # f = plt.figure(figsize=(20, 20))
#     # ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
#     # ax.imshow(search_region.img, cmap=plt.cm.binary)
#     # for line in top_boundaries:
#     #    ax.plot([i for i in range(search_region.img.shape[1])],[line for i in range(search_region.img.shape[1])],color='b')
#     # for line in bottom_boundaries:
#     #    ax.plot([i for i in range(search_region.img.shape[1])],[line for i in range(search_region.img.shape[1])],color='r')
#     #
#     #
#     # # print(f'textlines:{textlines}')
#     text_candidate_buckets = assign_characters_to_textlines(search_region.img, textlines, ccs)
#     # # print(f'example text_line ccs: {text_candidate_buckets[0].connected_components}')
#     mixed_text_candidates = [element for textline in text_candidate_buckets for element in textline]
#     # for _panel in mixed_text_candidates:
#     #     rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right-_panel.left, _panel.bottom-_panel.top, facecolor='r',edgecolor='b', alpha=0.35)
#     #     ax.add_patch(rect_bbox)
#     #
#     # plt.savefig('cond_chars_initial.tif')
#     #plt.show()
#
#
#
#     remaining_elements = set(ccs).difference(mixed_text_candidates)
#     text_candidate_buckets = assign_characters_proximity_search(search_region,
#                                                                 remaining_elements, text_candidate_buckets)
#
#     # print(f'buckets: {text_candidate_buckets}')
#     # print(f'buckets after filtering: {text_candidate_buckets}')
#     # print(f'example text_line ccs2: {text_candidate_buckets[0].connected_components}')
#     if len(text_candidate_buckets) > 2:
#         text_candidate_buckets = isolate_full_text_block(text_candidate_buckets, arrow)
#     # print(f'buckets: {text_candidate_buckets}')
#
#     transformed_textlines= []
#     for textline in text_candidate_buckets:
#         textline.connected_components = transform_panel_coordinates_to_expanded_rect(crop_dct['rectangle'],
#                                                                           Rect(0,0,0,0), textline.connected_components)
#         transformed_textlines.append(textline)
#
#     return transformed_textlines


# def attempt_remove_structure_parts(cropped_img, text_ccs):
#     """
#     Attempt to remove parts of structures from a cropped region containing conditions text.
#     :param np.ndarray cropped_img: array representing the cropped region
#     :param [Panels] text_ccs: text connected components detected during the raw line scan stage
#     :return np.ndarray: crop without the structure parts
#     """
#
#     crop_no_letters = erase_elements(Figure(cropped_img),text_ccs)
#     # f, ax = plt.subplots()
#     # ax.imshow(crop_no_letters.img)
#     # for _panel in text_ccs:
#     #    rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right-_panel.left,
#     #    _panel.bottom-_panel.top, facecolor='g',edgecolor='b',alpha=0.7)
#     #    ax.add_patch(rect_bbox)
#     # ax.set_title('characters removed')
#     # plt.show()
#     # skel_pixel_ratio = skeletonize_area_ratio(Figure(cropped_img),Rect(0,cropped_img.shape[1], 0, cropped_img.shape[0]))
#     # print(f'the skel-pixel ratio is {skel_pixel_ratio}')
#     ccs = label_and_get_ccs(crop_no_letters)
#     # print(ccs)
#     # print(f'type of ccs: {type(ccs)}')
#     # detect_structures(Figure(cropped_img), ccs)
#     skel_pixel_ratio = skeletonize_area_ratio(crop_no_letters, crop_no_letters.get_bounding_box()) # currently not used
#
#     # print(f'skel_pixel ratio: {skel_pixel_ratio}')
#     closed = binary_dilation(crop_no_letters.img,selem=disk(3)) #Decide based on skel-pixel ratio
#     # plt.imshow(closed)
#     # plt.show()
#     labelled = binary_tag(Figure(closed))
#     ccs = get_bounding_box(labelled)
#     structure_parts = [cc for cc in ccs if is_boundary_cc(cropped_img,cc)]
#     crop_no_structures = erase_elements(Figure(cropped_img),structure_parts)
#
#     # f, ax = plt.subplots()
#     # ax.imshow(crop_no_structures.img)
#     # for _panel in text_ccs:
#     #    rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right-_panel.left,
#     #    _panel.bottom-_panel.top, facecolor='none',edgecolor='b')
#     #    ax.add_patch(rect_bbox)
#     # ax.set_title('structures removed')
#     # plt.show()
#
#     return crop_no_structures


def identify_text_lines(crop):
    ccs = [cc for cc in crop.connected_components if cc.role != FigureRoleEnum.ARROW]  # filter out arrows

    if len(ccs) == 1:  # Special case
        only_cc = ccs[0]
        anchor = Point(only_cc.center[1], only_cc.center[0])
        return [(only_cc.top, only_cc.bottom, anchor)]
    if len(ccs) > 10:
        ccs = [cc for cc in ccs if cc.area > np.percentile([cc.area for cc in ccs], 0.2)]   # filter out all small ccs (e.g. dots)

    img = crop.img
    bottom_boundaries = [cc.bottom for cc in ccs]
    bottom_boundaries.sort()

    bottom_count = Counter(bottom_boundaries)
    bottom_boundaries = np.array([item for item in bottom_count.elements()]).reshape(-1, 1)

    little_data = len(ccs) < 10
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.005, 2.0, 100)},
                        cv=(len(bottom_boundaries) if little_data else 10))  # 10-fold cross-validation
    grid.fit(bottom_boundaries)
    best_bw = grid.best_params_['bandwidth']
    kde = KernelDensity(best_bw, kernel='exponential')
    kde.fit(bottom_boundaries)
    # print(f'params: {kde.get_params()}')
    rows = np.linspace(0, img.shape[0]+10, img.shape[0] + 11)
    logp_bottom = kde.score_samples(rows.reshape(-1, 1))

    heights = [cc.bottom - cc.top for cc in ccs]
    mean_height = np.mean(heights, dtype=np.uint32)
    bottom_lines, _ = find_peaks(logp_bottom, distance=mean_height*1.2)
    data = np.array([rows, logp_bottom])
    bottom_lines.sort()

    bucket_limits = find_minima_between_peaks(data, bottom_lines)
    buckets = np.split(rows, bucket_limits)
    # plt.plot(rows, logp, 'r')
    # plt.xlabel('Row')
    # plt.ylabel('logP(text_line)')
    # plt.scatter(kde_lower, [0 for elem in kde_lower], c='r', marker='+')
    # # plt.savefig('kde_2.tif')
    # plt.show()
    bucketed_chars = [[cc for cc in ccs if cc.bottom in bucket] for bucket in buckets]
    top_lines = [np.mean([cc.top for cc in bucket], dtype=int) for bucket in bucketed_chars]
    anchors = [sorted([cc for cc in bucket], key=lambda cc: cc.area)[-1].center for bucket in bucketed_chars]
    anchors = [Point(row=anchor[1], col=anchor[0]) for anchor in anchors]




    # f = plt.figure()
    # ax = f.add_axes([0.1, 0.1, 0.9, 0.9])
    # ax.imshow(img)
    # b = bucketed_chars[0]
    # for _panel in b:
    #     rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right-_panel.left, _panel.bottom-_panel.top, facecolor='none',edgecolor='r')
    #     ax.add_patch(rect_bbox)
    # plt.show()


    line_buckets = []
    # for peak in kde_lower:
    #     bucket = [cc for cc in ccs if cc.bottom in range(peak - 3, peak + 3)]
    #     line_buckets.append(bucket)

    # print(f'list of buckets: {line_buckets}')
    # print(len(line_buckets))
    # top_lines = []
    # # print(kde_lower)
    # for bucket, peak in zip(line_buckets, kde_lower):
    #     mean_height = np.mean([elem.height for elem in bucket])
    #     top_line = peak - mean_height
    #     top_lines.append(top_line)


    # f = plt.figure(figsize=(20, 20))
    # ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.imshow(img, cmap=plt.cm.binary)
    # for line in top_lines:
    #    ax.plot([i for i in range(img.shape[1])],[line for i in range(img.shape[1])],color='b')
    # for line in bottom_lines:
    #    ax.plot([i for i in range(img.shape[1])],[line for i in range(img.shape[1])],color='r')
    #
    # plt.show()
    # textlines = [TextLine(None,top,bottom) for top, bottom in zip(top_lines,bottom_lines)]
    return [line for line in zip(top_lines, bottom_lines, anchors)]


# def assign_characters_to_textlines(img, textlines, ccs, transform_from_crop=False, crop_rect=None):
#     text_candidate_buckets =[]
#     for textline in textlines:
#         textline_text_candidates = []
#
#         for cc in ccs:
#             if belongs_to_textline(img,cc,textline):
#                 textline_text_candidates.append(cc)
#
#             # elif is_small_textline_character(roi,cc,mean_character_area,text_line):
#             #     if cc not in textline_text_candidates: #avoid doubling
#             #         textline_text_candidates.append(cc)
#
#         textline_text_candidates = filter_distant_text_character(textline_text_candidates)
#         # print(f'text_line text cands: {textline_text_candidates}')
#         if transform_from_crop:
#             textline_text_candidates = transform_panel_coordinates_to_expanded_rect(
#                 crop_rect, Rect(0,0,0,0), textline_text_candidates) #Relative only so a placeholder Rect is the input
#
#         if textline_text_candidates: #If there are any candidates
#             textline.connected_components = textline_text_candidates
#             # print(f'components: {text_line.connected_components}')
#             text_candidate_buckets.append(textline)
#
#     return text_candidate_buckets


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
    # mixed_text_elements = [elem for text_line in text_buckets for elem in text_line]
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
    :param Figure fig: figure object containing image with the cc _panel
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
        # Check whether they share a text_line?
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


def clear_conditions_region(fig):
    """Removes connected components belonging to conditions and denoises the figure afterwards

    :param Figure fig: Analysed figure
    :return: new Figure object with conditions regions erased"""

    fig_no_cond = erase_elements(fig, [cc for cc in fig.connected_components
                                       if cc.role == FigureRoleEnum.ARROW or cc.role == FigureRoleEnum.CONDITIONSCHAR])

    area_threshold = fig.get_bounding_box().area / 30000
    # width_threshold = fig.get_bounding_box().width / 200
    noise = [panel for panel in fig_no_cond.connected_components if panel.area < area_threshold]

    return erase_elements(fig_no_cond, noise)



