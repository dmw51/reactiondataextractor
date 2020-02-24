"""
This file contains a pipeline necessary for OCR training. Images are pre-cropped to contain only relevant parts of
chemical diagrams. They might contain arrows, but no parts of structures are allowed.
The pipeline is to make sure that the images go through similar pre-processing steps compared to unseen data.

"""
import copy
import os
import logging
from PIL import Image

from utils.io import imread
from utils.processing import label_and_get_ccs, crop_rect
import conditions
from actions import find_solid_arrows, erase_elements
from models.arrows import SolidArrow
from models.segments import TextLine, Rect, Figure


def prepare_text_for_ocr(fig,arrow):
    """
    This is a function adapted from `conditions.scan_conditions_text` with arrow erasing as optional. Transformation
    to the  main coordinate system is omitted.
    :param Figure fig: Figure with text to be processed for OCR
    :param SolidArrow arrow: arrow to be erased, [] if none
    :return [TextLine]: list of textlines
    """

    fig = copy.deepcopy(fig)
    if arrows:
        fig = erase_elements(fig, arrows) # erase arrow at the very beginning

    text_block = fig.get_bounding_box()

    ccs = label_and_get_ccs(fig)
    top_boundaries, bottom_boundaries = conditions.identify_textlines(ccs, fig.img)

    textlines = [TextLine(0, fig.img.shape[1], upper, lower)
                 for upper, lower in zip(top_boundaries, bottom_boundaries)]

    text_candidate_buckets = conditions.assign_characters_to_textlines(fig.img, textlines, ccs)
    mixed_text_candidates = [element for textline in text_candidate_buckets for element in textline]

    remaining_elements = set(ccs).difference(mixed_text_candidates)
    if remaining_elements:
        text_candidate_buckets = conditions.assign_characters_proximity_search(fig,
                                                                           remaining_elements, text_candidate_buckets)

    return text_candidate_buckets


MAIN_DIR = os.getcwd()
PATH = os.path.join(MAIN_DIR, 'images', 'tess_train_images', 'high_res', 'crops')
for filename in os.listdir(PATH):
    p = os.path.join(PATH, filename)

    if os.path.isdir(p):
        continue

    p = os.path.join(PATH, filename)
    fig = imread(p)

    arrows = find_solid_arrows(fig, thresholds=None, min_arrow_lengths=None)

# Pass in the whole figure as a conditions region, no arrows
    textlines = prepare_text_for_ocr(fig, [])


    original_name = filename.split('.')[0]
    for idx, textline in enumerate(textlines):
        cropped_textline = crop_rect(fig.img, textline)['img']
        cropped_textline = Image.fromarray(cropped_textline)
        name = original_name + '_' + str(idx) + '.tif'

        cropped_textline.save(PATH+'/lines/'+name)





