"""
This file contains a pipeline necessary for OCR training. Images are pre-cropped to contain only relevant parts of
chemical diagrams. They might contain arrows, but no parts of structures are allowed.
The pipeline is to make sure that the images go through similar pre-processing steps compared to unseen data.
Once each crop is split into its component text lines, a ground-truth text file is created for each line.
For this purpose, a large text file `all_text` was created which contains all lines from all crops that had been
manually typed. The file is split line by line to give one ground truth text file per each line represented as an image.
"""
import copy
import os
import logging
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils.io import imread
from utils.processing import label_and_get_ccs, crop_rect
import conditions
from actions import find_solid_arrows, erase_elements
from models.arrows import SolidArrow
from models.segments import TextLine, Rect, Figure

MAIN_DIR = os.getcwd()
PATH = os.path.join(MAIN_DIR, 'images', 'tess_train_images', 'high_res', 'crops')
PATH_TO_TEXT_FILE = os.path.join(MAIN_DIR, 'images', 'tess_train_images', 'high_res', 'crops', 'text')
PADDING = 10  # This should be the same as in the OCR pre-processing step

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
    top_boundaries, bottom_boundaries = conditions.identify_text_lines(ccs, fig.img)

    textlines = [TextLine(0, fig.img.shape[1], upper, lower)
                 for upper, lower in zip(top_boundaries, bottom_boundaries)]

    text_candidate_buckets = conditions.assign_characters_to_textlines(fig.img, textlines, ccs)
    mixed_text_candidates = [element for textline in text_candidate_buckets for element in textline]

    remaining_elements = set(ccs).difference(mixed_text_candidates)
    if remaining_elements:
        text_candidate_buckets = conditions.assign_characters_proximity_search(fig,
                                                                           remaining_elements, text_candidate_buckets)

    return text_candidate_buckets


for filename in os.listdir(PATH):
    p = os.path.join(PATH, filename)

    if os.path.isdir(p):
        continue

    p = os.path.join(PATH, filename)
    fig = imread(p)

    arrows = find_solid_arrows(fig, thresholds=None, min_arrow_lengths=None)

# Pass in the whole figure as a conditions region, no arrows
    textlines = prepare_text_for_ocr(fig, arrows)

    original_name = filename.split('.')[0]
    for idx, textline in enumerate(textlines):
        cropped_textline = crop_rect(fig.img, textline)['img']

        cropped_textline = np.pad(cropped_textline,PADDING,mode='constant')
        cropped_textline = Image.fromarray(cropped_textline)

        name = original_name + '_' + str(idx) + '.tif'

        cropped_textline.save(PATH+'/lines/'+ name)

sorted_image_line_files = sorted(os.listdir(PATH+'/lines'))
all_text_path = PATH + '/text/all_text.txt'
# Create one text file per image of a text line
# Iterate over lines of `all_text` and create one ground_truth file with a name corresponding the image text line
with open(all_text_path, 'r') as all_text:
    for line_image_file, line_text in zip(sorted_image_line_files, all_text):
        ground_truth_text_filename = line_image_file.split('.')[0] + '.gt.txt'

        with open(PATH +'/lines/' + ground_truth_text_filename, 'w') as ground_truth:
            ground_truth.write(line_text)





