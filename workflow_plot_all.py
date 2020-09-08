import os
import copy
import logging

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from utils.io_ import imread
from utils.processing import erase_elements, binary_tag, get_bounding_box, crop_rect, create_megabox
from actions import (find_solid_arrows, segment, scan_all_reaction_steps, skeletonize_area_ratio)
from conditions import find_reaction_conditions, get_conditions
from chemschematicresolver.ocr import read_diag_text, read_label

level = 'DEBUG'
# create logger
logger = logging.getLogger('project')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(level)

# create formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

MAIN_DIR = os.getcwd()
PATH = os.path.join(MAIN_DIR, 'images', 'RDE_images', 'Easy', 'high_res')
lst =[]
for file in os.listdir(PATH):
    filename = file
    p = os.path.join(PATH, filename)
    fig = imread(p)
    global_skel_pixel_ratio = skeletonize_area_ratio(fig, fig.get_bounding_box())
    arrows = find_solid_arrows(fig, thresholds=None, min_arrow_lengths=None)
    labelled = binary_tag(copy.deepcopy(fig))
    initial_ccs = get_bounding_box(labelled)
    all_conditions_bboxes = []
    all_conditions = []
    for arrow in arrows:
        conditions_bboxes = find_reaction_conditions(fig, arrow, initial_ccs)
        conditions = get_conditions(fig, arrow, initial_ccs)
        all_conditions_bboxes.append(conditions_bboxes)
        all_conditions.append(conditions)



    cond_flat = [textline for conditions in all_conditions_bboxes for textline in conditions]
    fig_no_cond = erase_elements(fig, [*cond_flat, *arrows])
    steps = scan_all_reaction_steps(fig_no_cond, arrows, all_conditions, initial_ccs, global_skel_pixel_ratio)
    fig, ax = plt.subplots()
    ax.imshow(fig_no_cond.img)
    offset = 0
    # for step in steps:
    #
    #     raw_reacts = step.reactants.connected_components
    #     raw_prods = step.products.connected_components
    #
    #     # for panel in conditions:
    #     #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='y')
    #     #     ax.add_patch(rect_bbox)
    #
    #     for panel in raw_reacts:
    #         rect_bbox = Rectangle((panel.left + offset, panel.top + offset), panel.right - panel.left,
    #                               panel.bottom - panel.top, facecolor='none', edgecolor='m')
    #         ax.add_patch(rect_bbox)
    #
    #     for panel in raw_prods:
    #         rect_bbox = Rectangle((panel.left + offset, panel.top + offset), panel.right - panel.left,
    #                               panel.bottom - panel.top, facecolor='none', edgecolor='r')
    #         ax.add_patch(rect_bbox)
    #     offset += 3
    #
    # plt.show()
    # fig_noarrows = erase_elements(fig, arrows)
    # fig_noconditions = erase_elements(fig_noarrows, [all_conditions])
    # plt.imshow(fig_noconditions.img)
    # plt.show()
    # ccs = segment(fig_noarrows, arrows)
    # areas = [panel.area/fig.get_bounding_box().area for panel in ccs]
    # lst.append(areas)

    # steps = scan_all_reaction_steps(fig, arrows,all_conditions, ccs, global_skel_pixel_ratio)
    # f, ax = plt.subplots()
    # ax.imshow(fig.img,cmap=plt.cm.binary)
    # ax.set_title(filename)
    # for panel in ccs:
    #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='b')
    #     ax.add_patch(rect_bbox)
    #
    # for panel in arrows:
    #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='g')
    #     ax.add_patch(rect_bbox)
    # offset = -3
    # for step in steps:
    #     conditions = step.conditions.connected_components
    #     raw_reacts = step.reactants.connected_components
    #     raw_prods = step.products.connected_components
    #
    #     for panel in conditions:
    #         rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='y')
    #         ax.add_patch(rect_bbox)
    #         print(f'conditions panel: {panel}')
    #     for panel in raw_reacts:
    #         print(f'raw reacts panel : {panel}')
    #         rect_bbox = Rectangle((panel.left+offset, panel.top+offset), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='m')
    #         ax.add_patch(rect_bbox)
    #
    #     for panel in raw_prods:
    #         rect_bbox = Rectangle((panel.left+offset, panel.top+offset), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='r')
    #         ax.add_patch(rect_bbox)
    #     offset += 3
    # plt.show()
    # for conditions in all_conditions:
    #     print(f'conditions: {conditions}')
    #     for text_line in conditions:
    #         print(f'text_line: {text_line}')
    #         print(f'connected components: {text_line.connected_components}')
    #         text_line.adjust_left_right()
    #         text_block = create_megabox(text_line)
    #         print(text_block)
    #         cropped = crop_rect(fig_noarrows.img, text_block)
    #         plt.imshow(cropped['img'], cmap=plt.cm.binary)
    #         plt.show()
    #
    #         text = read_diag_text(fig_noarrows, text_block)
    #         all_text.append(text)

# print(all_text)
    # for panel in unclassified:
    #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='k')
    #     ax.add_patch(rect_bbox)
    #     for step in steps:
    #         cc, = step.conditions.connected_components
    #         print(cc)
    #         cropped = crop(fig_noarrows.img, left=cc.left, right=cc.right, top=cc.top, bottom=cc.bottom)
    #         plt.imshow(cropped, cmap=plt.cm.binary)
    #         plt.show()
    #
    #         text = read_diag_text(fig_noarrows, cc)
    #         print(text)
# # step1 = steps[0]
# # raw_prods = step1.products
# # isolated = isolate_patch(fig, raw_prods)
# # isolated = Figure(img=isolated)
# # isolated = binary_close(isolated, 5) #This works
# # labelled = binary_tag(isolated)
# # ccs = get_bounding_box(labelled)
#
# #ccs = segment(isolated,[]) # This is too general a procedure
# fig, ax = plt.subplots()
#
#
# plt.imshow(isolated.img)
# for panel in ccs:
#     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='b')
#     ax.add_patch(rect_bbox)
# plt.show()

