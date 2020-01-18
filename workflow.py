import os

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from utils.io import imread
from utils.processing import hide_arrows, crop
from actions import (find_solid_arrows, segment, find_step_reactants_and_products,
                     scan_all_reaction_steps, skeletonize_area_ratio)
from chemschematicresolver.ocr import read_diag_text, read_label
from models.segments import Figure

import pytesseract
import logging
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


PATH = os.path.join('C:/', 'Users', 'wilar', 'PycharmProjects', 'RDE', 'images', 'RDE_images', 'Easy\\')
for file in os.listdir(PATH):
    filename = file
    #
    p = os.path.join(PATH, filename)
    fig = imread(p)    #     print(fig.img.shape)
        # except Exception:
        #     print('couldnt read an image: ', p)
        #     n_corrupted +=1
        #     continue
    global_skel_pixel_ratio = skeletonize_area_ratio(fig, fig.get_bounding_box())
    arrows = find_solid_arrows(fig, thresholds=None, min_arrow_lengths=None)
    print(arrows)
    fig_noarrows = hide_arrows(fig, arrows)
    ccs = segment(fig_noarrows, arrows)
    steps = scan_all_reaction_steps(fig, arrows, ccs, global_skel_pixel_ratio)


    # plt.imshow(fig_noarrows.img, cmap=plt.cm.gray)
    # plt.show()
    #Function to find conditions - single arrow only?
    #raw = find_reactants_and_products(fig,conditions, arrows[0],arrows, ccs)
    # classified =set((*conditions, *raw.raw_prods, *raw.raw_reacts))
    # unclassified = get_unclassified_ccs(ccs, classified)
    # print('unclassified: ',unclassified)
    # #Find nearest cc to each unclassified cc
    # for cc in unclassified:
    #     classified = sorted(classified, key=lambda elem: elem.separation(cc))
    #     print(classified)
    #     nearest = classified[0]
    #
    #     print('nearest: ')
    #     print(nearest)
    #     print('----')
    #     groups = [conditions, raw.raw_reacts, raw.raw_prods]
    #     for group in groups:
    #         if nearest in group:
    #             group.add(cc)
# f,ax = plt.subplots(figsize=(30, 30))
# ax.imshow(fig.img)
# # ax.set_title(filename)
# for panel in ccs:
#     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='b')
#     ax.add_patch(rect_bbox)
#
# for panel in arrows:
#     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='g')
#     ax.add_patch(rect_bbox)
# # for step in steps:
# for step in steps:
#     conditions = step.conditions.connected_components
#     raw_reacts = step.reactants.connected_components
#     raw_prods = step.products.connected_components
#
#     for panel in conditions:
#         rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='y')
#         ax.add_patch(rect_bbox)
#
#     for panel in raw_reacts:
#        rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='m')
#        ax.add_patch(rect_bbox)
#
#     for panel in raw_prods:
#        rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='r')
#        ax.add_patch(rect_bbox)
# plt.show()
# for panel in unclassified:
#     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='k')
#     ax.add_patch(rect_bbox)
    for step in steps:
        cc, = step.conditions.connected_components
        print(cc)
        cropped = crop(fig_noarrows.img, left=cc.left, right=cc.right, top=cc.top, bottom=cc.bottom)
        plt.imshow(cropped, cmap=plt.cm.binary)
        plt.show()

        text = read_diag_text(fig_noarrows, cc)
        print(text)


