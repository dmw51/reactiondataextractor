import os
import copy


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pprint import pprint
from PIL import Image, ImageOps

from actions import (detect_structures, find_arrows, contextualize_species, get_conditions_smiles,
                     remove_redundant_characters, complete_reaction_steps, find_optimal_dilation_ksize)
from conditions import get_conditions
from models.segments import  FigureRoleEnum
from models.output import ReactionScheme
from utils.io_ import imread
from utils.processing import erase_elements

import chemschematicresolver as csr
import settings
from utils.io_ import extract_image
import logging
# level = 'DEBUG'
# # create logger
# logger = logging.getLogger('project')
# logger.setLevel(logging.DEBUG)
#
# # create console handler and set level to debug
# ch = logging.StreamHandler()
# ch.setLevel(level)
#
# # create formatter
# formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
#
# # add formatter to ch
# ch.setFormatter(formatter)
#
# # add ch to logger
# logger.addHandler(ch)
#
# MAIN_DIR = os.getcwd()
# #PATH = os.path.join(MAIN_DIR, 'images', 'Dummies')
# #PATH = os.path.join(MAIN_DIR, 'images', 'RDE_images', 'Easy')
# PATH = os.path.join(MAIN_DIR, 'images', 'RDE_images', 'case_studies_1st_year')
# filename = 'ol0c01288_0006.jpeg'
#
# filenames = ['jo0c01256_0011.jpeg']
# # for filename in os.listdir(PATH):
# for filename in filenames:
#
#     p = os.path.join(PATH, filename)
#     if os.path.isdir(p):
#         continue
#     fig = imread(p)
#     settings.main_figure.append(fig)
#     detect_structures(fig)   # This should be a Figure method; running during ''init''? - can't do, circular ref
#     find_optimal_dilation_ksize(fig)
#
#     # global_skel_pixel_ratio = skeletonize_area_ratio(fig, fig.get_bounding_box())
#     arrows = find_arrows(fig, int(fig.boundary_length))
#     structure_panels = contextualize_species(fig)
#
#     conditions = [get_conditions(fig, arrow) for arrow in arrows]
#
#     fig_no_cond = erase_elements(fig, [cc for cc in fig.connected_components
#                                        if cc.role == FigureRoleEnum.ARROW or cc.role == FigureRoleEnum.CONDITIONSCHAR])
#     fig_clean = remove_redundant_characters(fig_no_cond, fig_no_cond.connected_components)
#     processed = Image.fromarray(fig_clean.img).convert('RGB')
#     processed = ImageOps.invert(processed)
#
#     segmented_filepath = PATH+'/processed'+'.tif'
#     processed.save(segmented_filepath)
#
#     csr_out = csr.extract_image_rde(PATH+'/processed.tif', kernel_size=fig.kernel_size, debug=True, allow_wildcards=False)
#     conditions = get_conditions_smiles(csr_out, conditions)
#     os.remove(segmented_filepath)
#
#     steps = complete_reaction_steps(structure_panels, conditions)
#     [step.match_function_and_smiles(csr_out) for step in steps]
#     colors = ['r', 'g', 'y', 'm', 'b', 'c', 'k']
#
#     f = plt.figure()
#     ax = f.add_axes([0.1, 0.1, 0.9, .9])
#     ax.imshow(fig.img)
#     for panel in fig.connected_components:
#         if panel.role:
#             color = colors[panel.role.value]
#         else:
#             color = 'w'
#         rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor=color)
#         ax.add_patch(rect_bbox)
#     for step in steps:
#         for group in step:
#             for species in group:
#                 panel = species.panel
#                 rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
#                                       facecolor='none', edgecolor='m')
#                 ax.add_patch(rect_bbox)
#     plt.show()
#
#     scheme = ReactionScheme(steps)
#
#
#     print('procedure complete.')
#     settings.main_figure = []

log = logging.getLogger()
log.setLevel('DEBUG')

name = '10.1021_jacs.9b12546_1.jpg'
scheme = extract_image(name, debug=True)
