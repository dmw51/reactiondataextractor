import os
import copy


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pprint import pprint
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from models.segments import Figure, RoleEnum
from models.output import ReactionScheme
from utils.io import imread
from utils.processing import erase_elements, binary_close, get_bounding_box, flatten_list, label_and_get_ccs, detect_headers
from actions import detect_structures, label_and_get_ccs, find_arrows, contextualize_species, remove_redundant_characters, match_function_and_smiles
from conditions import get_conditions
from skimage.morphology import binary_closing, disk, skeletonize
import chemschematicresolver as csr

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

MAIN_DIR = os.getcwd()
#PATH = os.path.join(MAIN_DIR, 'images', 'Dummies')
#PATH = os.path.join(MAIN_DIR, 'images', 'RDE_images', 'Easy')
PATH = os.path.join(MAIN_DIR, 'images', 'RDE_images', 'case_studies_1st_year')
filename = 'ol0c01288_0006.jpeg'

filenames = ['ol0c01112_0006.jpeg']
for filename in os.listdir(PATH):
# for filename in filenames:
    p = os.path.join(PATH, filename)
    if os.path.isdir(p):
        continue
    fig = imread(p)
    # image = fig.img
    # image = np.invert(image)
    # ax.imshow(image)

    panels = label_and_get_ccs(fig)


    detect_structures(fig)
    initial_ccs = fig.connected_components
    # global_skel_pixel_ratio = skeletonize_area_ratio(fig, fig.get_bounding_box())
    arrows = find_arrows(fig, int(fig.boundary_length))

# # prod = arrow.prod_side
# # react = arrow.react_side
# structs = []
# for _ in range(1):
#     s = detect_structures(fig, initial_ccs)
#     # dists = np.hstack((dists,pairs))
#     # dists = np.transpose(pairs)
#     # structs.append(np.sum(elem if elem == -1 else 0 for elem in pairs[0,:]))
#     # sixes = sum(1 if elem == -6.0 else 0 for elem in structs)
# # print(f'structures: {structures}')
# # print(arrows)
# print(f'number of arrows: {len(arrows)}')
# plt.imshow(fig.img, cmap=plt.cm.binary)
# plt.show()
# for arrow in arrows:
#     y, x = list(zip(*arrow.pixels))
#     plt.scatter(x, y)
# plt.title('detected arrows')
# plt.show()
#
# plt.imshow(fig.img, cmap=plt.cm.binary)
# for arrow in arrows:
#
#     y, x = list(zip(*arrow.react_side))
#     plt.scatter(x, y, c='r', s=.1)
#     y, x = list(zip(*arrow.prod_side))
#     plt.scatter(x, y, c='b', s=.1)
# plt.title('detected sides')
# plt.show()
# ### TEMP
# fig_no_arrows = erase_elements(fig, arrows)
# # fig_no_arrows_closed = Figure(binary_closing(fig_no_arrows.img,selem=disk(2)))
# temp_ccs = label_and_get_ccs(fig_no_arrows)
# ccs, labels = detect_structures(fig_no_arrows, temp_ccs)
# f, ax = plt.subplots()
# colours = ['b', 'm', 'y','r']
# ax.imshow(fig_no_arrows.img, cmap='binary')
# for idx, panel in enumerate(temp_ccs):
#     #
#     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none', edgecolor=colours[labels[idx]])
#     ax.add_patch(rect_bbox)
#
# plt.show()
# ###

    all_conditions_bboxes = []
    all_conditions = []
    # f = plt.figure()
    # ax = f.add_axes([0.1, 0.1, 0.9, .9])
    # ax.imshow(fig.img)
    # plt.show()

    conditions = [get_conditions(fig, arrow) for arrow in arrows]
    steps = contextualize_species(fig, conditions)

    for step in steps:
        print(f'conditions: {step.conditions}')
        print(f'reactants: {step.reactants}')
        print(f'products: {step.products}')
    r = ReactionScheme(steps)
    # r.find_path(r.reactants, r.products)
    json = r.to_json()

    fig_no_cond = erase_elements(fig, [cc for cc in fig.connected_components if cc.role == RoleEnum.ARROW or cc.role == RoleEnum.CONDITIONSCHAR])
    fig_clean = remove_redundant_characters(fig_no_cond, fig_no_cond.connected_components)

    processed = Image.fromarray(fig_clean.img).convert('RGB')
    processed = ImageOps.invert(processed)

    segmented_filepath = PATH+'/processed'+'.tif'
    processed.save(segmented_filepath)

    csr_out, diags_ccs_ordered = csr.extract_image_rde(PATH+'/processed.tif', debug=True, allow_wildcards=False)
    os.remove(segmented_filepath)

    result = csr_out, diags_ccs_ordered
    for step in steps:
        ## Error is here:
        # print(f'all attributes of step {step}: {vars(step)}')

        match_function_and_smiles(step, result)

    print('done')




    # if crop:
        # f = plt.figure()
        # ax = f.add_axes([0.1, 0.1, 0.9, .9])
        # ax.imshow(crop.cropped_img)
        # for panel in crop.connected_components:
        #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='y')
        #     ax.add_patch(rect_bbox)
        # plt.show()
    # conditions_bboxes = find_step_conditions2(fig, arrow)
    # if conditions_bboxes:
    #     conditions = get_conditions(fig, arrow, initial_ccs)
    #     pprint(conditions)
    #     all_conditions_bboxes.append(conditions_bboxes)
    #     all_conditions.append(conditions)
    # else:
    #     all_conditions.append([])

# cond_flat = [text_line for conditions in all_conditions_bboxes for text_line in conditions] if all_conditions_bboxes else []
# fig_no_cond = erase_elements(fig, [*cond_flat,*arrows])
#
# leftover_ccs = label_and_get_ccs(fig_no_cond)
# fig_clean = remove_redundant_characters(fig_no_cond, leftover_ccs)
# fig_clean = remove_redundant_square_brackets(fig_clean, leftover_ccs)
# #headers = detect_headers(binary_close(fig_clean, 5))
# #boxes = detect_rectangle_boxes(fig_clean, greedy=True)
# #print(f'boxes: {boxes}')
# #print(f'headers: {headers}')
# #fig_clean = erase_elements(fig_clean, flatten_list(headers) + boxes)
#
#
# steps = scan_all_reaction_steps(fig_clean, arrows, all_conditions, initial_ccs, global_skel_pixel_ratio)
# f, ax = plt. subplots()
# #
# #
# ax.imshow(fig_clean.img)
#
# # for step in steps:
# #     offset = 0
# #     raw_reacts = step.reactants.connected_components
# #     raw_prods = step.products.connected_components
# #
# #     # for panel in conditions:
# #     #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='y')
# #     #     ax.add_patch(rect_bbox)
# #
# #     for panel in raw_reacts:
# #         rect_bbox = Rectangle((panel.left+offset, panel.top+offset), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='m')
# #         ax.add_patch(rect_bbox)
# #
# #     for panel in raw_prods:
# #         rect_bbox = Rectangle((panel.left+offset, panel.top+offset), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='r')
# #         ax.add_patch(rect_bbox)
# plt.show()

#
# processed = Image.fromarray(fig_clean.img).convert('RGB')
# processed = ImageOps.invert(processed)
#
# segmented_filepath = PATH+'/processed'+'.tif'
# processed.save(segmented_filepath)
#
# csr_out, diags_ccs_ordered = csr.extract_image_rde(PATH+'/processed.tif', debug=True, allow_wildcards=False)
# print(f'csr out: {csr_out}')
# print(f'ordered diags: {diags_ccs_ordered}')
#
# print(f'end here: {steps[0]}')
# print(steps[0].reactants[0])
# result = csr_out, diags_ccs_ordered
# os.remove(segmented_filepath)
# for step in steps:
#     print(f'all attributes of step {step}: {vars(step)}')
#     match_function_and_smiles(step, result)
#     for product in step.products:
#         print(f'step: {step}, product: {vars(product)}')
#     for reactant in step.reactants:
#         print(f'step: {step}, reactant: {vars(reactant)}')
#
#
#
#
# #
# # reaction = Reaction.from_reaction_steps(steps)
# # print(f'reaction: {reaction.steps}')
# #
# f, ax = plt.subplots(figsize=(10, 6))
# ax.imshow(fig.img, cmap=plt.cm.binary)
#
#
# for panel in arrows:
#     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',
#                           edgecolor='g')
#     ax.add_patch(rect_bbox)
#
# offset = 0
# print(f'all steps: {steps}')
# for step in steps:
#     # conditions_text = step.conditions.text_lines
#     reactants = step.reactants
#     products = step.products
#
#     # conditions_anchor = conditions_text[0]
#     # # ax.text(conditions_anchor.left-conditions_anchor.width//4, conditions_anchor.top - 2*conditions_anchor.height,
#     # #         str(step.conditions), color='b', size=8)
#     # for panel in conditions_text:
#     #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',
#     #                           edgecolor='y')
#     #     ax.add_patch(rect_bbox)
#
#     for reactant in reactants:
#         panel = reactant.connected_component
#         rect_bbox = Rectangle((panel.left+offset, panel.top+offset), panel.right-panel.left, panel.bottom-panel.top,
#                               facecolor='none', edgecolor='m')
#         ax.add_patch(rect_bbox)
#         # ax.text(panel.left, panel.top - panel.height//20, '%s \nlabel = %s' % (reactant.smiles, reactant.label,),
#         #         size=8)
#
#     for product in products:
#         panel = product.connected_component
#         print(f'product panel: {panel}')
#         rect_bbox = Rectangle((panel.left+offset, panel.top+offset), panel.right-panel.left, panel.bottom-panel.top,
#                               facecolor='none', edgecolor='r')
#         # ax.text(panel.left, panel.top - panel.height//20, '%s \nlabel = %s' % (product.smiles, product.label),
#         #         size=8)
#         ax.add_patch(rect_bbox)
#
#     offset += 5
#
# ax.set_axis_off()
# plt.savefig('out_consecutive_ABC.tif')
# plt.show()


















# plt.imshow(fig.img,cmap=plt.cm.binary)
# plt.savefig('original.jpg', format='jpg', dpi=1000)
# fig = preprocessing_remove_long_lines(fig)
# labelled = binary_tag(copy.deepcopy(fig))
# initial_ccs = get_bounding_box(labelled)
# global_skel_pixel_ratio = skeletonize_area_ratio(fig, fig.get_bounding_box())
# arrows = find_solid_arrows(fig, thresholds=None, min_arrow_lengths=None)
# all_conditions = []
# for arrow in arrows:
#     conditions = find_reaction_conditions(fig, arrow, initial_ccs, global_skel_pixel_ratio)
#     all_conditions.append(conditions)
# fig_noarrows = erase_elements(fig, arrows)
# fig_noconditions = erase_elements(fig_noarrows, *all_conditions)
# # plt.imshow(fig_noconditions.img,cmap=plt.cm.binary)
# # #plt.savefig('cleaned.jpg', format='jpg', dpi=1000)
# # plt.show()
# ccs = segment(fig_noconditions, arrows)
# steps = scan_all_reaction_steps(fig, arrows, all_conditions, ccs, global_skel_pixel_ratio)
# f, ax = plt.subplots()
# ax.imshow(fig_noconditions.img,cmap=plt.cm.binary)
# # ax.set_title(filename)
# all_text =[]
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
#     # for panel in conditions:
#     #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='y')
#     #     ax.add_patch(rect_bbox)
#
#     for panel in raw_reacts:
#         rect_bbox = Rectangle((panel.left+offset, panel.top+offset), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='m')
#         ax.add_patch(rect_bbox)
#
#     for panel in raw_prods:
#         rect_bbox = Rectangle((panel.left+offset, panel.top+offset), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='r')
#         ax.add_patch(rect_bbox)
#     offset += 3
#
#     # for panel in unclassified:
#     #     rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor='k')
#     #     ax.add_patch(rect_bbox)
#     #     for step in steps:
#     #         cc, = step.conditions.connected_components
#     #         print(cc)
#     #         cropped = crop(fig_noarrows.img, left=cc.left, right=cc.right, top=cc.top, bottom=cc.bottom)
#     #         plt.imshow(cropped, cmap=plt.cm.binary)
#     #         plt.show()
#     #
#     #         text = read_diag_text(fig_noarrows, cc)
#     #         print(text)
#
#     for text_line in conditions:
#         print(f'text_line: {text_line}')
#         print(f'connected components: {text_line.connected_components}')
#         text_line.adjust_left_right()
#         text_block = create_megabox(text_line)
#         print(f'text_line: {text_line}')
#         print(f'megabox: {text_block}')
#         cropped = crop_rect(fig_noarrows.img, text_block)
#         # plt.imshow(cropped['img'], cmap=plt.cm.binary)
#         # plt.show()
#
#         text = read_diag_text(fig_noarrows, text_block)
#         all_text.append(text)
# plt.show()
# print(all_text)
