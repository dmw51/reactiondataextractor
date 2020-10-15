
import copy


import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.ndimage import label
from skimage.transform import probabilistic_hough_line
from skimage.morphology import skeletonize as skeletonize_skimage
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN



from models.arrows import SolidArrow
from models.exceptions import NotAnArrowException, NoArrowsFoundException
from models.reaction import ReactionStep, Conditions, Diagram
from models.segments import Rect, Panel, Figure, FigureRoleEnum, ReactionRoleEnum, Crop
from models.utils import Point, Line
from utils.processing import approximate_line, pixel_ratio, binary_close, binary_floodfill, dilate_fragments
from utils.processing import (binary_tag, get_bounding_box, erase_elements,
                              isolate_patches, is_a_single_line, create_megabox)
from ocr import read_character
import settings

log = logging.getLogger('extract.actions')

# formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
# file_handler = logging.FileHandler('actions.log')
# file_handler.setFormatter(formatter)

# ch = logging.StreamHandler()
# ch.setFormatter(formatter)

# log.addHandler(file_handler)
# log.addHandler(ch)

# create console handler and set level
# level = 'DEBUG' if debug else 'INFO'
# ch = logging.StreamHandler()
# ch.setLevel(level)


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
        kernel = 4
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
        kernel = 25
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    # Using a binary floodfill to identify _panel regions
    fill_img = binary_floodfill(closed_fig)
    tag_img = binary_tag(fill_img)
    panels = get_bounding_box(tag_img)

    # Removing relatively tiny pixel islands that are determined to be noise
    area_threshold = bin_fig.get_bounding_box().area / 200
    width_threshold = bin_fig.get_bounding_box().width / 150
    panels = [panel for panel in panels if panel.area > area_threshold or panel.width > width_threshold]
    return set(panels)


def find_optimal_dilation_ksize(fig=None):
    """
    Use structural backbones to calculate local skeletonised-pixel ratio and find optimal dilation kernel size for
    structural segmentation. Each backbone is assigned its own dilation kernel to account for varying skel-pixel ratio
    around different backbones
    :param Figure fig: Analysed Figure object containing marked structural backbones if its ``connected_components``
    attribute
    :return dict: kernel sizes appropriate for each backbone
    """
    if fig is None:
        fig = settings.main_figure[0]

    backbones = [cc for cc in fig.connected_components if cc.role == FigureRoleEnum.STRUCTUREBACKBONE]

    kernel_sizes = {}
    for backbone in backbones:
        left, right, top, bottom = backbone
        crop_rect = Rect(left-50, right+50, top-50, bottom+50)
        p_ratio = skeletonize_area_ratio(fig, crop_rect)
        log.debug(f'found in-crop skel_pixel ratio: {p_ratio}')

        kernel_size = 4
        # mean_p_ratio = np.mean(p_ratios)
        if 0.016 < p_ratio < 0.02:
            kernel_size += 4
        elif 0.01 < p_ratio < 0.016:
            kernel_size += 8
        elif p_ratio < 0.01:
            kernel_size += 12

        kernel_sizes[backbone] = kernel_size
        log.info(f'Structure segmentation kernels:{kernel_sizes.values()}')

    fig.kernel_sizes = kernel_sizes
    return kernel_sizes


def skeletonize(fig):
    """
    A convenience function operating on Figure objects working similarly to skimage.morphology.skeletonize
    :param fig: analysed figure object
    :return: figure object with a skeletonised image
    """

    img = skeletonize_skimage(fig.img)

    return Figure(img, raw_img=fig.raw_img)


def skeletonize_area_ratio(fig, panel):
    """ Calculates the ratio of skeletonized image pixels to total number of pixels
    :param fig: Input figure
    :param panel: Original _panel object
    :return: Float : Ratio of skeletonized pixels to total area (see pixel_ratio)
    """

    skel_fig = skeletonize(fig)
    return pixel_ratio(skel_fig, panel)


def find_arrows(fig, min_arrow_length):
    """
    Arrow finding algorithm. Finds lines of length at least ``min_arrow length`` in ``fig`` and detects arrows
    using a rule-based algorithm. Can be extended to find other types of arrows
    :param Figure fig: analysed figure
    :param int min_arrow_length: minimum length of each arrow used by the Hough Transform
    :return: collection of found arrows
    """
    threshold = min_arrow_length//2

    arrows = find_solid_arrows(fig, threshold=threshold, min_arrow_length=min_arrow_length)

    if not arrows:
        log.warning('No arrows have been found in the image')
        raise NoArrowsFoundException('No arrows have been found')

    return list(set(arrows))


def find_solid_arrows(fig, threshold, min_arrow_length):
    """
    Finds all solid arrows in ``fig`` subject to ``threshold`` number of pixels and ``min_arrow_length`` minimum
    line length.
    :param Figure fig: input figure object
    :param int threshold: threshold number of pixels needed to define a line (Hough Transform param).
    :param int min_arrow_length: threshold length needed to define a line
    :return: collection of arrow objects
    """
    img = copy.deepcopy(fig.img)

    arrows = []
    skeletonized = skeletonize(fig)
    all_lines = probabilistic_hough_line(skeletonized.img, threshold=threshold, line_length=min_arrow_length, line_gap=3)
    for line in all_lines:
        # for line in all_lines:
        #     x, y = zip(*line)
        #     if abs(x[0] - x[1]) > 100:
        #         print('im the one')
        # isolated_fig = skeletonize(isolate_patches(fig, [cc]))
        # cc_lines = probabilistic_hough_line(fig.img, threshold=threshold, line_length=min_arrow_length, line_gap=3)
        # if len(cc_lines) > 1:
        #     print('stop')
        # if not cc_lines or (len(cc_lines) > 1 and not is_slope_consistent(cc_lines)):
        #     continue
        # if lines were found, 'break' these down and check again
        # shorter_lines = probabilistic_hough_line(isolated_fig.img, threshold=threshold//3, line_length=min_arrow_length//3)
        # if not shorter_lines or (len(shorter_lines) > 1 and not is_slope_consistent(shorter_lines)):
        #     continue

        points = [Point(row=y, col=x) for x, y in line]
        # Choose one of points to find the label and pixels in the image
        p1, p2 = points
        labelled_img, _ = label(img)
        p1_label = labelled_img[p1.row, p1.col]
        p2_label = labelled_img[p2.row, p2.col]
        if p1_label != p2_label: # Hough transform can find lines spanning several close ccs; these are discarded
            log.info('A false positive was found when detecting a line. Discarding...')
            continue
        else:
            parent_label = labelled_img[p1.row, p1.col]
            parent_panel = [cc for cc in fig.connected_components if p1.row in range(cc.top, cc.bottom+1) and
                                                                    p1.col in range(cc.left, cc.right+1)][0]
        if not is_a_single_line(skeletonized, parent_panel, min_arrow_length//2):
            continue
        # print('checking p1:...')
        # print(p1.row, p1. col)
        # print('should be (96, 226)')

        arrow_pixels = np.nonzero(labelled_img == parent_label)
        arrow_pixels = list(zip(*arrow_pixels))
        try:

            new_arrow = SolidArrow(arrow_pixels, line=approximate_line(p1, p2), panel=parent_panel)

        except NotAnArrowException as e:
            log.info('An arrow candidate was discarded - ' + str(e))
        else:
            arrows.append(new_arrow)
            parent_cc = [cc for cc in fig.connected_components if cc == new_arrow.panel][0]
            parent_cc.role = FigureRoleEnum.ARROW
    # lines = probabilistic_hough_line(skeleton, threshold=threshold, line_length=min_arrow_length)
    #print(lines)
    # labelled_img, _ = label(img)
    # arrows =[]
    # # plt.imshow(fig.img, cmap=plt.cm.binary)
    # # line1 = list(zip(*lines[0]))
    # # line2 = list(zip(*lines[1]))
    # # plt.plot(line1[0], line1[1])
    # # plt.plot(line2[0], line2[1])
    # # plt.axis('off')
    # # plt.show()
    # # plt.imshow(fig.img, cmap=plt.cm.binary)
    # # for line in lines:
    # #     x, y = list(zip(*line))
    # #     plt.plot(x,y)
    # # plt.title('detected lines')
    # # plt.show()
    #
    # for l in lines:
    #     points = [Point(row=y, col=x) for x, y in l]
    #     # Choose one of points to find the label and pixels in the image
    #     p1 = points[0]
    #     p2 = points[1]
    #     # p1_label = labelled_img[p1.row, p1.col]
    #     # p2_label = labelled_img[p2.row, p2.col]
    #     # if p1_label != p2_label: # Hough transform can find lines spanning several close ccs; these are discarded
    #     #     log.info('A false positive was found when detecting a line. Discarding...')
    #     #     continue
    #     #print('checking p1:...')
    #     #print(p1.row, p1. col)
    #     #print('should be (96, 226)')
    #     arrow_label = labelled_img[p1.row, p1.col]
    #
    #     arrow_pixels = np.nonzero(labelled_img == arrow_label)
    #     arrow_pixels = list(zip(*arrow_pixels))
    #     try:
    #         new_arrow = SolidArrow(arrow_pixels, line=approximate_line(*points))
    #     except NotAnArrowException as e:
    #         log.info('An arrow candidate was discarded - ' + str(e))
    #     else:
    #         arrows.append(new_arrow)
    # Filter poor arrow assignments based on aspect ratio
    # arrows = [arrow for arrow in arrows if arrow.aspect_ratio >5]  ## This is not valid for tilted arrows
    return list(set(arrows))


def complete_structures(fig: Figure):
    """
    Dilates a figure and uses structural backbones to find complete structures (backbones + superatoms etc.).

    Dilates a figure using a pre-derived dilation kernel. Checks which ccs correspond to backbones in the dilated figure.
    Assuming that this cc is the full (dilated) structure, compares it will all initial overlapping ccs to find original
    full structure in an image. Also assigns roles of all the smaller constituent ccs.
    :param Figure fig: analysed figure
    :return: [ReactionStep,...] collection of ReactionStep objects
    """

    backbones = [cc for cc in fig.connected_components if cc.role == FigureRoleEnum.STRUCTUREBACKBONE]
    # dilated_structure_panels, dilated_other = dilate_group_panels(fig, backbones)
    dilated_structure_panels, other_ccs = find_dilated_structures(fig, backbones)
    structure_panels = _complete_structures(fig, dilated_structure_panels)
    _assign_backbone_auxiliaries(fig, (dilated_structure_panels, other_ccs), structure_panels)  # Assigns cc roles
    # settings.main_figure[0] = fig   # This is an awful way to do it, losing conditions info (look at input fig)
    # # Need to somehow disentangle the different versions of ``fig`` - roles needed for later

    return structure_panels



def find_dilated_structures(fig, backbones):
    """
    Finds dilated structures by first dilating the image several times using backbone-specific kernel size.

    For each backbone, the figure is dilated using a backbone-specific kernel size. Dilated structure panel is then
    found based on comparison with the original backbone. A crop is made for each structure. If there are more than
    one connected component (the structure panel itself), e.g. a label, then this is noted and
    used later for role assignment.
    :param Figure fig: Analysed figure
    :param [Panel,...] backbones: Collection of panels containing structural backbones
    :return: (dilated_structure_panels, other_ccs) pair of collections containing the dilated panels and separate ccs
    present within these dilated panels
    """
    dilated_structure_panels = []
    other_ccs = []
    dilated_imgs = {}

    for backbone in backbones:
        ksize = fig.kernel_sizes[backbone]
        try:
            dilated_temp = dilated_imgs[ksize]
        except KeyError:
            dilated_temp = dilate_fragments(fig, ksize)
            dilated_imgs[ksize] = dilated_temp
        dilated_structure_panel = [cc for cc in dilated_temp.connected_components if cc.contains(backbone)][0]

        structure_crop = Crop(dilated_temp, dilated_structure_panel())
        other = [structure_crop.in_main_fig(c) for c in structure_crop.connected_components if structure_crop.in_main_fig(c) != dilated_structure_panel]
        other_ccs.extend(other)
        dilated_structure_panels.append(dilated_structure_panel)

    return dilated_structure_panels, other_ccs
    # dilated = dilate_fragments(fig, fig.kernel_sizes)
    #
    # dilated_structure_panels = []
    # non_structures = []
    # for cc in dilated.connected_components:
    #     if any(cc.contains(backbone) for backbone in backbones):
    #         dilated_structure_panels.append(cc)
    #     else:
    #         non_structures.append(cc)

    # return dilated_structure_panels, non_structures
    # return dilated_structure_panels

def scan_form_reaction_step(conditions, diags):
    """
    Scans an image around a single arrow to give reactants and products in a single reaction step
    :param Conditions conditions: Conditions object containing ``arrow`` around which the scan is performed
    :param [Diagram,...] diags: collection of all detected and recognised diagrams
    :return: a ReactionStep object
    """
    arrow = conditions.arrow
    endpoint1, endpoint2 = extend_line(conditions.arrow.line, extension=arrow.pixels[0].separation(arrow.pixels[-1]))
    react_side_point = conditions.arrow.react_side[0]
    endpoint1_close_to_react_side = endpoint1.separation(react_side_point) < endpoint2.separation(react_side_point)
    if endpoint1_close_to_react_side:
        react_endpoint, prod_endpoint = endpoint1, endpoint2
    else:
        react_endpoint, prod_endpoint = endpoint2, endpoint1

    initial_distance = 1.25 * np.sqrt(np.mean([diag.panel.area for diag in diags]))
    distance_fn = lambda diag: 1.5*np.sqrt(diag.panel.area)
    distances = initial_distance, distance_fn
    reactants = find_nearby_ccs(react_endpoint, diags, distances,
                                condition=lambda diag: diag.panel.role != ReactionRoleEnum.CONDITIONS)
    if not reactants:
        reactants = search_elsewhere('up-right', diags, conditions.arrow,  distances)


    products = find_nearby_ccs(prod_endpoint, diags, distances,
                               condition=lambda diag: diag.panel.role != ReactionRoleEnum.CONDITIONS)
    if not products:
        products = search_elsewhere('down-left', diags, conditions.arrow, distances)
    return ReactionStep(reactants, products, conditions=conditions)


def search_elsewhere(where, diags, arrow, distances):
    """
    Looks for structures in a different line of a multi-line reaction scheme.

    Assumes left-to-right reaction scheme. Estimates the optimal alternative search point using arrow and structures'
    coordinates. Performs a seach in the new spot.
    :param str where: Allows either 'down-left' to look below and to the left of arrow, or 'up-right' (above to the right)
    :param [Diagram,...]: Sequence of Diagram objects
    :param Arrow arrow: Original arrow, around which the search failed
    :param (float, lambda) distances: pair containing initial search distance and a distance function (usually same as
    in the parent function)
    :return: Collection of found species
    """
    assert where in ['down-left', 'up-right']
    fig = settings.main_figure[0]

    X = np.array([s.center[1] for s in diags] + [arrow.panel.center[1]]).reshape(-1, 1)  # the y-coordinate
    eps = np.mean([s.height for s in diags])
    dbscan = DBSCAN(eps=eps, min_samples=2)
    y = dbscan.fit_predict(X)
    num_labels = max(y) - min(y) + 1 # include outliers (label -1) if any
    arrow_label = y[-1]
    clustered = []
    for val in range(-1, num_labels):
        if val == arrow_label:
            continue  # discard this cluster - want to compare the arrow with other clusters only
        cluster = [centre for centre, label in zip (X, y) if label == val]
        if cluster:
            clustered.append(cluster)
    centres = [np.mean(cluster) for cluster in clustered]
    centres.sort()
    if where == 'down-left':
        move_to_vertical = [centre for centre in centres if centre > arrow.panel.center[1]][0]
        move_to_horizontal = np.mean([structure.width for structure in diags])
    elif where == 'up-right':
        move_to_vertical = [centre for centre in centres if centre < arrow.panel.center[1]][-1]
        move_to_horizontal = fig.img.shape[1] - np.mean([structure.width for structure in diags])
    else:
        raise ValueError("'where' takes in one of two values : ('down-left', 'up-right') only")
    species = find_nearby_ccs(Point(move_to_vertical, move_to_horizontal), diags, distances)

    return species




def _complete_structures(fig, dilated_structure_panels):
    """Uses ``dilated_structure_panels`` to find all constituent ccs of each chemical structure.

    Finds connected components belonging to a chemical structure and creates a large panel out of them. This effectively
    normalises panel sizes with respect to chosen dilation kernel sizes. Also sets the structure panel as the
    ``parent_panel`` of appropriate backbones
    :param Figure fig: Analysed figure
    :return: collection of Panels delineating complete chemical structures
    """

    # Look for the dilated structures by comparing with the original backbone ccs. Set aside structures that fully
    # overlap with small ccs (e.g. labels), but have not been connected by the dilation. (ambiguous)
    # Use the rest of structures to assign roles of smaller ccs directly (resolved). In the case of ambiguous
    # structures, carefully compare ccs so as to leave the disconnected entity intact.


    structure_panels = []
    for dilated_structure in dilated_structure_panels:
        constituent_ccs = [cc for cc in fig.connected_components if dilated_structure.contains(cc)]
        parent_structure_panel = create_megabox(constituent_ccs)
        structure_panels.append(parent_structure_panel)
        # backbone = [cc for cc in constituent_ccs if cc.role == FigureRoleEnum.STRUCTUREBACKBONE][0]
        #



    # f = plt.figure()
    # ax = f.add_axes([0,0,1,1])
    # ax.imshow(fig.img)
    # for _panel in structure_panels:
    #     rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right-_panel.left, _panel.bottom-_panel.top, facecolor='none',edgecolor='r')
    #     ax.add_patch(rect_bbox)
    # plt.show()
    #
    # f = plt.figure()
    # ax = f.add_axes([0,0,1,1])
    # ax.imshow(dilated.img)
    # for _panel in structure_panels:
    #     rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right-_panel.left, _panel.bottom-_panel.top, facecolor='none',edgecolor='r')
    #     ax.add_patch(rect_bbox)
    # plt.show()
    return structure_panels


def _assign_backbone_auxiliaries(fig, dilated_panels, structure_panels):
    """
    Takes in the ``structure_panels`` to assign roles to structural auxiliaries (solitary
    bondlines, superatom labels) found in ``figure`` based on comparison between each structure _panel and its
    corresponding backbone in ``figure``.
    :param Figure fig: analysed figure
    :param tuple(dilated_structures, dilated,non_structures) dilated_panels: pair containing dilated ccs classified into
    the two groups
    :return: None (mutates ''role'' attribute of each relevant connected component)
    """
    #TODO: Simplify/combine loops where appropriate to speed this up.
    dilated_structure_panels, other_ccs = dilated_panels
    parent_panel_pairs = [(parent, dilated) for parent in structure_panels for dilated in dilated_structure_panels if
                          dilated.contains(parent)]


    # resolved = []
    # ambiguous = []
    # for structure in dilated_structure_panels:
    #     disconnected_overlapping_ccs = [cc for cc in non_structures if structure.overlaps(cc)]
    #     if disconnected_overlapping_ccs:
    #         ambiguous.append((structure, disconnected_overlapping_ccs))
    #     else:
    #         resolved.append(structure)
    # log.debug('Found %d resolved and %d ambiguous structures.' % (len(resolved), len(ambiguous)))
    # resolved = [structure for structure in dilated_structures if not any(structure.contains(cc)
    #                                                                      for cc in dilated.connected_components)]
    # ambiguous_structures = [(structure, [cc for cc in dilated.connected_components if structure.contains(cc)])
    #                          for structure in dilated_structures]

    # [[setattr(auxiliary, 'role', FigureRoleEnum.STRUCTUREAUXILIARY) for auxiliary in fig.connected_components if
    #   resolved_structure.contains(auxiliary) and getattr(auxiliary, 'role') != FigureRoleEnum.STRUCTUREBACKBONE]
    #  for resolved_structure in resolved]
    for parent_panel in structure_panels:
        for cc in fig.connected_components:
            if parent_panel.contains(cc): # Set the parent panel for all
                setattr(cc, 'parent_panel', parent_panel)
                if cc.role != FigureRoleEnum.STRUCTUREBACKBONE:  # Set role for all except backbone which had been set
                    setattr(cc, 'role', FigureRoleEnum.STRUCTUREAUXILIARY)

    for cc in other_ccs:
        # ``other_ccs`` are dilated - find raw ccs in ``fig``
        fig_ccs = [fig_cc for fig_cc in fig.connected_components if  cc.contains(fig_cc)]
        # Reset roles for these - overlap with dilated structure panels, but they don't merge on dilation
        [setattr(fig_cc, 'role', None) for fig_cc in fig_ccs]


    # for structure in resolved:
    #     for cc in fig.connected_components:
    #         if cc.role != FigureRoleEnum.STRUCTUREBACKBONE and structure.contains(cc):
    #             setattr(cc, 'role', FigureRoleEnum.STRUCTUREAUXILIARY)
    #             parent_panel = [parent for parent, dilated in parent_panel_pairs if dilated == structure][0]
    #             setattr(cc, 'parent_panel', parent_panel)
    #
    # for structure, disconnected_overlapping_ccs in ambiguous:
    #     for cc in fig.connected_components:
    #         if cc.role != FigureRoleEnum.STRUCTUREBACKBONE:
    #             condition_1 = structure.contains(cc)
    #             condition_2 = lambda overlapping_cc: overlapping_cc.contains(cc)
    #
    #             if condition_1 and not (
    #             any(condition_2(overlapping_cc) for overlapping_cc in disconnected_overlapping_ccs)):
    #                 # include cc that are contained within structure and are either very small (parts of hashed bonds)
    #                 # or are glued to the structure in the dilated image
    #                 setattr(cc, 'role', FigureRoleEnum.STRUCTUREAUXILIARY)
    #                 parent_panel = [parent for parent, dilated in parent_panel_pairs if dilated == structure][0]
    #                 setattr(cc, 'parent_panel', parent_panel)

    log.info('Roles of structure auxiliaries have been assigned.')
# def find_step_reactants_and_products(fig, step_arrow, all_arrows, structures):
#     """
#     Finds reactants and products from ``structures`` of a single reaction step (around a single arrow) using
#     scanning in ``fig.img``. Scanning is terminated early if any of ``all_arrows`` is encountered
#     :param Figure fig: figure object being processed
#     :param Arrow step_arrow: Arrow object connecting the reactants and products
#     :param iterable all_arrows: a list of all arrows found
#     :param iterable structures: detected structures
#     :return: a dictionary with ``reactants``, ``products`` and ``first step`` flag
#     """
#     slope, intercept = get_line_parameters(step_arrow.line)



# def assign_to_nearest(structures, reactants, products, threshold=None):
#     """
#     This postrocessing function takes in unassigned structures and classified panels to perform a set difference.
#     It then assings the structures to the appropriate group based on the closest neighbour, subject to a threshold.
#     :param iterable structures: list of all detected structured
#     :param int threshold: maximum distance from nearest neighbour # Not used at this stage. Is it necessary?
#     # :param iterable conditions: List of conditions' panels of a reaction step # not used
#     :param iterable reactants: List of reactants' panels of a reaction step
#     :param iterable products: List of products' panels of a reaction step
#     :return dictionary: The modified groups
#     """
#
#     log.debug('Assigning connected components based on distance')
#     #print('assign: conditions set: ', conditions)
#     classified_structures = [*reactants, *products]
#     #print('diagonal lengths: ')
#     #print([cc.diagonal_length for cc in classified_ccs])
#     threshold =  0.5 * np.mean(([cc.diagonal_length for cc in classified_structures]))
#     unclassified =  [structure for structure in structures if structure not in classified_structures]
#     for cc in unclassified:
#         classified_structures.sort(key=lambda elem: elem.separation(cc))
#         nearest = classified_structures[0]
#         groups = [reactants, products]
#         for group in groups:
#             if nearest in group and nearest.separation(cc) < threshold:
#                 group.add(cc)
#                 log.info('assigning %s to group %s based on distance' % (cc, group))
#
#     return {'reactants':reactants, 'products':products}


def remove_redundant_characters(fig, ccs, chars_to_remove=None):
    """
    Removes reduntant characters such as '+' and '[', ']' from an image to facilitate resolution of diagrams and labels.
    This function takes in `ccs` which are ccs to be considered. It then closes all connected components in `fig.img1
    and compares connected components in `ccs` and closed image. This way, text characters belonging to structures are
    not considered.
    :param iterable ccs: iterable of Panels containing species to check
    :param chars_to_remove: characters to be removed
    :return: list connected components containing redundant characters
    """
    # TODO: Store closed image globally and use when needed?
    if chars_to_remove is None:
        chars_to_remove = '+[]'

    ccs_to_consider = [cc for cc in fig.connected_components if not cc.role]

    diags_to_erase = []
    for cc in ccs_to_consider:
        text_word = read_character(fig, cc)

        if text_word:
            text = text_word.text
            # print(f'recognised char: {text}')

            if any(redundant_char is text for redundant_char in chars_to_remove):
                diags_to_erase.append(cc)

    return erase_elements(fig, diags_to_erase)


def remove_redundant_square_brackets(fig, ccs):
    """
    Remove large, redundant square brackets, containing e.g. reaction conditions. These are not captured when parsing
    conditions' text (taller than a text line).
    :param Figure fig: Analysed figure
    :param [Panels] ccs: Iterable of Panels to consider for removal
    :return: Figure with square brackets removed
    """
    candidate_ccs = filter(lambda cc: cc.aspect_ratio > 5 or cc.aspect_ratio < 1 / 5, ccs)

    detected_lines = 0
    bracket_ccs = []

    # transform
    for cc in candidate_ccs:
        cc_fig = isolate_patches(fig,
                                 [cc])  # Isolate appropriate connected components in preparation for Hough
        # plt.imshow(cc_fig.img)
        # plt.show()
        line_length = (cc.width + cc.height) * 0.5  # since length >> width or vice versa, this is equal to ~0.8*length
        line = probabilistic_hough_line(cc_fig.img, line_length=int(line_length))
        if line:
            detected_lines += 1
            bracket_ccs.append(cc)

    print(bracket_ccs)
    if len(bracket_ccs) % 2 == 0:
        fig = erase_elements(fig, bracket_ccs)

    return fig


def detect_structures(fig ):
    """
    Detects structures based on parameters such as size, aspect ratio and number of detected lines

    :param Figure fig: analysed figure
    :return [Panels]: list of connected components classified as structures
    """
    ccs = fig.connected_components
    # Get a rough bond length (line length) value from the two largest structures
    ccs = sorted(ccs, key=lambda cc: cc.area, reverse=True)
    estimation_fig = skeletonize(isolate_patches(fig, ccs[:2]))
    length_scan_param = 0.02 * max(fig.width, fig.height)
    length_scan_start = length_scan_param if length_scan_param > 20 else 20
    min_line_lengths = np.linspace(length_scan_start, 3*length_scan_start, 20)
    # print(min_line_lengths)
    # min_line_lengths = list(range(20, 60, 2))
    num_lines = [(length, len(probabilistic_hough_line(estimation_fig.img, line_length=int(length), threshold=15))**2)
                    for length in min_line_lengths]
    # Choose the value where the number of lines starts to drop most rapidly and assign it as the boundary length
    (boundary_length,_), (_, _) = min(zip(num_lines, num_lines[1:]), key=lambda pair: pair[1][1] - pair[0][1])  # the key is
                                                                        # difference in number of detected lines
                                                                        # between adjacent pairs
    boundary_length = int(boundary_length)
    fig.boundary_length = boundary_length  # global estimation parameter
    # Use the length to find number of lines in each cc - this will be one of the used features
    cc_lines = []
    for cc in ccs:
        isolated_cc_fig = isolate_patches(fig, [cc])
        isolated_cc_fig = skeletonize(isolated_cc_fig)

        # angles = np.linspace(-np.pi, np.pi, 360)
        num_lines = len(probabilistic_hough_line(isolated_cc_fig.img,
                                                 line_length=boundary_length, threshold=10))
        cc_lines.append(num_lines)


    ##Case study only
    # skeleton_fig = skeletonize(fig)
    # f = plt.figure(figsize=(20, 20))
    # ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.imshow(fig.img, cmap=plt.cm.binary)
    # for line in all_lines:
    #
    #     x, y = list(zip(*line))
    #     ax.plot(x,y, 'r')
    #
    # plt.savefig('lines_structures.tif')
    # plt.show()
    ##Case study end

    cc_lines = np.array(cc_lines).reshape(-1,1)
    area = np.array([cc.area for cc in ccs]).reshape(-1, 1)
    aspect_ratio = np.array([cc.aspect_ratio for cc in ccs]).reshape(-1, 1)
    mean_area = np.mean(area)

    print(f'boundary: {boundary_length}')
    print(f'mean sqrt area: {np.sqrt(mean_area)}')

    data = np.hstack((cc_lines, area, aspect_ratio))
    # print(f'data: \n {data}')
    # print(f'data: {data}')
    data = MinMaxScaler().fit_transform(data)
    distances = np.array([(x, y, z, np.sqrt(np.sqrt(x**2 + y**2)+z**2)) for x,y,z in data])
    # print(f'transformed: \n {distances}')
    # print(f'transformed: {data}')
    # print(f'distances: {distances}')
    # data = data.clip(min=0)
    # data = cc_lines
    # print(f'data: {data}')

    labels = DBSCAN(eps=0.08, min_samples=20).fit_predict(data)

    colors = ['b', 'm', 'g', 'r']
    colors = ['b', 'm', 'g', 'r']
    paired = list(zip(ccs, labels))
    paired = [(cc, label) if cc.area > mean_area else (cc,0) for cc, label in paired]

    if False:
        f = plt.figure(figsize=(20, 20))
        ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.imshow(fig.img, cmap=plt.cm.binary)
        # ax.set_title('structure identification')
        for panel, label in paired:
            rect_bbox = Rectangle((panel.left, panel.top), panel.right-panel.left, panel.bottom-panel.top, facecolor='none',edgecolor=colors[label])
            ax.add_patch(rect_bbox)
        #plt.savefig('backbones.tif')
        plt.show()
    #

    ## TODO:  Currently it also detects arrows - filter the out (using a compound model - KMeans, another DBSCAN?)
    # ## Now exclude the aspect ratio to remove arrows
    # filtered = [_panel for _panel, label if labe == -1]
    # area = [cc.area for cc in filtered]
    # num_lines = []
    # data = data[:,:2]
    # labels = DBSCAN(eps=0.5, min_samples=2).fit_predict(data)
    # colors = ['b', 'm', 'g', 'r']
    # paired = list(zip(ccs, labels))
    # if True:
    #     f = plt.figure(figsize=(20, 20))
    #     ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    #     ax.imshow(fig.img, cmap=plt.cm.binary)
    #     ax.set_title('filtered')
    #     # ax.set_title('structure identification')
    #     for _panel, label in paired:
    #         rect_bbox = Rectangle((_panel.left, _panel.top), _panel.right-_panel.left, _panel.bottom-_panel.top, facecolor='none',edgecolor=colors[label])
    #         ax.add_patch(rect_bbox)
    #     #plt.savefig('backbones.tif')
    #     plt.show()
    structures = [panel for panel, label in paired if label == -1]
    structures = [panel for panel in structures if panel.aspect_ratio + 1/panel.aspect_ratio < 5]  # Remove long lines
    [setattr(structure, 'role', FigureRoleEnum.STRUCTUREBACKBONE) for structure in structures]

    return structures





    # # There are only two possible classes here: structures and text - arrows are excluded (for now?)
    # size = np.asarray([cc.area**2 for cc in ccs], dtype=float)
    # aspect_ratio = [cc.aspect_ratio for cc in ccs]
    # aspect_ratio = np.asarray([ratio + 1 / ratio for ratio in aspect_ratio],
    #                           dtype=float)  # Transform to weigh wide
    # print('aspect ratio: \n', aspect_ratio)
    # plt.hist(aspect_ratio)
    # plt.show()
    #
    # # and tall structures equally (as opposed to ratios around 1)
    # pixel_ratios = np.asarray([pixel_ratio(fig, cc) for cc in ccs])
    # data = np.vstack((size, aspect_ratio, pixel_ratios))
    # data = np.transpose(data)
    # print(np.mean(data, axis=0))
    # print(np.std(data, axis=0))
    # data -= np.mean(data, axis=0)
    # data /= np.std(data, axis=0)
    # data[:,2] = np.power(data[:, 2], 3)
    # # data[:, 2] = np.power(data[:, 2], 3)
    # f, ax = plt.subplots(2,2)
    # print(sorted(data[:,0]))
    # ax[0,0].scatter(data[:,0],data[:,1])
    # ax[0,1].scatter(data[:,1], data[:,2])
    # ax[1,0].scatter(data[:,0], data[:,2])
    # plt.show()
    # # print('size: \n', size)
    #
    # # print('pixel ratio: \n', pixel_ratio)
    # #
    # print(f'data:')
    # # print(data)
    # # print(data.shape)
    # #labels = KMeans(n_clusters=2, n_init=20).fit_predict(data)
    # eps = np.sum(np.std(data, axis=0))
    #
    # neigh = NearestNeighbors(n_neighbors=2)
    # nbrs = neigh.fit(data)
    # distances, indices = nbrs.kneighbors(data)
    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # print('distances: \n', distances)
    # _, bins, _ = plt.hist(distances)
    # plt.show()
    # eps = bins[1]
    # labels = DBSCAN(min_samples=5,eps=eps).fit_predict(data)
    # #labels = OPTICS().fit_predict(data)
    # print(labels)
    # return ccs, labels


def extend_line(line, extension=None):
    """
    Extends line in both directions. Output is a pair of points, each of which is further from an arrow (closer to
    reactants or products in the context of reactions).
    :param Line line: original Line object
    :param int extension: value dictated how far the new line should extend in each direction
    :return: two endpoints of a new line
    """

    if line.is_vertical:  # vertical line
        line.pixels.sort(key=lambda point: point.row)
        first_line_pixel = line.pixels[0]
        last_line_pixel = line.pixels[-1]
        if extension is None:
            extension = int((last_line_pixel.separation(first_line_pixel)) * 0.4)

        left_extended_point = Point(row=first_line_pixel.row - extension, col=first_line_pixel.col)
        right_extended_point = Point(row=last_line_pixel.row + extension, col=last_line_pixel.col)



    else:
        line.pixels.sort(key=lambda point: point.col)

        first_line_pixel = line.pixels[0]
        last_line_pixel = line.pixels[-1]
        if extension is None:
            extension = int((last_line_pixel.separation(first_line_pixel)) * 0.4)

        if line.slope == 0:
            left_extended_last_y = line.slope * (first_line_pixel.col - extension) + first_line_pixel.row
            right_extended_last_y = line.slope * (last_line_pixel.col + extension) + first_line_pixel.row

        else:
            left_extended_last_y = line.slope*(first_line_pixel.col-extension) + line.intercept
            right_extended_last_y = line.slope*(last_line_pixel.col+extension) + line.intercept

        left_extended_point = Point(row=left_extended_last_y, col=first_line_pixel.col-extension)
        right_extended_point = Point(row=right_extended_last_y, col=last_line_pixel.col+extension)

    # extended = approximate_line(first_line_pixel, left_extended_point) +\
    #            approximate_line(last_line_pixel, right_extended_point) + line
    #
    #
    # new_line = Line(extended)

    return (left_extended_point, right_extended_point)


def find_nearby_ccs(start, all_relevant_ccs, distances, role=None, condition=(lambda cc: True)):
    """
    Find all structures close to ``start`` position. All found structures are added to a queue and
    checked again to form a cluster of nearby structures.
    :param Point or (x,y) start: point where the search starts
    :param [Panel,...] all_relevant_ccs: list of all found structures
    :param type role: class specifying role of the ccs in the scheme (e.g. Diagram, Conditions)
    :param (float, lambda) distances: a tuple (maximum_initial_distance, distance_function) which specifies allowed
    distance from the starting point and a function defining cut-off distance for subsequent reference ccs
    :param lambda condition: optional condition to decide whether a connected component should be added to the frontier
                                                                                                                or not.
    :return: List of all nearby structures
    """
    max_initial_distance, distance_fn = distances
    frontier = []
    frontier.append(start)
    found_ccs = []
    visited = set()
    while frontier:
        reference = frontier.pop()
        visited.add(reference)
        # max_distance = max_initial_distance + 1.5 * np.sqrt(reference.area) \
        #                if isinstance(reference, Panel) else max_initial_distance
        max_distance = distance_fn(reference) if isinstance(reference, Panel) else max_initial_distance
        successors = [cc for cc in all_relevant_ccs if cc.separation(reference) < max_distance
                      and cc not in visited and condition(cc)]
        new_structures = [structure for structure in successors if structure not in found_ccs]
        frontier.extend(successors)
        found_ccs.extend(new_structures)

    if role is not None:
        [setattr(cc, 'role', role) for cc in found_ccs if not getattr(cc, 'role')]

    return found_ccs


def get_conditions_smiles(csr_output, conditions):
    """
    Appends structures found in the conditions region and transformed into smiles in ``cst_output`` to ``conditions``.

    :param (recognised, diagrams) csr_output: tuple containing a (recognised_structures, diagrams) pair of sequences
    :param [Conditions,...]  conditions: list of all Conditions object in an image
    """
    fig = settings.main_figure[0]
    conditions_structures = set([cc.parent_panel for cc in fig.connected_components if cc.parent_panel
                                 and cc.parent_panel.role == ReactionRoleEnum.CONDITIONS])

    if not conditions_structures:
        log.debug('Conditions remain unchanged after matching conditions smiles.')
        return conditions

    recognised, diagrams = csr_output
    for structure in conditions_structures:
        for idx, diag in enumerate(diagrams):
            if structure == Panel(diag.left, diag.right, diag.top, diag.bottom):
                label, smiles = recognised[idx]
                log.info('SMILES string was matched to a structure in the conditions region')

                closest_step_conditions = min(conditions, key=lambda step_conditions:
                                              structure.separation(step_conditions.anchor))
                closest_step_conditions.conditions_dct['other species'].append(smiles)

    return conditions
